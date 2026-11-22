"""Numerically trace the two residual-plus-LayerNorm equations in iTransformer Algorithm 1."""

from dataclasses import dataclass

import torch
from torch.overrides import TorchFunctionMode


@dataclass(eq=False)
class Node:
    op: str
    tensor: torch.Tensor
    inputs: tuple


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        yield from _tensors(tuple(value.values()))


_AFFINE = {"linear", "conv1d", "mm", "matmul", "bmm", "addmm"}
_ACTIVATION = {"gelu", "relu", "relu_", "leaky_relu", "silu", "sigmoid", "tanh", "elu", "mish"}
_VIEWS = {"contiguous", "clone", "reshape", "view", "transpose", "permute", "detach", "to"}
_NORMS = {"layer_norm", "native_layer_norm"}


class _Trace(TorchFunctionMode):
    def __init__(self, width, variables):
        self.width = width
        self.variables = variables
        self.nodes = {}
        self.norms = []

    def node(self, tensor):
        key = id(tensor)
        if key not in self.nodes:
            self.nodes[key] = Node("input", tensor, ())
        return self.nodes[key]

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        inputs = tuple(self.node(tensor) for tensor in _tensors((args, kwargs)))
        result = func(*args, **kwargs)
        name = func.__name__
        if name in {"scaled_dot_product_attention", "multi_head_attention_forward"}:
            name = "attention"
        if name in {"softmax", "_softmax"} and inputs and inputs[0].tensor.shape[-2:] == (
                self.variables, self.variables):
            name = "attention"
        if name in _NORMS:
            shape = args[1] if len(args) > 1 else kwargs["normalized_shape"]
            is_token_norm = tuple(shape) == (self.width,) and inputs[0].tensor.ndim >= 2
            main = result[0] if isinstance(result, tuple) else result
        for out in _tensors(result):
            # Eval-mode dropout and contiguous tensors often return the identical object.
            # Keep the old dependency in that case; an in-place add really changes it.
            if id(out) not in self.nodes or name in {"add_", "add"}:
                self.nodes[id(out)] = Node(name, out, inputs)
        if name in _NORMS and is_token_norm:
            self.norms.append((inputs[0], self.node(main)))
        return result


def _ancestry(start, stop=None):
    found = set()
    queue = [start]
    while queue:
        node = queue.pop()
        if node not in found:
            found.add(node)
            if node is not stop:
                queue.extend(node.inputs)
    return found


def _same(a, b):
    return a.shape == b.shape and torch.allclose(a.double(), b.double(), rtol=1e-5, atol=2e-6)


def _unwrap(node):
    while node.op in _VIEWS and node.inputs and _same(node.tensor, node.inputs[0].tensor):
        node = node.inputs[0]
    return node


def _sublayer(norm_input, previous, kind):
    """Compare the actual norm input with H + branch(H) at the observed execution."""
    summed = _unwrap(norm_input)
    if summed.op not in {"add", "add_"} or len(summed.inputs) != 2:
        # Unrecognized fused operations might implement the exact sum internally.
        branch = summed
        while branch.op in _VIEWS and branch.inputs:
            branch = branch.inputs[0]
        if branch.op in {"linear", "conv1d", "attention"} and not any(
                node.op in {"cat", "stack"} for node in branch.inputs):
            return "failed", f"LayerNorm read the direct {branch.op} output without a residual addition."
        return "inconclusive", f"Could not identify a residual addition before LayerNorm ({summed.op})."
    if summed.op == "add_":
        return "inconclusive", "The residual was added in place; its old numeric operands were not preserved."
    alternatives = []
    for skip, branch in (summed.inputs, tuple(reversed(summed.inputs))):
        if previous is not None and not _same(skip.tensor, previous.tensor):
            continue
        ancestors = _ancestry(branch, skip)
        if skip not in ancestors:
            continue
        ops = [n.op for n in ancestors if n is not skip]
        has_branch = ("attention" in ops if kind == "attention" else
                      sum(op in _AFFINE for op in ops) >= 2 and bool(set(ops) & _ACTIVATION))
        if has_branch and not set(ops) & _NORMS:
            alternatives.append((skip, branch))
    if not alternatives:
        if previous is not None and not any(_same(node.tensor, previous.tensor) for node in summed.inputs):
            return "failed", "Neither residual operand equals the preceding LayerNorm output."
        return "inconclusive", f"Could not establish a shared residual input and a distinct {kind} branch."
    for skip, branch in alternatives:
        if skip.tensor.shape == branch.tensor.shape == norm_input.tensor.shape:
            expected = skip.tensor.double() + branch.tensor.double()
            if _same(norm_input.tensor, expected):
                return "passed", "LayerNorm input matches the skip tensor plus the separately traced branch output."
    return "failed", "The observed LayerNorm input disagrees with the independent unscaled sum."


def check_residual_order(model, x, config):
    """Return a JSON-ready method-check row, without changing the model or training data."""
    row = {"criterion": "encoder_residual_order", "status": "inconclusive", "layers": []}
    trace = _Trace(config["d_model"], x.shape[-1])
    # The fused Transformer fast path hides its layer norms. Disabling it for this
    # synthetic pass exposes the public eager implementation (including built-in MHA).
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad(), trace:
            forecast = model(x)
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        row["detail"] = f"Could not inspect the eager computation: {type(exc).__name__}: {exc}"
        return row
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    output_path = _ancestry(trace.node(forecast))
    norms = [(a, b) for a, b in trace.norms if b in output_path]
    expected = config["e_layers"] * 2
    row["observed_token_norms"] = len(norms)
    if len(norms) not in {expected, expected + 1}:
        row["detail"] = f"Could not isolate {expected} encoder sublayer norms on the path to the forecast."
        return row
    # Allow a final encoder-wide LayerNorm after all complete residual blocks.
    previous = None
    for index in range(0, expected, 2):
        layer = {}
        for j, name in enumerate(("attention", "ffn")):
            norm_input, norm_output = norms[index + j]
            status, detail = _sublayer(norm_input, previous, name)
            layer[name] = {"status": status, "detail": detail}
            previous = norm_output
        row["layers"].append(layer)
    statuses = {branch["status"] for layer in row["layers"] for branch in layer.values()}
    row["status"] = "failed" if "failed" in statuses else "inconclusive" if "inconclusive" in statuses else "passed"
    row["detail"] = "Checked both post-residual normalization inputs in each observed block on a synthetic forward pass."
    return row
