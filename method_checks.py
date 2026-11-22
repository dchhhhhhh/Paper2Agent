"""Synthetic method checks for the Traffic iTransformer reproduction, independent of labels and checkpoints."""

import argparse
import hashlib
import itertools
import json
import math
import sys
import types
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode


SEED = 7631


def entry(criterion, status, detail, **evidence):
    return dict(criterion=criterion, status=status, detail=detail, **evidence)


def reference(q, k, v):
    """Evaluate each head independently in float64, with no mask or dropout."""
    scores = q.double() @ k.double().transpose(-1, -2) / math.sqrt(q.shape[-1])
    return scores.softmax(dim=-1) @ v.double()


def matches(actual, expected):
    return torch.allclose(actual.double(), expected.double(), rtol=2e-4, atol=2e-5)


class AttentionTrace(TorchFunctionMode):
    """Observe SDPA calls (including imported aliases) without replacing their output."""

    def __init__(self, variables, width):
        self.variables = variables
        self.width = width
        self.calls = []

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if func is F.scaled_dot_product_attention:
            operands = [args[i] if i < len(args) else kwargs[key]
                        for i, key in enumerate(("query", "key", "value"))]
            q, k, v = operands
            shape_ok = all(t.shape == q.shape for t in (k, v)) and q.ndim >= 3
            tokens_ok = shape_ok and q.shape[-2] == self.variables
            width_ok = shape_ok and q.shape[-1] == self.width
            heads = math.prod(q.shape[:-2]) if shape_ok else 0
            formula_ok = tokens_ok and width_ok and matches(result, reference(q, k, v))
            self.calls.append(dict(query_shape=list(q.shape), key_shape=list(k.shape), tokens_ok=tokens_ok, width_ok=width_ok,
                                   formula_ok=formula_ok, batch_heads=heads))
        return result


class HistoryReductions(TorchFunctionMode):
    """Evidence of an optional history-level transform when a model declares no flag."""

    def __init__(self, history):
        self.history = history
        self.operations = set()

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        if func.__name__ in {"mean", "var", "std", "var_mean", "std_mean"}:
            tensor = args[0] if args else kwargs.get("input")
            dim = args[1] if len(args) > 1 else kwargs.get("dim")
            if isinstance(dim, (list, tuple)) and len(dim) == 1:
                dim = dim[0]
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 3 and type(dim) is int:
                if dim % tensor.ndim != 0 and tensor.shape[dim] == self.history:
                    self.operations.add(func.__name__)
        return result


def count_projection_matches(records, x, config):
    """Check packed or separate Q/K/V Linear projections before the output map.

    This also recognizes explicit matmul/softmax implementations. A new packing
    convention that does not match any candidate supplies no positive evidence.
    """
    batch, _, variables = x.shape
    width, heads = config["d_model"], config["n_heads"]
    token_shape = (batch, variables, width)
    candidates = []
    separate = {}
    for i, (inputs, projected) in enumerate(records):
        if tuple(inputs.shape) != token_shape:
            continue
        if tuple(projected.shape) == (batch, variables, 3 * width):
            major = projected.reshape(batch, variables, 3, heads, -1).unbind(2)
            interleaved = projected.reshape(batch, variables, heads, 3, -1).unbind(3)
            candidates.append((i, (major, interleaved)))
        if tuple(projected.shape) == token_shape:
            separate.setdefault(id(inputs), []).append((i, projected.reshape(batch, variables, heads, -1)))
    for group in separate.values():
        if len(group) == 3:
            candidates.append((group[-1][0], ([projection for _, projection in group],)))

    # A layer must match a distinct subsequent output-projection input.
    consumed = set()
    for pos, layouts in sorted(candidates, key=lambda item: item[0]):
        possible = [(i, inputs) for i, (inputs, _) in enumerate(records) if i > pos
                    and i not in consumed and tuple(inputs.shape) == token_shape]
        found = False
        for layout in layouts:
            for q, k, v in itertools.permutations(layout):
                q, k, v = (t.transpose(1, 2) for t in (q, k, v))
                merged = reference(q, k, v).transpose(1, 2).reshape(token_shape)
                for i, actual in possible:
                    if matches(actual, merged):
                        consumed.add(i)
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return len(consumed)


def check_builtin_mha(layer, args, kwargs, result, batch, variables, config):
    actual = result[0]
    operands = [args[i] if i < len(args) else kwargs[name] for i, name in enumerate(("query", "key", "value"))]
    if not layer.batch_first:
        actual = actual.transpose(0, 1)
        operands = [value.transpose(0, 1) for value in operands]
    width, heads = config["d_model"], config["n_heads"]
    if any(t.shape != (batch, variables, width) for t in operands) or layer.num_heads != heads:
        return False
    weights = (layer.in_proj_weight.chunk(3) if layer.in_proj_weight is not None
               else (layer.q_proj_weight, layer.k_proj_weight, layer.v_proj_weight))
    biases = layer.in_proj_bias.chunk(3) if layer.in_proj_bias is not None else (None,) * 3
    per_head = []
    for tensor, weight, bias in zip(operands, weights, biases):
        proj = F.linear(tensor.double(), weight.double(), None if bias is None else bias.double())
        per_head.append(proj.reshape(batch, variables, heads, -1).transpose(1, 2))
    combined = reference(*per_head).transpose(1, 2).reshape(batch, variables, width)
    output = layer.out_proj
    expected = F.linear(combined, output.weight.double(), None if output.bias is None else output.bias.double())
    return matches(actual, expected)


def trace_forward(model, x, config, version):
    records = []
    builtin_matches = []
    trace = AttentionTrace(x.shape[-1], config["d_model"] // config["n_heads"])
    def record(layer, args, kwargs, output):
        records.append((args[0] if args else kwargs["input"], output))
    def record_mha(layer, args, kwargs, output):
        builtin_matches.append(check_builtin_mha(layer, args, kwargs, output, x.shape[0], x.shape[-1], config))
    with ExitStack() as stack:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                stack.callback(module.register_forward_hook(record, with_kwargs=True).remove)
            if isinstance(module, nn.MultiheadAttention) and version == 2:
                stack.callback(module.register_forward_hook(record_mha, with_kwargs=True).remove)
        with trace:
            forecast = model(x)
    layers = config["e_layers"]
    heads = config["n_heads"]
    projected = count_projection_matches(records, x, config)
    evidence = dict(expected_layers=layers, matched_projection_inputs=projected, builtin_mha=builtin_matches,
                    sdpa_calls=trace.calls)
    if version == 2 and (projected == layers or len(builtin_matches) == layers and all(builtin_matches)):
        # Chunking queries and moving the scaling outside SDPA can change all
        # local call shapes/values while leaving complete per-layer outputs exact.
        verdict = entry("variable_attention", "passed",
                        "The full projection-level numerical reference matched every configured layer.", **evidence)
    elif (version == 2 and any(c["query_shape"] != c["key_shape"] for c in trace.calls)
          and all(builtin_matches)):
        verdict = entry("variable_attention", "inconclusive",
                        "Attention used asymmetric operands. The full projection did not establish equivalence.", **evidence)
    elif not all(builtin_matches) or any(not c["formula_ok"] for c in trace.calls):
        verdict = entry("variable_attention", "failed",
                        "Observed an attention axis or output inconsistent with unmasked scaled attention over variables.",
                        **evidence)
    elif (len(builtin_matches) == layers or projected == layers or (trace.calls and sum(c["batch_heads"] for c in trace.calls)
                                  == x.shape[0] * heads * layers)):
        verdict = entry("variable_attention", "passed",
                        "The numerical reference matched for the configured attention depth and variable count.", **evidence)
    else:
        verdict = entry("variable_attention", "inconclusive",
                        "Insufficient independent numerical evidence for every configured layer; review this implementation.",
                        **evidence)
    return forecast, verdict


def valid_output(forecast, x, horizon):
    return (isinstance(forecast, torch.Tensor) and forecast.is_floating_point()
            and list(forecast.shape) == [x.shape[0], horizon, x.shape[-1]]
            and bool(torch.isfinite(forecast).all()))


def finite_measurement(value):
    number = float(value)
    return number if math.isfinite(number) else None


def check_behavior(model, config, version):
    history, horizon = config["seq_len"], config["pred_len"]
    # Distinguish token, batch and head axes. Each call changes the number of
    # available sensors, as required by the training/inference protocol.
    variables = min(n for n in range(7, 20) if n not in (config["n_heads"], history, horizon))
    x = torch.randn(3, history, variables)
    original = model(x)
    contracted = model(x[:, :, :variables - 2].contiguous())
    good = valid_output(original, x, horizon) and valid_output(contracted, x[:, :, :variables - 2], horizon)
    checks = [entry("output_contract", "passed" if good else "failed",
                    "Checked finite floating forecasts at two supplied sensor counts.",
                    input_shape=list(x.shape), expected_horizon=horizon, sensor_counts=[variables, variables - 2])]
    if not good:
        return checks
    try:
        _, attention = trace_forward(model, x, config, version)
    except Exception as exc:
        attention = entry("variable_attention", "inconclusive",
                          f"Normal forward passed; attention inspection unavailable: {type(exc).__name__}: {exc}")
    checks.append(attention)
    if version == 2:
        from residual_check import check_residual_order
        try:
            checks.append(check_residual_order(model, x, config))
        except Exception as exc:
            checks.append(entry("encoder_residual_order", "inconclusive",
                                f"Normal forward passed; block inspection unavailable: {type(exc).__name__}: {exc}"))
    perm = torch.arange(variables - 1, -1, -1)
    actual = model(x[:, :, perm])
    expected = original[:, :, perm]
    checks.append(entry("sensor_permutation", "passed" if matches(actual, expected) else "failed",
                        "Reorder the input sensors and require the same reordering in the forecasts.",
                        max_abs_error=finite_measurement((actual - expected).abs().max())))
    if version == 2:
        changed_batch = x.clone()
        changed_batch[0] = torch.randn_like(x[0])
        unchanged_samples = model(changed_batch)[1:]
        checks.append(entry("batch_independence", "passed" if matches(unchanged_samples, original[1:]) else "failed",
                            "Changing one window must leave the other samples' forecasts unchanged.",
                            max_abs_error=finite_measurement((unchanged_samples - original[1:]).abs().max())))

    # Change the temporal pattern of one sensor, not just its level, because
    # per-window normalization intentionally removes levels before attention.
    perturbed = x.clone()
    perturbed[:, :, 0] = x[:, :, 0].roll(1, dims=1)
    def cross_effect(forecast):
        return float((model(perturbed)[:, :, 1:] - forecast[:, :, 1:]).abs().max())
    effect = cross_effect(original)
    if effect <= 1e-7:
        # Valid forecasters can deliberately zero-initialize an output head or
        # residual gate. Exercise trainable zero tensors once at nonzero values.
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad and not bool(parameter.count_nonzero()):
                    parameter.normal_(0.0, 0.05)
        effect = cross_effect(model(x))
        original = model(x)
    checks.append(entry("cross_sensor_effect", "passed" if math.isfinite(effect) and effect > 1e-7 else "failed",
                        "Changing one sensor's temporal pattern must reach at least one other forecast.",
                        max_effect_on_other_sensors=finite_measurement(effect)))

    norm = config.get("use_norm", getattr(model, "use_norm", None)) if version == 2 else config.get("use_norm", True)
    if norm is not False:
        offset = torch.randn(x.shape[0], 1, variables) * 3
        scale = torch.linspace(0.75, 1.75, variables).reshape(1, 1, -1)
        actual = model(x * scale + offset)
        expected = original * scale + offset
        ok = torch.allclose(actual, expected, rtol=4e-4, atol=2e-4)
        if norm is None and not ok:
            observer = HistoryReductions(history)
            try:
                with observer:
                    model(x)
            except Exception as exc:
                checks.append(entry("history_scale_restore", "inconclusive",
                                    f"Optional normalization could not be inspected: {type(exc).__name__}: {exc}"))
            else:
                if observer.operations:
                    checks.append(entry("history_scale_restore", "inconclusive",
                                        "History statistics were computed without a declared normalization flag; "
                                        "verify whether the forecasts should restore the sensor scales.",
                                        observed_reductions=sorted(observer.operations),
                                        max_abs_error=finite_measurement((actual - expected).abs().max())))
                else:
                    checks.append(entry("history_scale_restore", "not_required",
                                        "Optional normalization was not declared and no history reductions were observed."))
        else:
            checks.append(entry("history_scale_restore", "passed" if ok else "failed",
                                "Change each sensor's offset/scale independently and check its restoration.",
                                max_abs_error=finite_measurement((actual - expected).abs().max())))
    else:
        checks.append(entry("history_scale_restore", "not_required", "Per-window normalization explicitly disabled."))
    return checks


def check_model(path, config, card=None, protocol_version=2):
    """Audit one generated Forecaster. Unsupported attention needs independent review."""
    path = Path(path).resolve()
    if protocol_version not in (1, 2):
        raise ValueError("Unknown method-check protocol version")
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text(encoding="utf-8-sig"))
    for name in ("d_model", "n_heads", "e_layers", "seq_len", "pred_len"):
        if type(config.get(name)) is not int or config[name] <= 0:
            raise ValueError(f"Expected positive integer config value for {name}")
    if config["d_model"] % config["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    if card is not None:
        from experiment_runner import real_preflight
        from task_input import load_task
        task = load_task(Path(card))
    source = path.read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    module = types.ModuleType("p2a_method_check_" + sha)
    module.__file__ = str(path)
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.random.default_generator.manual_seed(SEED)
            torch.set_num_threads(1)
            # Execute precisely the hashed bytes; repair attempts may have the
            # same size and mtime as the preceding version in a Python cache.
            checks = []
            try:
                exec(compile(source, str(path), "exec"), module.__dict__)
                model = module.Forecaster(dict(config)).cpu().eval()
            except Exception as exc:
                checks.append(entry("model_execution", "failed", f"{type(exc).__name__}: {exc}"))
            else:
                if card is not None:
                    real = real_preflight(model, task, config)
                    checks.append(entry("real_training_window", real["status"], real.get("reason") or
                                        "Checked forecast dimensions and finiteness on a real training window.",
                                        evidence=real))
                try:
                    checks.extend(check_behavior(model, config, protocol_version))
                except Exception as exc:
                    checks.append(entry("model_execution", "failed", f"{type(exc).__name__}: {exc}"))
    finally:
        torch.set_num_threads(threads)
        if previous is None:
            sys.modules.pop(module.__name__, None)
        else:
            sys.modules[module.__name__] = previous
    statuses = {record["status"] for record in checks}
    status = "failed" if "failed" in statuses else "inconclusive" if "inconclusive" in statuses else "passed"
    return dict(passed=status == "passed", status=status, checks=checks, model_sha256=sha,
                seed=SEED, protocol_version=protocol_version,
                input="real training window and synthetic" if card is not None else "synthetic",
                torch=str(torch.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "config", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--card", type=Path, help="Also run the ordinary real-training-window preflight")
    parser.add_argument("--protocol-version", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    result = check_model(args.model, args.config, args.card, args.protocol_version)
    temp = args.report.with_suffix(args.report.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.report)
    print(result["status"])


if __name__ == "__main__":
    main()
