"""Compare pre-training gates on labeled edits of the frozen Traffic implementation."""

import argparse
import importlib.util
import json
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from experiment_runner import preflight, read_series, sha256, write_json
from method_checks import check_model
from task_input import load_task


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "runs/traffic96_001"
SDPA = """attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )"""


@dataclass(frozen=True)
class Case:
    name: str
    label: str
    description: str
    # Each source fragment must match exactly once in the archived generation.
    edits: tuple = ()


CASES = (
    Case("archived_generation", "correct", "Unchanged archived generator output; positive control for the selected mechanisms."),
    Case("explicit_attention", "correct", "Write scaled softmax attention with tensor operations in place of SDPA.",
         ((SDPA, """scores = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = F.dropout(scores.softmax(-1), p=self.attention_dropout, training=self.training)
        attended = weights @ value"""),)),
    Case("renamed_projections", "correct", "Rename both attention projections; preserve weights and operations.",
         (("self.qkv =", "self.input_map ="), ("self.qkv(tokens)", "self.input_map(tokens)"),
          ("self.output_projection =", "self.merge ="), ("self.output_projection(attended)", "self.merge(attended)"))),
    Case("no_variable_mixing", "incorrect", "Prior probe: pass values through and ignore query/key.",
         ((SDPA, "attended = value"),)),
    Case("attend_over_heads", "incorrect", "Prior probe: swap head and variable axes around SDPA.",
         ((SDPA, """attended = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            dropout_p=self.attention_dropout if self.training else 0.0
        ).transpose(1, 2)"""),)),
    Case("causal_sensor_mask", "incorrect", "Apply a triangular mask following sensor order.",
         (("is_causal=False,", "is_causal=True,"),)),
    Case("unscaled_dot_product", "incorrect", "Remove division by sqrt(head dimension).",
         (("is_causal=False,", "is_causal=False, scale=1.0,"),)),
    Case("uniform_attention", "incorrect", "Average variable values independently of query and key.",
         ((SDPA, "attended = value.mean(dim=2, keepdim=True).expand_as(value)"),)),
    Case("missing_window_mean", "incorrect", "Omit the historical mean when restoring the forecast scale.",
         (("prediction.to(dtype=mean.dtype) * scale + mean", "prediction.to(dtype=mean.dtype) * scale"),)),
    Case("missing_ffn_residual", "incorrect", "Drop the feedforward residual required by Algorithm 1.",
         (("tokens + self.ffn_residual_dropout(self.ffn(tokens))", "self.ffn_residual_dropout(self.ffn(tokens))"),)),
    Case("truncate_forecast", "incorrect", "Return one fewer forecast step than requested.",
         (("return prediction.contiguous()", "return prediction[:, :-1].contiguous()"),)),
    Case("nonfinite_forecast", "incorrect", "Return NaN forecasts of the expected shape.",
         (("return prediction.contiguous()", "return prediction * float('nan')"),)),
)

# Added after reviewing the first comparison. Keep their denominator separate.
ADDITIONAL_CONTROLS = (
    Case("chunked_queries", "correct", "Split Q along the variable axis, attend to every K/V in each chunk, concatenate.",
         ((SDPA, """attended = torch.cat([F.scaled_dot_product_attention(
            q_part, key, value, dropout_p=self.attention_dropout if self.training else 0.0
        ) for q_part in query.split(3, dim=-2)], dim=-2)"""),)),
    Case("prescale_query", "correct", "Apply the scale to Q before SDPA and request unit scale inside SDPA.",
         ((SDPA, """attended = F.scaled_dot_product_attention(
            query / (self.head_dim ** 0.5), key, value, scale=1.0,
            dropout_p=self.attention_dropout if self.training else 0.0)"""),)),
)


def edit_source(original, case):
    source = original
    for before, after in case.edits:
        if source.count(before) != 1:
            raise ValueError(f"{case.name}: edit no longer matches the frozen model exactly once")
        source = source.replace(before, after)
    return source


def load_model(path, config):
    torch.manual_seed(config["seed"])
    module_name = "smoke_model_" + sha256(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.Forecaster(config).eval()


def smoke(model, series, config):
    try:
        return preflight(model, series, config, torch.device("cpu"))
    except (RuntimeError, ValueError, TypeError, IndexError, AttributeError) as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


@torch.inference_mode()
def verify_fixtures(reference, candidate, inputs, case):
    """Check our edit actually changes predictions (or preserves them for a control)."""
    same = []
    for x in inputs:
        left, right = reference(x), candidate(x)
        same.append(left.shape == right.shape and torch.allclose(left, right, atol=3e-6, rtol=2e-5))
    if all(same) != (case.label == "correct"):
        raise ValueError(f"{case.name}: intervention does not match fixture expectation")
    return same


def summarize(rows, gate):
    matrix = {label: Counter(row[gate]["status"] for row in rows if row["label"] == label)
              for label in ("correct", "incorrect")}
    return {
        "correct_accepted": matrix["correct"]["passed"], "false_rejects": matrix["correct"]["failed"],
        "correct_inconclusive": matrix["correct"]["inconclusive"],
        "incorrect_rejected": matrix["incorrect"]["failed"], "false_accepts": matrix["incorrect"]["passed"],
        "incorrect_inconclusive": matrix["incorrect"]["inconclusive"],
    }


def compare(card, *, controls_only=False, protocol_version=2):
    files = [ROOT / "method_checks.py", Path(__file__), ROOT / "experiment_runner.py"]
    if protocol_version == 2:
        files.append(ROOT / "residual_check.py")
    code_hashes = {p.name: sha256(p) for p in files}
    config_path = FROZEN / "config.json"
    source_path = FROZEN / "model.py"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = source_path.read_text(encoding="utf-8")
    torch.set_num_threads(config["threads"])
    task = load_task(card)
    history = config["seq_len"]
    # Keep just the first training window plus its target for all paired cases.
    first_window = read_series(task, config, "train")[:history + config["pred_len"]].clone()
    generator = torch.Generator().manual_seed(4309)
    controls = [torch.randn(2, history, n, generator=generator) for n in (5, 9)]
    reference = load_model(source_path, config)
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for case in ADDITIONAL_CONTROLS if controls_only else CASES + ADDITIONAL_CONTROLS:
            variant = Path(directory) / (case.name + ".py")
            variant.write_text(edit_source(source, case), encoding="utf-8", newline="\n")
            model = load_model(variant, config)
            same = verify_fixtures(reference, model, controls, case)
            baseline = smoke(model, first_window, config)
            if baseline["status"] == "passed":
                strengthened = check_model(variant, config, protocol_version=protocol_version)
            else:
                strengthened = {"status": "failed", "reason": "Failed the common real-data smoke gate."}
            if strengthened["status"] not in {"passed", "failed", "inconclusive"}:
                raise ValueError("Unknown method check status")
            rows.append({"case": case.name, "label": case.label, "intervention": case.description,
                         "variant_sha256_lf": sha256(variant), "same_as_archived_on_control_inputs": same,
                         "baseline": baseline, "strengthened": strengthened})
    extra_names = {case.name for case in ADDITIONAL_CONTROLS}
    primary = [row for row in rows if row["case"] not in extra_names]
    extra = [row for row in rows if row["case"] in extra_names]
    if any(sha256(p) != code_hashes[p.name] for p in files):
        raise RuntimeError("Evaluation code changed during the comparison; rerun on the finalized files.")
    return {
        "task_id": task.task_id, "torch": str(torch.__version__), "fixture_source_sha256": sha256(source_path),
        "configuration_sha256": sha256(config_path), "traffic_sha256": sha256(task.data_path),
        "code_sha256": code_hashes,
        "protocol_version": protocol_version,
        "gate_calibration": ("Initial criteria, prior to the comparison."
                             if protocol_version == 1 else
                             "Revised after observations on the primary faults and the two additional equivalents; "
                             "results on these cases are regression evidence."),
        "seed_for_initial_weights": config["seed"], "fixture_control_seed": 4309,
        "real_smoke_input_shape": list(first_window[:history].unsqueeze(0).shape),
        "baseline": "Archived fit() pre-training smoke: first real training window, shape, floating dtype, finite.",
        "strengthened": "Same smoke gate, then deterministic generated-model method checks.",
        "scope": "Constructed source variants of one architecture, labeled for the selected requirements. "
                 "Two attention faults were known from the earlier probe. Controls are numerically checked on "
                 "two synthetic inputs with equal initial weights. No training, validation or test scoring.",
        "summary": {gate: summarize(primary, gate) for gate in ("baseline", "strengthened")}, "cases": primary,
        "additional_equivalent_implementations": {
            "scope": "Two mathematically equivalent implementations added after the initial 12-case comparison.",
            "summary": {gate: summarize(extra, gate) for gate in ("baseline", "strengthened")}, "cases": extra,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, default=ROOT / "examples/itransformer_traffic/task.json")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--controls-only", action="store_true", help="Run the two additional positive controls only.")
    parser.add_argument("--protocol-version", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    report = compare(args.card, controls_only=args.controls_only, protocol_version=args.protocol_version)
    output = args.out or ROOT / "evidence" / ("quality_positive_controls.json" if args.controls_only
                                               else "quality_comparison.json")
    write_json(output, report)
    if not args.controls_only:
        print(json.dumps(report["summary"], indent=2))
    print(json.dumps(report["additional_equivalent_implementations"]["summary"], indent=2))


if __name__ == "__main__":
    main()
