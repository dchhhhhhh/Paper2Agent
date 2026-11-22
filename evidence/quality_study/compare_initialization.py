"""Compare the selected candidates before training on a fixed synthetic batch."""

import json
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from compare_method_checks import load_model
from run_experiment import digest, save


if __name__ == "__main__":
    config = json.loads((ROOT/"examples/itransformer_traffic/cpu_first_run.json").read_text(encoding="utf8"))
    torch.set_num_threads(config["threads"])
    paths = [ROOT/"runs"/name/"model.py" for name in ("quality_baseline_01", "quality_reviewed_01")]
    models = [load_model(path, config) for path in paths]
    params = [list(model.parameters()) for model in models]
    same_tensors = len(params[0]) == len(params[1]) and all(torch.equal(a, b) for a, b in zip(*params))
    seed = 7435
    x = torch.randn(3, config["seq_len"], 11, generator=torch.Generator().manual_seed(seed))
    with torch.inference_mode():
        a, b = [model(x) for model in models]
    save(HERE/"initialization.json", {"model_sha256": [digest(p) for p in paths], "parameter_counts":
         [sum(p.numel() for p in group) for group in params], "weight_seed": config["seed"],
         "all_parameter_tensors_equal_in_registration_order": same_tensors, "input_seed": seed,
         "input_shape": list(x.shape), "max_forecast_difference": float((a-b).abs().max())})
