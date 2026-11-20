"""Fit a generated forecasting model, select on validation, then score a frozen model."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import platform
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from task_input import load_task


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_series(task, config, split):
    """Use the task's exact preprocessing. Keep just this split as contiguous float32."""
    loader_root = task.dataloader_path.parent.parent
    sys.path.insert(0, str(loader_root))
    try:
        factory = importlib.import_module("data_provider.data_factory")
        if Path(factory.__file__).resolve() != task.dataloader_path:
            raise ValueError("Imported data loader differs from the one on the task card.")
        args = SimpleNamespace(
            data="custom", root_path=str(task.data_path.parent), data_path=task.data_path.name,
            seq_len=task.lookback_len, label_len=task.label_len, pred_len=task.pred_len,
            features="M", target="OT", embed="timeF", freq="h", batch_size=config["batch_size"],
            num_workers=0,
        )
        dataset, _ = factory.data_provider(args, split)
    finally:
        sys.path.pop(0)
    # Ignore the upstream DataLoader's drop_last. Visit each upstream window exactly once.
    values = torch.from_numpy(np.ascontiguousarray(dataset.data_x, dtype=np.float32))
    count = len(values) - task.lookback_len - task.pred_len + 1
    if count < 1 or count != len(dataset) or not torch.isfinite(values).all():
        raise ValueError(f"Invalid {split} series or window count.")
    for index in (0, count - 1):
        upstream_x, upstream_y, _, _ = dataset[index]
        start = index + task.lookback_len
        if not np.array_equal(upstream_x.astype("float32"), values[index:start].numpy()):
            raise ValueError(f"{split}: input window does not match supplied data loader")
        if not np.array_equal(upstream_y[-task.pred_len:].astype("float32"), values[start:start + task.pred_len].numpy()):
            raise ValueError(f"{split}: target window does not match supplied data loader")
    return values


def batches(values, config, *, training, generator=None):
    history, horizon = config["seq_len"], config["pred_len"]
    size = config["batch_size"] if training else config["eval_batch_size"]
    total = len(values) - history - horizon + 1
    order = torch.randperm(total, generator=generator) if training else torch.arange(total)
    time_offsets = torch.arange(history + horizon)
    for indices in order.split(size):
        starts = indices[:, None] + time_offsets
        variables = config["variables_per_train_window"]
        if training and variables < values.shape[1]:
            # Independent uniform subset without replacement for each window.
            columns = torch.rand((len(indices), values.shape[1]), generator=generator).topk(variables, dim=1).indices
            window = values[starts[:, :, None], columns[:, None, :]]
        else:
            window = values[starts]
        yield window[:, :history], window[:, history:]


def predict(model, x, target, device):
    prediction = model(x.to(device))
    if prediction.shape != target.shape or not prediction.is_floating_point():
        raise ValueError(f"Forecast shape/type {tuple(prediction.shape)} {prediction.dtype}; expected {tuple(target.shape)}.")
    return prediction


@torch.inference_mode()
def evaluate(model, values, config, device, *, predictions=None):
    model.eval()
    sum_squared_error = 0.0
    elements = windows = 0
    for x, target in batches(values, config, training=False):
        forecast = predict(model, x, target, device).float().cpu()
        difference = forecast.double() - target.double()
        sum_squared_error += difference.square().sum().item()
        if predictions is not None:
            predictions[windows:windows + len(x)] = forecast.numpy()
        elements += target.numel()
        windows += len(x)
    if not elements or not math.isfinite(sum_squared_error):
        raise ValueError("Nonfinite score or empty evaluation split.")
    return {"sse": sum_squared_error, "elements": elements, "windows": windows, "mse": sum_squared_error / elements}


def fit(args, log):
    task = load_task(args.card)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for key, expected in (("seq_len", task.lookback_len), ("label_len", task.label_len), ("pred_len", task.pred_len)):
        if config[key] != expected:
            raise ValueError(f"{key} disagrees with task card")
    for key in ("epochs", "threads", "batch_size", "eval_batch_size", "variables_per_train_window"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    schedule = config.get("lr_schedule", "constant")
    if schedule not in {"constant", "halve_after_epoch"}:
        raise ValueError("Unsupported lr_schedule")
    patience = config.get("patience") or config["epochs"]
    if type(patience) is not int or patience <= 0:
        raise ValueError("patience must be a positive integer")
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads"])
    generator = torch.Generator().manual_seed(seed)
    log("start", config=config, device=str(device), python=platform.python_version(), torch=torch.__version__,
        model_sha256=sha256(args.model), config_sha256=sha256(args.config), card_sha256=sha256(args.card),
        data_sha256=sha256(task.data_path), loader_sha256=sha256(task.dataloader_path), paper_sha256=sha256(task.paper_path))
    spec = importlib.util.spec_from_file_location("generated_forecast_module", args.model)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.Forecaster(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train = read_series(task, config, "train")
    val = read_series(task, config, "val")
    variables = train.shape[1]
    if config["variables_per_train_window"] > variables or val.shape[1] != variables:
        raise ValueError("Variable count differs from the configured training subset or between splits.")
    training_windows = len(train) - config["seq_len"] - config["pred_len"] + 1
    # Exercise the generated model on real training data before the first optimizer step.
    model.eval()
    smoke_x = train[:config["seq_len"]].unsqueeze(0)
    smoke_y = train[config["seq_len"]:config["seq_len"] + config["pred_len"]].unsqueeze(0)
    with torch.no_grad():
        smoke_prediction = predict(model, smoke_x, smoke_y, device)
        if not torch.isfinite(smoke_prediction).all():
            raise ValueError("Generated model returned nonfinite predictions on training preflight.")
    log("inputs_ready", train_windows=training_windows, variables=variables,
        parameters=sum(p.numel() for p in model.parameters()), preflight_split="train",
        preflight_input=list(smoke_x.shape), preflight_output=list(smoke_prediction.shape))
    best = math.inf
    best_epoch = 0
    checkpoint = args.out / "best.pt"
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_sse = 0.0
        train_count = train_windows = 0
        last_progress = time.monotonic()
        for x, target in batches(train, config, training=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            forecast = predict(model, x, target, device)
            loss = torch.nn.functional.mse_loss(forecast, target.to(device))
            value = loss.item()
            if not math.isfinite(value):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            train_sse += value * target.numel()
            train_count += target.numel()
            train_windows += len(x)
            if time.monotonic() - last_progress >= 30:
                log("train_progress", epoch=epoch, windows=train_windows, total=training_windows)
                last_progress = time.monotonic()
        if train_windows != training_windows:
            raise ValueError("Training window coverage mismatch.")
        log("validation_started", epoch=epoch, train_windows=train_windows)
        validation = evaluate(model, val, config, device)
        improved = validation["mse"] < best
        if improved:
            best_epoch, best = epoch, validation["mse"]
            tmp = checkpoint.with_suffix(".pt.tmp")
            torch.save(model.state_dict(), tmp)
            tmp.replace(checkpoint)
        log("epoch", epoch=epoch, lr=optimizer.param_groups[0]["lr"], train_windows=train_windows,
            train_mse=train_sse / train_count, val=validation, best_checkpoint=improved)
        if epoch - best_epoch >= patience:
            break
        if schedule == "halve_after_epoch":
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
    del train, val, optimizer
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    log("test_started", selected_epoch=best_epoch, val_mse=best, checkpoint_sha256=sha256(checkpoint))
    test = read_series(task, config, "test")
    history, horizon = config["seq_len"], config["pred_len"]
    windows = len(test) - history - horizon + 1
    shape = (windows, horizon, variables)
    required = (math.prod(shape) + test[history:].numel()) * 4 + 1024 * 1024
    if shutil.disk_usage(args.out).free < required:
        raise OSError(f"Saving predictions and truths requires {required} free bytes.")
    temporary = args.out / "predictions.partial.npy"
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    scores = evaluate(model, test, config, device, predictions=output)
    output.flush()
    del output
    if scores["windows"] != windows or scores["elements"] != math.prod(shape):
        raise ValueError("Incomplete test coverage.")
    temporary.replace(args.out / "predictions.npy")
    np.save(args.out / "test_truth_series.npy", test[history:].numpy(), allow_pickle=False)
    metrics = {
        **scores, "test_mse": scores["mse"], "split": "test", "scale": "per-variable training StandardScaler",
        "best_epoch": best_epoch, "best_validation_mse": best, "epochs_completed": epoch,
        "forecast_shape": shape, "predictions": "predictions.npy", "truth_series": "test_truth_series.npy",
        "truth_index": "truth_series[window + horizon_offset, variable]", "checkpoint": "best.pt",
        "checkpoint_sha256": sha256(checkpoint), "model_sha256": sha256(args.model), "task_id": task.task_id,
    }
    write_json(args.out / "metrics.json", metrics)
    log("complete", test_mse=scores["mse"], test_sse=scores["sse"], test_elements=scores["elements"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("card", "model", "config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    expected_files = ("metrics.json", "best.pt", "predictions.npy", "predictions.partial.npy", "test_truth_series.npy")
    if any((args.out / name).exists() for name in expected_files):
        parser.error("Output already contains training artifacts. Use a fresh attempt directory.")
    started = time.monotonic()
    with (args.out / "training.jsonl").open("x", encoding="utf-8", buffering=1) as events:
        def log(event, **details):
            record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": round(time.monotonic() - started, 3), **details}
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            events.write(line + "\n")
            print(line, flush=True)

        try:
            fit(args, log)
        except BaseException as exc:
            log("failed", exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
