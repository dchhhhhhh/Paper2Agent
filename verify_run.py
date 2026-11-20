"""Independently audit a completed Traffic-96 run without loading its model code."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


# These counts belong to the exact 17,544-row CSV already verified in step 1.
TRAFFIC_SHA256 = "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"
ROWS = 17_544
CHANNELS = 862


def require(condition, message):
    if not condition:
        raise ValueError(message)


def match_score(actual, expected, name):
    require(math.isfinite(actual) and math.isfinite(expected) and
            math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
            f"{name} differs: {actual} vs {expected}")


def audit_training(out, metrics):
    with (out / "training.jsonl").open(encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream]
    names = [item["event"] for item in events]
    epochs = [item for item in events if item["event"] == "epoch"]
    require(bool(epochs), "No completed epochs.")
    milestones = [name for name in names if name != "train_progress"]
    expected = ["start", "inputs_ready"] + ["validation_started", "epoch"] * len(epochs)
    require(milestones == expected + ["test_started", "complete"], "Unexpected stage ordering.")
    start, inputs, test_start, complete = events[0], events[1], events[-2], events[-1]
    config = start["config"]
    require(start["data_sha256"] == TRAFFIC_SHA256, "Input digest differs from the Traffic source.")
    require(start["model_sha256"] == metrics["model_sha256"], "Model digest changed within run.")
    require(metrics["task_id"] == "reproducebench-itransformer-traffic-96" and metrics["split"] == "test",
            "Unexpected task or split.")
    require((config["seq_len"], config["label_len"], config["pred_len"]) == (96, 48, 96),
            "Unexpected window specification for Traffic-96.")
    require(inputs["variables"] == CHANNELS, "Data channel count differs from the task.")
    history, horizon = config["seq_len"], config["pred_len"]
    # The supplied loader adds history before the validation/test split borders.
    train_rows, test_rows = int(ROWS * 0.7), int(ROWS * 0.2)
    train_windows = train_rows - history - horizon + 1
    val_windows = ROWS - train_rows - test_rows - horizon + 1
    test_windows = test_rows - horizon + 1
    require(inputs["train_windows"] == train_windows, "Train windows differ from the source split.")
    best_val = math.inf
    best_epoch = 0
    validations = [e for e in events if e["event"] == "validation_started"]
    for i, (epoch, launch) in enumerate(zip(epochs, validations), 1):
        require(epoch["epoch"] == launch["epoch"] == i, "Duplicate, skipped or reordered epoch.")
        require(epoch["train_windows"] == launch["train_windows"] == train_windows,
                f"Epoch {i} did not record the complete training window count.")
        val = epoch["val"]
        require(val["windows"] == val_windows and val["elements"] == val_windows * horizon * CHANNELS,
                f"Epoch {i} has incomplete validation coverage.")
        require(val["sse"] >= 0 and math.isfinite(epoch["train_mse"]), f"Invalid epoch {i} metrics.")
        match_score(val["sse"] / val["elements"], val["mse"], f"Epoch {i} validation MSE")
        improved = val["mse"] < best_val
        require(epoch["best_checkpoint"] is improved, f"Epoch {i} checkpoint-selection mismatch.")
        if improved:
            best_epoch, best_val = i, val["mse"]
    require(metrics["epochs_completed"] == len(epochs) <= config["epochs"], "Epoch total mismatch.")
    if len(epochs) < config["epochs"]:
        require(len(epochs) - best_epoch >= (config.get("patience") or config["epochs"]),
                "Run ended before epoch budget or early-stopping criterion.")
    require(metrics["best_epoch"] == test_start["selected_epoch"] == best_epoch,
            "Test checkpoint did not come from the best validation epoch.")
    match_score(metrics["best_validation_mse"], best_val, "Selected validation loss")
    match_score(test_start["val_mse"], best_val, "Logged validation loss")
    require(metrics["checkpoint"] == "best.pt", "Unexpected checkpoint location.")
    with (out / "best.pt").open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    require(digest == metrics["checkpoint_sha256"] == test_start["checkpoint_sha256"],
            "Checkpoint file differs from the frozen evaluation checkpoint.")
    return {
        "epochs_checked": len(epochs), "best_epoch": best_epoch, "best_validation_mse": best_val,
        "train_windows_per_epoch": train_windows, "validation_windows_per_epoch": val_windows,
        "test_windows": test_windows, "checkpoint_sha256": digest,
    }, complete


def audit(out):
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    training, final_event = audit_training(out, metrics)
    require(metrics["predictions"] == "predictions.npy" and metrics["truth_series"] == "test_truth_series.npy",
            "Unexpected evaluation artifact paths.")
    require(metrics["truth_index"] == "truth_series[window + horizon_offset, variable]",
            "Unknown target indexing.")
    forecasts = np.load(out / "predictions.npy", mmap_mode="r", allow_pickle=False)
    truth = np.load(out / "test_truth_series.npy", mmap_mode="r", allow_pickle=False)
    windows = training["test_windows"]
    shape = (windows, 96, CHANNELS)
    require(forecasts.shape == tuple(metrics["forecast_shape"]) == shape, "Incomplete forecast array.")
    require(truth.shape == (windows + 95, CHANNELS), "Compressed ground-truth series has wrong dimensions.")
    require(forecasts.dtype == truth.dtype == np.float32, "Expected float32 evaluation artifacts.")
    # Rebuild each horizon by its offset in the compact ground-truth series.
    offsets = np.arange(96)[None, :]
    sse_chunks = []
    for first in range(0, windows, 8):
        end = min(windows, first + 8)
        expected = truth[np.arange(first, end)[:, None] + offsets]
        residual = np.subtract(forecasts[first:end], expected, dtype=np.float64)
        sse_chunks.append(float(np.square(residual, out=residual).sum(dtype=np.float64)))
    sse = math.fsum(sse_chunks)
    count = math.prod(shape)
    mse = sse / count
    require(metrics["windows"] == windows and metrics["elements"] == count == final_event["test_elements"],
            "Test sample or element count mismatch.")
    match_score(sse, metrics["sse"], "Recomputed test SSE")
    match_score(sse, final_event["test_sse"], "Recomputed/logged test SSE")
    for recorded in (metrics["mse"], metrics["test_mse"], final_event["test_mse"]):
        match_score(mse, recorded, "Recomputed test MSE")
    return {"status": "verified", "task_id": metrics["task_id"], **training, "test_shape": list(shape),
            "test_sse": sse, "test_elements": count, "test_mse": mse, "scale": metrics["scale"],
            "reported_mse": metrics["test_mse"], "absolute_mse_difference": abs(mse - metrics["test_mse"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Completed training output directory")
    parser.add_argument("--report", required=True, type=Path, help="New audit report path")
    args = parser.parse_args()
    try:
        with args.report.open("x", encoding="utf-8") as report:
            try:
                outcome = audit(args.out)
            except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
                outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            text = json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False)
            report.write(text + "\n")
            print(text)
    except OSError as exc:
        print(f"Could not create audit report: {exc}", file=sys.stderr)
        return 1
    return 0 if outcome["status"] == "verified" else 1


if __name__ == "__main__":
    sys.exit(main())
