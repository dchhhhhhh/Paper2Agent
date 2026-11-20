"""Adapter for the data_provider(args, flag) interface published with iTransformer."""

import csv
import importlib
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from task_input import Task


def inspect_training_batch(task: Task) -> dict:
    import torch

    loader_root = task.dataloader_path.parent.parent
    if task.dataloader_path != loader_root / "data_provider" / "data_factory.py":
        raise ValueError("当前适配器需要论文作者目录中的 data_provider/data_factory.py。")
    sys.path.insert(0, str(loader_root))
    try:
        factory = importlib.import_module("data_provider.data_factory")
        if Path(factory.__file__).resolve() != task.dataloader_path:
            raise ValueError("导入的加载器与任务卡指定文件不一致。")
    finally:
        sys.path.pop(0)

    with task.data_path.open(encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
    if len(header) != 863 or header.count("date") != 1 or "OT" not in header or len(set(header)) != len(header):
        raise ValueError("需要 Traffic 多变量表头，包含 date 和 862 个不同的变量（含 OT）。")
    args = SimpleNamespace(
        data="custom", root_path=str(task.data_path.parent), data_path=task.data_path.name,
        seq_len=task.lookback_len, label_len=task.label_len, pred_len=task.pred_len,
        features="M", target="OT", embed="timeF", freq="h", batch_size=2, num_workers=0,
    )
    with torch.random.fork_rng(devices=[]), redirect_stdout(sys.stderr):
        torch.random.default_generator.manual_seed(0)
        dataset, loader = factory.data_provider(args, "train")
        batch = next(iter(loader), None)
    if batch is None or len(batch) != 4:
        raise ValueError("训练加载器需要返回一批输入、标签及两组时间特征。")

    x, y, x_time, y_time = batch
    variables = len(header) - 1
    expected_x = (args.batch_size, args.seq_len, variables)
    expected_y = (args.batch_size, args.label_len + args.pred_len, variables)
    if tuple(x.shape) != expected_x or tuple(y.shape) != expected_y:
        raise ValueError(f"窗口维度不符，读到输入 {tuple(x.shape)}，标签 {tuple(y.shape)}。")
    for marks, values in ((x_time, x), (y_time, y)):
        if marks.ndim != 3 or marks.shape[:2] != values.shape[:2]:
            raise ValueError("时间特征的批次或时间维不匹配。")
    if not all(torch.isfinite(t).all().item() for t in batch):
        raise ValueError("批次含有 NaN 或无穷大。")
    if not torch.equal(x[:, -args.label_len:], y[:, :args.label_len]):
        raise ValueError("标签前缀与输入结尾的历史观测不一致。")
    if not torch.equal(x_time[:, -args.label_len:], y_time[:, :args.label_len]):
        raise ValueError("标签前缀的时间特征与输入末尾不一致。")
    return {
        "split": "train", "sample_seed": 0, "train_windows": len(dataset),
        "axes": ["batch", "time", "variable"], "input_shape": list(x.shape),
        "label_shape": list(y.shape), "forecast_target_shape": list(y[:, args.label_len:].shape),
        "input_time_features_shape": list(x_time.shape), "label_time_features_shape": list(y_time.shape),
        "finite": True, "history_overlap_steps": args.label_len,
        "history_values_and_times_match": True,
    }
