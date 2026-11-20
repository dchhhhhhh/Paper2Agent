"""Check local inputs and inspect one real training batch; publish a JSON result."""

import argparse
import json
import sys
from pathlib import Path

from task_input import InputProblem, load_task
from train_batch import inspect_training_batch


def check(card: Path) -> tuple[dict, int]:
    report = {"stage": "input_check", "model_calls": 0}
    try:
        task = load_task(card)
    except InputProblem as exc:
        if exc.task_id:
            report["task_id"] = exc.task_id
        return {**report, "status": exc.status, "issues": exc.issues}, 2
    report.update(task_id=task.task_id, stage="train_batch_check")
    try:
        batch = inspect_training_batch(task)
    except Exception as exc:
        return {**report, "status": "batch_check_failed",
                "issues": [{"field": "train_batch", "message": f"{type(exc).__name__}: {exc}"}]}, 1
    return {**report, "status": "train_batch_checked", "batch": batch}, 0


def main() -> int:
    parser = argparse.ArgumentParser(description="校验任务输入并读取一个真实训练批次")
    parser.add_argument("card", type=Path, help="JSON 任务卡")
    parser.add_argument("--report", type=Path, help="将本次检查写入新的 UTF-8 JSON 文件")
    args = parser.parse_args()
    report, code = check(args.card)
    result = json.dumps(report, ensure_ascii=False, indent=2)
    print(result)
    if args.report:
        try:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(result + "\n")
        except OSError as exc:
            print(f"无法写入新的检查记录：{exc}", file=sys.stderr)
            return 1
    return code


if __name__ == "__main__":
    sys.exit(main())
