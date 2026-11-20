"""Read and validate the first supported task without loading ML dependencies."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Task:
    task_id: str
    paper_path: Path
    data_path: Path
    dataloader_path: Path
    lookback_len: int
    label_len: int
    pred_len: int


class InputProblem(Exception):
    def __init__(self, status: str, issues: list[dict], task_id: str | None = None):
        super().__init__(status)
        self.status = status
        self.issues = issues
        self.task_id = task_id


def load_task(card_path: Path) -> Task:
    try:
        card = json.loads(card_path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise InputProblem("invalid_task", [{"field": "task_card", "message": str(exc)}]) from exc
    if not isinstance(card, dict):
        raise InputProblem("invalid_task", [{"field": "task_card", "message": "任务卡须为 JSON 对象。"}])

    issues = []
    task_id = card.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        issues.append({"field": "task_id", "message": "需要非空任务编号。"})
    experiment = card.get("experiment")
    if not isinstance(experiment, dict):
        experiment = {}
    scope = {
        "method": "iTransformer", "dataset": "Traffic", "metric": "MSE",
        "lookback_len": 96, "label_len": 48, "pred_len": 96, "evaluation_split": "test",
    }
    for name, expected in scope.items():
        value = experiment.get(name)
        if type(value) is not type(expected) or value != expected:
            issues.append({"field": f"experiment.{name}", "message": f"当前任务需要 {expected}，读到 {value!r}。"})
    if card.get("dataloader_entrypoint") != "data_provider":
        issues.append({"field": "dataloader_entrypoint", "message": "需要 data_provider 入口。"})
    if issues:
        raise InputProblem("invalid_task", issues, task_id if isinstance(task_id, str) else None)

    paths = {}
    for field in ("paper_path", "data_path", "dataloader_path"):
        value = card.get(field)
        if not isinstance(value, str) or not value.strip():
            issues.append({"field": field, "message": "请填写本地文件路径。"})
            continue
        try:
            path = (card_path.parent / value).resolve()
            with path.open("rb") as stream:
                head = stream.read(5)
            if not head:
                raise ValueError("文件为空。")
            if field == "paper_path" and head != b"%PDF-":
                raise ValueError("论文文件缺少 PDF 标识。")
        except FileNotFoundError:
            issues.append({"field": field, "path": value, "message": "指定的文件不存在。"})
            continue
        except (OSError, ValueError, RuntimeError) as exc:
            issues.append({"field": field, "path": value, "message": str(exc)})
            continue
        paths[field] = path
    if issues:
        raise InputProblem("waiting_for_input", issues, task_id)
    return Task(task_id, **paths, lookback_len=experiment["lookback_len"],
                label_len=experiment["label_len"], pred_len=experiment["pred_len"])
