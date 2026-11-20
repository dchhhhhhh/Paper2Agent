"""Orchestrate paper analysis, model generation and a train/validation/test attempt."""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from model_call import invoke_model
from task_input import load_task


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def event(run, stage, **fields):
    line = {"time": datetime.now(timezone.utc).isoformat(), "stage": stage, **fields}
    with (run / "workflow.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(stage, flush=True)


def object_schema(fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def generate(args):
    from pypdf import PdfReader

    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    task = load_task(args.card)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    manifest = {"task_sha256": digest(args.card), "paper_sha256": digest(task.paper_path),
                "config_sha256": digest(args.config), "task_id": task.task_id}
    record = run / "inputs.json"
    if record.exists() and json.loads(record.read_text(encoding="utf8")) != manifest:
        raise ValueError("Use a fresh run directory for changed inputs")
    save(record, manifest)
    if (run / "model.py").exists() or (run / "fit").exists():
        raise ValueError("Generated code or training artifacts already exist. Use a new run directory.")
    shutil.copyfile(args.config, run / "config.json")
    source = PdfReader(task.paper_path)
    pages = "\n\n".join(f"[PDF page {n}]\n{source.pages[n-1].extract_text()}" for n in config["paper_pages"])
    card = json.loads(args.card.read_text(encoding="utf-8-sig"))
    spec_file = run / "paper_spec.json"
    if not spec_file.exists():
        event(run, "paper_analysis_requested")
        text = {"type": "string"}
        texts = {"type": "array", "items": text}
        analysis_schema = object_schema({"method": text, "requirements_with_page_evidence": texts,
                                         "data_and_evaluation": texts, "ambiguous_or_tunable": texts})
        prompt = ("根据以下公开论文的选页和目标实验，独立提取可实施的方法规格。给出变量轴、归一化、"
                  "编码层、残差、激活、预测头的依据和具体页码。区分论文设置和运行选择。输出中文 JSON。\n"
                  + json.dumps(card["experiment"], ensure_ascii=False) + "\n" + pages)
        answer = invoke_model(stage="analysis", prompt=prompt, log_dir=run/"calls", schema=analysis_schema)
        save(spec_file, json.loads(answer))
        event(run, "paper_analysis_saved", spec_sha256=digest(spec_file))
    event(run, "code_generation_requested")
    prompt = ("Implement the forecasting model described by this paper analysis. Return one complete Python module "
              "under the python field. Export Forecaster(config: dict) as a torch.nn.Module, forward(x) maps "
              "[B,seq_len,N] to [B,pred_len,N] for variable N, including training with subsets. Use real multi-head "
              "attention and torch.nn.functional.scaled_dot_product_attention for an economical exact CPU implementation. "
              "Honor d_model,n_heads,e_layers,d_ff,dropout. Use only torch and built-in Python, no file or network I/O, "
              "no training script, external models, known performance values or test-data access. "
              "Implement paper mechanisms, document dimensions, and use dropout only during training.\nSpec:\n"
              + spec_file.read_text(encoding="utf8") + "\nFrozen CPU experiment configuration:\n"
              + json.dumps(config))
    answer = invoke_model(stage="generation", prompt=prompt, log_dir=run/"calls",
                          schema=object_schema({"python": {"type": "string"}}))
    code = json.loads(answer)["python"].strip() + "\n"
    tree = ast.parse(code)
    if not any(isinstance(node, ast.ClassDef) and node.name == "Forecaster" for node in tree.body):
        raise ValueError("Generated module does not declare Forecaster")
    (run / "model.py").write_text(code, encoding="utf-8")
    event(run, "generated", model_sha256=digest(run/"model.py"), spec_sha256=digest(spec_file))


def train(args):
    run = args.run
    manifest = json.loads((run/"inputs.json").read_text(encoding="utf8"))
    task = load_task(args.card)
    if (digest(args.card), digest(task.paper_path), digest(run/"config.json")) != (
        manifest["task_sha256"], manifest["paper_sha256"], manifest["config_sha256"]
    ):
        raise ValueError("Task, paper or configuration changed after generation")
    with (run / "workflow.jsonl").open(encoding="utf-8") as stream:
        generated = [json.loads(line) for line in stream if line.strip()]
    if not any(row.get("stage") == "generated" and row["model_sha256"] == digest(run/"model.py") for row in generated):
        raise ValueError("Generated-code hash does not match recorded model output")
    event(run, "training_dispatched")
    cmd = [args.python, "-B", "-X", "utf8", str(Path(__file__).with_name("experiment_runner.py")),
           "--card", str(args.card), "--config", str(run/"config.json"), "--model", str(run/"model.py"),
           "--out", str(run/"fit")]
    subprocess.run(cmd, check=True)
    metrics = json.loads((run/"fit/metrics.json").read_text(encoding="utf8"))
    event(run, "test_measured", test_mse=metrics["test_mse"], elements=metrics["elements"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "train", "all"])
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--python", default=sys.executable, help="Interpreter with torch for training")
    args = parser.parse_args()
    if args.stage in {"generate", "all"} and args.config is None:
        parser.error("--config is required to generate")
    try:
        if args.stage in {"generate", "all"}:
            generate(args)
        if args.stage in {"train", "all"}:
            train(args)
    except Exception as exc:
        if args.run.is_dir():
            event(args.run, "failed", request=args.stage, error=str(exc))
        raise


if __name__ == "__main__":
    main()
