"""Orchestrate paper analysis, model generation and a train/validation/test attempt."""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
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


@contextmanager
def exclusive_run(run):
    run.mkdir(parents=True, exist_ok=True)
    # OS locks are released even if a process is interrupted during a model call.
    with (run / ".workflow.lock").open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another workflow owns {run}") from exc
        yield


def publish_candidate(run, code, spec_hash, **fields):
    # Record the new immutable digest before switching the active source. If the
    # process stops between the two operations, verify_inputs can finish it.
    pending = run/"model.pending.py"
    pending.write_text(code, encoding="utf8")
    event(run, "generated", model_sha256=digest(pending), spec_sha256=spec_hash, **fields)
    pending.replace(run/"model.py")


def generate(args):
    from pypdf import PdfReader

    run = args.run
    run.mkdir(parents=True, exist_ok=True)
    task = load_task(args.card)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    manifest = {"task_sha256": digest(args.card), "paper_sha256": digest(task.paper_path),
                "config_sha256": digest(args.config), "task_id": task.task_id}
    if args.protocol == "reviewed":
        manifest["protocol"] = "reviewed-v1"
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
    draft = run / "paper_spec_draft.json" if args.protocol == "reviewed" else spec_file
    if not draft.exists():
        event(run, "paper_analysis_requested")
        text = {"type": "string"}
        texts = {"type": "array", "items": text}
        analysis_schema = object_schema({"method": text, "requirements_with_page_evidence": texts,
                                         "data_and_evaluation": texts, "ambiguous_or_tunable": texts})
        prompt = ("根据以下公开论文的选页和目标实验，独立提取可实施的方法规格。给出变量轴、归一化、"
                  "编码层、残差、激活、预测头的依据和具体页码。区分论文设置和运行选择。输出中文 JSON。\n"
                  + json.dumps(card["experiment"], ensure_ascii=False) + "\n" + pages)
        answer = invoke_model(stage="analysis", prompt=prompt, log_dir=run/"calls", schema=analysis_schema)
        save(draft, json.loads(answer))
        event(run, "paper_analysis_saved", spec_sha256=digest(draft))
    if args.protocol == "reviewed":
        from spec_review import review_spec

        events = [json.loads(line) for line in (run/"workflow.jsonl").read_text(encoding="utf8").splitlines()]
        if any(row["stage"] == "spec_held" for row in events):
            raise ValueError("This specification was held; inspect its findings before starting a new run")
        previous = [row for row in events if row["stage"] == "spec_approved"]
        if previous:
            if digest(spec_file) != previous[-1]["spec_sha256"]:
                raise ValueError("The approved specification changed")
        else:
            event(run, "spec_review_requested")
            try:
                approved = review_spec(run=run, spec=json.loads(draft.read_text(encoding="utf8")),
                                       pages=pages, experiment=card["experiment"], config=config)
            except ValueError as exc:
                event(run, "spec_held", reason=str(exc), draft_sha256=digest(draft))
                raise
            save(spec_file, approved)
            event(run, "spec_approved", spec_sha256=digest(spec_file))
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
    if args.protocol == "baseline":
        tree = ast.parse(code)
        if not any(isinstance(node, ast.ClassDef) and node.name == "Forecaster" for node in tree.body):
            raise ValueError("Generated module does not declare Forecaster")
    publish_candidate(run, code, digest(spec_file))


def verify_inputs(args):
    run = args.run
    manifest = json.loads((run/"inputs.json").read_text(encoding="utf8"))
    task = load_task(args.card)
    if (digest(args.card), digest(task.paper_path), digest(run/"config.json")) != (
        manifest["task_sha256"], manifest["paper_sha256"], manifest["config_sha256"]
    ):
        raise ValueError("Task, paper or configuration changed after generation")
    with (run / "workflow.jsonl").open(encoding="utf-8") as stream:
        generated = [json.loads(line) for line in stream if line.strip()]
    records = [row for row in generated if row.get("stage") == "generated"]
    pending = run/"model.pending.py"
    if pending.exists() and records and digest(pending) == records[-1]["model_sha256"]:
        pending.replace(run/"model.py")
    model_hash = digest(run/"model.py")
    if not records or records[-1]["model_sha256"] != model_hash:
        raise ValueError("Generated-code hash does not match recorded model output")
    spec_hash = digest(run/"paper_spec.json")
    if spec_hash != records[-1]["spec_sha256"]:
        raise ValueError("Specification differs from the input to code generation")
    approvals = [row for row in generated if row.get("stage") == "spec_approved"]
    if manifest.get("protocol") == "reviewed-v1" and (not approvals or approvals[-1]["spec_sha256"] != spec_hash):
        raise ValueError("Missing approval for current specification")
    if approvals and manifest.get("protocol") != "reviewed-v1":
        raise ValueError("Manifest protocol differs from the recorded review")
    return manifest, records


def assess(args):
    run = args.run
    manifest, records = verify_inputs(args)
    if manifest.get("protocol") != "reviewed-v1":
        raise ValueError("The method gate applies to reviewed runs")
    attempt = run / "quality" / f"candidate_{len(records):02d}"
    attempt.mkdir(parents=True, exist_ok=True)
    # Keep each draft, including a rejected one, with the observations that led to its verdict.
    shutil.copyfile(run / "model.py", attempt / "model.py")
    report = attempt / "checks.json"
    pending = attempt / "checks.pending.json"
    cmd = [args.python, "-B", "-X", "utf8", str(Path(__file__).with_name("method_checks.py")),
           "--model", str(attempt/"model.py"), "--config", str(run/"config.json"), "--report", str(pending),
           "--card", str(args.card)]
    subprocess.run(cmd, check=True, timeout=180)
    result = json.loads(pending.read_text(encoding="utf8"))
    if result.get("status") not in {"passed", "failed", "inconclusive"}:
        raise ValueError("Unknown method verdict")
    if result.get("model_sha256") != digest(attempt/"model.py"):
        raise ValueError("Method check evaluated different model bytes")
    result.update(spec_sha256=digest(run/"paper_spec.json"),
                  config_sha256=digest(run/"config.json"))
    save(report, result)
    pending.unlink()
    event(run, "model_checked", status=result["status"], report=report.relative_to(run).as_posix(),
          model_sha256=result["model_sha256"])
    return result


MAX_CODE_REPAIRS = 2


def repair_source(*, failures, spec_file, config_file, source_file, calls):
    prompt = ("Repair only the evidence-backed failures in this PyTorch Forecaster. Preserve its API and "
              "working mechanisms. Use only torch and built-ins, no file, network or test data access. "
              "Return the complete repaired module as JSON under python. Address the underlying paper "
              "mechanism for arbitrary supported inputs.\nObserved failures:\n" + json.dumps(failures) +
              "\nTask specification:\n" + spec_file.read_text(encoding="utf8") +
              "\nFrozen configuration:\n" + config_file.read_text(encoding="utf8") +
              "\nPrevious candidate:\n" + source_file.read_text(encoding="utf8"))
    answer = invoke_model(stage="repair", prompt=prompt, log_dir=calls,
                          schema=object_schema({"python": {"type": "string"}}))
    return json.loads(answer)["python"].strip() + "\n"


def repair(args):
    run = args.run
    manifest, records = verify_inputs(args)
    if manifest.get("protocol") != "reviewed-v1" or (run/"fit").exists():
        raise ValueError("Repair requires a reviewed candidate before training")
    if len(records) > MAX_CODE_REPAIRS:
        raise ValueError("Repair limit reached; candidate held for review")
    attempt = run / "quality" / f"candidate_{len(records):02d}"
    report = json.loads((attempt / "checks.json").read_text(encoding="utf8"))
    if (report["status"] != "failed" or report["model_sha256"] != digest(run/"model.py")
            or report["spec_sha256"] != digest(run/"paper_spec.json")
            or report["config_sha256"] != digest(run/"config.json")):
        raise ValueError("Repair needs a failed report for the current code, spec and config")
    failures = [check for check in report["checks"] if check["status"] == "failed"]
    if not failures:
        raise ValueError("No diagnosed failure supplied to the repair stage")
    code = repair_source(failures=failures, spec_file=run/"paper_spec.json", config_file=run/"config.json",
                         source_file=attempt/"model.py", calls=run/"calls")
    publish_candidate(run, code, report["spec_sha256"], source="repair", previous_model_sha256=report["model_sha256"])


def train(args):
    run = args.run
    manifest, _ = verify_inputs(args)
    if manifest.get("protocol") == "reviewed-v1":
        verdict = assess(args)
        while verdict["status"] == "failed":
            repair(args)
            verdict = assess(args)
        if verdict["status"] != "passed":
            raise ValueError("Method gate inconclusive; candidate held for independent review")
    event(run, "training_dispatched")
    cmd = [args.python, "-B", "-X", "utf8", str(Path(__file__).with_name("experiment_runner.py")),
           "--card", str(args.card), "--config", str(run/"config.json"), "--model", str(run/"model.py"),
           "--out", str(run/"fit")]
    subprocess.run(cmd, check=True)
    metrics = json.loads((run/"fit/metrics.json").read_text(encoding="utf8"))
    event(run, "test_measured", test_mse=metrics["test_mse"], elements=metrics["elements"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["generate", "check", "repair", "train", "all"])
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--protocol", choices=["baseline", "reviewed"], default=None,
                        help="Generation protocol; default reviewed. Older runs retain their protocol.")
    parser.add_argument("--python", default=sys.executable, help="Interpreter with torch for training")
    args = parser.parse_args()
    if args.stage in {"generate", "all"} and args.config is None:
        parser.error("--config is required to generate")
    if args.protocol and args.stage not in {"generate", "all"}:
        parser.error("--protocol applies only when generating a new run")
    with exclusive_run(args.run):
        if args.protocol is None:
            saved = args.run/"inputs.json"
            if saved.exists():
                recorded = json.loads(saved.read_text(encoding="utf8")).get("protocol")
                if recorded not in (None, "reviewed-v1"):
                    raise ValueError(f"Unknown saved protocol: {recorded}")
                args.protocol = "reviewed" if recorded else "baseline"
            else:
                args.protocol = "reviewed"
        try:
            if args.stage in {"generate", "all"}:
                generate(args)
            if args.stage == "check" and assess(args)["status"] != "passed":
                return 2
            if args.stage == "repair":
                repair(args)
            if args.stage in {"train", "all"}:
                train(args)
        except Exception as exc:
            event(args.run, "failed", request=args.stage, error=str(exc))
            raise


if __name__ == "__main__":
    sys.exit(main())
