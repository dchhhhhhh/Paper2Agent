"""Blind an individual generated model and review it without exposing its study arm."""

import argparse
import ast
import json
import secrets
from pathlib import Path
from types import SimpleNamespace

from model_call import invoke_model
from run_experiment import digest, exclusive_run, object_schema, save, verify_inputs
from task_input import load_task


RUBRIC = {
    "tokens_and_head": "Each sensor's history is a token. Embedding and horizon head are shared; N may vary.",
    "attention": "Each block applies unmasked, appropriately scaled multi-head dot-product attention over sensors per example.",
    "encoder": "Algorithm 1: attention and position-wise FFN each have a residual followed by hidden-axis LayerNorm.",
    "forecast": "History only, direct [B,S,N] output; if instance normalization is used, restore its scale. Disable evaluation dropout.",
}


def anonymous_source(path):
    # ast.unparse drops comments. Remove true docstrings as well, keeping code and literals.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node, clean=False) is not None:
                node.body.pop(0)
                if not node.body and not isinstance(node, ast.Module):
                    node.body.append(ast.Pass())
    return ast.unparse(tree) + "\n"


def prepare(run, root):
    path = run / "model.py"
    sha = digest(path)
    mapping = run / "blind_submission.json"
    if mapping.exists():
        old = json.loads(mapping.read_text(encoding="utf-8"))
        if old["original_sha256"] != sha:
            raise ValueError("A different source was previously submitted for this run")
        if digest(root/old["id"]/"source.py") != old["anonymous_sha256"]:
            raise ValueError("Anonymous source differs from its first submission")
        return old["id"]
    anonymous = anonymous_source(path)
    token = secrets.token_hex(6)
    folder = root / token
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "source.py").write_text(anonymous, encoding="utf8", newline="\n")
    save(mapping, {"id": token, "original_sha256": sha, "anonymous_sha256": digest(folder/"source.py")})
    return token


def judge(folder, card_path, config_path):
    from pypdf import PdfReader

    source = folder / "source.py"
    if (folder/"judgment.json").exists():
        raise ValueError("Existing blinded judgment; retain the first submission")
    task = load_task(card_path)
    pdf = PdfReader(task.paper_path)
    pages = "\n".join(f"[PDF page {n}]\n{pdf.pages[n-1].extract_text()}" for n in (4, 5, 13))
    code = source.read_text(encoding="utf8")
    indexed = "\n".join(f"{i:03} {line}" for i, line in enumerate(code.splitlines(), 1))
    prompt = ("Independently audit one anonymous iTransformer PyTorch module. "
              "You only have the source, task, local config, and primary paper excerpts. "
              "For each criterion decide pass, critical_error, or unclear and justify it in Chinese using actual code lines. "
              "Accept equivalent implementations; dataset-level preprocessing, disabling optional input normalization, "
              "and choosing a narrow model are valid. Score only these four criteria. "
              "The criteria concern model.py, not the external loader or optimizer.\n"
              + json.dumps(RUBRIC) + "\nConfig:\n" + config_path.read_text(encoding="utf8")
              + "\nTask:\n" + json.dumps(json.loads(card_path.read_text(encoding="utf8"))["experiment"], ensure_ascii=False)
              + "\nExcerpts:\n" + pages + "\nAnonymous numbered source:\n" + indexed)
    row = object_schema({"verdict": {"type":"string", "enum":["pass", "critical_error", "unclear"]},
                         "source_lines": {"type": "array", "items": {"type": "integer"}},
                         "reason": {"type":"string"}})
    reply = invoke_model(stage="blind_review", prompt=prompt, log_dir=folder/"calls",
                         schema=object_schema(dict.fromkeys(RUBRIC, row)))
    findings = json.loads(reply)
    if set(findings) != set(RUBRIC):
        raise ValueError("Blinded response omitted a criterion")
    for item in findings.values():
        if item["verdict"] not in {"pass", "critical_error", "unclear"} or not item["reason"].strip():
            raise ValueError("Incomplete blinded finding")
        if not item["source_lines"] or not all(type(n) is int and 1 <= n <= len(code.splitlines())
                                               for n in item["source_lines"]):
            raise ValueError("Invalid source line references in blinded findings")
    votes = {row["verdict"] for row in findings.values()}
    overall = "critical_error" if "critical_error" in votes else "unclear" if "unclear" in votes else "pass"
    save(folder/"judgment.json", {"anonymous_sha256": digest(source), "verdict": overall, "criteria": findings})
    return overall


def release(run, card):
    """Record the arm's decision independently of its blinded source review."""
    import torch
    from compare_method_checks import load_model
    from experiment_runner import real_preflight

    model = run / "model.py"
    config = json.loads((run/"config.json").read_text(encoding="utf8"))
    torch.set_num_threads(config["threads"])
    manifest, _ = verify_inputs(SimpleNamespace(card=card, run=run))
    inputs = {"model_sha256": digest(model), "config_sha256": digest(run/"config.json"),
              "spec_sha256": digest(run/"paper_spec.json"), "card_sha256": digest(card)}
    if manifest.get("protocol") == "reviewed-v1":
        checks = sorted((run/"quality").glob("candidate_*/checks.json"))
        report = json.loads(checks[-1].read_text(encoding="utf8")) if checks else {}
        if not all(report.get(field) == inputs[field] for field in ("model_sha256", "config_sha256", "spec_sha256")):
            raise ValueError("Missing verdict for the final candidate")
        inputs["checks_sha256"] = digest(checks[-1])
        result = {"status": report["status"], "report": checks[-1].relative_to(run).as_posix()}
    saved = run/"release.json"
    if saved.exists():
        old = json.loads(saved.read_text(encoding="utf8"))
        if any(old.get(key, value) != value for key, value in inputs.items()) or old["model_sha256"] != inputs["model_sha256"]:
            raise ValueError("Recorded release refers to different evidence")
        if not all(key in old for key in inputs):
            # Add provenance to the pilot's already recorded preflight without re-scoring.
            save(saved, {**old, **inputs})
        return old["status"]
    if manifest.get("protocol") != "reviewed-v1":
        task = load_task(card)
        try:
            module = load_model(model, config)
        except Exception as exc:
            result = {"status": "failed", "reason": str(exc)}
        else:
            result = real_preflight(module, task, config)
    save(saved, {**result, **inputs})
    return result["status"]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run", type=Path, required=True)
    prep.add_argument("--root", type=Path, required=True)
    review = sub.add_parser("judge")
    for arg in ("folder", "card", "config"):
        review.add_argument("--" + arg, type=Path, required=True)
    decide = sub.add_parser("release")
    for arg in ("run", "card"):
        decide.add_argument("--" + arg, type=Path, required=True)
    args = cli.parse_args()
    with exclusive_run(args.folder if args.stage == "judge" else args.run):
        if args.stage == "prepare":
            print(prepare(args.run, args.root))
        elif args.stage == "judge":
            print(judge(args.folder, args.card, args.config))
        else:
            print(release(args.run, args.card))


if __name__ == "__main__":
    main()
