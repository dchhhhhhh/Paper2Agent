"""Assemble the four prespecified trials; verify source, judgment and metric links."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "runs"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(run):
    events = [json.loads(line) for line in (run/"workflow.jsonl").read_text(encoding="utf8").splitlines()]
    calls = [read(path) for path in (run/"calls").glob("*/call.json")]
    row = {"run": run.name, "completed_calls": [c["stage"] for c in sorted(calls, key=lambda c: c["started_utc"])
                                                if c["status"] == "completed"],
           "failed_calls": [c["stage"] for c in calls if c["status"] == "failed"]}
    if not (run/"model.py").exists():
        if events[-1]["stage"] not in {"failed", "spec_held"}:
            raise ValueError(f"{run.name} is still active")
        return {**row, "release": "no_source", "reason": events[-1].get("error") or events[-1].get("reason"),
                "blind_verdict": None}
    release = read(run/"release.json")
    submission = read(run/"blind_submission.json")
    folder = HERE/"blind"/submission["id"]
    judgment = read(folder/"judgment.json")
    sha = digest(run/"model.py")
    if sha != release["model_sha256"] or sha != submission["original_sha256"]:
        raise ValueError(f"{run.name} candidate changed")
    if digest(folder/"source.py") != judgment["anonymous_sha256"] or judgment["anonymous_sha256"] != submission["anonymous_sha256"]:
        raise ValueError(f"{run.name} blind file changed")
    if "report" in release and digest(run/release["report"]) != release["checks_sha256"]:
        raise ValueError(f"{run.name} method report changed")
    row.update(release=release["status"], blind_id=submission["id"], blind_verdict=judgment["verdict"],
               source_sha256=sha)
    metric = run/"fit/metrics.json"
    if metric.exists():
        scores = read(metric)
        verification = read(run/"verification.json")
        if (scores["model_sha256"] != sha or verification["status"] != "verified"
                or verification["reported_mse"] != scores["test_mse"]):
            raise ValueError(f"{run.name} test metric has not been verified")
        row.update(test_mse=scores["test_mse"], best_validation_mse=scores["best_validation_mse"],
                   best_epoch=scores["best_epoch"], test_elements=scores["elements"],
                   independent_test_mse=verification["test_mse"])
    return row


if __name__ == "__main__":
    arms = {}
    for arm in ("baseline", "reviewed"):
        rows = [summarize(RUNS/f"quality_{arm}_{index:02d}") for index in (1, 2)]
        delivered = [r for r in rows if r["release"] == "passed"]
        arms[arm] = {"attempted": len(rows), "delivered": len(delivered),
                     "source_reviewed": sum(r["blind_verdict"] is not None for r in rows),
                     "delivered_with_critical_error": sum(r["blind_verdict"] == "critical_error" for r in delivered),
                     "delivered_unclear": sum(r["blind_verdict"] == "unclear" for r in delivered), "trials": rows}
    (HERE/"summary.json").write_text(json.dumps(arms, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
