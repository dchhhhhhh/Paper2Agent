"""Replay a known axis error through the same diagnostic repair prompt as production."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from run_experiment import repair_source, save

OUT = Path(__file__).resolve().parent / "axis_repair"
ARCHIVE = ROOT / "runs/traffic96_001"
CONFIG = ARCHIVE / "config.json"
CARD = ROOT / "examples/itransformer_traffic/task.json"


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=["before", "repair", "after"])
    stage = cli.parse_args().stage
    OUT.mkdir(exist_ok=True)
    if stage == "repair":
        before = json.loads((OUT/"before.json").read_text(encoding="utf-8"))
        failed = [item for item in before["checks"] if item["status"] == "failed"]
        if not failed:
            raise ValueError("No diagnosed fault")
        if (OUT/"after.py").exists():
            raise ValueError("Preserve the first repair attempt")
        source = repair_source(failures=failed, spec_file=ARCHIVE/"paper_spec.json", config_file=CONFIG,
                               source_file=OUT/"before.py", calls=OUT/"calls")
        (OUT/"after.py").write_text(source, encoding="utf8")
        return
    from method_checks import check_model
    if stage == "before":
        from compare_method_checks import CASES, edit_source
        case = next(c for c in CASES if c.name == "attend_over_heads")
        mutated = edit_source((ARCHIVE/"model.py").read_text(encoding="utf8"), case)
        (OUT/"before.py").write_text(mutated, encoding="utf8")
    result = check_model(OUT/(stage + ".py"), CONFIG, CARD)
    save(OUT/(stage + ".json"), result)
    print(result["status"])


if __name__ == "__main__":
    main()
