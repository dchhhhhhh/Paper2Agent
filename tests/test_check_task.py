"""Exercise input validation through the same CLI used for a real batch."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parents[1]
ENTRY = PROJECT / "check_task.py"
EXAMPLE = PROJECT / "examples" / "itransformer_traffic" / "task.json"
PATH_FIELDS = ("paper_path", "data_path", "dataloader_path")


class CheckTaskCliTests(unittest.TestCase):
    def setUp(self):
        self.task = json.loads(EXAMPLE.read_text(encoding="utf-8-sig"))
        for field in PATH_FIELDS:
            self.task[field] = str((EXAMPLE.parent / self.task[field]).resolve())

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def run_input_check(self):
        task_file = self.directory / "task.json"
        task_file.write_text(json.dumps(self.task, ensure_ascii=False), encoding="utf-8")
        report_file = self.directory / "report.json"
        # Put a detectable stub at the first import location. Input validation
        # must finish before any attempt to import the training runtime.
        torch_marker = self.directory / "torch-imported"
        (self.directory / "torch.py").write_text(
            "from pathlib import Path\n"
            "Path(__file__).with_name('torch-imported').touch()\n"
            "raise RuntimeError('torch imported during input validation')\n",
            encoding="utf-8",
        )
        env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(self.directory))
        result = subprocess.run(
            [sys.executable, str(ENTRY), str(task_file), "--report", str(report_file)],
            cwd=self.directory, env=env, capture_output=True, encoding="utf-8",
            timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse(torch_marker.exists(), "input validation tried to import torch")
        report = json.loads(report_file.read_text(encoding="utf-8-sig"))
        self.assertEqual(report["stage"], "input_check")
        self.assertEqual(report["model_calls"], 0)
        return report

    def test_missing_data_names_the_field_and_waits(self):
        self.task["data_path"] = str(self.directory / "missing.csv")
        report = self.run_input_check()
        self.assertEqual(report["status"], "waiting_for_input")
        self.assertIn("data_path", [issue["field"] for issue in report["issues"]])

    def test_unfilled_paths_from_original_template_are_reported(self):
        for field in PATH_FIELDS:
            self.task[field] = None
        report = self.run_input_check()
        self.assertEqual(report["status"], "waiting_for_input")
        fields = {issue["field"] for issue in report["issues"]}
        self.assertTrue(set(PATH_FIELDS).issubset(fields), fields)

    def test_unsupported_forecast_length_stops_before_loading(self):
        self.task["experiment"]["pred_len"] = 192
        report = self.run_input_check()
        self.assertEqual(report["status"], "invalid_task")


if __name__ == "__main__":
    unittest.main()
