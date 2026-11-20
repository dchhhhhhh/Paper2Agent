import json
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]


class RealTrafficAcceptance(unittest.TestCase):
    def test_complete_example_returns_one_aligned_training_batch(self):
        example = PROJECT / "examples/itransformer_traffic/task.json"
        result = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(PROJECT / "check_task.py"), str(example)],
            capture_output=True, encoding="utf-8", timeout=180, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "train_batch_checked")
        self.assertEqual(report["model_calls"], 0)
        batch = report["batch"]
        self.assertEqual(batch["train_windows"], 12089)
        self.assertEqual(batch["input_shape"], [2, 96, 862])
        self.assertEqual(batch["label_shape"], [2, 144, 862])
        self.assertEqual(batch["forecast_target_shape"], [2, 96, 862])
        self.assertTrue(batch["finite"])
        self.assertTrue(batch["history_values_and_times_match"])
