import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as flow
from verify_run import audit_generation


class ReviewWorkflowTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.run = Path(tmp.name) / "run"
        self.run.mkdir()
        self.args = SimpleNamespace(run=self.run, card=self.run/"task.json", python=sys.executable)
        self.paper = self.run/"paper.pdf"
        for path in (self.paper, self.args.card, self.run/"config.json", self.run/"paper_spec.json"):
            path.write_text("{}", encoding="utf8")
        task = SimpleNamespace(paper_path=self.paper)
        mock_task = patch.object(flow, "load_task", return_value=task)
        mock_task.start()
        self.addCleanup(mock_task.stop)
        flow.save(self.run/"inputs.json", {"protocol":"reviewed-v1", "task_sha256":flow.digest(self.args.card),
                  "paper_sha256":flow.digest(self.paper), "config_sha256":flow.digest(self.run/"config.json")})
        self.spec = flow.digest(self.run/"paper_spec.json")
        flow.event(self.run, "spec_approved", spec_sha256=self.spec)
        flow.publish_candidate(self.run, "candidate = 1\n", self.spec)

    def test_uncertain_or_stale_observation_never_dispatches_repair(self):
        with patch.object(flow, "assess", return_value={"status":"inconclusive"}), patch.object(flow, "repair") as repair:
            with self.assertRaisesRegex(ValueError, "inconclusive"):
                flow.train(self.args)
            repair.assert_not_called()
        (self.run/"paper_spec.json").write_text("changed", encoding="utf8")
        with self.assertRaisesRegex(ValueError, "Specification differs"):
            flow.verify_inputs(self.args)

    def test_repair_diagnosis_tracks_two_candidates_and_recovers_pending_publication(self):
        first = self.run/"model.py"
        sha = flow.digest(first)
        attempt = self.run/"quality/candidate_01"
        attempt.mkdir(parents=True)
        (attempt/"model.py").write_bytes(first.read_bytes())
        failure = {"criterion":"axis", "status":"failed", "detail":"mixed heads"}
        flow.save(attempt/"checks.json", {"status":"failed", "model_sha256":sha,
                  "spec_sha256":self.spec, "config_sha256":flow.digest(self.run/"config.json"), "checks":[failure]})
        with patch.object(flow, "repair_source", return_value="candidate = 2\n") as model:
            flow.repair(self.args)
        self.assertEqual(model.call_args.kwargs["failures"], [failure])
        self.assertEqual(flow.digest(attempt/"model.py"), sha)
        first.replace(self.run/"model.pending.py")
        _, versions = flow.verify_inputs(self.args)
        self.assertEqual(versions[-1]["previous_model_sha256"], sha)
        self.assertEqual(len(versions), 2)
        self.assertEqual(first.read_text(encoding="utf8"), "candidate = 2\n")
        # The independent fit audit must recognize a final repair response.
        folder = self.run/"calls/repair-abc"
        folder.mkdir(parents=True)
        flow.save(folder/"call.json", {"status":"completed", "stage":"repair", "attempt_id":"repair-abc"})
        (folder/"response.txt").write_text(json.dumps({"python":"candidate = 2"}), encoding="utf8")
        result = audit_generation(self.run/"fit", {"model_sha256":flow.digest(first)})
        self.assertEqual(result["generation_call_id"], "repair-abc")


if __name__ == "__main__":
    unittest.main()
