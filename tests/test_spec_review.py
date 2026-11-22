"""Specification acceptance requires traceable evidence and an independent re-review."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spec_review


def findings():
    output = {}
    anchors = {
        "pdf/": ("pdf/4", "attention among variates"),
        "task/": ("task/metric", "MSE"),
        "config/": ("config/d_model", "32"),
    }
    for key, (prefix, _) in spec_review._CRITERIA.items():
        source, quote = anchors[prefix]
        output[key] = {"verdict": "pass", "targets": ["requirements_with_page_evidence/0"],
                       "explanation": "Supported in this test fixture.",
                       "evidence": [{"source": source, "quote": quote}]}
    return output


class TestSpecReview(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_dir = Path(directory.name)
        self.spec = {"method": "iTransformer", "requirements_with_page_evidence": ["original"],
                     "data_and_evaluation": ["keep this"], "ambiguous_or_tunable": []}
        self.args = dict(run=self.run_dir, spec=self.spec, pages="[PDF page 4]\nattention  among\nvariates",
                         experiment={"metric": "MSE"}, config={"d_model": 32})

    def test_flagged_entry_is_repaired_then_reviewed_fresh(self):
        rejected, approved = findings(), findings()
        rejected["variable_attention"]["verdict"] = "revise"
        edits = {"edits": [{"target": "requirements_with_page_evidence/0", "text": "correct axis"}]}
        with patch.object(spec_review, "invoke_model", side_effect=map(json.dumps, [rejected, edits, approved])) as call:
            fixed = spec_review.review_spec(**self.args)
        self.assertEqual(fixed["requirements_with_page_evidence"], ["correct axis"])
        self.assertEqual(fixed["data_and_evaluation"], ["keep this"])
        self.assertEqual(self.spec["requirements_with_page_evidence"], ["original"])
        self.assertEqual([c.kwargs["stage"] for c in call.call_args_list],
                         ["spec_review", "spec_repair", "spec_review"])
        self.assertEqual(len(list(self.run_dir.rglob("approved.json"))), 1)

    def test_invented_quote_cannot_approve_or_trigger_spec_repair(self):
        fabricated = findings()
        fabricated["encoder_head"]["evidence"][0]["quote"] = "a false citation"
        with patch.object(spec_review, "invoke_model", return_value=json.dumps(fabricated)) as call:
            with self.assertRaisesRegex(ValueError, "Unverifiable"):
                spec_review.review_spec(**self.args)
        self.assertEqual(call.call_count, 2)
        self.assertFalse(list(self.run_dir.rglob("approved.json")))

    def test_resume_after_repair_retries_only_interrupted_review(self):
        rejected = findings()
        rejected["variable_attention"]["verdict"] = "revise"
        edits = {"edits": [{"target": "requirements_with_page_evidence/0", "text": "repaired axis"}]}
        responses = [json.dumps(rejected), json.dumps(edits), TimeoutError("review backend interrupted")]
        with patch.object(spec_review, "invoke_model", side_effect=responses) as call:
            with self.assertRaises(TimeoutError):
                spec_review.review_spec(**self.args)
            self.assertEqual(call.call_count, 3)
        with patch.object(spec_review, "invoke_model", return_value=json.dumps(findings())) as resumed:
            fixed = spec_review.review_spec(**self.args)
        resumed.assert_called_once()
        self.assertEqual(resumed.call_args.kwargs["stage"], "spec_review")
        self.assertIn("repaired axis", resumed.call_args.kwargs["prompt"])
        self.assertEqual(fixed["requirements_with_page_evidence"], ["repaired axis"])
        with patch.object(spec_review, "invoke_model") as third:
            self.assertEqual(spec_review.review_spec(**self.args), fixed)
            third.assert_not_called()
        self.assertEqual(len(list(self.run_dir.rglob("session.json"))), 1)

    def test_restarting_does_not_refresh_invalid_citation_or_repair_budget(self):
        invalid = findings()
        invalid["encoder_head"]["evidence"][0]["quote"] = "not in supplied source"
        with patch.object(spec_review, "invoke_model", side_effect=[json.dumps(invalid), TimeoutError()]):
            with self.assertRaises(TimeoutError):
                spec_review.review_spec(**self.args)
        with patch.object(spec_review, "invoke_model", return_value=json.dumps(invalid)) as retried:
            with self.assertRaisesRegex(ValueError, "Unverifiable"):
                spec_review.review_spec(**self.args)
            retried.assert_called_once()
        with patch.object(spec_review, "invoke_model") as repeat:
            with self.assertRaisesRegex(ValueError, "Unverifiable"):
                spec_review.review_spec(**self.args)
            with self.assertRaisesRegex(ValueError, "budget changed"):
                spec_review.review_spec(**self.args, max_repairs=2)
            repeat.assert_not_called()

    def test_changed_paper_material_uses_new_session_and_preserves_legacy_audit(self):
        old = self.run_dir / "spec_reviews" / "review-old-random-name" / "approved.json"
        old.parent.mkdir(parents=True)
        old.write_text("legacy record", encoding="utf-8")
        with patch.object(spec_review, "invoke_model", return_value=json.dumps(findings())) as call:
            spec_review.review_spec(**self.args)
            changed = {**self.args, "pages": self.args["pages"] + "\nUpdated paper excerpt."}
            spec_review.review_spec(**changed)
            self.assertEqual(call.call_count, 2)
        self.assertEqual(len(list(self.run_dir.rglob("session.json"))), 2)
        self.assertEqual(old.read_text(encoding="utf-8"), "legacy record")

    def test_missing_material_or_exhausted_revisions_blocks_generation(self):
        for reject in ["unresolved", "revise"]:
            with self.subTest(reject=reject):
                report = findings()
                report["forecast_target"]["verdict"] = reject
                with patch.object(spec_review, "invoke_model", return_value=json.dumps(report)) as call:
                    with self.assertRaisesRegex(ValueError, "forecast_target"):
                        spec_review.review_spec(**{**self.args, "run": self.run_dir / reject}, max_repairs=0)
                self.assertEqual(call.call_count, 1)
        self.assertFalse(list(self.run_dir.rglob("approved.json")))

    def test_declines_unrelated_or_missing_edits(self):
        with self.assertRaisesRegex(ValueError, "exactly the flagged"):
            spec_review._apply_edits(self.spec, [{"target": "method", "text": "other"}],
                                     {"data_and_evaluation/0"})


if __name__ == "__main__":
    unittest.main()
