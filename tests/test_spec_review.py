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

    def test_missing_material_or_exhausted_revisions_blocks_generation(self):
        for reject in ["unresolved", "revise"]:
            with self.subTest(reject=reject):
                report = findings()
                report["forecast_target"]["verdict"] = reject
                with patch.object(spec_review, "invoke_model", return_value=json.dumps(report)) as call:
                    with self.assertRaisesRegex(ValueError, "forecast_target"):
                        spec_review.review_spec(**self.args, max_repairs=0)
                self.assertEqual(call.call_count, 1)
        self.assertFalse(list(self.run_dir.rglob("approved.json")))

    def test_declines_unrelated_or_missing_edits(self):
        with self.assertRaisesRegex(ValueError, "exactly the flagged"):
            spec_review._apply_edits(self.spec, [{"target": "method", "text": "other"}],
                                     {"data_and_evaluation/0"})


if __name__ == "__main__":
    unittest.main()
