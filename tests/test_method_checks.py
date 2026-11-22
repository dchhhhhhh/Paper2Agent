"""Checks for alternate public attention APIs and abstaining on unsupported instrumentation."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from method_checks import check_behavior, trace_forward


class TinyForecaster(nn.Module):
    use_norm = False

    def __init__(self, builtin=False, causal=False):
        super().__init__()
        self.builtin = builtin
        self.causal = causal
        self.embed = nn.Linear(8, 12)
        self.forecast = nn.Linear(12, 4)
        if builtin:
            self.mha = nn.MultiheadAttention(12, 3, batch_first=True)
        else:
            self.fused_projection = nn.Linear(12, 36)
            self.merge = nn.Linear(12, 12)

    def forward(self, x):
        z = self.embed(input=x.transpose(1, 2))
        b, n, d = z.shape
        if self.builtin:
            mask = torch.ones(n, n, dtype=torch.bool).triu(1) if self.causal else None
            attended, _ = self.mha(z, z, z, attn_mask=mask)
        else:
            q, k, v = self.fused_projection(z).reshape(b, n, 3, 3, 4).permute(2, 0, 3, 1, 4).unbind(0)
            # Mathematically equivalent to one unmasked call at the default scale.
            result = [F.scaled_dot_product_attention(tile * 0.5, k, v, scale=1.0) for tile in q.split(3, -2)]
            attended = self.merge(torch.cat(result, -2).transpose(1, 2).reshape(b, n, d))
        return self.forecast(attended + z).transpose(1, 2)


class TestMethodChecks(unittest.TestCase):
    config = dict(seq_len=8, pred_len=4, n_heads=3, d_model=12, e_layers=1)

    def test_other_attention_implementations(self):
        x = torch.randn(3, 8, 7)
        with torch.inference_mode():
            for builtin, causal, status in ((True, False, "passed"), (True, True, "failed"), (False, False, "passed")):
                with self.subTest(builtin=builtin, causal=causal):
                    _, result = trace_forward(TinyForecaster(builtin, causal).eval(), x, self.config, 2)
                    self.assertEqual(result["status"], status, result)

    def test_observer_error_abstains_after_successful_forward(self):
        with torch.inference_mode(), patch("method_checks.trace_forward", side_effect=TypeError("unsupported trace")):
            checks = check_behavior(TinyForecaster().eval(), self.config, 1)
        self.assertEqual(next(c["status"] for c in checks if c["criterion"] == "output_contract"), "passed")
        self.assertEqual(next(c["status"] for c in checks if c["criterion"] == "variable_attention"), "inconclusive")


if __name__ == "__main__":
    unittest.main()
