"""Detect omitted residuals and pre-norm with equivalent implementations as controls."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from residual_check import check_residual_order


class Forecast(nn.Module):
    def __init__(self, block_factory):
        super().__init__()
        self.embed = nn.Linear(6, 8)
        self.layers = nn.Sequential(block_factory(), block_factory())
        self.head = nn.Linear(8, 3)

    def forward(self, history):
        return self.head(self.layers(self.embed(history.transpose(1, 2)))).transpose(1, 2)


class FunctionalBlock(nn.Module):
    def __init__(self, mode="correct"):
        super().__init__()
        self.mode = mode
        self.a = nn.MultiheadAttention(8, 2, batch_first=True)
        self.up = nn.Linear(8, 13)
        self.down = nn.Linear(13, 8)

    def forward(self, h):
        a = self.a(h, h, h, need_weights=False)[0]
        u = F.layer_norm(a if self.mode == "omit_attention_skip" else a + h, (8,))
        f = self.down(F.relu(self.up(u)))
        return F.layer_norm(f if self.mode == "omit_ffn_skip" else f + u, (8,))


class TestResidualCheck(unittest.TestCase):
    def setUp(self):
        self.config = {"e_layers": 2, "d_model": 8}
        torch.manual_seed(17)
        self.x = torch.randn(2, 6, 5)

    def check(self, block):
        return check_residual_order(Forecast(block).eval(), self.x, self.config)

    def test_builtin_and_functional_postnorm(self):
        for block in [lambda: nn.TransformerEncoderLayer(8, 2, 13, batch_first=True), FunctionalBlock]:
            result = self.check(block)
            self.assertEqual(result["status"], "passed", result)
            self.assertEqual(len(result["layers"]), 2)

    def test_absent_attention_or_ffn_skip(self):
        for mode, component in [("omit_attention_skip", "attention"), ("omit_ffn_skip", "ffn")]:
            result = self.check(lambda: FunctionalBlock(mode))
            self.assertEqual(result["status"], "failed", result)
            self.assertTrue(all(layer[component]["status"] == "failed" for layer in result["layers"]), result)

    def test_pre_norm_rejected(self):
        result = self.check(lambda: nn.TransformerEncoderLayer(8, 2, 13, batch_first=True, norm_first=True))
        self.assertEqual(result["status"], "failed", result)

    def test_unobserved_implementation_requires_review(self):
        self.assertEqual(self.check(nn.Identity)["status"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
