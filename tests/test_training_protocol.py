"""Regression checks for window coverage, sampled variates and published test MSE."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_runner import batches, evaluate


class ZeroForecast(torch.nn.Module):
    def forward(self, observations):
        return torch.zeros_like(observations[:, :2])


class TestTrainingProtocol(unittest.TestCase):
    def setUp(self):
        self.config = dict(seq_len=3, pred_len=2, batch_size=4, eval_batch_size=4,
                           variables_per_train_window=2)
        # A measurement encodes its time and its sensor, allowing both axes to be checked.
        self.values = 10 * torch.arange(10)[:, None] + torch.arange(3)[None, :]
        self.values = self.values.float()

    def test_training_samples_the_same_variables_for_input_and_target(self):
        seen = []
        generator = torch.Generator().manual_seed(17)
        for observations, target in batches(self.values, self.config, training=True, generator=generator):
            self.assertEqual(observations.shape[1:], (3, 2))
            self.assertEqual(target.shape[1:], (2, 2))
            for x, y in zip(observations, target):
                start = int(x[0, 0]) // 10
                sensors = (x[0].long() % 10).tolist()
                self.assertEqual(len(set(sensors)), 2)
                self.assertTrue(torch.equal(x, self.values[start:start + 3, sensors]))
                self.assertTrue(torch.equal(y, self.values[start + 3:start + 5, sensors]))
                seen.append(start)
        self.assertEqual(sorted(seen), list(range(6)))

    def test_evaluation_scores_all_windows_horizons_and_variables(self):
        # Six windows form a batch of four followed by two. Including the last
        # row catches dropping that final batch or averaging per-batch MSEs.
        forecasts = np.full((6, 2, 3), np.nan, dtype=np.float32)
        result = evaluate(ZeroForecast(), self.values, self.config,
                          torch.device("cpu"), predictions=forecasts)
        self.assertTrue(np.all(forecasts == 0))
        self.assertEqual(result["windows"], 6)
        self.assertEqual(result["elements"], 6 * 2 * 3)
        truth_series = self.values[3:].numpy().astype(np.float64)
        truth = np.stack([truth_series[window:window + 2] for window in range(6)])
        independent_sse = np.square(forecasts.astype(np.float64) - truth).sum()
        self.assertEqual(result["sse"], independent_sse)
        self.assertEqual(result["mse"], independent_sse / truth.size)


if __name__ == "__main__":
    unittest.main()
