from __future__ import annotations

import unittest

from src.core.scoring import brier_score, clip_probability, log_loss


class ScoringTests(unittest.TestCase):
    def test_brier_score(self) -> None:
        self.assertAlmostEqual(brier_score(0.8, 1.0), 0.04)
        self.assertAlmostEqual(brier_score(0.2, 0.0), 0.04)

    def test_log_loss(self) -> None:
        value = log_loss(0.8, 1.0)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 0.3)

    def test_probability_clip(self) -> None:
        self.assertGreater(clip_probability(0.0), 0.0)
        self.assertLess(clip_probability(1.0), 1.0)


if __name__ == "__main__":
    unittest.main()

