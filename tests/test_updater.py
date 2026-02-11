from __future__ import annotations

import unittest

from src.core.updater import (
    apply_calibrator,
    fit_histogram_calibrator,
    multiplicative_weights_update,
    normalize_weights,
)


class UpdaterTests(unittest.TestCase):
    def test_multiplicative_weights_update(self) -> None:
        weights = {"a": 0.5, "b": 0.5}
        losses = {"a": 0.1, "b": 0.4}
        updated = multiplicative_weights_update(weights, losses, eta=1.0)
        self.assertGreater(updated["a"], updated["b"])
        self.assertAlmostEqual(sum(updated.values()), 1.0, places=6)

    def test_fit_and_apply_calibrator(self) -> None:
        points = [(0.1, 0.0), (0.15, 0.0), (0.85, 1.0), (0.9, 1.0)]
        payload = fit_histogram_calibrator(points, bins=5)
        calibrated_low = apply_calibrator(payload, 0.1)
        calibrated_high = apply_calibrator(payload, 0.9)
        self.assertLess(calibrated_low, calibrated_high)

    def test_normalize_weights_zero_case(self) -> None:
        norm = normalize_weights({"a": 0.0, "b": -1.0})
        self.assertAlmostEqual(norm["a"] + norm["b"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

