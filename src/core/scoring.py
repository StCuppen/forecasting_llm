from __future__ import annotations

import math


def clip_probability(p: float, epsilon: float = 1e-9) -> float:
    return min(max(float(p), epsilon), 1.0 - epsilon)


def brier_score(p: float, y: float) -> float:
    return (float(p) - float(y)) ** 2


def log_loss(p: float, y: float, epsilon: float = 1e-9) -> float:
    p = clip_probability(p, epsilon=epsilon)
    y = float(y)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))

