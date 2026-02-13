from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List


@dataclass
class ForecastRun:
    model: str
    probability: float
    reasoning: str
    token_usage: int


@dataclass
class AggregatedForecast:
    p_raw: float
    p_calibrated: float
    dispersion: float
    confidence_class: str
    individual_runs: List[ForecastRun]
    n_runs: int
    n_trimmed: int


def trimmed_mean(probabilities: List[float], trim_fraction: float = 0.2) -> float:
    if not probabilities:
        raise ValueError("trimmed_mean requires at least one probability")
    sorted_p = sorted(probabilities)
    n = len(sorted_p)
    trim_count = max(1, int(n * trim_fraction))
    if n < 4:
        trim_count = 0
    trimmed = sorted_p[trim_count : n - trim_count] if trim_count > 0 else sorted_p
    return sum(trimmed) / len(trimmed)


def extremize(p: float, k: float = 1.73) -> float:
    if p <= 0.001 or p >= 0.999:
        return p
    log_odds = math.log(p / (1 - p))
    adjusted = log_odds * k
    return 1 / (1 + math.exp(-adjusted))


def compute_dispersion(probabilities: List[float]) -> float:
    n = len(probabilities)
    if n < 2:
        return 0.0
    mean = sum(probabilities) / n
    variance = sum((p - mean) ** 2 for p in probabilities) / (n - 1)
    return math.sqrt(variance)


def classify_confidence(dispersion: float, evidence_score: float, signal_strength: float) -> str:
    agreement = max(0.0, min(1.0, 1.0 - dispersion * 5.0))
    evidence = max(0.0, min(1.0, evidence_score))
    signal = max(0.0, min(1.0, signal_strength / 0.30))
    composite = 0.45 * evidence + 0.35 * agreement + 0.20 * signal
    if composite >= 0.72 and signal_strength >= 0.12:
        return "high"
    if composite <= 0.40:
        return "low"
    return "medium"


def aggregate_forecasts(
    runs: List[ForecastRun],
    extremize_k: float = 1.73,
    trim_fraction: float = 0.2,
    evidence_score: float = 0.5,
) -> AggregatedForecast:
    if not runs:
        raise ValueError("aggregate_forecasts requires at least one run")
    probabilities = [r.probability for r in runs]
    p_raw = trimmed_mean(probabilities, trim_fraction)
    # Dampen extremization by evidence quality: weak evidence → less pull from 50%
    effective_k = 1.0 + (extremize_k - 1.0) * max(0.0, min(1.0, evidence_score))
    p_calibrated = extremize(p_raw, effective_k)
    dispersion = compute_dispersion(probabilities)
    confidence = classify_confidence(
        dispersion=dispersion,
        evidence_score=evidence_score,
        signal_strength=abs(p_raw - 0.5),
    )
    p_calibrated = max(0.01, min(0.99, p_calibrated))
    n_trimmed = max(1, int(len(runs) * trim_fraction)) * 2 if len(runs) >= 4 else 0
    return AggregatedForecast(
        p_raw=p_raw,
        p_calibrated=p_calibrated,
        dispersion=dispersion,
        confidence_class=confidence,
        individual_runs=runs,
        n_runs=len(runs),
        n_trimmed=n_trimmed,
    )
