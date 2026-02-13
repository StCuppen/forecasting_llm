from bot.aggregation import (
    ForecastRun,
    aggregate_forecasts,
    classify_confidence,
    compute_dispersion,
    extremize,
    trimmed_mean,
)


def test_trimmed_mean_with_five_values() -> None:
    probs = [0.1, 0.2, 0.6, 0.8, 0.9]
    # Drop 0.1 and 0.9 -> mean([0.2, 0.6, 0.8]) = 0.5333...
    assert abs(trimmed_mean(probs, 0.2) - (1.6 / 3.0)) < 1e-9


def test_extremize_moves_away_from_half() -> None:
    assert extremize(0.7, 1.73) > 0.7
    assert extremize(0.3, 1.73) < 0.3
    assert abs(extremize(0.5, 1.73) - 0.5) < 1e-12


def test_compute_dispersion_zero_for_singleton() -> None:
    assert compute_dispersion([0.4]) == 0.0


def test_aggregate_forecasts_clamps_probability() -> None:
    runs = [
        ForecastRun(model="a", probability=0.99, reasoning="", token_usage=10),
        ForecastRun(model="b", probability=0.98, reasoning="", token_usage=10),
        ForecastRun(model="c", probability=0.97, reasoning="", token_usage=10),
        ForecastRun(model="d", probability=0.96, reasoning="", token_usage=10),
        ForecastRun(model="e", probability=0.95, reasoning="", token_usage=10),
    ]
    out = aggregate_forecasts(runs=runs)
    assert 0.01 <= out.p_calibrated <= 0.99
    assert out.n_runs == 5
    assert out.n_trimmed == 2


def test_confidence_requires_signal_strength_for_high() -> None:
    # Low dispersion and decent evidence but weak signal should not be "high".
    assert classify_confidence(dispersion=0.03, evidence_score=0.8, signal_strength=0.04) != "high"
    assert classify_confidence(dispersion=0.03, evidence_score=0.8, signal_strength=0.2) == "high"


def test_evidence_weighted_extremize_dampens_with_low_evidence() -> None:
    runs = [
        ForecastRun(model="a", probability=0.15, reasoning="", token_usage=10),
        ForecastRun(model="b", probability=0.15, reasoning="", token_usage=10),
    ]
    full_ev = aggregate_forecasts(runs=runs, evidence_score=1.0)
    low_ev = aggregate_forecasts(runs=runs, evidence_score=0.3)
    # Low evidence result should be closer to raw 0.15 (less extremized)
    assert abs(low_ev.p_calibrated - 0.15) < abs(full_ev.p_calibrated - 0.15), (
        f"low_ev={low_ev.p_calibrated:.4f} should be closer to 0.15 than full_ev={full_ev.p_calibrated:.4f}"
    )


def test_evidence_weighted_extremize_identity_at_zero() -> None:
    runs = [
        ForecastRun(model="a", probability=0.30, reasoning="", token_usage=10),
    ]
    result = aggregate_forecasts(runs=runs, evidence_score=0.0)
    # effective_k = 1.0 + (1.73 - 1.0) * 0.0 = 1.0, which is identity
    assert abs(result.p_calibrated - result.p_raw) < 0.01, (
        f"Expected near-identity at evidence_score=0, got p_raw={result.p_raw:.4f}, p_cal={result.p_calibrated:.4f}"
    )
