from datetime import datetime, timedelta, timezone

from bot.aggregation import AggregatedForecast, ForecastRun
from bot.publish_gate import (
    EvidenceItem,
    SpecLockResult,
    evaluate_publish_gate,
    shrink_probability,
)


def _sample_forecast() -> AggregatedForecast:
    runs = [
        ForecastRun(model="m1", probability=0.55, reasoning="", token_usage=10),
        ForecastRun(model="m2", probability=0.58, reasoning="", token_usage=10),
        ForecastRun(model="m3", probability=0.57, reasoning="", token_usage=10),
    ]
    return AggregatedForecast(
        p_raw=0.566,
        p_calibrated=0.60,
        dispersion=0.02,
        confidence_class="high",
        individual_runs=runs,
        n_runs=3,
        n_trimmed=0,
    )


def test_publish_gate_publish_action_with_good_inputs() -> None:
    now = datetime.now(timezone.utc)
    evidence = [
        EvidenceItem(
            content="Recent official release",
            source_url="https://www.bls.gov/news.release/cpi.htm",
            source_name="bls.gov",
            retrieved_at=now,
            published_at=now - timedelta(days=2),
            relevance_score=0.9,
            is_primary_source=True,
        ),
        EvidenceItem(
            content="Secondary analysis",
            source_url="https://www.reuters.com/world/us/example/",
            source_name="Reuters",
            retrieved_at=now,
            published_at=now - timedelta(days=1),
            relevance_score=0.7,
            is_primary_source=False,
        ),
        EvidenceItem(
            content="Third corroborating source",
            source_url="https://www.bloomberg.com/economics/cpi/",
            source_name="Bloomberg",
            retrieved_at=now,
            published_at=now - timedelta(days=1),
            relevance_score=0.8,
            is_primary_source=False,
        ),
    ]
    gate = evaluate_publish_gate(
        forecast=_sample_forecast(),
        evidence=evidence,
        spec_lock=SpecLockResult(confidence=0.9, canonical_text="spec"),
        question_deadline=now + timedelta(days=10),
    )
    assert gate.action in {"publish", "publish_low_confidence"}
    assert 0.0 <= gate.gate_score <= 1.0


def test_shrink_probability_moves_toward_base_rate() -> None:
    p = 0.8
    out = shrink_probability(p=p, gate_score=0.4, base_rate=0.5)
    assert 0.5 < out < 0.8


def test_gate_hard_floor_no_primary_sources() -> None:
    from bot.publish_gate import compute_evidence_sufficiency

    now = datetime.now(timezone.utc)
    evidence = [
        EvidenceItem(
            content="x",
            source_url=f"https://site{i}.com/page",
            source_name=f"site{i}",
            retrieved_at=now,
            published_at=now,
            relevance_score=0.9,
            is_primary_source=False,
        )
        for i in range(6)
    ]
    score = compute_evidence_sufficiency(evidence)
    assert score <= 0.5, f"Expected <= 0.5 with 0 primary sources, got {score}"


def test_gate_hard_floor_low_mean_relevance() -> None:
    from bot.publish_gate import compute_evidence_sufficiency

    now = datetime.now(timezone.utc)
    evidence = [
        EvidenceItem(
            content="x",
            source_url=f"https://site{i}.com/page",
            source_name=f"site{i}",
            retrieved_at=now,
            published_at=now,
            relevance_score=0.5,
            is_primary_source=True,
        )
        for i in range(4)
    ]
    score = compute_evidence_sufficiency(evidence)
    assert score <= 0.6, f"Expected <= 0.6 with mean relevance < 0.7, got {score}"


def test_gate_abstain_fewer_than_3_evidence() -> None:
    now = datetime.now(timezone.utc)
    evidence = [
        EvidenceItem(
            content="x",
            source_url="https://gov.us/x",
            source_name="gov",
            retrieved_at=now,
            published_at=now,
            relevance_score=0.9,
            is_primary_source=True,
        ),
    ]
    gate = evaluate_publish_gate(
        forecast=_sample_forecast(),
        evidence=evidence,
        spec_lock=SpecLockResult(confidence=0.9, canonical_text="spec"),
        question_deadline=now + timedelta(days=10),
    )
    assert gate.action == "abstain"


def test_agreement_neutral_at_n1() -> None:
    now = datetime.now(timezone.utc)
    single_run_forecast = AggregatedForecast(
        p_raw=0.60,
        p_calibrated=0.65,
        dispersion=0.0,
        confidence_class="medium",
        individual_runs=[ForecastRun(model="m1", probability=0.60, reasoning="", token_usage=10)],
        n_runs=1,
        n_trimmed=0,
    )
    evidence = [
        EvidenceItem(
            content="x",
            source_url=f"https://site{i}.com/page",
            source_name=f"site{i}",
            retrieved_at=now,
            published_at=now,
            relevance_score=0.8,
            is_primary_source=True if i == 0 else False,
        )
        for i in range(4)
    ]
    gate = evaluate_publish_gate(
        forecast=single_run_forecast,
        evidence=evidence,
        spec_lock=SpecLockResult(confidence=0.9, canonical_text="spec"),
        question_deadline=now + timedelta(days=10),
    )
    assert gate.model_agreement == 0.5, f"Expected 0.5 at n=1, got {gate.model_agreement}"
    assert any("Single model run" in r for r in gate.reasons)
