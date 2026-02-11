from __future__ import annotations

from typing import Any

from src.core.schemas import Diagnostic, EvidenceBundle, Prediction, Question, Resolution, Score


ERROR_TAXONOMY = {
    "overconfidence",
    "underconfidence",
    "evidence_gap",
    "timing_miss",
    "source_conflict",
    "resolver_ambiguity",
    "execution_issue",
}


def _recommended_patch(error_type: str) -> str:
    patches = {
        "overconfidence": (
            "Checklist:\n"
            "- Add explicit base-rate prior before final aggregation.\n"
            "- Enforce uncertainty floor near event cutoffs.\n"
            "- Require counter-hypothesis paragraph in prompt."
        ),
        "underconfidence": (
            "Checklist:\n"
            "- Increase weight for decisive, high-trust evidence.\n"
            "- Add trigger conditions for tail outcomes.\n"
            "- Calibrate with recent domain-specific outcomes."
        ),
        "evidence_gap": (
            "Checklist:\n"
            "- Increase retrieval breadth and diversity.\n"
            "- Add source freshness and authority constraints.\n"
            "- Fail forecast when evidence bundle is below minimum size."
        ),
        "timing_miss": (
            "Checklist:\n"
            "- Add explicit timeline decomposition.\n"
            "- Track leading indicators with time-to-event scoring.\n"
            "- Penalize stale evidence in ranking."
        ),
        "source_conflict": (
            "Checklist:\n"
            "- Add contradiction detection stage.\n"
            "- Prefer primary sources for resolution-critical facts.\n"
            "- Route unresolved conflicts to manual bucket."
        ),
        "resolver_ambiguity": (
            "Checklist:\n"
            "- Mark resolver as manual-review required.\n"
            "- Capture objective fallback sources in resolver_config.\n"
            "- Prevent scoring until confidence threshold is met."
        ),
        "execution_issue": (
            "Checklist:\n"
            "- Add retry/backoff and rate-limit handling.\n"
            "- Capture failure class in run metadata.\n"
            "- Requeue failed question with jittered delay."
        ),
    }
    return patches.get(error_type, patches["execution_issue"])


def diagnose_prediction(
    question: Question,
    prediction: Prediction,
    resolution: Resolution,
    score: Score,
    evidence: EvidenceBundle | None,
    llm_assisted: bool = False,
) -> Diagnostic:
    # deterministic first-pass
    p = prediction.p_ens
    y = resolution.y
    error_type = "execution_issue"
    if resolution.status != "resolved" or resolution.resolution_confidence < 0.5:
        error_type = "resolver_ambiguity"
    elif p >= 0.8 and y == 0:
        error_type = "overconfidence"
    elif p <= 0.2 and y == 1:
        error_type = "underconfidence"
    elif (evidence is None) or len(evidence.items) < 3:
        error_type = "evidence_gap"
    elif score.brier_ens > 0.25 and abs(p - y) > 0.5:
        error_type = "timing_miss"

    if error_type not in ERROR_TAXONOMY:
        error_type = "execution_issue"

    evidence_count = len(evidence.items) if evidence else 0
    top_sources = []
    if evidence:
        top_sources = [f"{item.url}#{item.rank}" for item in evidence.items[:3]]

    notes = [
        f"Prediction={p:.3f}, outcome={y:.1f}, brier={score.brier_ens:.3f}.",
        f"Evidence bundle items={evidence_count}; top refs={top_sources or ['none']}.",
        f"Resolution confidence={resolution.resolution_confidence:.2f}; resolver={question.resolver_type}.",
    ]

    # Optional LLM assist can extend notes, but deterministic path remains default.
    if llm_assisted:
        notes.append("LLM-assisted augmentation disabled in V1 deterministic mode unless explicitly implemented.")

    return Diagnostic(
        question_id=question.id,
        error_type=error_type,
        structured_notes=notes,
        recommended_patch=_recommended_patch(error_type),
    )

