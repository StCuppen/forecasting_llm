from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher

from src.core.schemas import Question, QuestionCandidate
from src.core.utils import jaccard_similarity, normalize_title


@dataclass
class DedupeDecision:
    duplicate_of: str | None
    reason: str
    score: float


def _title_similarity(a: str, b: str) -> float:
    a_norm = normalize_title(a)
    b_norm = normalize_title(b)
    seq = SequenceMatcher(a=a_norm, b=b_norm).ratio()
    token = jaccard_similarity(a_norm.split(), b_norm.split())
    return 0.6 * seq + 0.4 * token


def find_duplicate(
    candidate: QuestionCandidate,
    existing: list[Question],
    title_threshold: float = 0.88,
    max_time_delta: timedelta = timedelta(hours=18),
) -> DedupeDecision:
    best_match_id: str | None = None
    best_score = 0.0
    best_reason = "no_match"
    for question in existing:
        similarity = _title_similarity(candidate.title, question.title)
        if similarity < title_threshold:
            continue

        if candidate.close_time and question.close_time:
            delta = abs(candidate.close_time - question.close_time)
            if delta > max_time_delta:
                continue

        if candidate.resolve_time_expected and question.resolve_time_expected:
            delta_resolve = abs(candidate.resolve_time_expected - question.resolve_time_expected)
            if delta_resolve > max_time_delta:
                continue

        if similarity > best_score:
            best_score = similarity
            best_match_id = question.id
            best_reason = "title_and_time_match"

    return DedupeDecision(
        duplicate_of=best_match_id,
        reason=best_reason,
        score=best_score,
    )

