from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QuestionStub:
    id: str
    title: str
    resolution_criteria: str = ""


@dataclass
class FamilyDetection:
    family_type: str
    members: List[str]
    score: float
    details: dict[str, Any] = field(default_factory=dict)


def enforce_sum_to_one(probabilities: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, v) for v in probabilities.values())
    if total <= 0:
        n = len(probabilities) or 1
        return {k: 1.0 / n for k in probabilities}
    return {k: max(0.0, v) / total for k, v in probabilities.items()}


def _pava_non_decreasing(values: List[float]) -> List[float]:
    if not values:
        return []
    blocks: List[Tuple[float, int]] = [(v, 1) for v in values]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            total_weight = blocks[i][1] + blocks[i + 1][1]
            avg = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / total_weight
            blocks[i : i + 2] = [(avg, total_weight)]
            if i > 0:
                i -= 1
        else:
            i += 1
    out: List[float] = []
    for avg, weight in blocks:
        out.extend([avg] * weight)
    return out


def enforce_monotonicity(ladder: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not ladder:
        return []
    sorted_ladder = sorted(ladder, key=lambda x: x[0])
    thresholds = [x[0] for x in sorted_ladder]
    probs = [x[1] for x in sorted_ladder]
    fitted_neg = _pava_non_decreasing([-p for p in probs])
    fitted = [-v for v in fitted_neg]
    return list(zip(thresholds, fitted))


def _enforce_non_decreasing(ladder: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not ladder:
        return []
    sorted_ladder = sorted(ladder, key=lambda x: x[0])
    thresholds = [x[0] for x in sorted_ladder]
    probs = [x[1] for x in sorted_ladder]
    fitted = _pava_non_decreasing(probs)
    return list(zip(thresholds, fitted))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _extract_threshold_info(question: QuestionStub) -> tuple[str, float, str] | None:
    text = f"{question.title}\n{question.resolution_criteria}"
    pattern = re.compile(
        r"(?:\b(exceed|over|above|at least|>=|greater than)\b\s*([-+]?\d[\d,]*(?:\.\d+)?))|"
        r"(?:\b(below|under|at most|<=|less than)\b\s*([-+]?\d[\d,]*(?:\.\d+)?))",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None

    direction = "ge"
    raw_threshold = ""
    if match.group(1):
        direction = "ge"
        raw_threshold = match.group(2) or ""
    elif match.group(3):
        direction = "le"
        raw_threshold = match.group(4) or ""
    if not raw_threshold:
        return None
    try:
        threshold = float(raw_threshold.replace(",", ""))
    except ValueError:
        return None

    stem_text = pattern.sub(" ", text)
    stem_text = re.sub(r"\b\d[\d,]*(?:\.\d+)?\b", " ", stem_text)
    stem = _normalize_text(stem_text)
    if len(stem) < 8:
        return None
    return direction, threshold, stem


def detect_threshold_ladders(questions: List[QuestionStub]) -> list[FamilyDetection]:
    groups: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for q in questions:
        info = _extract_threshold_info(q)
        if info is None:
            continue
        direction, threshold, stem = info
        groups.setdefault((direction, stem), []).append((q.id, threshold))

    families: list[FamilyDetection] = []
    for (direction, stem), members in groups.items():
        if len(members) < 2:
            continue
        sorted_members = sorted(members, key=lambda x: x[1])
        member_ids = [m[0] for m in sorted_members]
        threshold_map = {m[0]: float(m[1]) for m in sorted_members}
        families.append(
            FamilyDetection(
                family_type="threshold_ladder",
                members=member_ids,
                score=min(1.0, 0.45 + 0.1 * len(member_ids)),
                details={
                    "direction": direction,
                    "stem": stem[:120],
                    "thresholds": threshold_map,
                },
            )
        )
    return families


def detect_threshold_ladder(questions: List[QuestionStub]) -> Optional[FamilyDetection]:
    families = detect_threshold_ladders(questions)
    if not families:
        return None
    return sorted(families, key=lambda f: f.score, reverse=True)[0]


def _extract_winner_signature(question: QuestionStub) -> tuple[str, str] | None:
    title = question.title.strip()
    match = re.search(r"^\s*will\s+(.+?)\s+win\b(.*)$", title, re.IGNORECASE)
    if not match:
        return None
    candidate = _normalize_text(match.group(1))
    suffix = _normalize_text(match.group(2) + " " + question.resolution_criteria)
    if not candidate or len(candidate) < 2 or len(suffix) < 8:
        return None
    return candidate, suffix


def detect_mutually_exclusive_pairs(questions: List[QuestionStub]) -> list[FamilyDetection]:
    groups: dict[str, list[tuple[str, str]]] = {}
    for q in questions:
        sig = _extract_winner_signature(q)
        if sig is None:
            continue
        candidate, suffix = sig
        groups.setdefault(suffix, []).append((q.id, candidate))

    out: list[FamilyDetection] = []
    for suffix, members in groups.items():
        if len(members) != 2:
            continue
        distinct_candidates = {candidate for _, candidate in members}
        if len(distinct_candidates) != 2:
            continue
        ids = [mid for mid, _ in members]
        out.append(
            FamilyDetection(
                family_type="mutually_exclusive_pair",
                members=ids,
                score=0.8,
                details={
                    "stem": suffix[:120],
                    "candidates": {mid: cand for mid, cand in members},
                },
            )
        )
    return out


def detect_multiple_choice_sets(questions: List[QuestionStub]) -> list[FamilyDetection]:
    option_pattern = re.compile(r"^(.*?)(?:\s*[-:|]\s*)(?:option|choice)\s+(.+)$", re.IGNORECASE)
    groups: dict[str, list[tuple[str, str]]] = {}
    for q in questions:
        title = q.title.strip()
        match = option_pattern.search(title)
        if not match:
            continue
        stem = _normalize_text(match.group(1) + " " + q.resolution_criteria)
        option = _normalize_text(match.group(2))
        if len(stem) < 20 or len(option) < 1:
            continue
        groups.setdefault(stem, []).append((q.id, option))

    out: list[FamilyDetection] = []
    for stem, members in groups.items():
        if len(members) < 2:
            continue
        out.append(
            FamilyDetection(
                family_type="multiple_choice_set",
                members=[mid for mid, _ in members],
                score=min(1.0, 0.4 + 0.1 * len(members)),
                details={
                    "stem": stem[:120],
                    "options": {mid: opt for mid, opt in members},
                },
            )
        )
    return out


def detect_question_families(questions: List[QuestionStub]) -> list[FamilyDetection]:
    families: list[FamilyDetection] = []
    families.extend(detect_multiple_choice_sets(questions))
    families.extend(detect_threshold_ladders(questions))
    families.extend(detect_mutually_exclusive_pairs(questions))
    return families


def project_family_constraints(
    probabilities: Dict[str, float],
    families: List[FamilyDetection],
) -> tuple[Dict[str, float], list[dict[str, Any]]]:
    projected = {qid: max(0.0, min(1.0, float(p))) for qid, p in probabilities.items()}
    adjustments: list[dict[str, Any]] = []

    for family in families:
        family_type = family.family_type
        if family_type in {"multiple_choice_set", "mutually_exclusive_pair"}:
            subset = {qid: projected[qid] for qid in family.members if qid in projected}
            if len(subset) < 2:
                continue
            normalized = enforce_sum_to_one(subset)
            for qid, new_p in normalized.items():
                before = projected[qid]
                projected[qid] = new_p
                if abs(new_p - before) > 1e-9:
                    adjustments.append(
                        {
                            "question_id": qid,
                            "family_type": family_type,
                            "before": before,
                            "after": new_p,
                            "members": list(family.members),
                        }
                    )
            continue

        if family_type == "threshold_ladder":
            thresholds = family.details.get("thresholds", {})
            direction = str(family.details.get("direction", "ge")).lower()
            ladder_rows: list[tuple[float, float, str]] = []
            for qid in family.members:
                if qid not in projected:
                    continue
                if qid not in thresholds:
                    continue
                ladder_rows.append((float(thresholds[qid]), projected[qid], qid))
            if len(ladder_rows) < 2:
                continue
            ladder_rows.sort(key=lambda x: x[0])
            ladder = [(thr, prob) for thr, prob, _ in ladder_rows]
            if direction == "le":
                fixed = _enforce_non_decreasing(ladder)
            else:
                fixed = enforce_monotonicity(ladder)
            for idx, (_, _, qid) in enumerate(ladder_rows):
                before = projected[qid]
                after = max(0.0, min(1.0, float(fixed[idx][1])))
                projected[qid] = after
                if abs(after - before) > 1e-9:
                    adjustments.append(
                        {
                            "question_id": qid,
                            "family_type": family_type,
                            "before": before,
                            "after": after,
                            "members": list(family.members),
                            "threshold": float(ladder_rows[idx][0]),
                            "direction": direction,
                        }
                    )

    return projected, adjustments
