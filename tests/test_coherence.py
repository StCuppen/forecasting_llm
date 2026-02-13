from bot.coherence import (
    FamilyDetection,
    QuestionStub,
    detect_mutually_exclusive_pairs,
    detect_threshold_ladder,
    enforce_monotonicity,
    enforce_sum_to_one,
    project_family_constraints,
)


def test_enforce_sum_to_one_normalizes() -> None:
    probs = {"A": 0.2, "B": 0.5, "C": 0.6}
    norm = enforce_sum_to_one(probs)
    assert abs(sum(norm.values()) - 1.0) < 1e-12
    assert all(v >= 0.0 for v in norm.values())


def test_enforce_monotonicity_makes_decreasing_ladder() -> None:
    ladder = [(100.0, 0.40), (200.0, 0.45), (300.0, 0.30)]
    fixed = enforce_monotonicity(ladder)
    probs = [p for _, p in fixed]
    assert probs[0] >= probs[1] >= probs[2]


def test_detect_threshold_ladder_groups_members() -> None:
    questions = [
        QuestionStub(id="q1", title="Will CPI exceed 2.0% in June 2026?"),
        QuestionStub(id="q2", title="Will CPI exceed 2.5% in June 2026?"),
        QuestionStub(id="q3", title="Unrelated title"),
    ]
    family = detect_threshold_ladder(questions)
    assert family is not None
    assert family.family_type == "threshold_ladder"
    assert set(family.members) == {"q1", "q2"}


def test_detect_mutually_exclusive_pair() -> None:
    questions = [
        QuestionStub(id="a", title="Will Team A win the 2026 finals?"),
        QuestionStub(id="b", title="Will Team B win the 2026 finals?"),
    ]
    families = detect_mutually_exclusive_pairs(questions)
    assert len(families) == 1
    assert families[0].family_type == "mutually_exclusive_pair"
    assert set(families[0].members) == {"a", "b"}


def test_project_family_constraints_normalizes_pair() -> None:
    probs = {"a": 0.7, "b": 0.6}
    families = [FamilyDetection(family_type="mutually_exclusive_pair", members=["a", "b"], score=0.8)]
    projected, adjustments = project_family_constraints(probs, families)
    assert abs(projected["a"] + projected["b"] - 1.0) < 1e-12
    assert len(adjustments) >= 1
