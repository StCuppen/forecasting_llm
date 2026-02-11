from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from src.core.dedupe import find_duplicate
from src.core.schemas import Question, QuestionCandidate


class DedupeTests(unittest.TestCase):
    def test_duplicate_match(self) -> None:
        existing = [
            Question(
                id="manifold:1",
                source="manifold",
                source_id="1",
                title="Will Bitcoin be above $100k by Friday?",
                description="",
                close_time=datetime(2026, 2, 14, tzinfo=timezone.utc),
                resolve_time_expected=datetime(2026, 2, 14, tzinfo=timezone.utc),
                tags=["crypto"],
                resolver_type="manifold_api",
                resolver_config={},
                status="open",
            )
        ]
        candidate = QuestionCandidate(
            source="polymarket",
            source_id="abc",
            title="Will Bitcoin be above 100k by Friday",
            description="",
            close_time=datetime(2026, 2, 14, 3, tzinfo=timezone.utc),
            resolve_time_expected=datetime(2026, 2, 14, 3, tzinfo=timezone.utc),
            tags=["crypto"],
            resolver_type="polymarket_api",
            resolver_config={},
            status="open",
        )
        decision = find_duplicate(candidate, existing)
        self.assertEqual(decision.duplicate_of, "manifold:1")

    def test_no_duplicate_far_time(self) -> None:
        existing = [
            Question(
                id="manifold:1",
                source="manifold",
                source_id="1",
                title="Will Bitcoin be above $100k by Friday?",
                description="",
                close_time=datetime(2026, 3, 14, tzinfo=timezone.utc),
                resolve_time_expected=datetime(2026, 3, 14, tzinfo=timezone.utc),
                tags=["crypto"],
                resolver_type="manifold_api",
                resolver_config={},
                status="open",
            )
        ]
        candidate = QuestionCandidate(
            source="polymarket",
            source_id="abc",
            title="Will Bitcoin be above 100k by Friday",
            description="",
            close_time=datetime(2026, 2, 14, tzinfo=timezone.utc),
            resolve_time_expected=datetime(2026, 2, 14, tzinfo=timezone.utc),
            tags=["crypto"],
            resolver_type="polymarket_api",
            resolver_config={},
            status="open",
        )
        decision = find_duplicate(candidate, existing)
        self.assertIsNone(decision.duplicate_of)


if __name__ == "__main__":
    unittest.main()

