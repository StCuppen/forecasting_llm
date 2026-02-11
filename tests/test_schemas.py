from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.core.schemas import EvidenceBundle, EvidenceItem, Prediction, Question


class SchemaTests(unittest.TestCase):
    def test_question_roundtrip(self) -> None:
        q = Question(
            id="metaculus:123",
            source="metaculus",
            source_id="123",
            title="Test title",
            description="Test description",
            close_time=datetime(2026, 2, 15, tzinfo=timezone.utc),
            resolve_time_expected=datetime(2026, 2, 16, tzinfo=timezone.utc),
            tags=["macro", "rates"],
            resolver_type="metaculus_api",
            resolver_config={"endpoint": "https://example"},
            status="open",
        )
        clone = Question.from_record(q.to_record())
        self.assertEqual(q.id, clone.id)
        self.assertEqual(q.title, clone.title)
        self.assertEqual(q.tags, clone.tags)

    def test_prediction_roundtrip(self) -> None:
        p = Prediction(
            question_id="q1",
            run_id="r1",
            made_at=datetime(2026, 2, 11, tzinfo=timezone.utc),
            p_ens=0.61,
            p_agents={"a": 0.6, "b": 0.62},
            model_versions={"a": "m1", "b": "m2"},
            evidence_bundle_id="e1",
            cost_estimate=0.2,
            latency=4.5,
            forecast_context={"prompt_template_version": "v1"},
        )
        clone = Prediction.from_record(p.to_record())
        self.assertAlmostEqual(p.p_ens, clone.p_ens)
        self.assertEqual(set(p.p_agents.keys()), set(clone.p_agents.keys()))

    def test_evidence_bundle_roundtrip(self) -> None:
        bundle = EvidenceBundle(
            bundle_id="b1",
            items=[
                EvidenceItem(
                    url="https://example.com",
                    retrieved_at=datetime(2026, 2, 11, tzinfo=timezone.utc),
                    snippet_hash="abc",
                    trust_score=0.8,
                    rank=1,
                    snippet="snippet",
                )
            ],
            archived_text_hashes=["xyz"],
        )
        clone = EvidenceBundle.from_record(bundle.to_record())
        self.assertEqual(bundle.bundle_id, clone.bundle_id)
        self.assertEqual(len(clone.items), 1)
        self.assertEqual(clone.items[0].url, "https://example.com")


if __name__ == "__main__":
    unittest.main()

