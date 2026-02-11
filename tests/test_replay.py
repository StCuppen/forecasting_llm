from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.core.replay import replay_output_shape
from src.core.schemas import EvidenceBundle, EvidenceItem, Prediction


class ReplayTests(unittest.TestCase):
    def test_replay_output_shape(self) -> None:
        prediction = Prediction(
            question_id="q1",
            run_id="r1",
            made_at=datetime(2026, 2, 11, tzinfo=timezone.utc),
            p_ens=0.63,
            p_agents={"a": 0.6, "b": 0.66},
            model_versions={"a": "m1", "b": "m2"},
            evidence_bundle_id="b1",
            cost_estimate=None,
            latency=3.2,
            forecast_context={"prompt_template_version": "v1"},
        )
        bundle = EvidenceBundle(
            bundle_id="b1",
            items=[
                EvidenceItem(
                    url="https://a.example",
                    retrieved_at=datetime(2026, 2, 11, tzinfo=timezone.utc),
                    snippet_hash="h1",
                    trust_score=0.8,
                    rank=1,
                    snippet="A",
                ),
                EvidenceItem(
                    url="https://b.example",
                    retrieved_at=datetime(2026, 2, 11, tzinfo=timezone.utc),
                    snippet_hash="h2",
                    trust_score=0.7,
                    rank=2,
                    snippet="B",
                ),
            ],
        )
        shape = replay_output_shape(prediction, bundle, config_version="league-v1")
        self.assertEqual(shape["run_id"], "r1")
        self.assertEqual(shape["agent_count"], 2)
        self.assertEqual(shape["evidence_count"], 2)
        self.assertEqual(shape["config_version"], "league-v1")


if __name__ == "__main__":
    unittest.main()

