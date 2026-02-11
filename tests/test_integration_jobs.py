from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from src.core.sqlite_storage import SQLiteStorage
from src.core.utils import utc_now
from src.core.schemas import QuestionCandidate, ResolutionCandidate
from src.jobs.forecast_open import run_forecast_open
from src.jobs.ingest_7day import run_ingest
from src.jobs.resolve_due import run_resolve_due
from src.jobs.score_and_diagnose import run_score_and_diagnose
from src.jobs.update_online import run_update_online


class _FakeConnector:
    def __init__(self, candidate: QuestionCandidate, resolution: ResolutionCandidate) -> None:
        self._candidate = candidate
        self._resolution = resolution

    def list_candidates(self, window_days: int = 7) -> list[QuestionCandidate]:
        return [self._candidate]

    def fetch_details(self, source_id: str) -> QuestionCandidate | None:
        return self._candidate

    def get_resolution(self, source_id: str) -> ResolutionCandidate:
        return self._resolution


class IntegrationJobTests(unittest.TestCase):
    def test_ingest_forecast_resolve_score_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = tmp_path / "league.toml"
            db_path = tmp_path / "league.sqlite3"
            md_dir = tmp_path / "predictions" / "feedback_loop"
            cfg.write_text(
                f"""
db_path = "{db_path.as_posix()}"
window_days = 7
[forecast]
dry_run_default = true
apply_calibration = false
max_questions_per_tick = 10
prediction_log_dir = "{md_dir.as_posix()}"
write_prediction_markdown = true
[sources.metaculus]
enabled = true
[updater]
eta = 0.4
default_weight = 1.0
[domain_keywords]
macro = ["macro"]
""",
                encoding="utf-8",
            )

            candidate = QuestionCandidate(
                source="metaculus",
                source_id="q1",
                title="Will CPI come in below 3% this week?",
                description="macro test",
                close_time=utc_now() - timedelta(hours=1),
                resolve_time_expected=utc_now() - timedelta(minutes=1),
                tags=["macro"],
                resolver_type="metaculus_api",
                resolver_config={},
                status="open",
                raw_payload={"id": "q1"},
            )
            resolution = ResolutionCandidate(
                source="metaculus",
                source_id="q1",
                status="resolved",
                resolved_at=utc_now(),
                y=1.0,
                resolution_confidence=0.95,
                raw_payload={"resolution": "YES"},
            )
            fake = _FakeConnector(candidate, resolution)

            with patch("src.jobs.ingest_7day.build_connectors", return_value={"metaculus": fake}):
                ingest_result = run_ingest(config_path=str(cfg))
            self.assertEqual(ingest_result["ingested"], 1)

            forecast_result = run_forecast_open(config_path=str(cfg), dry_run=True)
            self.assertEqual(forecast_result["forecasted"], 1)
            md_files = list(md_dir.glob("*.md"))
            self.assertEqual(len(md_files), 1)
            self.assertIn("Feedback Loop Prediction", md_files[0].read_text(encoding="utf-8"))

            with patch("src.jobs.resolve_due.build_connectors", return_value={"metaculus": fake}):
                resolve_result = run_resolve_due(config_path=str(cfg))
            self.assertEqual(resolve_result["resolved"], 1)

            score_result = run_score_and_diagnose(config_path=str(cfg))
            self.assertEqual(score_result["scored"], 1)

            update_result = run_update_online(config_path=str(cfg))
            self.assertTrue("macro" in update_result or "general" in update_result)

            store = SQLiteStorage(str(db_path))
            store.init()
            try:
                rows = store.list_recent_resolved_with_scores(limit=10)
                self.assertEqual(len(rows), 1)
            finally:
                store.close()

    def test_weekly_prediction_limit_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = tmp_path / "league.toml"
            db_path = tmp_path / "league.sqlite3"
            md_dir = tmp_path / "predictions" / "feedback_loop"
            cfg.write_text(
                f"""
db_path = "{db_path.as_posix()}"
window_days = 7
[forecast]
dry_run_default = true
apply_calibration = false
max_questions_per_tick = 10
weekly_prediction_limit = 1
prediction_log_dir = "{md_dir.as_posix()}"
write_prediction_markdown = true
[sources.metaculus]
enabled = true
[updater]
eta = 0.4
default_weight = 1.0
""",
                encoding="utf-8",
            )

            base_time = utc_now()
            candidate_1 = QuestionCandidate(
                source="metaculus",
                source_id="q1",
                title="Q1",
                description="d1",
                close_time=base_time + timedelta(hours=1),
                resolve_time_expected=base_time + timedelta(hours=2),
                tags=["macro"],
                resolver_type="metaculus_api",
                resolver_config={},
                status="open",
                raw_payload={"id": "q1"},
            )
            candidate_2 = QuestionCandidate(
                source="metaculus",
                source_id="q2",
                title="Q2",
                description="d2",
                close_time=base_time + timedelta(hours=1),
                resolve_time_expected=base_time + timedelta(hours=2),
                tags=["macro"],
                resolver_type="metaculus_api",
                resolver_config={},
                status="open",
                raw_payload={"id": "q2"},
            )

            class _FakeMultiConnector:
                def list_candidates(self, window_days: int = 7):
                    return [candidate_1, candidate_2]

                def fetch_details(self, source_id: str):
                    return candidate_1 if source_id == "q1" else candidate_2

                def get_resolution(self, source_id: str):
                    return ResolutionCandidate(source="metaculus", source_id=source_id, status="unresolved")

            with patch("src.jobs.ingest_7day.build_connectors", return_value={"metaculus": _FakeMultiConnector()}):
                ingest_result = run_ingest(config_path=str(cfg))
            self.assertEqual(ingest_result["ingested"], 2)

            first = run_forecast_open(config_path=str(cfg), dry_run=True)
            self.assertEqual(first["forecasted"], 1)
            second = run_forecast_open(config_path=str(cfg), dry_run=True)
            self.assertEqual(second["forecasted"], 0)
            self.assertGreaterEqual(second["skipped_due_weekly_limit"], 1)


if __name__ == "__main__":
    unittest.main()
