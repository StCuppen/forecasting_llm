from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.schemas import (
    Diagnostic,
    EvidenceBundle,
    Prediction,
    Question,
    Resolution,
    Score,
)
from src.core.storage import Storage
from src.core.utils import to_iso, utc_now


class SQLiteStorage(Storage):
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def init(self) -> None:
        migration_dir = Path("migrations")
        for sql_file in sorted(migration_dir.glob("*.sql")):
            sql = sql_file.read_text(encoding="utf-8")
            self.conn.executescript(sql)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def _json(self, value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _from_row_question(self, row: sqlite3.Row) -> Question:
        return Question.from_record(
            {
                "id": row["id"],
                "source": row["source"],
                "source_id": row["source_id"],
                "title": row["title"],
                "description": row["description"],
                "close_time": row["close_time"],
                "resolve_time_expected": row["resolve_time_expected"],
                "tags": json.loads(row["tags_json"]),
                "resolver_type": row["resolver_type"],
                "resolver_config": json.loads(row["resolver_config_json"]),
                "status": row["status"],
                "duplicate_of": row["duplicate_of"],
                "raw_payload": json.loads(row["raw_payload_json"]) if row["raw_payload_json"] else None,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
        )

    def _from_row_prediction(self, row: sqlite3.Row) -> Prediction:
        return Prediction.from_record(
            {
                "id": row["id"],
                "question_id": row["question_id"],
                "run_id": row["run_id"],
                "made_at": row["made_at"],
                "p_ens": row["p_ens"],
                "p_agents": json.loads(row["p_agents_json"]),
                "model_versions": json.loads(row["model_versions_json"]),
                "evidence_bundle_id": row["evidence_bundle_id"],
                "cost_estimate": row["cost_estimate"],
                "latency": row["latency"],
                "forecast_context": json.loads(row["forecast_context_json"]),
                "calibrator_version": row["calibrator_version"],
            }
        )

    def upsert_question(self, question: Question) -> None:
        now_iso = to_iso(utc_now())
        self.conn.execute(
            """
            INSERT INTO questions (
                id, source, source_id, title, description, close_time,
                resolve_time_expected, tags_json, resolver_type, resolver_config_json,
                status, duplicate_of, raw_payload_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                description=excluded.description,
                close_time=excluded.close_time,
                resolve_time_expected=excluded.resolve_time_expected,
                tags_json=excluded.tags_json,
                resolver_type=excluded.resolver_type,
                resolver_config_json=excluded.resolver_config_json,
                status=excluded.status,
                duplicate_of=excluded.duplicate_of,
                raw_payload_json=excluded.raw_payload_json,
                updated_at=excluded.updated_at
            """,
            (
                question.id,
                question.source,
                question.source_id,
                question.title,
                question.description,
                to_iso(question.close_time),
                to_iso(question.resolve_time_expected),
                self._json(question.tags),
                question.resolver_type,
                self._json(question.resolver_config),
                question.status,
                question.duplicate_of,
                self._json(question.raw_payload) if question.raw_payload is not None else None,
                to_iso(question.created_at),
                now_iso,
            ),
        )
        self.conn.execute("DELETE FROM question_tags WHERE question_id = ?", (question.id,))
        for tag in question.tags:
            self.conn.execute(
                "INSERT OR IGNORE INTO question_tags(question_id, tag) VALUES (?, ?)",
                (question.id, tag),
            )
        self.conn.commit()

    def list_questions(self, statuses: list[str] | None = None) -> list[Question]:
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            rows = self.conn.execute(
                f"SELECT * FROM questions WHERE status IN ({placeholders}) ORDER BY updated_at DESC",
                tuple(statuses),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM questions ORDER BY updated_at DESC").fetchall()
        return [self._from_row_question(row) for row in rows]

    def list_open_unforecasted_questions(self, limit: int | None = None) -> list[Question]:
        query = """
            SELECT q.*
            FROM questions q
            LEFT JOIN predictions p ON p.id = (
                SELECT p2.id FROM predictions p2
                WHERE p2.question_id = q.id
                ORDER BY p2.made_at DESC, p2.id DESC
                LIMIT 1
            )
            WHERE q.status = 'open' AND q.duplicate_of IS NULL AND p.id IS NULL
            ORDER BY q.resolve_time_expected ASC
        """
        params: list[Any] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [self._from_row_question(row) for row in rows]

    def list_due_unresolved_questions(self, now: datetime) -> list[Question]:
        rows = self.conn.execute(
            """
            SELECT q.*
            FROM questions q
            LEFT JOIN resolutions r ON r.question_id = q.id
            WHERE q.status IN ('open', 'forecasted')
              AND q.duplicate_of IS NULL
              AND q.resolve_time_expected IS NOT NULL
              AND q.resolve_time_expected <= ?
              AND r.question_id IS NULL
            ORDER BY q.resolve_time_expected ASC
            """,
            (to_iso(now),),
        ).fetchall()
        return [self._from_row_question(row) for row in rows]

    def list_resolved_unscored_questions(self) -> list[Question]:
        rows = self.conn.execute(
            """
            SELECT q.*
            FROM questions q
            JOIN resolutions r ON r.question_id = q.id
            LEFT JOIN scores s ON s.question_id = q.id
            WHERE q.status = 'resolved'
              AND q.duplicate_of IS NULL
              AND s.question_id IS NULL
            ORDER BY r.resolved_at ASC
            """
        ).fetchall()
        return [self._from_row_question(row) for row in rows]

    def mark_question_status(self, question_id: str, status: str) -> None:
        self.conn.execute(
            "UPDATE questions SET status = ?, updated_at = ? WHERE id = ?",
            (status, to_iso(utc_now()), question_id),
        )
        self.conn.commit()

    def set_question_duplicate(self, question_id: str, duplicate_of: str) -> None:
        self.conn.execute(
            "UPDATE questions SET duplicate_of = ?, updated_at = ? WHERE id = ?",
            (duplicate_of, to_iso(utc_now()), question_id),
        )
        self.conn.commit()

    def get_question(self, question_id: str) -> Question | None:
        row = self.conn.execute(
            "SELECT * FROM questions WHERE id = ?",
            (question_id,),
        ).fetchone()
        return self._from_row_question(row) if row else None

    def upsert_evidence_bundle(self, bundle: EvidenceBundle) -> None:
        self.conn.execute(
            """
            INSERT INTO evidence_bundles(bundle_id, items_json, archived_text_hashes_json, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(bundle_id) DO UPDATE SET
                items_json=excluded.items_json,
                archived_text_hashes_json=excluded.archived_text_hashes_json
            """,
            (
                bundle.bundle_id,
                self._json([item.to_record() for item in bundle.items]),
                self._json(bundle.archived_text_hashes or []),
                to_iso(bundle.created_at),
            ),
        )
        self.conn.commit()

    def get_evidence_bundle(self, bundle_id: str) -> EvidenceBundle | None:
        row = self.conn.execute(
            "SELECT * FROM evidence_bundles WHERE bundle_id = ?",
            (bundle_id,),
        ).fetchone()
        if row is None:
            return None
        return EvidenceBundle.from_record(
            {
                "bundle_id": row["bundle_id"],
                "items": json.loads(row["items_json"]),
                "archived_text_hashes": json.loads(row["archived_text_hashes_json"]),
                "created_at": row["created_at"],
            }
        )

    def insert_prediction(self, prediction: Prediction) -> int:
        self.conn.execute(
            """
            INSERT INTO predictions (
                question_id, run_id, made_at, p_ens, p_agents_json, model_versions_json,
                evidence_bundle_id, cost_estimate, latency, forecast_context_json, calibrator_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(question_id, run_id) DO UPDATE SET
                p_ens=excluded.p_ens,
                p_agents_json=excluded.p_agents_json,
                model_versions_json=excluded.model_versions_json,
                evidence_bundle_id=excluded.evidence_bundle_id,
                cost_estimate=excluded.cost_estimate,
                latency=excluded.latency,
                forecast_context_json=excluded.forecast_context_json,
                calibrator_version=excluded.calibrator_version
            """,
            (
                prediction.question_id,
                prediction.run_id,
                to_iso(prediction.made_at),
                prediction.p_ens,
                self._json(prediction.p_agents),
                self._json(prediction.model_versions),
                prediction.evidence_bundle_id,
                prediction.cost_estimate,
                prediction.latency,
                self._json(prediction.forecast_context),
                prediction.calibrator_version,
            ),
        )
        row = self.conn.execute(
            "SELECT id FROM predictions WHERE question_id = ? AND run_id = ?",
            (prediction.question_id, prediction.run_id),
        ).fetchone()
        self.conn.commit()
        return int(row["id"])

    def get_latest_prediction(self, question_id: str) -> Prediction | None:
        row = self.conn.execute(
            """
            SELECT *
            FROM predictions
            WHERE question_id = ?
            ORDER BY made_at DESC, id DESC
            LIMIT 1
            """,
            (question_id,),
        ).fetchone()
        if row is None:
            return None
        return self._from_row_prediction(row)

    def count_predictions_since(self, since: datetime) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM predictions WHERE made_at >= ?",
            (to_iso(since),),
        ).fetchone()
        return int(row["c"]) if row else 0

    def insert_resolution(self, resolution: Resolution) -> None:
        self.conn.execute(
            """
            INSERT INTO resolutions (
                question_id, resolved_at, y, resolver_payload_raw_json, resolution_confidence, status
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(question_id) DO UPDATE SET
                resolved_at=excluded.resolved_at,
                y=excluded.y,
                resolver_payload_raw_json=excluded.resolver_payload_raw_json,
                resolution_confidence=excluded.resolution_confidence,
                status=excluded.status
            """,
            (
                resolution.question_id,
                to_iso(resolution.resolved_at),
                resolution.y,
                self._json(resolution.resolver_payload_raw),
                resolution.resolution_confidence,
                resolution.status,
            ),
        )
        self.conn.commit()

    def get_resolution(self, question_id: str) -> Resolution | None:
        row = self.conn.execute(
            "SELECT * FROM resolutions WHERE question_id = ?",
            (question_id,),
        ).fetchone()
        if row is None:
            return None
        return Resolution.from_record(
            {
                "id": row["id"],
                "question_id": row["question_id"],
                "resolved_at": row["resolved_at"],
                "y": row["y"],
                "resolver_payload_raw": json.loads(row["resolver_payload_raw_json"]),
                "resolution_confidence": row["resolution_confidence"],
                "status": row["status"],
            }
        )

    def insert_score(self, score: Score) -> None:
        self.conn.execute(
            """
            INSERT INTO scores (
                question_id, scored_at, brier_ens, logloss_ens,
                brier_agents_json, logloss_agents_json, aggregates_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(question_id) DO UPDATE SET
                scored_at=excluded.scored_at,
                brier_ens=excluded.brier_ens,
                logloss_ens=excluded.logloss_ens,
                brier_agents_json=excluded.brier_agents_json,
                logloss_agents_json=excluded.logloss_agents_json,
                aggregates_json=excluded.aggregates_json
            """,
            (
                score.question_id,
                to_iso(score.scored_at),
                score.brier_ens,
                score.logloss_ens,
                self._json(score.brier_agents),
                self._json(score.logloss_agents),
                self._json(score.aggregates),
            ),
        )
        self.conn.commit()

    def insert_diagnostic(self, diagnostic: Diagnostic) -> None:
        self.conn.execute(
            """
            INSERT INTO diagnostics (
                question_id, error_type, structured_notes_json, recommended_patch, created_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(question_id) DO UPDATE SET
                error_type=excluded.error_type,
                structured_notes_json=excluded.structured_notes_json,
                recommended_patch=excluded.recommended_patch,
                created_at=excluded.created_at
            """,
            (
                diagnostic.question_id,
                diagnostic.error_type,
                self._json(diagnostic.structured_notes),
                diagnostic.recommended_patch,
                to_iso(diagnostic.created_at),
            ),
        )
        self.conn.commit()

    def list_recent_resolved_with_scores(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                q.id AS question_id,
                q.title AS title,
                q.tags_json AS tags_json,
                r.y AS y,
                s.brier_ens AS brier_ens,
                s.logloss_ens AS logloss_ens,
                p.p_ens AS p_ens,
                r.resolved_at AS resolved_at
            FROM questions q
            JOIN resolutions r ON r.question_id = q.id
            JOIN scores s ON s.question_id = q.id
            JOIN predictions p ON p.id = (
                SELECT p2.id FROM predictions p2
                WHERE p2.question_id = q.id
                ORDER BY p2.made_at DESC, p2.id DESC
                LIMIT 1
            )
            ORDER BY r.resolved_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "question_id": row["question_id"],
                    "title": row["title"],
                    "tags": json.loads(row["tags_json"]),
                    "y": float(row["y"]),
                    "brier_ens": float(row["brier_ens"]),
                    "logloss_ens": float(row["logloss_ens"]),
                    "p_ens": float(row["p_ens"]),
                    "resolved_at": row["resolved_at"],
                }
            )
        return output

    def list_scores_after(self, last_score_id: int) -> list[Score]:
        rows = self.conn.execute(
            "SELECT * FROM scores WHERE id > ? ORDER BY id ASC",
            (last_score_id,),
        ).fetchall()
        scores: list[Score] = []
        for row in rows:
            scores.append(
                Score.from_record(
                    {
                        "id": row["id"],
                        "question_id": row["question_id"],
                        "scored_at": row["scored_at"],
                        "brier_ens": row["brier_ens"],
                        "logloss_ens": row["logloss_ens"],
                        "brier_agents": json.loads(row["brier_agents_json"]),
                        "logloss_agents": json.loads(row["logloss_agents_json"]),
                        "aggregates": json.loads(row["aggregates_json"]),
                    }
                )
            )
        return scores

    def get_scores_for_calibration(self, domain_tag: str, limit: int) -> list[tuple[float, float]]:
        rows = self.conn.execute(
            """
            SELECT
                p.p_ens AS p_ens,
                r.y AS y,
                s.aggregates_json AS aggregates_json
            FROM scores s
            JOIN resolutions r ON r.question_id = s.question_id
            JOIN predictions p ON p.id = (
                SELECT p2.id FROM predictions p2
                WHERE p2.question_id = s.question_id
                ORDER BY p2.made_at DESC, p2.id DESC
                LIMIT 1
            )
            ORDER BY s.id DESC
            LIMIT ?
            """,
            (limit * 5,),
        ).fetchall()
        points: list[tuple[float, float]] = []
        for row in rows:
            aggregates = json.loads(row["aggregates_json"])
            if aggregates.get("domain_tag") == domain_tag:
                points.append((float(row["p_ens"]), float(row["y"])))
            if len(points) >= limit:
                break
        return points

    def get_state_value(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM updater_state WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row["value"]) if row else None

    def set_state_value(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO updater_state(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self.conn.commit()

    def get_domain_weights(self, domain_tag: str) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT agent_name, weight FROM weights WHERE domain_tag = ?",
            (domain_tag,),
        ).fetchall()
        return {str(row["agent_name"]): float(row["weight"]) for row in rows}

    def save_domain_weights(self, domain_tag: str, weights: dict[str, float]) -> None:
        now_iso = to_iso(utc_now())
        for agent_name, weight in weights.items():
            self.conn.execute(
                """
                INSERT INTO weights(domain_tag, agent_name, weight, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(domain_tag, agent_name) DO UPDATE SET
                    weight=excluded.weight,
                    updated_at=excluded.updated_at
                """,
                (domain_tag, agent_name, weight, now_iso),
            )
        self.conn.commit()

    def save_calibrator(self, domain_tag: str, version: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO calibrators(domain_tag, version, fitted_at, payload_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(domain_tag) DO UPDATE SET
                version=excluded.version,
                fitted_at=excluded.fitted_at,
                payload_json=excluded.payload_json
            """,
            (domain_tag, version, to_iso(utc_now()), self._json(payload)),
        )
        self.conn.commit()

    def get_calibrator(self, domain_tag: str) -> tuple[str, dict[str, Any]] | None:
        row = self.conn.execute(
            "SELECT version, payload_json FROM calibrators WHERE domain_tag = ?",
            (domain_tag,),
        ).fetchone()
        if row is None:
            return None
        return str(row["version"]), json.loads(row["payload_json"])
