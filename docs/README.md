# 7-Day Forecasting League (V1)

This module adds a SQLite-first forecasting league pipeline that:

1. Ingests 7-day markets/questions from multiple sources.
2. Forecasts open questions with reproducibility metadata.
3. Resolves due questions from source APIs.
4. Scores predictions and stores diagnostics.
5. Updates per-domain ensemble weights online.
6. Optionally fits weekly calibration models.

## Repository Layout

- `src/connectors/`
  - `metaculus.py`, `kalshi.py`, `polymarket.py`, `manifold.py`
  - shared connector interface in `base.py`
- `src/core/`
  - schemas, config, storage, scoring, dedupe, diagnostics, updater, replay, dashboard
- `src/jobs/`
  - cron-friendly jobs and a unified CLI/tick runner
- `migrations/`
  - SQL schema for SQLite (`001_init.sql`)
- `tests/`
  - unit + mocked integration + replay tests

## Quick Start

1. Install deps:
```bash
poetry install
```

2. Configure:
- Edit `league.toml` (sources, db path, windows, updater/calibration knobs).

3. Run one end-to-end dry run:
```bash
poetry run python -m src.jobs.cli tick --dry-run --config league.toml
```

This command runs ingest -> forecast -> resolve -> score -> update and prints a lightweight dashboard.

## Job Commands

Run jobs individually:

```bash
poetry run python -m src.jobs.cli ingest_7day --config league.toml
poetry run python -m src.jobs.cli forecast_open --dry-run --config league.toml
poetry run python -m src.jobs.cli resolve_due --config league.toml
poetry run python -m src.jobs.cli score_and_diagnose --config league.toml
poetry run python -m src.jobs.cli update_online --config league.toml
poetry run python -m src.jobs.cli weekly_calibrate --config league.toml
```

## Reproducibility Guarantees

Each prediction stores:

- `run_id`, `made_at`
- ensemble + per-agent probabilities
- model versions (`model_versions`)
- `forecast_context` with prompt/pipeline version metadata
- evidence snapshot bundle id
- markdown rationale artifact path (`forecast_context.prediction_markdown_path`)

Each evidence bundle stores per-item:

- source URL
- retrieval timestamp
- snippet hash
- trust score and rank

Resolution and scoring are stored separately and only applied after resolution to avoid leakage.

## Typical Workflows

Local iteration (safe):
```bash
poetry run python -m src.jobs.cli tick --dry-run --config league.toml
```

Production tick (live forecast calls):
```bash
poetry run python -m src.jobs.cli tick --config league.toml
```

Prediction rationale markdown files are written to:
- `predictions/feedback_loop/`
This folder is dedicated to the feedback-loop pipeline outputs.

Weekly calibration:
```bash
poetry run python -m src.jobs.cli weekly_calibrate --config league.toml
```

## Notes

- SQLite is the default backend through `src/core/sqlite_storage.py`.
- Connector failures are non-fatal during ingest; failures are counted and skipped.
- Ambiguous resolutions are routed to `manual_review` question status.
- Dashboard text includes last resolved and biggest misses.
