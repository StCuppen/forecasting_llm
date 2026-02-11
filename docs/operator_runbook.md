# Operator Runbook

## Daily Operation

Run one tick:
```bash
poetry run python -m src.jobs.cli tick --config league.toml
```

Recommended cadence:
- `ingest_7day`: every 1-6 hours
- `forecast_open`: every 1-6 hours
- `resolve_due`: every 1-6 hours
- `score_and_diagnose`: every 1-6 hours
- `update_online`: after scoring
- `weekly_calibrate`: weekly

Artifacts:
- Forecast rationale markdown is written to `predictions/feedback_loop/` for this pipeline.

Limit controls:
- Rolling 7-day forecast cap is controlled by `forecast.weekly_prediction_limit` (default `25`).
- When the cap is hit, `forecast_open` returns with `skipped_due_weekly_limit > 0`.

## Safe Local/Dry Run

```bash
poetry run python -m src.jobs.cli tick --dry-run --config league.toml
```

Dry run avoids live model calls and generates deterministic pseudo-forecasts.

## Debugging Resolver Issues

1. Check due unresolved:
```bash
poetry run python -m src.jobs.cli resolve_due --config league.toml
```

2. Inspect question row in SQLite:
- `questions.status` should become `resolved` or `manual_review`.
- `manual_review` means ambiguous or low-confidence resolver output.

3. Verify raw payload storage:
- `questions.raw_payload_json` (ingest snapshot)
- `resolutions.resolver_payload_raw_json` (resolved cases)

4. If API shape changed:
- update source connector parsing logic in `src/connectors/<source>.py`
- preserve raw payload for audit.

## Adding a New Connector

1. Add `src/connectors/<new_source>.py` implementing:
- `list_candidates(window_days)`
- `fetch_details(source_id)`
- `get_resolution(source_id)`

2. Update `src/connectors/__init__.py` factory.

3. Add config section to `league.toml`:
```toml
[sources.new_source]
enabled = true
base_url = "..."
timeout_seconds = 20
max_retries = 3
cache_ttl_seconds = 300
```

4. Add mocked integration test covering ingest + resolve path.

## Failure Modes and Actions

- `429` / rate-limit:
  - increase connector backoff / cache TTL
  - stagger cron runs
- zero forecasts:
  - ensure open questions are not all duplicates or already forecasted
  - run `ingest_7day` first
- no scoring:
  - verify `resolve_due` produced `resolved` rows
  - check prediction exists for resolved question

## Recovery

- Re-run missed stages idempotently:
  - ingest, resolve, score, update are upsert-based
- For a specific question:
  - manually set `questions.status='open'` to re-forecast
  - remove stale rows cautiously from `predictions`/`scores` if needed
