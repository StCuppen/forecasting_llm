# Tournament workflow diagnosis (2026-02-10)

## Root causes behind "successful workflow" but no predictions

1. **Wrong/default tournament target in tournament mode**
   - The bot used `MetaculusApi.CURRENT_AI_COMPETITION_ID` unless explicitly overridden.
   - In `forecasting_tools` v0.2.31 this value is pinned to older AI Benchmark constants, so runs can silently target the wrong tournament.

2. **Insufficient logging visibility in workflow artifacts**
   - Workflow originally captured only stdout into `bot.log`.
   - Python logging writes to stderr by default, so retrieval/submission lines were absent from artifacts and summary greps.

3. **No hard guardrails for "must retrieve and post"**
   - Even if zero questions were retrieved or zero submissions were attempted, the run could still report success.

## What was changed

- Added explicit tournament selection via `--tournament-id` in `bot/main_forecast_pipeline.py`.
- Added runtime guardrails:
  - `--min-questions`
  - `--min-post-attempts`
- Added structured run stats logging (`Retrieved ... questions from tournament`, `Run Stats: ...`) and explicit per-report error lines.
- Updated `.github/workflows/run_tournament.yaml`:
  - new dispatch inputs: `tournament_id`, `min_questions`, `min_post_attempts`
  - pass these flags to the bot
  - capture both stdout and stderr (`2>&1 | tee bot.log`)
  - include forecast failure counts in step summary

## Operational note

With workflow defaults (`min_questions=1`, `min_post_attempts=1`), the run now **fails fast** if it does not actually retrieve at least one question and attempt at least one post.
