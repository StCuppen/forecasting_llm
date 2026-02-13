# Conventions

## Model Roster (as of 2026-02-12)

| Slot | Model ID | Notes |
|------|----------|-------|
| Primary researcher | `moonshotai/kimi-k2.5` | Best efficiency + independent thinking. 2-3 iterations typical. Paid only (`:free` returns 404). |
| Primary reasoner | `google/gemini-2.5-flash` | Thorough, fast, good for planner/search/audit utility calls. |
| Budget / third voice | `openai/gpt-5-mini` | Fast and usable as a second opinion; avoid over-weighting shallow outputs. |

Default lean roster in code: 2x Kimi K2.5, 2x GPT-5 Mini, 1x Gemini 2.5 Flash.

## Model ID Gotchas
- `google/gemini-3-flash-preview` is valid on OpenRouter.
- `openai/gpt-5-mini` may return 400 on some OpenRouter accounts; verify account access.
- `moonshotai/kimi-k2.5:free` returns 404; use paid variant only.

## Lean Pipeline Rules
- Keep legacy staged/ReAct mode available as fallback, but default path is lean ensemble.
- Search planning should be adaptive (`SEARCH(...)` list), not fixed basket decomposition.
- Log both `planned_queries` and `executed_queries` for inspectability.
- Keep evidence visibility high: include top evidence rows with source/date/primary/relevance/bearing/snippet hash.

## Confidence and Gating
- Treat confidence as process quality, not forecast extremity alone.
- Include signal strength (`|p-0.5|`) and an `informativeness` label (`weak_view`/`strong_view`) in summaries.
- Gate logs should include both component scores and raw metrics (`evidence_count`, `distinct_sources`, `freshness_days`, etc.).
- Low-confidence gate action shrinks probability toward base rate/0.5.

## MC/Numeric Rules
- MC: aggregate per-option from model outputs, normalize to sum to 1.
- Numeric: aggregate percentiles 10/25/50/75/90, enforce monotonicity, clamp to bounds.
- Numeric unit and order-of-magnitude checks are required before accepting parsed output.
- Fallback distributions are safety net only when parsing/validation fails.

## Audit/Signpost Rules
- Red-team critique should leave a log artifact even if no change (`ran | no_change`).
- Claim audit must label claims `evidence_backed` vs `assumption`.
- Signpost extraction target is 3-5 events; keep a repair/fallback path when extraction is weak.
- `update_online` should evaluate signposts and trigger re-forecasting when they fire.

## Config
- `league.toml`: `dry_run_default=true` by default; explicitly set false for live runs.
- Tournament runtime: keep `folder_to_save_reports_to=None` to avoid package-name saver errors.
- Rolling forecast cap: 25 predictions per 7-day window.
