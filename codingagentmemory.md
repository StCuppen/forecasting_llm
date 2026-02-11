
## Session Update (2026-02-11, Codex GPT-5) - Safe Guardrail Modules + Posted Test Forecast

### Requested model roster corrected
Default ensemble models in `run_ensemble_forecast(...)` are now:
- `moonshotai/kimi-k2.5`
- `google/gemini-2.5-flash`
- `openai/gpt-5-mini`

### Implemented (v1 = warnings/flags first, no automatic probability edits)

1. Canonical spec lens (not cage)
- Added `CanonicalSpec` extraction step (`extract_canonical_spec`) before model runs.
- Injected canonical spec text into ReAct prompt.
- Added required `Spec echo` in final format.
- Added spec consistency classifier with statuses: `OK`, `MINOR_DRIFT`, `MAJOR_DRIFT`.
- For `MAJOR_DRIFT`, runs one repair pass; if still major drift, warns and flags (`SPEC_REPAIR_FAILED`) rather than hard fail.

2. Evidence ledger (structured memory)
- Added automatic `evidence_ledger` entries per retrieved source:
  - `id`, `source_type`, `url_ref`, `retrieved_at`, `snippet`, `directness_tag`.
- Added directness tagging (`DIRECT` / `PROXY` / `CONTEXT`) from source domain heuristics.

3. Validators with warnings/risk flags
- Added soft validators that output `warnings[]` and `risk_flags[]` on every model output:
  - spec drift warnings (`SPEC_MAJORDRIFT`)
  - numeric provenance warning + orphan count (`ORPHAN_NUMERICS_HIGH`)
  - market hallucination warning if odds are mentioned but no direct market snapshot (`MARKET_HALLUCINATION`)
  - proxy-heavy evidence warning (`PROXY_HEAVY`)
- V1 behavior intentionally does not auto-change forecast probability from validator warnings.

4. Numeric policy: prove it or label it
- Added numeric-claim extractor + provenance classifier (`sourced` / `assumption` / `orphan`).
- Logs orphan counts into diagnostics and warnings banner.

5. Market snapshot module with explicit none case
- Added `market_snapshot = {found: bool, items: [...]}` built from retrieved search result text.
- Prompt now explicitly says to write `No direct market found.` when direct market odds are unavailable.

6. Outlier handling via cross-exam
- Added optional outlier interrogation (`outlier_xexam`) when model probability differs from median by threshold.
- Outlier prompt requires top 3 drivers each tied to `LEDGER-x` or `ASSUMPTION`.
- If grounding is weak, outlier is downweighted (interrogate, then discount).

7. Feature-flag plumbing
- Added runtime flags:
  - `spec_lock`
  - `evidence_ledger`
  - `numeric_provenance`
  - `market_snapshot`
  - `outlier_xexam`
- Flags are passed through ensemble + per-model runs and recorded in run outputs.

8. Ablation runner scaffold
- Added `scripts/run_forecast_ablation.py`:
  - runs flag combos on holdout questions from JSONL dataset (`question`, `y`),
  - reports avg Brier/logloss,
  - reports `spec_mismatch_rate`, `orphan_numeric_rate`, `market_hallucination_rate`, token/runtime metrics,
  - outputs JSON results for comparison.

### Test forecast run (posted)
Question URL (resolves within next week):
- `https://www.metaculus.com/questions/41501/china-1-for-japan-visitors-in-jan-2026/`

Run output file:
- `forecasts/forecast_20260211_225612_httpswwwmetaculuscomquestions41501china-1-.txt`

Per-model outputs:
- Kimi K2.5: 3%
- Gemini 2.5 Flash: 1%
- GPT-5 Mini: 20%
- Judge synthesis: 2%
- Final posted prediction: 2%

Posting result:
- Forecast was posted successfully to Metaculus (`question_id=41230`).

### Output analysis
- Spec echo appears in model outputs and spec framing reduced target ambiguity.
- Validators are surfacing meaningful issues:
  - all models received proxy-heavy warnings (`PROXY_HEAVY`) due source mix,
  - GPT-5 Mini produced high orphan-numeric count and got `ORPHAN_NUMERICS_HIGH`.
- Judge discounted the outlier (GPT-5 Mini at 20%) and converged near Kimi/Gemini low probabilities.
- Inspectability improved significantly: each model now carries structured diagnostics (spec status, numeric provenance report, market snapshot state, evidence ledger).

### Follow-up bugfixes after run
- Fixed Metaculus URL enrichment scoping bug caused by local import shadowing.
- Fixed comment-post call signature to use `post_id` for `post_question_comment(...)`.

## Session Update (2026-02-12, Codex GPT-5) - Tournament Workflow Reliability + Date-Awareness Check

### Tournament workflow status
- Reproduced a real tournament runtime failure during local workflow-equivalent execution:
  - Crash: `RuntimeError: Package name not found`
  - Source: `forecasting_tools` internal report saver path when `folder_to_save_reports_to` is set.
- Fix applied in `bot/main_forecast_pipeline.py`:
  - Set `folder_to_save_reports_to=None` (we already persist outputs via `--output-file`).
- Added safe tournament throttle for operational testing and incident response:
  - New CLI arg: `--max-open-questions` (default 0 = no cap)
  - Wired through `scan_tournament(..., max_open_questions=...)`.
- Updated GitHub workflow `.github/workflows/run_tournament.yaml`:
  - Added `workflow_dispatch` input `max_open_questions`
  - Passes through to bot CLI.

### Verified tournament path (local smoke test)
- Command executed in workflow-equivalent mode:
  - tournament: `metaculus-cup-spring-2026`
  - `question_types=binary`
  - `force_repost=true`
  - `min_questions=1`
  - `min_post_attempts=1`
  - `max_open_questions=1`
- Observed successful operational stats in logs:
  - `Retrieved 1 questions from tournament`
  - `post_attempts=1`
  - `errors=0`
  - `Submission Attempt: posting forecast | url=https://www.metaculus.com/questions/41595`
- Log file: `tournament_smoke_after_fix.log`.

### Date-awareness check ("did models catch January already passed?")
- On question `https://www.metaculus.com/questions/41501/china-1-for-japan-visitors-in-jan-2026/`, models did account for timeline:
  - Kimi explicitly: **"No January 2026 data released yet (as of Feb 11, 2026)"**
  - Gemini/GPT also framed resolution as January data pending JNTO release.
- Conclusion: yes, they recognized January was in the past while official data publication was still pending.

### Additional fixes
- Metaculus URL enrichment scope bug fixed in ensemble runner (avoid local import shadowing).
- Comment posting call fixed to use `post_id` for `MetaculusApi.post_question_comment(...)`.

### Workflow Summary Guardrail Fix (2026-02-12, follow-up)
- GitHub workflow run `21927552614` succeeded end-to-end (run job completed, post_attempts=1).
- A subsequent run failed in summary step when zero posts were expected because questions were already forecasted.
- Updated `.github/workflows/run_tournament.yaml` summary logic to parse `skipped_already_forecasted` and avoid false failure when all supported questions were skipped for that reason.
- The guardrail still fails true silent failures (supported open > 0, posted = 0, and not explained by already-forecasted skips).
