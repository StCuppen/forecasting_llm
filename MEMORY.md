# Agent Memory (Local Only)

Purpose: keep durable lessons for this repo so repeated failures are avoided.
Visibility: local-only via `.git/info/exclude`.

## Operating Notes
- Date base: use absolute dates in analysis and status checks.
- Verify latest workflow behavior from artifacts/logs, not only green status.
- If run fetched questions but attempts=0, inspect `bot.log` for skip reason.

## Current State (2026-02-11)
- Latest 12 `run_tournament.yaml` runs (08:15Z to 11:00Z) all fetched 8 questions.
- Each run found 8 OPEN, with 4 supported binary and 4 unsupported non-binary.
- Posting attempts logged: 0 across all 12 runs.
- Raw `bot.log` shows: `Skipping 4 previously forecasted questions`.
- Guardrail bug in workflow summary step:
  - `supported_open` parsing produced text (`Found 8 OPEN questions (4 supported binary)`) instead of an integer.
  - Bash numeric check then errored (`syntax error in expression`) and did not fail the job.
  - Example run: `21902447702` (`2026-02-11T11:00:15Z`).
- Fix implemented locally (2026-02-11):
  - Resolved merge conflicts in `.github/workflows/run_tournament.yaml` and `bot/main_forecast_pipeline.py`.
  - Workflow summary parsing now extracts numeric `supported_open` robustly and validates numeric shape before comparison.
  - Added `question_types` workflow input and bot CLI flag (`all` or `binary`).
  - Bot now supports non-binary processing paths:
    - multiple choice: uniform fallback prior
    - numeric: bounded monotonic fallback distribution
  - Tournament path now uses `scan_tournament(..., question_types=...)` with explicit tournament ID selection.
- Local working tree currently has unresolved conflicts in:
  - none (conflicts resolved in local edits)

## Fast Checks
1. `git status --short`
2. `python scripts/analyze_tournament_runs.py --repo StCuppen/forecasting_llm --workflow run_tournament.yaml --limit 12 --method artifacts`
3. Inspect one artifact `bot.log` directly if attempts are zero.

## Next Actions
- Push changes and run one dispatch with:
  - `question_types=all`
  - `force_repost=true`
  - `min_questions=1`
  - `min_post_attempts=1`
- Validate latest run via `scripts/analyze_tournament_runs.py` and one raw artifact `bot.log`.
- Keep this file updated after every workflow diagnosis/fix.

## Session Update (2026-02-11, Codex GPT-5)
- Built and pushed a V1 7-day feedback-loop league pipeline to `main` in three commits:
  - `f674f9b`: core league implementation (connectors, core modules, jobs, migrations, docs, tests).
  - `0bdd3af`: rolling forecast cap of 25 predictions per 7-day window.
  - `898b19a`: feedback-loop artifact mode tagging + lessons doc + ignore `forecasts/`.
- Added dedicated prediction artifact logging under `predictions/feedback_loop/` with markdown output containing:
  - probability outputs (ensemble + agents),
  - model/version metadata,
  - evidence snapshot metadata,
  - rationale text.
- Verified one live artifact with actual forecast content:
  - `predictions/feedback_loop/20260211T202357.022577+0000_manifold_gu5zc9ZPcc_1d1235ee.md`
  - includes `Mode: live` and populated probability/rationale fields.
- Recorded key reliability lesson:
  - "artifact written" is not enough; validate presence of actual forecast fields (`p_ens` + rationale) before treating a run as successful.

## Session Update (2026-02-11, Claude Opus 4.6) — ReAct Agent Repair

### Problem Diagnosed
The live artifact (`20260211T202357...manifold_gu5zc9ZPcc_1d1235ee.md`) revealed the prediction pipeline was broken:
- Both models (MiMo v2 Flash, Gemini 3 Flash) output **0.500000** (the default fallback).
- Both hit `[Agent reached iteration limit]` — neither produced `FINAL_FORECAST`.
- Evidence URLs were massively duplicated (~40 entries, ~15 unique).
- Gemini output contained raw `SEARCH()` commands and Chinese characters in the rationale.

### Root Causes Identified
1. **Single search per iteration**: `re.search()` captured only the first `SEARCH()` per response; models emitting multiple searches had all but the first ignored, wasting turns.
2. **Overly complex ReAct prompt**: 8 mandatory sections (pathway tables, gate analysis, market calibration) — too heavy for free/preview models. They spent all 5 iterations searching and never reached structured output.
3. **Late force-completion**: Models were only forced to conclude on iteration 4-5, by which point the budget was already burned.
4. **Tight token budget**: 8,000 max_tokens per response was insufficient for structured output.
5. **No evidence dedup**: Same URLs appended on every search hit.

### Fixes Applied (files: `bot/agent/agent_experiment.py`, `bot/agent/prompts.py`)

| Fix | Detail |
|-----|--------|
| Multi-search extraction | `re.search()` → `re.findall()`: now processes ALL `SEARCH()` calls per response, batches results back |
| Evidence dedup | Added `seen_urls` set; only new URLs appended to `all_sources` |
| Earlier force-conclusion | Now forces conclusion when `search_count >= 3 AND iteration >= 3`, not just on iteration 4-5 |
| 3-tier probability fallback | (1) Dedicated low-temp LLM call asking for just a number, (2) regex scan of conversation for probability statements, (3) 0.5 as true last resort |
| Token budget increase | 8,000 → 16,000 max_tokens per response |
| Simplified ReAct prompt | Reduced from 8 sections to 4: resolution criteria, key evidence, reasoning, FINAL_FORECAST. Added "Do NOT output 50% unless genuinely a coin flip" guardrail. Tells models they can issue multiple SEARCH() calls. |

### Key Lessons
- **Free/preview models need simple prompts**: Pathway tables and gate analysis sound great in theory but MiMo and Gemini Flash can't follow them reliably. Keep the output format to: criteria, evidence, reasoning, probability.
- **Always process ALL tool calls per response**: If the agent emits 3 SEARCH() calls, process all 3 — don't waste an iteration on each.
- **Force conclusion proactively**: Don't wait until the last iteration. After 3 successful searches, the model has enough data — push it to decide.
- **Never trust 0.5 as a real forecast**: 0.5 almost always means extraction failed. The forced-extraction fallback (separate low-temp LLM call) is a safety net that should rarely be needed but catches failures.
- **Dedup evidence at ingestion time**: Building a `seen_urls` set is trivial and prevents the 3-5x URL duplication problem.

## Session Update (2026-02-11, Claude Opus 4.6) — Ensemble Validation Run

### Model Changes
- **Removed**: `xiaomi/mimo-v2-flash:free` — returns 404, model no longer available on OpenRouter
- **Added**: `moonshotai/kimi-k2.5` — reasoning model, strong structured output, efficient (2 iterations typical)
- **Added**: `openai/gpt-4o-mini` — fast, cheap, but weaker reasoning than Kimi/Gemini
- **Kept**: `google/gemini-3-flash-preview` — most thorough researcher (7 searches, named athletes, sport-specific analysis)
- **Note**: `openai/gpt-5-mini` returns 400 on this OpenRouter account. `moonshotai/kimi-k2.5:free` returns 404. Only paid Kimi works.

### Ensemble Filtering Fix
- Added failed-model filter: results with `probability == 0.5 AND search_count == 0` are excluded from the ensemble average
- Previously a dead model returning default 0.5 would drag a correct 98% down to 74%

### Validation Run: "Will Norway have the Most Gold Medals in the 2026 Winter Olympics?"
Results: Kimi 88%, Gemini 92%, GPT-4o Mini 65% → Ensemble 81.7%

**Per-model quality ranking:**
1. **Kimi K2.5**: Best efficiency + independent thinking. Found current standings (7-4 lead), Polymarket odds (91-93%), and discounted slightly below market. 3 searches, 2 iterations.
2. **Gemini 3 Flash**: Most thorough. 7 searches, found athlete-specific data, modeled "perfect storm" loss scenarios. Had one empty response costing an extra iteration.
3. **GPT-4o Mini**: Weakest. Used generic betting odds (-180 → 64%) without finding the actual current medal standings. Underinformed estimate.

### Key Findings
- **Simplified prompt works**: All 3 models followed the 4-section format and produced FINAL_FORECAST reliably
- **Multi-search extraction works**: Kimi and Gemini both issued 3-4 searches in first response, all processed correctly
- **GPT-4o Mini is a weak link**: Doesn't research deeply enough, anchors to generic odds rather than current event data
- **Equal weighting is naive**: A model with 7 deep searches should count more than one with 3 shallow ones

### Improvements Implemented (same session, second round)

| Improvement | Detail |
|-------------|--------|
| Parallel model execution | `asyncio.gather()` runs all 3 models simultaneously. Cut ~4min → ~2.3min on DHS question. |
| Search-depth weighted ensemble | Weight = `max(1, search_count)`. Kimi (8 searches) gets 57%, Gemini (5) gets 36%, GPT-4o Mini (1) gets 7%. Prevents shallow models from dragging down well-researched ones. |
| "Current state" prompt nudge | Added step 1 in WORKFLOW: "SEARCH for the current status/state of the topic". GPT-4o Mini now searches for current state first. |
| Empty response retry | When model returns empty, `iteration -= 1` before retry. Gemini's empty response no longer wastes an iteration slot. |

### Validation Run: "Will Congress fund the Department of Homeland Security by February 13?"

| Model | Prob | Searches | Iters | Tokens | Weight |
|-------|------|----------|-------|--------|--------|
| Kimi K2.5 | 35% | 8 | 4 | 67K | 57% |
| Gemini 3 Flash | 35% | 5 | 2 | 24K | 36% |
| GPT-4o Mini | 30% | 1 | 2 | 5K | 7% |
| **Ensemble** | **34.6%** | 14 | — | — | — |

**Quality assessment**: All 3 models converge on 30-35% — genuine agreement that a DHS shutdown is more likely than not. Kimi identified Polymarket at 56% shutdown probability and discounted. Gemini modeled specific pathways (30% "Recess CR", 5% sudden compromise, 65% strategic lapse). GPT-4o Mini still the weakest but at least in the right range.

**Total time**: 141s (parallel), down from ~4min sequential.

### Current Model Ranking (for this codebase)
1. **Kimi K2.5** — Best balance of research depth + reasoning quality + independent thinking
2. **Gemini 3 Flash** — Most thorough researcher, occasional empty responses, good pathway analysis
3. **GPT-4o Mini** — Fast but shallow. Adequate as a third voice, not as primary

### Remaining Open Issues
- `openai/gpt-5-mini` returns 400 on current OpenRouter account — revisit when access is available
- MWU weight updater needs resolution data to calibrate — run `resolve_due` + `score_and_diagnose` regularly
- Consider replacing GPT-4o Mini with DeepSeek R1 (`deepseek/deepseek-r1-0528:free`) if available for deeper reasoning

## Session Update (2026-02-11, Codex GPT-5) - Improvement Loop Continuation

### What Was Implemented

1. Per-iteration search cap in ReAct loop:
- Added `max_searches_per_iteration=3` to `run_react_agent(...)`.
- If a model emits many `SEARCH()` calls in one response, only up to 3 are executed.
- Added explicit system logging when SEARCH cap is applied.

2. Hard force-finalization behavior:
- Added `should_force` gating before executing searches.
- If the model already has enough evidence (iteration/search thresholds met), additional `SEARCH()` calls are ignored and the model is forced to output `FINAL_FORECAST`.
- This closes the loophole where models kept searching despite finalization prompts.

3. Reliability-aware fallback weighting:
- Added reliability classification for each model output:
  - `LOW`: forced extraction / iteration-limit style outputs.
  - `MEDIUM`: very high token usage (>= 60k total tokens).
  - `HIGH`: normal completion.
- Replaced fallback weighted average from pure `search_count` to quality-aware weight:
  - search contribution is capped (`min(search_count, 8)`)
  - forced extraction gets penalized (`x0.35`)
  - very high token usage gets a mild penalty (`x0.85` at >= 70k)

4. Judge prompt reliability guidance:
- Updated judge prompt to explicitly use reliability tags and discount `reliability=LOW` outliers unless corroborated.

5. Judge probability extraction robustness:
- Judge parser now strips markdown `*` and supports flexible `FINAL_PROBABILITY` patterns.
- Fixes missed extraction when judge outputs markdown like `**FINAL_PROBABILITY:** 31%`.

### Test Forecast Runs (Continuation)

Question tested:
- "Will the next US Retail Sales report (for January or a February flash estimate) show growth of at least 0.4%?"

Run A (before force-finalization guard):
- Output log: `forecasts/forecast_20260211_213552_Will_the_next_US_Retail_Sales_report_for_January.txt`
- Final prediction: 27.0%
- Weighted average: 31.1%
- Model metrics:
  - Kimi K2.5: 50% (11 searches, reliability=LOW)
  - Gemini 3 Flash: 25% (9 searches, reliability=MEDIUM)
  - GPT-4o Mini: 30% (3 searches, reliability=HIGH)
- Observation: still too much searching for Kimi/Gemini; Kimi ended in forced extraction.

Run B (after force-finalization guard):
- Output log: `forecasts/forecast_20260211_214128_Will_the_next_US_Retail_Sales_report_for_January.txt`
- Final prediction: 34.2%
- Weighted average: 34.2%
- Model metrics:
  - Kimi K2.5: 30% (6 searches, reliability=MEDIUM)
  - Gemini 3 Flash: 28% (6 searches, reliability=MEDIUM)
  - GPT-4o Mini: 55% (3 searches, reliability=HIGH)
- Observation: search behavior improved materially (11/9 down to 6/6), no forced extraction in this run, and total runtime dropped (~188s to ~112s).

### Key Lessons From This Loop
- Limiting SEARCH burst size alone is not enough; models can still keep searching unless post-threshold SEARCH calls are explicitly blocked.
- Reliability signals should be first-class in ensemble logic (forced extraction should never earn high influence from raw search volume).
- Judge output parsing must tolerate markdown formatting to avoid silent fallback to weighted averages.

## Session Update (2026-02-12)
- Locked default ensemble model roster to: `moonshotai/kimi-k2.5`, `google/gemini-2.5-flash`, `openai/gpt-5-mini`.
- Added inspectability modules with feature flags (spec lock, evidence ledger, numeric provenance, market snapshot, outlier cross-exam).
- Posted test forecast: Metaculus post `41501` (question id `41230`) with final prediction 2%.
- Tournament workflow reliability fix:
  - Removed fragile internal package report save path by setting `folder_to_save_reports_to=None` in `main_forecast_pipeline`.
  - Added `--max-open-questions` CLI and workflow input for controllable tournament runs.
  - Local smoke test confirmed tournament path can complete and post with `post_attempts=1`.
- Date-awareness verification:
  - Models recognized January 2026 had passed and treated the question as pending publication of January data (not as a future month event).
