# Lessons Learned

## Agent Behavior
- Lean adaptive search planning is materially better than fixed pre-baked basket plans for this codebase.
- Keep the planner output tied to execution logs (`planned` vs `executed`) or debugging becomes guesswork.
- If a model run produces probability ~0.5 with thin reasoning, check parse quality and evidence freshness before trusting it.

## Transparency and Diagnostics
- "Evidence items: N" without evidence rows is not verifiable. Include a compact top-evidence section every run.
- Gate scores need decomposition plus metrics; a single scalar gate score hides failure modes.
- Red-team, audit, and signpost passes should always leave a footprint even when they do not change output.

## Evidence Quality
- Flat relevance scores (all 0.6) mean the pipeline can't prioritize — evidence must be LLM-scored after extraction.
- Page chrome (nav bars, cookie banners) in evidence bundles means scraping worked but extraction didn't. Always run extraction after fetch.
- Primary source detection by domain suffix only (.gov/.edu) misses company canonical sources. Use question-aware heuristic lookup table.
- Count-based evidence_score (`len(items)/6`) rewards noise. Use mean relevance score instead.

## Calibration and Gate Rigor
- Extremizing weak evidence amplifies noise. Dampen extremization by `effective_k = 1.0 + (k-1.0) * evidence_score`.
- Gate must have hard floors: 0 primary sources → cap evidence_sufficiency at 0.5; mean_relevance < 0.7 → cap at 0.6; < 3 evidence items → abstain.
- Model agreement at n=1 is tautologically 1.0; set to neutral 0.5 to avoid false confidence signals.
- Signpost magnitudes must be anchored to final (post-calibration) probability and clipped to [1%, 99%]. Truncate pp values toward zero.

## Confidence Semantics
- High model agreement does not guarantee a strong signal; near-coinflip forecasts can still be high process quality.
- Keep `confidence_class` separate from `informativeness` so downstream consumers can distinguish "well-processed" from "strong directional view".

## Coherence
- Family projection is a cheap reliability win when questions are related (threshold ladders, mutual exclusivity).
- Apply coherence projection after per-question forecast generation and log before/after probabilities.
- Avoid false positives by keeping family detection conservative.

## Numeric/MC Quality
- MC/numeric fallback distributions should be last-resort safety only.
- Numeric validation needs both bounds and unit/order-of-magnitude checks; otherwise parse-success can still be semantically wrong.
- Enforce percentile monotonicity after any clamping step.

## Infrastructure
- `RuntimeError: Package name not found` came from `forecasting_tools` report saver; use `folder_to_save_reports_to=None`.
- Workflow summary checks must treat skip-only runs as valid outcomes when no new open questions require posting.
- Verify outcomes from artifacts/logs, not only CI green status.

## Validation and Regression
- Keep a tight regression map: evidence pipeline, gate floors, and aggregation calibration each need dedicated tests.
- Lock primary-source heuristics with explicit domain fixtures so template edits cannot silently reduce recall.
- Lock relevance scoring behavior with extracted-claim fixtures; avoid tests that pass on raw-page fallback alone.
- Assert forced abstain for `evidence_count < 3` independently of other gate component values.
- Assert evidence_sufficiency caps for `primary_sources == 0` and `mean_relevance < 0.7` with clear expected ceilings.
- Assert single-run forecasts (`n_runs <= 1`) return neutral agreement `0.5`, not perfect agreement.
- Assert extremization interpolation so weak evidence drives `effective_k` toward `1.0`.
- After report refactors, test for section presence so structured markdown output stays machine-auditable.
- Keep baseline fixture forecasts to recognize pre-upgrade artifacts (proxy-only evidence, flat relevance, tautological agreement).
- Track suite totals after major pipeline changes (current run: 41 tests passing).

## Date Awareness
- Always anchor relative terms ("next month", "today") to explicit dates in logs or summaries.
- For short-horizon questions, evidence staleness penalties matter more; prioritize freshest direct sources.
