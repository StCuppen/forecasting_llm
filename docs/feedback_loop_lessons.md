# Feedback Loop Lessons

## 2026-02-11 - Dry-Run Artifacts Mistaken for Real Forecasts

- Observation: `predictions/feedback_loop/` contained only dry-run forecasts, which looked structurally correct but were not live model outputs.
- Risk: false confidence in forecasting quality and feedback-loop signal.
- Action taken:
  - Added explicit `Mode: dry-run|live` marker to every markdown artifact.
  - Added `forecast_context.dry_run` flag in stored prediction metadata.
  - Ran/verified live forecasting path separately for confirmation.
- Ongoing rule:
  - Only use `Mode: live` artifacts for lessons, diagnostics, and updater feedback.

