---
description: Run forecasts on custom Metaculus URLs for testing
---

# Test Forecast Workflow

Run the forecasting bot on specific Metaculus question URLs for testing and debugging.

## Prerequisites

1. Ensure your `.env` file has the required API keys:
   - `METACULUS_TOKEN` (required for fetching questions)
   - `OPENROUTER_API_KEY` (required for LLM inference)
   - `EXA_API_KEY` (primary search - recommended)
   - `SERPER_API_KEY` (fallback search)

2. Install dependencies if not already done:
   ```bash
   poetry install
   ```

---

## Option 1: Test Mode (No Posting)

Run a forecast locally without posting to Metaculus. Good for debugging.

```bash
poetry run python bot/main_forecast_pipeline.py --mode test_questions
```

This uses the hardcoded example question in the code. To test your own URL, edit `EXAMPLE_QUESTIONS` in `main_forecast_pipeline.py` or use Option 2.

---

## Option 2: Custom URLs (Posts to Metaculus)

Run forecasts on specific question URLs. **This WILL post predictions to Metaculus.**

// turbo
```bash
poetry run python bot/main_forecast_pipeline.py --mode urls --urls "https://www.metaculus.com/questions/XXXXX/"
```

### Multiple URLs

Comma-separated, no spaces around commas:

```bash
poetry run python bot/main_forecast_pipeline.py --mode urls --urls "https://www.metaculus.com/questions/123/,https://www.metaculus.com/questions/456/"
```

### Force Re-Post (Override Skip)

If you've already forecasted on a question but want to update:

```bash
poetry run python bot/main_forecast_pipeline.py --mode urls --urls "https://www.metaculus.com/questions/XXXXX/" --force-repost
```

---

## Option 3: Dry-Run Custom URLs (No Posting)

To test on custom URLs WITHOUT posting, temporarily modify the code:

1. Open `bot/main_forecast_pipeline.py`
2. Find the `elif run_mode == "urls":` block (~line 373)
3. Change `template_bot.publish_reports_to_metaculus = True` to `False`
4. Run your command
5. **Remember to revert the change!**

Or create a quick test script:

```python
# test_single_question.py
import asyncio
from bot.main_forecast_pipeline import TemplateForecaster
from forecasting_tools import MetaculusApi

url = "https://www.metaculus.com/questions/XXXXX/"

bot = TemplateForecaster(
    research_reports_per_question=1,
    predictions_per_research_report=1,
    publish_reports_to_metaculus=False,  # <-- No posting
    skip_previously_forecasted_questions=False,
)

question = MetaculusApi.get_question_by_url(url)
reports = asyncio.run(bot.forecast_questions([question], return_exceptions=True))
print(reports)
```

---

## Output Locations

| Output | Location |
|--------|----------|
| Forecast reports JSON | `./forecast_reports.json` (default) or `--output-file` |
| Saved reports folder | `./forecast_reports/` |
| Detailed agent logs | `./forecasts/forecast_{timestamp}_{question}.txt` |

---

## Example: Full Test Run

```bash
# 1. Activate poetry shell (optional, for convenience)
poetry shell

# 2. Run on a specific question (will post!)
python bot/main_forecast_pipeline.py --mode urls --urls "https://www.metaculus.com/questions/578/human-extinction-by-2100/"

# 3. Check the output
cat forecast_reports.json
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `METACULUS_TOKEN not set` | Add to `.env` |
| `OPENROUTER_API_KEY not set` | Add to `.env` - required for LLM calls |
| `No EXA or Serper key` | At least one search provider needed |
| `Already forecasted` | Use `--force-repost` flag |
| Empty probability (0.5 fallback) | Check logs for probability extraction failures |

---

## Quick Reference

```bash
# Test mode (no posting, uses example question)
poetry run python bot/main_forecast_pipeline.py --mode test_questions

# Custom URL (POSTS to Metaculus)
poetry run python bot/main_forecast_pipeline.py --mode urls --urls "URL1,URL2"

# Force re-post
poetry run python bot/main_forecast_pipeline.py --mode urls --urls "URL" --force-repost

# Tournament mode (scheduled)
poetry run python bot/main_forecast_pipeline.py --mode tournament
```
