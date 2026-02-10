"""
Analyze GitHub Actions "Run Tournament" workflow runs for:
- whether questions were found (open/upcoming announced)
- whether submission attempts were made
- whether errors occurred

This requires a GitHub token with permissions to read Actions logs for the repo.

Usage (PowerShell):
  $env:GITHUB_TOKEN="..."; python scripts/analyze_tournament_runs.py --repo StCuppen/forecasting_llm --limit 50
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class RunSummary:
    run_id: int
    html_url: str
    created_at: str | None
    conclusion: str | None
    status: str | None
    announced_upcoming: int
    announced_open: int
    fetched_total: int
    submission_attempts: int
    submission_skips: int
    errors: int


def _gh_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "forecasting_llm-main/analyze_tournament_runs",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_json(url: str, headers: dict[str, str], params: dict[str, Any] | None = None) -> Any:
    r = requests.get(url, headers=headers, params=params, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text[:200]}")
    return r.json()


def _download_bytes(url: str, headers: dict[str, str]) -> bytes:
    r = requests.get(url, headers=headers, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"GET {url} -> {r.status_code}: {r.text[:200]}")
    return r.content


def _parse_logs_zip(content: bytes) -> dict[str, int]:
    z = zipfile.ZipFile(io.BytesIO(content))
    # Patterns correspond to the repo's current logging + workflow summary parsing.
    fetched_re = re.compile(r"Fetched (\d+) questions \(Open \+ Upcoming\)")
    found_upcoming_re = re.compile(r"FOUND (\d+) UPCOMING QUESTIONS")
    found_open_re = re.compile(r"Found (\d+) OPEN questions", re.IGNORECASE)
    attempt_re = re.compile(r"Submission Attempt: posting forecast")
    skip_re = re.compile(r"Submission Skipped:")
    err_re = re.compile(r"Error while processing question url:", re.IGNORECASE)

    fetched_total = 0
    announced_upcoming = 0
    announced_open = 0
    attempts = 0
    skips = 0
    errors = 0

    for name in z.namelist():
        if not name.lower().endswith(".txt"):
            continue
        txt = z.read(name).decode("utf-8", errors="ignore")

        for m in fetched_re.finditer(txt):
            fetched_total = max(fetched_total, int(m.group(1)))
        for m in found_upcoming_re.finditer(txt):
            announced_upcoming = max(announced_upcoming, int(m.group(1)))
        for m in found_open_re.finditer(txt):
            announced_open = max(announced_open, int(m.group(1)))

        attempts += len(attempt_re.findall(txt))
        skips += len(skip_re.findall(txt))
        errors += len(err_re.findall(txt))

    return {
        "fetched_total": fetched_total,
        "announced_upcoming": announced_upcoming,
        "announced_open": announced_open,
        "attempts": attempts,
        "skips": skips,
        "errors": errors,
    }

def _parse_artifact_zip(content: bytes) -> dict[str, int]:
    """
    Artifact zip typically contains bot.log plus output folders.
    We parse any .log/.txt files found (especially bot.log).
    """
    z = zipfile.ZipFile(io.BytesIO(content))
    fetched_re = re.compile(r"Fetched (\d+) questions \(Open \+ Upcoming\)")
    found_upcoming_re = re.compile(r"FOUND (\d+) UPCOMING QUESTIONS")
    found_open_re = re.compile(r"Found (\d+) OPEN questions", re.IGNORECASE)
    attempt_re = re.compile(r"Submission Attempt: posting forecast")
    skip_re = re.compile(r"Submission Skipped:")
    err_re = re.compile(r"Error while processing question url:", re.IGNORECASE)

    fetched_total = 0
    announced_upcoming = 0
    announced_open = 0
    attempts = 0
    skips = 0
    errors = 0

    for name in z.namelist():
        low = name.lower()
        if not (low.endswith(".txt") or low.endswith(".log")):
            continue
        txt = z.read(name).decode("utf-8", errors="ignore")

        for m in fetched_re.finditer(txt):
            fetched_total = max(fetched_total, int(m.group(1)))
        for m in found_upcoming_re.finditer(txt):
            announced_upcoming = max(announced_upcoming, int(m.group(1)))
        for m in found_open_re.finditer(txt):
            announced_open = max(announced_open, int(m.group(1)))

        attempts += len(attempt_re.findall(txt))
        skips += len(skip_re.findall(txt))
        errors += len(err_re.findall(txt))

    return {
        "fetched_total": fetched_total,
        "announced_upcoming": announced_upcoming,
        "announced_open": announced_open,
        "attempts": attempts,
        "skips": skips,
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="StCuppen/forecasting_llm", help="owner/repo")
    ap.add_argument("--workflow", default="run_tournament.yaml", help="workflow file name")
    ap.add_argument("--limit", type=int, default=50, help="max runs to scan")
    ap.add_argument(
        "--method",
        choices=["auto", "logs", "artifacts"],
        default="auto",
        help="Prefer run logs zip, workflow artifacts zip, or auto fallback",
    )
    ap.add_argument("--sleep", type=float, default=0.2, help="sleep between API calls")
    args = ap.parse_args()

    token = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
    if not token:
        print("Missing GITHUB_TOKEN (or GITHUB_PAT) env var.", file=sys.stderr)
        return 2

    headers = _gh_headers(token)
    runs_url = f"https://api.github.com/repos/{args.repo}/actions/workflows/{args.workflow}/runs"

    data = _get_json(runs_url, headers=headers, params={"per_page": min(100, args.limit)})
    workflow_runs = data.get("workflow_runs") or []
    workflow_runs = workflow_runs[: args.limit]

    summaries: list[RunSummary] = []

    for run in workflow_runs:
        run_id = int(run["id"])
        html_url = run.get("html_url") or ""
        created_at = run.get("created_at")
        conclusion = run.get("conclusion")
        status = run.get("status")

        stats: dict[str, int] | None = None
        errors: list[str] = []

        if args.method in ("auto", "logs"):
            logs_url = f"https://api.github.com/repos/{args.repo}/actions/runs/{run_id}/logs"
            try:
                content = _download_bytes(logs_url, headers=headers)
                stats = _parse_logs_zip(content)
            except Exception as e:
                errors.append(f"log_download_failed={e}")

        if stats is None and args.method in ("auto", "artifacts"):
            # Artifacts listing is often accessible without auth, but downloads require auth.
            arts_url = f"https://api.github.com/repos/{args.repo}/actions/runs/{run_id}/artifacts"
            try:
                arts = _get_json(arts_url, headers=headers)
                artifacts = arts.get("artifacts") or []
                # Prefer the run_tournament artifact names used in this repo.
                preferred = None
                for a in artifacts:
                    if (a.get("name") or "").lower() in ("bot-log", "minibench-bot-log", "forecast-log"):
                        preferred = a
                        break
                if preferred is None and artifacts:
                    preferred = artifacts[0]
                if preferred is None:
                    raise RuntimeError("no artifacts found")
                artifact_id = int(preferred["id"])
                zip_url = f"https://api.github.com/repos/{args.repo}/actions/artifacts/{artifact_id}/zip"
                content = _download_bytes(zip_url, headers=headers)
                stats = _parse_artifact_zip(content)
            except Exception as e:
                errors.append(f"artifact_download_failed={e}")

        if stats is None:
            stats = {
                "fetched_total": 0,
                "announced_upcoming": 0,
                "announced_open": 0,
                "attempts": 0,
                "skips": 0,
                "errors": 0,
            }
            if errors:
                print(f"run_id={run_id} " + " ".join(errors), file=sys.stderr)

        summaries.append(
            RunSummary(
                run_id=run_id,
                html_url=html_url,
                created_at=created_at,
                conclusion=conclusion,
                status=status,
                announced_upcoming=stats["announced_upcoming"],
                announced_open=stats["announced_open"],
                fetched_total=stats["fetched_total"],
                submission_attempts=stats["attempts"],
                submission_skips=stats["skips"],
                errors=stats["errors"],
            )
        )
        time.sleep(args.sleep)

    # Print a compact table (TSV)
    print(
        "\t".join(
            [
                "run_id",
                "created_at",
                "status",
                "conclusion",
                "fetched_total",
                "ann_open",
                "ann_upcoming",
                "attempts",
                "skips",
                "errors",
                "url",
            ]
        )
    )
    for s in summaries:
        print(
            "\t".join(
                [
                    str(s.run_id),
                    s.created_at or "",
                    s.status or "",
                    s.conclusion or "",
                    str(s.fetched_total),
                    str(s.announced_open),
                    str(s.announced_upcoming),
                    str(s.submission_attempts),
                    str(s.submission_skips),
                    str(s.errors),
                    s.html_url,
                ]
            )
        )

    # Highlight failure cases: announced but zero attempts
    failures = [
        s
        for s in summaries
        if (s.announced_open > 0 or s.announced_upcoming > 0 or s.fetched_total > 0)
        and s.submission_attempts == 0
    ]
    if failures:
        print("\nFailures (announced/fetched but 0 submission attempts):", file=sys.stderr)
        for s in failures[:25]:
            print(
                f"- run_id={s.run_id} created_at={s.created_at} fetched={s.fetched_total} "
                f"open={s.announced_open} upcoming={s.announced_upcoming} errors={s.errors} url={s.html_url}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
