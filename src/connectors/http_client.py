from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from src.core.utils import stable_hash


class SimpleHttpClient:
    def __init__(
        self,
        timeout_seconds: int = 20,
        max_retries: int = 3,
        cache_ttl_seconds: int = 300,
        cache_dir: str = ".cache/league_http",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_path = Path(cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, url: str, params: dict[str, Any] | None) -> Path:
        cache_key = stable_hash(url + "|" + json.dumps(params or {}, sort_keys=True))
        return self.cache_path / f"{cache_key}.json"

    def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | list[Any]:
        cache_file = self._cache_file(url, params)
        if cache_file.exists():
            age_seconds = time.time() - cache_file.stat().st_mtime
            if age_seconds <= self.cache_ttl_seconds:
                return json.loads(cache_file.read_text(encoding="utf-8"))

        delay = 1.0
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(
                        f"retryable status={response.status_code}", response=response
                    )
                response.raise_for_status()
                payload = response.json()
                cache_file.write_text(json.dumps(payload), encoding="utf-8")
                return payload
            except Exception as exc:
                last_error = exc
                time.sleep(delay)
                delay *= 2
        if last_error is None:
            raise RuntimeError("Unexpected HTTP client failure with no exception.")
        raise last_error

