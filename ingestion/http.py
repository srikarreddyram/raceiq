"""Small shared HTTP helper for the REST-based ingestion sources.

Ergast, OpenF1, and the weather API are all "GET JSON, retry on failure"
clients, so that logic lives here once instead of three times.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from ingestion.config import IngestionConfig

logger = logging.getLogger(__name__)


def get_json(url: str, config: IngestionConfig, params: dict[str, Any] | None = None) -> Any:
    """GET a JSON endpoint, retrying only on transient failures.

    A 4xx response (bad request, no data for this key) will not change on
    retry, so it's raised immediately instead of burning the retry budget —
    only network errors and 5xx responses are retried with backoff.
    """
    last_error: Exception | None = None

    for attempt in range(1, config.max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=config.request_timeout_seconds)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as error:
            if error.response is not None and 400 <= error.response.status_code < 500:
                raise
            last_error = error
        except requests.RequestException as error:
            last_error = error

        logger.warning(
            "Request failed (attempt %s/%s) for %s: %s", attempt, config.max_retries, url, last_error
        )
        if attempt < config.max_retries:
            time.sleep(2**attempt)  # exponential backoff: 2s, 4s, 8s...

    assert last_error is not None
    raise last_error
