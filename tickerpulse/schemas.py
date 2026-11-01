"""Message contracts shared by producers, the Spark job and analytics.

Every message carries a `ticker` field, so one topic per source serves the
whole universe (the ticker is also used as the Kafka message key).
"""
from __future__ import annotations

from datetime import datetime, timezone

# Field name -> Spark SQL type string (consumed by streaming/spark_job.py)
SCHEMAS: dict[str, dict[str, str]] = {
    "prices": {
        "ticker": "string", "ts": "string",
        "open": "double", "high": "double", "low": "double", "close": "double",
        "volume": "long",
    },
    "reddit": {
        "ticker": "string", "id": "string", "ts": "string", "subreddit": "string",
        "title": "string", "body": "string", "score": "int", "num_comments": "int",
    },
    "stocktwits": {
        "ticker": "string", "id": "long", "ts": "string", "author": "string",
        "body": "string", "label": "string", "likes": "int",
    },
    "news": {
        "ticker": "string", "ts": "string", "publisher": "string",
        "title": "string", "summary": "string", "url": "string",
    },
}

# Field that is converted to a proper timestamp downstream.
TIME_FIELD = "ts"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
