"""Thin MongoDB helpers used by the Spark job and the analytics layer."""
from __future__ import annotations

import pandas as pd

from .config import Settings, settings as default_settings


def get_db(cfg: Settings = default_settings):
    from pymongo import MongoClient

    return MongoClient(cfg.mongo_uri, serverSelectionTimeoutMS=5000)[cfg.mongo_db]


def ensure_indexes(db) -> None:
    """Unique keys make stream restarts idempotent (no duplicate documents)."""
    db["prices"].create_index([("ticker", 1), ("ts", 1)], unique=True)
    db["reddit"].create_index([("ticker", 1), ("id", 1)], unique=True)
    db["stocktwits"].create_index([("ticker", 1), ("id", 1)], unique=True)
    db["news"].create_index([("ticker", 1), ("url", 1)], unique=True)


def load_frame(db, collection: str, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load a collection into a DataFrame with a parsed UTC `ts` column."""
    query = {} if since is None else {"ts_parsed": {"$gte": since.to_pydatetime()}}
    df = pd.DataFrame(list(db[collection].find(query, {"_id": 0})))
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts"])
    return df
