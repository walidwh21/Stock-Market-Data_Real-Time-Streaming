"""StockTwits symbol streams (messages carry a bullish/bearish label when set)."""
from __future__ import annotations

from typing import Iterable

import requests

from .base import Producer


class StockTwitsProducer(Producer):
    source = "stocktwits"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.interval = self.cfg.social_interval

    def fetch(self, ticker: str) -> Iterable[dict]:
        resp = requests.get(
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            headers={"User-Agent": "tickerpulse/1.0"}, timeout=15,
        )
        resp.raise_for_status()
        for m in resp.json().get("messages", []):
            sentiment = (m.get("entities") or {}).get("sentiment") or {}
            yield {
                "ticker": ticker,
                "id": int(m["id"]),
                "ts": m["created_at"],
                "author": (m.get("user") or {}).get("username", ""),
                "body": m.get("body", ""),
                "label": (sentiment.get("basic") or "").lower(),   # 'bullish' | 'bearish' | ''
                "likes": int((m.get("likes") or {}).get("total", 0)),
            }

    def record_key(self, record: dict) -> str:
        return f"{record['ticker']}:{record['id']}"
