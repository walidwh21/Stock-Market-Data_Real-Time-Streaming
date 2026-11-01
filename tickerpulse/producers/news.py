"""Company news headlines via yfinance."""
from __future__ import annotations

from typing import Iterable

from .base import Producer


class NewsProducer(Producer):
    source = "news"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.interval = self.cfg.news_interval

    def fetch(self, ticker: str) -> Iterable[dict]:
        import yfinance as yf

        for item in yf.Ticker(ticker).news or []:
            c = item.get("content", {})
            url = (c.get("canonicalUrl") or {}).get("url")
            if not url:
                continue
            yield {
                "ticker": ticker,
                "ts": c.get("pubDate", ""),
                "publisher": (c.get("provider") or {}).get("displayName", ""),
                "title": c.get("title", ""),
                "summary": c.get("summary", ""),
                "url": url,
            }

    def record_key(self, record: dict) -> str:
        return f"{record['ticker']}:{record['url']}"
