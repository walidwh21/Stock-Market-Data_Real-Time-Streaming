"""Recent Reddit posts mentioning each company (public JSON search endpoint)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import requests

from ..config import company_name
from .base import Producer

HEADERS = {"User-Agent": "tickerpulse/1.0 (research project)"}
SUBREDDITS = ("stocks", "investing", "wallstreetbets")


class RedditProducer(Producer):
    source = "reddit"

    def __init__(self, *a, subreddits=SUBREDDITS, **kw):
        super().__init__(*a, **kw)
        self.interval = self.cfg.social_interval
        self.subreddits = subreddits

    def fetch(self, ticker: str) -> Iterable[dict]:
        query = f'"{ticker}" OR "{company_name(ticker)}"'
        for sub in self.subreddits:
            resp = requests.get(
                f"https://www.reddit.com/r/{sub}/search.json",
                params={"q": query, "restrict_sr": 1, "sort": "new", "limit": 50},
                headers=HEADERS, timeout=15,
            )
            if resp.status_code != 200:
                continue
            for child in resp.json().get("data", {}).get("children", []):
                p = child["data"]
                yield {
                    "ticker": ticker,
                    "id": p["id"],
                    "ts": datetime.fromtimestamp(p["created_utc"], timezone.utc).isoformat(),
                    "subreddit": sub,
                    "title": p.get("title", ""),
                    "body": p.get("selftext", ""),
                    "score": int(p.get("score", 0)),
                    "num_comments": int(p.get("num_comments", 0)),
                }

    def record_key(self, record: dict) -> str:
        return f"{record['ticker']}:{record['id']}"
