"""Intraday prices via yfinance (latest 1-minute bar per ticker)."""
from __future__ import annotations

from typing import Iterable

from .base import Producer


class PriceProducer(Producer):
    source = "prices"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.interval = self.cfg.price_interval

    def fetch(self, ticker: str) -> Iterable[dict]:
        import yfinance as yf

        # period="1d", interval="1m"  ->  today's minute bars (not "one month")
        bars = yf.Ticker(ticker).history(period="1d", interval="1m")
        if bars.empty:
            return []
        last = bars.iloc[-1]
        return [{
            "ticker": ticker,
            "ts": bars.index[-1].isoformat(),
            "open": float(last["Open"]), "high": float(last["High"]),
            "low": float(last["Low"]), "close": float(last["Close"]),
            "volume": int(last["Volume"]),
        }]

    def record_key(self, record: dict) -> str:
        return f"{record['ticker']}@{record['ts']}"
