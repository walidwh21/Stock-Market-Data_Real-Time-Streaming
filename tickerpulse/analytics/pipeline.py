"""End-to-end analytics run: load data -> features -> forecast -> notes.

    python -m tickerpulse.analytics.pipeline                 # read from MongoDB
    python -m tickerpulse.analytics.pipeline --demo          # synthetic data, no services needed
    python -m tickerpulse.analytics.pipeline --schedule 08:30
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import settings
from . import features as F
from .forecast import ForecastResult, run_forecast
from .report import build_note, send_webhook


def daily_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Collapse intraday bars to daily OHLCV per ticker."""
    p = prices.copy()
    p["day"] = p["ts"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    g = p.sort_values("ts").groupby(["ticker", "day"])
    return g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                 close=("close", "last"), volume=("volume", "sum")).reset_index()


def load_from_mongo():
    from ..storage import get_db, load_frame

    db = get_db(settings)
    return tuple(load_frame(db, c) for c in ("prices", "reddit", "stocktwits", "news"))


def analyze(prices_raw, reddit, stocktwits, news, tickers):
    daily = daily_prices(prices_raw) if "ts" in prices_raw.columns and "day" not in prices_raw.columns else prices_raw
    feats = F.build_features(reddit, stocktwits, news)
    results: dict[str, ForecastResult] = {}
    notes: dict[str, str] = {}
    for t in tickers:
        px = daily[daily["ticker"] == t]
        try:
            res = run_forecast(t, px, feats[feats["ticker"] == t])
        except ValueError as exc:
            print(f"skip {t}: {exc}")
            continue
        results[t] = res
        notes[t] = build_note(res, feats, px)
    return daily, feats, results, notes


def run(demo: bool, tickers: list[str], out_dir: Path) -> None:
    if demo:
        from ..demo.synthetic import generate

        prices, reddit, stocktwits, news = generate(tickers)
    else:
        prices, reddit, stocktwits, news = load_from_mongo()
    _, _, results, notes = analyze(prices, reddit, stocktwits, news, tickers)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for t, note in notes.items():
        path = out_dir / f"{t}_{stamp}.txt"
        path.write_text(note, encoding="utf-8")
        print(f"wrote {path}")
        send_webhook(note)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the TickerPulse analytics pipeline")
    ap.add_argument("--tickers", nargs="*", default=settings.tickers)
    ap.add_argument("--demo", action="store_true", help="use synthetic data (no Kafka/Mongo needed)")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--schedule", metavar="HH:MM", help="run daily at this local time")
    args = ap.parse_args()
    out = Path(args.out)

    if not args.schedule:
        run(args.demo, args.tickers, out)
        return
    print(f"Scheduler active - running daily at {args.schedule}. Ctrl+C to stop.")
    last_run = None
    while True:
        now = datetime.now()
        if now.strftime("%H:%M") == args.schedule and last_run != now.date():
            run(args.demo, args.tickers, out)
            last_run = now.date()
        time.sleep(20)


if __name__ == "__main__":
    main()
