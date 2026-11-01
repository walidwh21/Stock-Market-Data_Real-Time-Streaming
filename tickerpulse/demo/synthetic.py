"""Synthetic market + social data so the whole analytics stack runs offline.

Prices follow a seeded random walk whose daily return is weakly influenced by
the previous day's crowd sentiment - enough for the sentiment features to
carry a little signal, as they would in a real feed. ALL DATA IS FAKE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import catalyst_keywords, company_name

START_PRICE = {"NVDA": 118, "TSLA": 245, "GOOGL": 168, "AAPL": 212, "MSFT": 415, "AMZN": 182}
DRIFT = {"NVDA": 0.0014, "TSLA": 0.0002, "GOOGL": 0.0007, "AAPL": 0.0005, "MSFT": 0.0006, "AMZN": 0.0008}
VOLATILITY = {"NVDA": 0.026, "TSLA": 0.031, "GOOGL": 0.016, "AAPL": 0.013, "MSFT": 0.012, "AMZN": 0.017}

BULL = ["{c} looks very bullish, strong {k} momentum", "buying more {t}, {k} is a huge catalyst 🚀",
        "{t} breakout incoming, calls printing", "upgrade on {c} - {k} demand is strong",
        "{c} beat expectations, record growth 📈"]
BEAR = ["{t} looks overvalued, bubble risk", "selling my {t}, {k} delays are a worry",
        "{c} miss and downgrade, not good 📉", "puts on {t}, panic in the market",
        "lawsuit and recall news hit {c}, weak outlook"]
NEUT = ["{t} earnings next week, watching {k}", "anyone following {c} today?",
        "{t} flat today, volume is average", "reading about {k} at {c}"]
SUBS = ["stocks", "investing", "wallstreetbets"]


def generate(tickers: list[str], days: int = 150, seed: int = 11):
    rng = np.random.default_rng(seed)
    biz = pd.bdate_range(end=pd.Timestamp("2026-10-30"), periods=days)
    price_rows, reddit_rows, st_rows, news_rows = [], [], [], []
    sid = 0

    for t in tickers:
        price = START_PRICE.get(t, 100.0)
        vol = VOLATILITY.get(t, 0.02)
        mood_prev = 0.0
        regime = np.zeros(days)                                       # slow, mean-reverting crowd mood
        for j in range(1, days):
            regime[j] = 0.92 * regime[j - 1] + rng.normal(0, 0.12)
        for i, day in enumerate(biz):
            mood = float(np.clip(regime[i] + rng.normal(0, 0.30), -1, 1))
            ret = DRIFT.get(t, 0.0005) + 0.006 * mood_prev + rng.normal(0, vol)
            o = price
            price *= float(np.exp(ret))
            hi, lo = max(o, price) * (1 + abs(rng.normal(0, vol / 3))), min(o, price) * (1 - abs(rng.normal(0, vol / 3)))
            volume = int(rng.lognormal(16.5, 0.25) * (1 + 3 * abs(ret) / vol * 0.2))
            price_rows.append({"ticker": t, "ts": day + pd.Timedelta(hours=20), "open": o, "high": hi,
                               "low": lo, "close": price, "volume": volume})
            mood_prev = mood

            n = int(np.clip(rng.poisson(14) * (1 + 1.5 * abs(mood)), 4, 60))
            kws = catalyst_keywords(t)
            for _ in range(n):
                r = rng.random()
                pool = BULL if r < 0.5 + 0.45 * mood else BEAR if r < 0.88 else NEUT
                text = rng.choice(pool).format(t=t, c=company_name(t), k=rng.choice(kws))
                ts = day + pd.Timedelta(minutes=int(rng.integers(0, 1380)))
                sid += 1
                if rng.random() < 0.55:
                    lab = "bullish" if pool is BULL else "bearish" if pool is BEAR else ""
                    st_rows.append({"ticker": t, "id": sid, "ts": ts, "author": f"user{sid % 97}",
                                    "body": text, "label": lab if rng.random() < 0.6 else "",
                                    "likes": int(rng.poisson(2))})
                else:
                    reddit_rows.append({"ticker": t, "id": f"r{sid}", "ts": ts, "subreddit": rng.choice(SUBS),
                                        "title": text, "body": "", "score": int(rng.poisson(25)),
                                        "num_comments": int(rng.poisson(8))})
            for _ in range(rng.poisson(1.5)):
                pool = BULL if mood > 0.15 else BEAR if mood < -0.15 else NEUT
                news_rows.append({"ticker": t, "ts": day + pd.Timedelta(hours=int(rng.integers(8, 20))),
                                  "publisher": rng.choice(["Newswire", "MarketDesk", "Finance Daily"]),
                                  "title": rng.choice(pool).format(t=t, c=company_name(t), k=rng.choice(kws)),
                                  "summary": "", "url": f"https://news.example/{t}/{day:%Y%m%d}/{rng.integers(1e6)}"})

    def frame(rows):
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    prices = frame(price_rows)
    prices["day"] = prices["ts"].dt.tz_localize(None).dt.normalize()
    return prices, frame(reddit_rows), frame(st_rows), frame(news_rows)
