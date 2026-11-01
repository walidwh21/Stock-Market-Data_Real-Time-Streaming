"""Turn raw social / news records into daily per-ticker sentiment features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import catalyst_keywords
from . import sentiment as S

FEATURE_COLUMNS = [
    "sentiment_mean", "sentiment_dispersion", "fear_score", "catalyst_buzz",
    "engagement_sentiment", "bull_ratio", "post_count", "attention_surge",
]


def unify_text_sources(reddit: pd.DataFrame, stocktwits: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Stack the three text sources into one frame: ticker, ts, text, weight, label."""
    parts = []
    if reddit is not None and len(reddit):
        parts.append(pd.DataFrame({
            "ticker": reddit["ticker"], "ts": reddit["ts"],
            "text": reddit["title"].fillna("") + " " + reddit["body"].fillna(""),
            "weight": reddit["score"].clip(lower=0) + reddit["num_comments"].clip(lower=0),
            "label": "", "source": "reddit"}))
    if stocktwits is not None and len(stocktwits):
        parts.append(pd.DataFrame({
            "ticker": stocktwits["ticker"], "ts": stocktwits["ts"],
            "text": stocktwits["body"].fillna(""), "weight": stocktwits["likes"].clip(lower=0),
            "label": stocktwits["label"].fillna(""), "source": "stocktwits"}))
    if news is not None and len(news):
        parts.append(pd.DataFrame({
            "ticker": news["ticker"], "ts": news["ts"],
            "text": news["title"].fillna("") + ". " + news["summary"].fillna(""),
            "weight": 1, "label": "", "source": "news"}))
    if not parts:
        return pd.DataFrame(columns=["ticker", "ts", "text", "weight", "label", "source"])
    return pd.concat(parts, ignore_index=True)


def score_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-row `score`, `fear_hits` and `catalyst_hits` columns."""
    out = df.copy()
    out["score"] = out["text"].map(S.score)
    manual = out["label"].map(S.label_to_score)          # human labels override the lexicon
    out["score"] = manual.fillna(out["score"])
    out["fear_hits"] = out["text"].map(lambda t: S.count_any(t, S.FEAR_WORDS))
    out["catalyst_hits"] = [
        S.count_any(t, catalyst_keywords(tk)) for t, tk in zip(out["text"], out["ticker"])
    ]
    out["day"] = out["ts"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    return out


def daily_features(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored rows to one row per (ticker, day)."""
    if scored.empty:
        return pd.DataFrame(columns=["ticker", "day", *FEATURE_COLUMNS])

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["weight"].astype(float) + 1.0              # +1 so zero-engagement posts still count
        labeled = g["label"].isin(["bullish", "bearish"])
        return pd.Series({
            "sentiment_mean": g["score"].mean(),
            "sentiment_dispersion": g["score"].std(ddof=0),
            "fear_score": g["fear_hits"].sum() / max(len(g), 1),
            "catalyst_buzz": g["catalyst_hits"].sum() / max(len(g), 1),
            "engagement_sentiment": float(np.average(g["score"], weights=w)),
            "bull_ratio": (g.loc[labeled, "label"] == "bullish").mean() if labeled.any() else 0.5,
            "post_count": len(g),
        })

    feats = scored.groupby(["ticker", "day"]).apply(agg, include_groups=False).reset_index()
    feats = feats.sort_values(["ticker", "day"])
    ma = feats.groupby("ticker")["post_count"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    feats["attention_surge"] = (feats["post_count"] - ma) / ma.replace(0, np.nan)
    feats["attention_surge"] = feats["attention_surge"].fillna(0.0)
    return feats.reset_index(drop=True)


def build_features(reddit, stocktwits, news) -> pd.DataFrame:
    return daily_features(score_rows(unify_text_sources(reddit, stocktwits, news)))
