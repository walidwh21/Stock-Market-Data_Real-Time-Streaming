"""Dependency-free sentiment scoring tuned for retail-investor language.

Score range is [-1, 1]. Handles negation ("not bullish"), intensifiers
("very bullish") and market slang. If NLTK's VADER is installed it can be
blended in via `score(text, use_vader=True)`.
"""
from __future__ import annotations

import math
import re

LEXICON: dict[str, float] = {
    # bullish
    "bullish": 2.0, "bull": 1.5, "long": 1.0, "calls": 1.0, "buy": 1.5, "buying": 1.5,
    "moon": 2.0, "rocket": 2.0, "breakout": 1.5, "rally": 1.8, "surge": 1.8, "soar": 2.0,
    "beat": 1.5, "beats": 1.5, "strong": 1.2, "growth": 1.0, "upgrade": 1.6, "outperform": 1.6,
    "record": 1.2, "green": 1.0, "ripping": 1.5, "undervalued": 1.4, "gem": 1.5, "love": 1.2,
    "great": 1.2, "winning": 1.3, "accumulate": 1.3, "dip": 0.4,
    # bearish
    "bearish": -2.0, "bear": -1.5, "short": -1.3, "puts": -1.0, "sell": -1.5, "selling": -1.5,
    "crash": -2.5, "dump": -2.0, "plunge": -2.2, "tank": -1.8, "tanking": -1.8, "drop": -1.2,
    "miss": -1.5, "weak": -1.2, "downgrade": -1.6, "underperform": -1.6, "red": -1.0,
    "overvalued": -1.4, "bubble": -1.8, "fraud": -2.5, "lawsuit": -1.5, "recall": -1.5,
    "bagholder": -1.5, "fud": -1.0, "panic": -2.0, "recession": -1.8, "layoffs": -1.3,
    "fear": -1.2, "worried": -1.2, "scam": -2.3,
}
NEGATIONS = {"not", "no", "never", "isn't", "wasn't", "don't", "doesn't", "won't", "cannot", "can't", "without"}
INTENSIFIERS = {"very": 1.3, "extremely": 1.5, "super": 1.4, "highly": 1.3, "really": 1.2, "slightly": 0.7}
FEAR_WORDS = {"crash", "bubble", "panic", "dump", "puts", "sell", "bear", "bearish",
              "plunge", "recession", "fraud", "layoffs", "fear", "worried", "scam", "tank", "tanking"}

_TOKEN = re.compile(r"[a-z']+")
_EMOJI = {"🚀": 2.0, "📈": 1.5, "💎": 1.2, "🔥": 1.0, "📉": -1.5, "🩸": -1.5, "💀": -1.5, "😱": -1.2}


def tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def score(text: str, use_vader: bool = False) -> float:
    """Return a sentiment score in [-1, 1]."""
    toks = tokens(text)
    total, hits = 0.0, 0
    for i, tok in enumerate(toks):
        base = LEXICON.get(tok)
        if base is None:
            continue
        window = toks[max(0, i - 3):i]
        if any(w in NEGATIONS for w in window):
            base = -0.8 * base
        for w in window:
            base *= INTENSIFIERS.get(w, 1.0)
        total += base
        hits += 1
    for ch, val in _EMOJI.items():
        n = (text or "").count(ch)
        total += val * n
        hits += n
    lex = 0.0 if hits == 0 else math.tanh(total / (hits ** 0.5 * 2.0))
    if use_vader:
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            return 0.5 * lex + 0.5 * SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
        except Exception:
            pass
    return lex


def count_any(text: str, words) -> int:
    ws = set(words)
    return sum(1 for t in tokens(text) if t in ws)


def label_to_score(label: str) -> float | None:
    """Map a human label (StockTwits-style) to a score, or None if unlabeled."""
    return {"bullish": 0.8, "bearish": -0.8}.get((label or "").lower())
