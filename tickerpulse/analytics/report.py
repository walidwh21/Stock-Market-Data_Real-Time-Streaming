"""Turn forecasts + sentiment features into a readable analyst note.

If an OpenAI-compatible LLM endpoint is configured (LLM_API_URL / LLM_API_KEY /
LLM_MODEL) the note is written by the model; otherwise a deterministic,
rule-based note is produced so the pipeline works fully offline.
"""
from __future__ import annotations

import json

import pandas as pd
import requests

from ..config import Settings, company_name, settings as default_settings
from .forecast import ForecastResult

DISCLAIMER = ("Educational output from an automated pipeline. Not investment advice; "
              "forecasts are statistical estimates and can be wrong.")

SYSTEM_PROMPT = """You are a careful quantitative market analyst.
Given a short price history, model forecasts and daily sentiment features for one stock,
write a concise note with exactly these sections:
OUTLOOK: BULLISH | NEUTRAL | BEARISH
CONVICTION: LOW | MEDIUM | HIGH
KEY DRIVERS: 3-4 bullets grounded only in the supplied numbers
SENTIMENT READ: 2 bullets (dispersion, fear, catalyst buzz, attention)
RISKS: 2-3 bullets
WATCH LEVELS: support / resistance inferred from the supplied prices
Do not invent data. State uncertainty plainly. Never promise returns."""


def _signal(res: ForecastResult, latest: pd.Series | None) -> tuple[str, str, float]:
    exp_ret = res.forecast.loc[3, "Ensemble"] / res.last_close - 1
    band = (res.forecast.loc[3, "high"] - res.forecast.loc[3, "low"]) / res.last_close / 2
    z = exp_ret / band if band > 0 else 0.0
    tilt = 0.0
    if latest is not None:
        tilt = 0.5 * latest.get("sentiment_mean", 0) + 0.3 * latest.get("engagement_sentiment", 0)
    score = z + tilt
    outlook = "BULLISH" if score > 0.35 else "BEARISH" if score < -0.35 else "NEUTRAL"
    conviction = "HIGH" if abs(score) > 1.0 else "MEDIUM" if abs(score) > 0.5 else "LOW"
    return outlook, conviction, float(exp_ret)


def rule_based_note(res: ForecastResult, feats: pd.DataFrame | None) -> str:
    latest = None
    if feats is not None and len(feats):
        f = feats[feats["ticker"] == res.ticker].sort_values("day")
        latest = f.iloc[-1] if len(f) else None
    outlook, conviction, exp_ret = _signal(res, latest)
    fc = res.forecast
    m = res.metrics
    lines = [
        f"{res.ticker} ({company_name(res.ticker)}) - analyst note as of {res.last_day:%Y-%m-%d}",
        "",
        f"OUTLOOK: {outlook}",
        f"CONVICTION: {conviction}",
        "",
        "PRICE FORECAST",
        f"- Last close: ${res.last_close:,.2f}",
    ]
    for h, row in fc.iterrows():
        lines.append(f"- +{h}d: ${row['Ensemble']:,.2f}  (80% range ${row['low']:,.2f} - ${row['high']:,.2f})")
    lines += ["", "KEY DRIVERS",
              f"- Ensemble expects {exp_ret:+.2%} over 3 days.",
              f"- Walk-forward 1-day MAPE: ensemble {m.loc['Ensemble','MAPE_1d']:.2f}% vs persistence "
              f"{m.loc['Persistence','MAPE_1d']:.2f}%.",
              f"- Directional accuracy on the test window: {m.loc['Ensemble','DirAcc']:.0f}%.",
              f"- Most influential input: {res.importances.index[0]}."]
    if latest is not None:
        lines += ["", "SENTIMENT READ",
                  f"- Mean sentiment {latest['sentiment_mean']:+.2f}, dispersion {latest['sentiment_dispersion']:.2f} "
                  f"({'divided crowd' if latest['sentiment_dispersion'] > 0.45 else 'broad agreement'}).",
                  f"- Fear score {latest['fear_score']:.2f}, catalyst buzz {latest['catalyst_buzz']:.2f}, "
                  f"attention surge {latest['attention_surge']:+.0%}."]
    lines += ["", "RISKS",
              "- Short-horizon price models have limited skill; treat the range, not the point estimate.",
              "- Sentiment can reverse quickly on news; social data is noisy and can be manipulated.",
              "", DISCLAIMER]
    return "\n".join(lines)


def _llm_prompt(res: ForecastResult, feats: pd.DataFrame | None, prices: pd.DataFrame) -> str:
    tail = prices.sort_values("day").tail(10)[["day", "close", "volume"]].round(2)
    tail["day"] = tail["day"].dt.strftime("%Y-%m-%d")
    f = None
    if feats is not None and len(feats):
        f = feats[feats["ticker"] == res.ticker].sort_values("day").tail(7).round(3)
        f = f.assign(day=f["day"].dt.strftime("%Y-%m-%d"))
    return json.dumps({
        "ticker": res.ticker, "company": company_name(res.ticker),
        "recent_prices": tail.to_dict("records"),
        "forecast": res.forecast.round(2).reset_index().to_dict("records"),
        "walk_forward_metrics": res.metrics.round(3).reset_index().to_dict("records"),
        "sentiment_features_last_7d": [] if f is None else f.to_dict("records"),
    }, indent=1)


def llm_note(res, feats, prices, cfg: Settings = default_settings) -> str:
    resp = requests.post(
        cfg.llm_api_url,
        headers={"Authorization": f"Bearer {cfg.llm_api_key}", "Content-Type": "application/json"},
        json={"model": cfg.llm_model, "temperature": 0.2,
              "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user", "content": _llm_prompt(res, feats, prices)}]},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip() + "\n\n" + DISCLAIMER


def build_note(res: ForecastResult, feats, prices, cfg: Settings = default_settings) -> str:
    if cfg.llm_api_url and cfg.llm_api_key and cfg.llm_model:
        try:
            return llm_note(res, feats, prices, cfg)
        except Exception as exc:  # fall back rather than fail the whole run
            return rule_based_note(res, feats) + f"\n\n(LLM unavailable: {exc})"
    return rule_based_note(res, feats)


def send_webhook(text: str, cfg: Settings = default_settings) -> bool:
    if not cfg.report_webhook_url:
        return False
    r = requests.post(cfg.report_webhook_url, json={"text": text[:3500]}, timeout=15)
    return r.ok
