"""Multi-horizon price forecasting with walk-forward validation.

Two complementary models are compared against a persistence baseline and
averaged into an ensemble:
  * Ridge regression      - linear, stable on short histories
  * Gradient boosting     - non-linear, can use sentiment interactions
Targets are log-returns for horizons 1..H days, which keeps the problem
scale-free so the same code works for any ticker.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS

HORIZONS = (1, 2, 3)
MODEL_NAMES = ("Persistence", "Ridge", "GradBoost", "Ensemble")


def make_dataset(prices: pd.DataFrame, feats: pd.DataFrame | None) -> pd.DataFrame:
    """Daily frame with technical + sentiment features and forward-return targets."""
    df = prices.sort_values("day").set_index("day")[["close", "volume"]].copy()
    r = np.log(df["close"]).diff()
    for lag in (1, 2, 3, 5, 10):
        df[f"ret_{lag}"] = np.log(df["close"] / df["close"].shift(lag))
    df["vol_10"] = r.rolling(10).std()
    df["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    df["ma_gap"] = df["close"] / df["close"].rolling(10).mean() - 1
    if feats is not None and len(feats):
        f = feats.set_index("day")[FEATURE_COLUMNS].reindex(df.index).ffill(limit=3)
        for c in FEATURE_COLUMNS:
            df[c] = f[c]
    for c in FEATURE_COLUMNS:
        if c not in df:
            df[c] = 0.0
    for h in HORIZONS:
        df[f"y{h}"] = np.log(df["close"].shift(-h) / df["close"])
    return df


def feature_names(df: pd.DataFrame) -> list[str]:
    skip = {"close", "volume", *[f"y{h}" for h in HORIZONS]}
    return [c for c in df.columns if c not in skip]


def _models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=25.0)),
        "GradBoost": GradientBoostingRegressor(
            n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=7),
    }


@dataclass
class ForecastResult:
    ticker: str
    last_close: float
    last_day: pd.Timestamp
    forecast: pd.DataFrame                    # index = horizon (1..H), columns = Ridge, GradBoost, Ensemble, low, high
    metrics: pd.DataFrame                     # index = model, columns = MAPE_1d, RMSE_1d, DirAcc
    walk_forward: pd.DataFrame                # per-day actual vs predicted (1-day horizon)
    importances: pd.Series = field(default_factory=pd.Series)


def run_forecast(ticker: str, prices: pd.DataFrame, feats: pd.DataFrame | None,
                 test_days: int = 30, refit_every: int = 5) -> ForecastResult:
    data = make_dataset(prices, feats)
    X_cols = feature_names(data)
    usable = data.dropna(subset=X_cols + ["y1"]).copy()
    if len(usable) < test_days + 40:
        raise ValueError(f"{ticker}: need at least {test_days + 40} usable days, got {len(usable)}")

    # ---------- walk-forward evaluation on the 1-day horizon ----------
    start = len(usable) - test_days
    rows = []
    models = None
    for i in range(start, len(usable)):
        if models is None or (i - start) % refit_every == 0:
            train = usable.iloc[:i]
            models = {n: m.fit(train[X_cols], train["y1"]) for n, m in _models().items()}
        x = usable.iloc[[i]][X_cols]
        pr = {n: float(m.predict(x)[0]) for n, m in models.items()}
        pr["Ensemble"] = 0.5 * (pr["Ridge"] + pr["GradBoost"])
        pr["Persistence"] = 0.0                       # "tomorrow == today"
        rows.append({"day": usable.index[i], "actual": usable["y1"].iloc[i],
                     "close": usable["close"].iloc[i], **pr})
    wf = pd.DataFrame(rows).set_index("day")

    actual_next = wf["close"] * np.exp(wf["actual"])
    metrics = {}
    for name in MODEL_NAMES:
        pred_price = wf["close"] * np.exp(wf[name])
        err = pred_price - actual_next
        metrics[name] = {
            "MAPE_1d": float((err.abs() / actual_next).mean() * 100),
            "RMSE_1d": float(np.sqrt((err ** 2).mean())),
            "DirAcc": float((np.sign(wf[name]) == np.sign(wf["actual"])).mean() * 100)
            if name != "Persistence" else float("nan"),
        }
    metrics = pd.DataFrame(metrics).T

    # ---------- final fit on everything, forecast H days ahead ----------
    last = data.dropna(subset=X_cols).iloc[[-1]]
    resid = (wf["actual"] - wf["Ensemble"]).std()
    out, importances = {}, None
    for h in HORIZONS:
        train = data.dropna(subset=X_cols + [f"y{h}"])
        fitted = {n: m.fit(train[X_cols], train[f"y{h}"]) for n, m in _models().items()}
        p = {n: float(m.predict(last[X_cols])[0]) for n, m in fitted.items()}
        p["Ensemble"] = 0.5 * (p["Ridge"] + p["GradBoost"])
        band = 1.28 * resid * np.sqrt(h)               # ~80% interval, random-walk scaling
        base = float(last["close"].iloc[0])
        out[h] = {
            "Ridge": base * np.exp(p["Ridge"]), "GradBoost": base * np.exp(p["GradBoost"]),
            "Ensemble": base * np.exp(p["Ensemble"]),
            "low": base * np.exp(p["Ensemble"] - band), "high": base * np.exp(p["Ensemble"] + band),
        }
        if h == 1:
            importances = pd.Series(fitted["GradBoost"].feature_importances_, index=X_cols)
    forecast = pd.DataFrame(out).T
    forecast.index.name = "horizon_days"

    return ForecastResult(
        ticker=ticker, last_close=float(last["close"].iloc[0]), last_day=last.index[0],
        forecast=forecast, metrics=metrics, walk_forward=wf,
        importances=importances.sort_values(ascending=False))
