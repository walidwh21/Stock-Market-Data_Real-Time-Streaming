"""Fast offline tests:  python -m pytest -q   (or: python tests/test_core.py)"""
import numpy as np
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tickerpulse.analytics import sentiment as S
from tickerpulse.analytics.features import build_features
from tickerpulse.analytics.forecast import run_forecast
from tickerpulse.config import catalyst_keywords, company_name, settings
from tickerpulse.demo.synthetic import generate


def test_sentiment_polarity_and_negation():
    assert S.score("very bullish, buying calls") > 0.3
    assert S.score("panic selling, bubble about to crash") < -0.3
    assert S.score("not bullish at all") < S.score("bullish")
    assert S.score("") == 0.0


def test_config_is_generic():
    assert "TSLA" in settings.tickers and company_name("TSLA") == "Tesla"
    assert company_name("XYZ") == "XYZ"
    assert catalyst_keywords("UNKNOWN")


def test_features_and_forecast_roundtrip():
    prices, reddit, st, news = generate(["TSLA"], days=120)
    feats = build_features(reddit, st, news)
    assert {"sentiment_mean", "attention_surge"} <= set(feats.columns)
    res = run_forecast("TSLA", prices, feats, test_days=20)
    assert list(res.forecast.index) == [1, 2, 3]
    assert (res.forecast["low"] < res.forecast["high"]).all()
    assert np.isfinite(res.metrics.loc["Ensemble", "MAPE_1d"])


if __name__ == "__main__":
    test_sentiment_polarity_and_negation(); test_config_is_generic(); test_features_and_forecast_roundtrip()
    print("all tests passed")
