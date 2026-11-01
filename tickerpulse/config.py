"""Central, environment-driven configuration.

Nothing here is hard-coded to a company, account or machine: change the
universe with the TICKERS variable and everything else follows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # .env support is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# Friendly names used to build social-media search queries.
# Unknown tickers simply fall back to the ticker symbol itself.
COMPANY_NAMES = {
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "GOOGL": "Google Alphabet",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "AMD": "AMD",
    "NFLX": "Netflix",
}

# Product / catalyst keywords per ticker, used by the "catalyst buzz" feature.
CATALYST_KEYWORDS = {
    "NVDA": ["blackwell", "h100", "h200", "rubin", "gpu"],
    "TSLA": ["cybertruck", "robotaxi", "fsd", "optimus", "model y"],
    "GOOGL": ["gemini", "waymo", "tpu", "search", "cloud"],
    "AAPL": ["iphone", "vision pro", "services", "m4", "siri"],
    "MSFT": ["copilot", "azure", "openai", "windows", "xbox"],
    "AMZN": ["aws", "prime", "kuiper", "alexa", "trainium"],
}
DEFAULT_CATALYSTS = ["launch", "earnings", "guidance", "partnership", "acquisition"]


def _split(value: str) -> list[str]:
    return [v.strip().upper() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    tickers: list[str] = field(
        default_factory=lambda: _split(os.getenv("TICKERS", "NVDA,TSLA,GOOGL,AAPL,MSFT,AMZN"))
    )
    kafka_bootstrap: str = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    kafka_bootstrap_internal: str = os.getenv("KAFKA_BOOTSTRAP_INTERNAL", "kafka:29092")
    topic_prefix: str = os.getenv("TOPIC_PREFIX", "tp")
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    mongo_db: str = os.getenv("MONGO_DB", "tickerpulse")
    price_interval: int = int(os.getenv("PRICE_INTERVAL", "60"))
    social_interval: int = int(os.getenv("SOCIAL_INTERVAL", "120"))
    news_interval: int = int(os.getenv("NEWS_INTERVAL", "300"))
    llm_api_url: str = os.getenv("LLM_API_URL", "")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    report_webhook_url: str = os.getenv("REPORT_WEBHOOK_URL", "")

    # ---- derived helpers -------------------------------------------------
    def topic(self, source: str) -> str:
        """Kafka topic name for a data source, e.g. 'tp.prices'."""
        return f"{self.topic_prefix}.{source}"

    @property
    def sources(self) -> dict[str, str]:
        """Mongo collection name -> Kafka topic."""
        return {s: self.topic(s) for s in ("prices", "reddit", "stocktwits", "news")}


def company_name(ticker: str) -> str:
    return COMPANY_NAMES.get(ticker.upper(), ticker.upper())


def catalyst_keywords(ticker: str) -> list[str]:
    return CATALYST_KEYWORDS.get(ticker.upper(), DEFAULT_CATALYSTS)


settings = Settings()
