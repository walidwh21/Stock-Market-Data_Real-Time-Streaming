"""Shared plumbing for all producers: polling loop, de-duplication, Kafka sink."""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Iterable

from ..config import Settings, settings as default_settings

log = logging.getLogger("tickerpulse.producer")


class Producer(ABC):
    """Poll an external source for every ticker and publish new records to Kafka."""

    source: str = ""        # 'prices' | 'reddit' | 'stocktwits' | 'news'
    interval: int = 60      # seconds between polls

    def __init__(self, cfg: Settings = default_settings, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run
        self.topic = cfg.topic(self.source)
        self._seen: set[str] = set()
        self._kafka = None if dry_run else self._connect()

    # ---- to implement ----------------------------------------------------
    @abstractmethod
    def fetch(self, ticker: str) -> Iterable[dict]:
        """Return raw records for one ticker (must follow tickerpulse.schemas)."""

    def record_key(self, record: dict) -> str | None:
        """Unique id used for de-duplication. Return None to disable."""
        return None

    # ---- shared behaviour ------------------------------------------------
    def _connect(self):
        from kafka import KafkaProducer  # imported lazily so --dry-run needs no broker

        return KafkaProducer(
            bootstrap_servers=self.cfg.kafka_bootstrap.split(","),
            key_serializer=lambda k: k.encode(),
            value_serializer=lambda v: json.dumps(v, default=str).encode(),
            linger_ms=50,
        )

    def publish(self, record: dict) -> None:
        if self.dry_run:
            print(f"[{self.topic}] {json.dumps(record, default=str)[:160]}")
        else:
            self._kafka.send(self.topic, key=record["ticker"], value=record)

    def run_once(self) -> int:
        sent = 0
        for ticker in self.cfg.tickers:
            try:
                for rec in self.fetch(ticker):
                    key = self.record_key(rec)
                    if key is not None:
                        if key in self._seen:
                            continue
                        self._seen.add(key)
                    self.publish(rec)
                    sent += 1
            except Exception as exc:  # one bad ticker must not stop the loop
                log.warning("%s/%s failed: %s", self.source, ticker, exc)
        if self._kafka:
            self._kafka.flush()
        return sent

    def run_forever(self) -> None:
        log.info("%s producer started for %s", self.source, ", ".join(self.cfg.tickers))
        while True:
            n = self.run_once()
            log.info("%s: published %d new records", self.source, n)
            time.sleep(self.interval)
