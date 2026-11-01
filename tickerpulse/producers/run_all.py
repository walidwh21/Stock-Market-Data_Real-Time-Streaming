"""Run every producer concurrently:  python -m tickerpulse.producers.run_all"""
from __future__ import annotations

import argparse
import logging
import threading

from ..config import settings
from .news import NewsProducer
from .prices import PriceProducer
from .reddit import RedditProducer
from .stocktwits import StockTwitsProducer

ALL = {"prices": PriceProducer, "reddit": RedditProducer,
       "stocktwits": StockTwitsProducer, "news": NewsProducer}


def main() -> None:
    ap = argparse.ArgumentParser(description="Start TickerPulse data producers")
    ap.add_argument("--only", nargs="*", choices=ALL, help="subset of sources to run")
    ap.add_argument("--dry-run", action="store_true", help="print records instead of sending to Kafka")
    ap.add_argument("--once", action="store_true", help="poll each source one time and exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    producers = [ALL[name](settings, dry_run=args.dry_run) for name in (args.only or ALL)]

    if args.once:
        for p in producers:
            p.run_once()
        return

    threads = [threading.Thread(target=p.run_forever, daemon=True, name=p.source) for p in producers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
