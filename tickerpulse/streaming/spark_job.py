"""Structured Streaming job: Kafka (4 topics) -> MongoDB (4 collections).

Submit with:
    spark-submit --master spark://localhost:7077 \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
      -m tickerpulse.streaming.spark_job

Each micro-batch is upserted with pymongo, keyed on the natural id of the
record, so replays after a restart never create duplicates.
"""
from __future__ import annotations

import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ..config import settings
from ..schemas import SCHEMAS
from ..storage import ensure_indexes, get_db

# Natural key used for upserts, per source.
UPSERT_KEYS = {
    "prices": ("ticker", "ts"),
    "reddit": ("ticker", "id"),
    "stocktwits": ("ticker", "id"),
    "news": ("ticker", "url"),
}

log = logging.getLogger("tickerpulse.spark")


def ddl(source: str) -> str:
    return ", ".join(f"{name} {typ}" for name, typ in SCHEMAS[source].items())


def make_writer(source: str):
    keys = UPSERT_KEYS[source]

    def write_batch(batch_df, batch_id: int) -> None:
        rows = [r.asDict() for r in batch_df.collect()]
        if not rows:
            return
        from pymongo import UpdateOne

        col = get_db(settings)[source]
        ops = [UpdateOne({k: r[k] for k in keys}, {"$set": r}, upsert=True) for r in rows]
        col.bulk_write(ops, ordered=False)
        log.info("%s batch %s: upserted %d rows", source, batch_id, len(ops))

    return write_batch


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ensure_indexes(get_db(settings))

    spark = SparkSession.builder.appName("tickerpulse-stream").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    queries = []
    for source, topic in settings.sources.items():
        raw = (
            spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", settings.kafka_bootstrap_internal)
            .option("subscribe", topic)
            .option("startingOffsets", "latest")
            .load()
        )
        parsed = (
            raw.select(F.from_json(F.col("value").cast("string"), ddl(source)).alias("d"))
            .select("d.*")
            .dropna(subset=["ticker"])
        )
        q = (
            parsed.writeStream.foreachBatch(make_writer(source))
            .option("checkpointLocation", f"/tmp/tickerpulse/checkpoints/{source}")
            .trigger(processingTime="10 seconds")
            .start()
        )
        queries.append(q)
        log.info("stream started: %s -> %s", topic, source)

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
