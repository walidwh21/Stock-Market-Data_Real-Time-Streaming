#!/usr/bin/env bash
# Create the Kafka topics used by TickerPulse (idempotent).
set -euo pipefail

CONTAINER="${KAFKA_CONTAINER:-tp-kafka}"
PREFIX="${TOPIC_PREFIX:-tp}"
BOOTSTRAP="localhost:9092"

echo "Waiting for Kafka in container '$CONTAINER'..."
until docker exec "$CONTAINER" kafka-topics --list --bootstrap-server "$BOOTSTRAP" >/dev/null 2>&1; do
  sleep 2
done

for source in prices reddit stocktwits news; do
  docker exec "$CONTAINER" kafka-topics --create --if-not-exists \
    --topic "${PREFIX}.${source}" --bootstrap-server "$BOOTSTRAP" \
    --partitions 3 --replication-factor 1
done

echo "Topics:"
docker exec "$CONTAINER" kafka-topics --list --bootstrap-server "$BOOTSTRAP"
