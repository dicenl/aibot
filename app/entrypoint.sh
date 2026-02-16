#!/usr/bin/env sh
set -eu
: "${RUN_EVERY_SECONDS:=3600}"

echo "CryptoBot loop started. Interval=${RUN_EVERY_SECONDS}s"
while true; do
  date -u
  python /app/bot.py
  sleep "${RUN_EVERY_SECONDS}"
done
