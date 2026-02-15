#!/usr/bin/env sh
set -eu

: "${RUN_EVERY_SECONDS:=900}"  # 900 = 15 min

echo "CryptoBot loop started. Interval=${RUN_EVERY_SECONDS}s"
while true; do
  date -u
  python /app/bot.py || true
  sleep "${RUN_EVERY_SECONDS}"
done
