import os
import json
from datetime import datetime, timezone

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

from python_bitvavo_api.bitvavo import Bitvavo


# ----------------------------
# Optional AI (second opinion)
# ----------------------------
def ai_advice(snapshot: dict) -> dict:
    """
    Optional AI second opinion.
    Returns JSON dict:
      {"action":"BUY|SELL|HOLD","confidence":0..1,"reason":"short"}
    Falls back to HOLD if disabled or error.
    """
    use_ai = os.getenv("USE_AI", "false").lower() == "true"
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not use_ai or not api_key:
        return {"action": "HOLD", "confidence": 0.50, "reason": "AI disabled or missing OPENAI_API_KEY"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system = (
            "You are a conservative crypto trading advisor. "
            "Respond with strict JSON only: "
            '{"action":"BUY|SELL|HOLD","confidence":0..1,"reason":"short"} '
            "Prefer HOLD if uncertain."
        )

        user = (
            "Given this market snapshot, return ONE conservative action.\n\n"
            f"{json.dumps(snapshot, indent=2)}"
        )

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )

        content = (resp.choices[0].message.content or "").strip()
        return json.loads(content)

    except Exception as e:
        return {"action": "HOLD", "confidence": 0.40, "reason": f"AI error: {e}"}


# ----------------------------
# Data + indicators
# ----------------------------
def to_df(candles) -> pd.DataFrame:
    """
    Bitvavo candles: [timestamp, open, high, low, close, volume]
    """
    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
    df["ema_20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=close, window=50).ema_indicator()
    return df


# ----------------------------
# Baseline strategy (simple)
# ----------------------------
def baseline_rule(s: dict) -> dict:
    """
    Conservative baseline:
    - BUY: RSI < 30 AND EMA20 > EMA50 (oversold pullback in uptrend)
    - SELL: RSI > 70 AND EMA20 < EMA50 (overbought bounce in downtrend)
    - else HOLD
    """
    rsi = s.get("rsi_14")
    ema20 = s.get("ema_20")
    ema50 = s.get("ema_50")

    if rsi is None or ema20 is None or ema50 is None:
        return {"action": "HOLD", "confidence": 0.40, "reason": "Not enough indicator history"}

    if rsi < 30 and ema20 > ema50:
        return {"action": "BUY", "confidence": 0.65, "reason": "RSI oversold in uptrend (EMA20>EMA50)"}

    if rsi > 70 and ema20 < ema50:
        return {"action": "SELL", "confidence": 0.65, "reason": "RSI overbought in downtrend (EMA20<EMA50)"}

    return {"action": "HOLD", "confidence": 0.55, "reason": "No strong signal"}


# ----------------------------
# Helpers
# ----------------------------
def parse_symbols() -> list[str]:
    """
    Use SYMBOLS="ADA-EUR,ETC-EUR,..." if provided, otherwise fall back to SYMBOL.
    """
    symbols_env = os.getenv("SYMBOLS", "").strip()
    if symbols_env:
        symbols = [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
        return symbols
    return [os.getenv("SYMBOL", "BTC-EUR").strip().upper()]


def safe_float(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None


# ----------------------------
# Main
# ----------------------------
def main():
    symbols = parse_symbols()
    interval = os.getenv("INTERVAL", "1h").strip()
    limit = int(os.getenv("CANDLE_LIMIT", "300"))

    bitvavo = Bitvavo({
        "APIKEY": os.getenv("BITVAVO_API_KEY", ""),
        "APISECRET": os.getenv("BITVAVO_API_SECRET", ""),
        "RESTURL": "https://api.bitvavo.com/v2",
        "WSURL": "wss://ws.bitvavo.com/v2/",
        "ACCESSWINDOW": 10000,
    })

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n=== CryptoBot run @ {now} | interval={interval} | limit={limit} | symbols={len(symbols)} ===")

    summary = []

    for symbol in symbols:
        try:
            candles = bitvavo.candles(symbol, interval, {"limit": limit})

            # Bitvavo can return dict with error
            if isinstance(candles, dict) and candles.get("error"):
                err = candles
                print(f"\n[{symbol}] Bitvavo error: {err}")
                summary.append((symbol, "ERROR", 0.0, "Bitvavo error"))
                continue

            # Or sometimes empty list
            if not isinstance(candles, list) or len(candles) < 60:
                print(f"\n[{symbol}] Not enough candle data returned (len={len(candles) if hasattr(candles,'__len__') else 'n/a'})")
                summary.append((symbol, "ERROR", 0.0, "Not enough candle data"))
                continue

            df = compute_indicators(to_df(candles))
            last = df.iloc[-1].to_dict()

            snapshot = {
                "symbol": symbol,
                "interval": interval,
                "timestamp_utc": last["ts"].isoformat(),
                "close": safe_float(last["close"]),
                "rsi_14": safe_float(last.get("rsi_14")),
                "ema_20": safe_float(last.get("ema_20")),
                "ema_50": safe_float(last.get("ema_50")),
            }

            baseline = baseline_rule(snapshot)
            ai = ai_advice({**snapshot, "baseline": baseline})

            print("\n" + "=" * 72)
            print(f"{symbol} | close={snapshot['close']} | ts={snapshot['timestamp_utc']}")
            print(f"RSI14={snapshot['rsi_14']} EMA20={snapshot['ema_20']} EMA50={snapshot['ema_50']}")
            print("- Baseline:", baseline)
            print("- AI:", ai)
            print("=" * 72)

            summary.append((symbol, ai.get("action", "HOLD"), float(ai.get("confidence", 0.0)), ai.get("reason", "")))

        except Exception as e:
            print(f"\n[{symbol}] Exception: {e}")
            summary.append((symbol, "ERROR", 0.0, f"Exception: {e}"))

    # Summary table
    print("\n--- Summary (AI output) ---")
    for sym, action, conf, reason in summary:
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
        short_reason = (reason or "").strip()
        if len(short_reason) > 120:
            short_reason = short_reason[:120] + "…"
        print(f"{sym:10s}  {action:5s}  conf={conf_str}  {short_reason}")

    print("--- End ---\n")


if __name__ == "__main__":
    main()
