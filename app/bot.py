import os
import json
from datetime import datetime, timezone

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

from python_bitvavo_api.bitvavo import Bitvavo


def ai_advice(snapshot: dict) -> dict:
    """
    Optional AI second opinion.
    Returns JSON dict: {"action":"BUY|SELL|HOLD","confidence":0..1,"reason":"..."}
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
            '{"action":"BUY|SELL|HOLD","confidence":0..1,"reason":"short"}'
        )
        user = (
            "Given this market snapshot, return ONE conservative action. "
            "Prefer HOLD if uncertain.\n\n"
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

        content = resp.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        return {"action": "HOLD", "confidence": 0.40, "reason": f"AI error: {e}"}


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


def main():
    symbol = os.getenv("SYMBOL", "BTC-EUR")
    interval = os.getenv("INTERVAL", "1h")
    limit = int(os.getenv("CANDLE_LIMIT", "300"))

    bitvavo = Bitvavo({
        "APIKEY": os.getenv("BITVAVO_API_KEY", ""),
        "APISECRET": os.getenv("BITVAVO_API_SECRET", ""),
        "RESTURL": "https://api.bitvavo.com/v2",
        "WSURL": "wss://ws.bitvavo.com/v2/",
        "ACCESSWINDOW": 10000,
    })

    candles = bitvavo.candles(symbol, interval, {"limit": limit})
    if isinstance(candles, dict) and candles.get("error"):
        raise RuntimeError(f"Bitvavo error: {candles}")

    df = compute_indicators(to_df(candles))

    last = df.iloc[-1].to_dict()
    snapshot = {
        "symbol": symbol,
        "interval": interval,
        "timestamp_utc": last["ts"].isoformat(),
        "close": float(last["close"]),
        "rsi_14": None if pd.isna(last["rsi_14"]) else float(last["rsi_14"]),
        "ema_20": None if pd.isna(last["ema_20"]) else float(last["ema_20"]),
        "ema_50": None if pd.isna(last["ema_50"]) else float(last["ema_50"]),
    }

    baseline = baseline_rule(snapshot)
    ai = ai_advice({**snapshot, "baseline": baseline})

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print("=" * 72)
    print(f"{now} | {symbol} | interval={interval} | close={snapshot['close']}")
    print(f"RSI14={snapshot['rsi_14']} EMA20={snapshot['ema_20']} EMA50={snapshot['ema_50']}")
    print("- Baseline:", baseline)
    print("- AI:", ai)
    print("=" * 72)


if __name__ == "__main__":
    main()
