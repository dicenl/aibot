import os
import json
import requests
from datetime import datetime, timezone

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

from python_bitvavo_api.bitvavo import Bitvavo


# =========================
# Config helpers
# =========================
def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def parse_symbols() -> list[str]:
    """
    Use SYMBOLS="ADA-EUR,ETC-EUR,..." if provided, otherwise fall back to SYMBOL.
    """
    symbols_env = os.getenv("SYMBOLS", "").strip()
    if symbols_env:
        return [s.strip().upper() for s in symbols_env.split(",") if s.strip()]
    return [os.getenv("SYMBOL", "BTC-EUR").strip().upper()]


def safe_float(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return float(x)
    except Exception:
        return None


# =========================
# Telegram
# =========================
def telegram_send(message: str) -> bool:
    enabled = env_bool("TELEGRAM_ENABLED", True)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not enabled:
        return False
    if not token or not chat_id:
        # Don't raise; just make it obvious in logs
        print("Telegram not sent: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True,
    }

    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"Telegram send failed: {r.status_code} {r.text}")
            return False
        return True
    except Exception as e:
        print(f"Telegram send exception: {e}")
        return False


# =========================
# Optional AI (second opinion)
# =========================
def ai_advice(snapshot: dict) -> dict:
    """
    Optional AI second opinion.
    Returns JSON dict:
      {"action":"BUY|SELL|HOLD","confidence":0..1,"reason":"short"}
    Falls back to HOLD if disabled or error.
    """
    use_ai = env_bool("USE_AI", False)
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


# =========================
# State (for on-change alerts)
# =========================
def load_state() -> dict:
    """
    Loads last actions per symbol from STATE_FILE.
    Default location: /data/state.json (bind a volume to /data!)
    """
    path = os.getenv("STATE_FILE", "/data/state.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    path = os.getenv("STATE_FILE", "/data/state.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
    except Exception as e:
        print(f"Warning: could not save state to {path}: {e}")


# =========================
# Data + indicators
# =========================
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


# =========================
# Baseline strategy (simple)
# =========================
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


# =========================
# Main
# =========================
def main():
    symbols = parse_symbols()
    interval = os.getenv("INTERVAL", "1h").strip()
    limit = int(os.getenv("CANDLE_LIMIT", "300"))

    # Telegram behavior
    send_all = env_bool("TELEGRAM_SEND_ALL", False)         # send HOLD too
    on_change = env_bool("TELEGRAM_ON_CHANGE", True)        # only alert if action changed
    conf_threshold = float(os.getenv("TELEGRAM_MIN_CONF", "0.0"))  # filter on AI confidence

    # Load previous state if we want "on change" behavior
    state = load_state() if on_change else {}

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

            if isinstance(candles, dict) and candles.get("error"):
                print(f"\n[{symbol}] Bitvavo error: {candles}")
                summary.append((symbol, "ERROR", 0.0, "Bitvavo error"))
                continue

            if not isinstance(candles, list) or len(candles) < 60:
                print(f"\n[{symbol}] Not enough candle data returned (len={len(candles) if hasattr(candles,'__len__') else 'n/a'})")
                summary.append((symbol, "ERROR", 0.0, "Not enough candle data"))
                continue

            df = compute_indicators(to_df(candles))

            # Ensure we take the most recent candle by timestamp
            df = df.sort_values("ts").reset_index(drop=True)
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

        # Normalize AI fields
        action = (ai.get("action") or "HOLD").upper()
        try:
            conf = float(ai.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        reason = str(ai.get("reason", "") or "")

        # If AI is disabled, don't leak that message into Telegram
        if reason.startswith("AI disabled") or "missing OPENAI_API_KEY" in reason:
            reason = baseline.get("reason", "No strong signal")


#            ai = ai_advice({**snapshot, "baseline": baseline})
#
#            # Normalize AI fields
#            action = (ai.get("action") or "HOLD").upper()
#            try:
#                conf = float(ai.get("confidence", 0.0))
#            except Exception:
#                conf = 0.0
#            reason = str(ai.get("reason", ""))

            print("\n" + "=" * 72)
            print(f"{symbol} | close={snapshot['close']} | ts={snapshot['timestamp_utc']}")
            print(f"RSI14={snapshot['rsi_14']} EMA20={snapshot['ema_20']} EMA50={snapshot['ema_50']}")
            print("- Baseline:", baseline)
            print("- AI:", {"action": action, "confidence": conf, "reason": reason})
            print("=" * 72)

            summary.append((symbol, action, conf, reason))

        except Exception as e:
            print(f"\n[{symbol}] Exception: {e}")
            summary.append((symbol, "ERROR", 0.0, f"Exception: {e}"))

    # Console Summary
    print("\n--- Summary (AI output) ---")
    for sym, action, conf, reason in summary:
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
        short_reason = (reason or "").strip()
        if len(short_reason) > 120:
            short_reason = short_reason[:120] + "…"
        print(f"{sym:10s}  {action:5s}  conf={conf_str}  {short_reason}")

    # Telegram summary (default: BUY/SELL/ERROR; optional HOLD)
    lines = []
    for sym, action, conf, reason in summary:
        prev = state.get(sym) if on_change else None

        if action == "ERROR":
            lines.append(f"❗ {sym}: ERROR - {reason}")
            state[sym] = action
            continue

        # confidence filter (only applies to non-HOLD)
        if action in ("BUY", "SELL") and conf < conf_threshold:
            continue

        # Only send if changed
        if on_change and prev == action:
            continue

        # Default: don't spam HOLD
        if (not send_all) and action == "HOLD":
            state[sym] = action  # still update state so we can detect HOLD->BUY later
            continue

        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
        lines.append(f"{emoji} {sym}: {action} ({conf:.2f}) - {reason}")
        state[sym] = action

    if lines:
        header = f"📊 CryptoBot {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ({interval})"
        ok = telegram_send(header + "\n" + "\n".join(lines))
        print(f"Telegram sent: {ok}")

    # Persist state (for on-change)
    if on_change:
        save_state(state)

    print("--- End ---\n")


if __name__ == "__main__":
    main()
