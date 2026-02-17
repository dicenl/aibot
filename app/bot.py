import os
import json
import requests
from datetime import datetime, timezone

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

from python_bitvavo_api.bitvavo import Bitvavo


# =========================
# Helpers
# =========================
def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def parse_symbols() -> list[str]:
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
        print("Telegram: missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "disable_web_page_preview": True}
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
# AI (optional)
# =========================
def ai_advice(snapshot: dict) -> dict:
    use_ai = env_bool("USE_AI", False)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not use_ai or not api_key:
        return {"action": "HOLD", "confidence": 0.50, "reason": "AI disabled or missing OPENAI_API_KEY"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content":
                 "You are a conservative crypto trading advisor. "
                 "Respond with strict JSON only: "
                 '{"action":"BUY|SELL|HOLD","confidence":0..1,"reason":"short"} '
                 "Prefer HOLD if uncertain."},
                {"role": "user", "content": json.dumps(snapshot, indent=2)},
            ],
            temperature=0.2,
        )
        return json.loads((resp.choices[0].message.content or "").strip())
    except Exception as e:
        return {"action": "HOLD", "confidence": 0.40, "reason": f"AI error: {e}"}


# =========================
# State (on-change)
# =========================
def load_state() -> dict:
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
# Indicators
# =========================
def to_df(candles) -> pd.DataFrame:
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
# Baseline with EMA20 slope + stretch TP
# =========================
def baseline_rule(s: dict) -> dict:
    rsi = s.get("rsi_14")
    ema20 = s.get("ema_20")
    ema50 = s.get("ema_50")
    close = s.get("close")
    slope = s.get("ema20_slope_pct")
    dist = s.get("dist_ema20_pct")

    if rsi is None or ema20 is None or ema50 is None or close is None or slope is None or dist is None:
        return {"action": "HOLD", "confidence": 0.40, "reason": "Not enough indicator history"}

    buy_rsi = float(os.getenv("BUY_RSI", "35"))
    tp_rsi = float(os.getenv("TP_RSI", "70"))
    tp_pct = float(os.getenv("TAKE_PROFIT_PCT", "2.5"))
    down_sell_rsi = float(os.getenv("DOWN_SELL_RSI", "45"))

    min_up_slope = float(os.getenv("EMA_SLOPE_MIN_UP", "0.15"))
    min_down_slope = float(os.getenv("EMA_SLOPE_MIN_DOWN", "0.15"))

    in_uptrend = ema20 > ema50
    in_downtrend = ema20 < ema50

    # BUY: dip in uptrend AND EMA20 rising enough
    if in_uptrend and slope >= min_up_slope and rsi < buy_rsi:
        return {
            "action": "BUY",
            "confidence": 0.72,
            "reason": f"Dip in uptrend (RSI<{buy_rsi:g}, slope +{slope:.2f}%)"
        }

    # SELL: blow-off top (extreme overbought, even if slope turning)
    extreme_rsi = float(os.getenv("EXTREME_TP_RSI", "80"))
    extreme_tp_pct = float(os.getenv("EXTREME_TP_PCT", "5.0"))

    if ema20 > ema50 and rsi > extreme_rsi and dist_ema20_pct >= extreme_tp_pct:
        return {
            "action": "SELL",
            "confidence": 0.75,
            "reason": f"Blow-off top (RSI>{extreme_rsi:g} & +{dist_ema20_pct:.1f}%>EMA20)"
        }

    # SELL: take profit in uptrend ONLY if stretched above EMA20 AND EMA20 rising enough
    if in_uptrend and slope >= min_up_slope and rsi > tp_rsi and dist >= tp_pct:
        return {
            "action": "SELL",
            "confidence": 0.68,
            "reason": f"Overbought + stretched (+{dist:.1f}%>EMA20, slope +{slope:.2f}%)"
        }

    # SELL: weakness in downtrend ONLY if EMA20 falling enough
    if in_downtrend and slope <= -min_down_slope and rsi < down_sell_rsi:
        return {
            "action": "SELL",
            "confidence": 0.60,
            "reason": f"Weak in downtrend (RSI<{down_sell_rsi:g}, slope {slope:.2f}%)"
        }

    return {"action": "HOLD", "confidence": 0.55, "reason": "No strong signal"}


# =========================
# Main
# =========================
def main():
    symbols = parse_symbols()
    interval = os.getenv("INTERVAL", "1h").strip()
    limit = int(os.getenv("CANDLE_LIMIT", "300"))

    send_all = env_bool("TELEGRAM_SEND_ALL", False)
    on_change = env_bool("TELEGRAM_ON_CHANGE", True)
    conf_threshold = float(os.getenv("TELEGRAM_MIN_CONF", "0.0"))

    slope_n = int(os.getenv("EMA_SLOPE_LOOKBACK", "5"))

    state = load_state() if on_change else {}

    bitvavo = Bitvavo({
        "APIKEY": os.getenv("BITVAVO_API_KEY", ""),
        "APISECRET": os.getenv("BITVAVO_API_SECRET", ""),
        "RESTURL": "https://api.bitvavo.com/v2",
        "WSURL": "wss://ws.bitvavo.com/v2/",
        "ACCESSWINDOW": 10000,
    })

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n=== CryptoBot run @ {now} | interval={interval} | limit={limit} | symbols={len(symbols)} | slope_n={slope_n} ===")

    summary = []

    for symbol in symbols:
        try:
            candles = bitvavo.candles(symbol, interval, {"limit": limit})

            if isinstance(candles, dict) and candles.get("error"):
                msg = f"Bitvavo error: {candles}"
                print(f"[{symbol}] {msg}")
                summary.append((symbol, "ERROR", 0.0, msg))
                continue

            if not isinstance(candles, list) or len(candles) < 80:
                msg = f"Not enough candles (len={len(candles) if hasattr(candles,'__len__') else 'n/a'})"
                print(f"[{symbol}] {msg}")
                summary.append((symbol, "ERROR", 0.0, msg))
                continue

            df = compute_indicators(to_df(candles)).sort_values("ts").reset_index(drop=True)
            df_ok = df.dropna(subset=["rsi_14", "ema_20", "ema_50"])

            if len(df_ok) <= slope_n:
                msg = "Not enough indicator history (for EMA slope)"
                print(f"[{symbol}] {msg}")
                summary.append((symbol, "ERROR", 0.0, msg))
                continue

            last_row = df_ok.iloc[-1]
            prev_row = df_ok.iloc[-1 - slope_n]

            close = float(last_row["close"])
            ema20_now = float(last_row["ema_20"])
            ema50_now = float(last_row["ema_50"])
            rsi = float(last_row["rsi_14"])

            ema20_prev = float(prev_row["ema_20"])
            slope = ((ema20_now / ema20_prev) - 1.0) * 100.0 if ema20_prev else 0.0
            dist = ((close / ema20_now) - 1.0) * 100.0 if ema20_now else 0.0

            snapshot = {
                "symbol": symbol,
                "interval": interval,
                "timestamp_utc": last_row["ts"].isoformat(),
                "close": close,
                "rsi_14": rsi,
                "ema_20": ema20_now,
                "ema_50": ema50_now,
                "ema20_slope_pct": slope,
                "dist_ema20_pct": dist,
            }

            baseline = baseline_rule(snapshot)
            ai = ai_advice({**snapshot, "baseline": baseline})

            action = (ai.get("action") or "HOLD").upper()
            try:
                conf = float(ai.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            ai_reason = str(ai.get("reason", "") or "")

            # Telegram reason: never send AI-disabled text
            if ai_reason.startswith("AI disabled") or "missing OPENAI_API_KEY" in ai_reason:
                tg_reason = baseline.get("reason", "No strong signal")
            else:
                tg_reason = ai_reason or baseline.get("reason", "No strong signal")

            # Log per coin
            print(
                f"[{symbol}] close={close:.4f} RSI={rsi:.2f} "
                f"EMA20={ema20_now:.4f} EMA50={ema50_now:.4f} "
                f"dist={dist:.2f}% slope={slope:.2f}% -> {action} ({conf:.2f}) | {tg_reason}"
            )

            summary.append((symbol, action, conf, tg_reason))

        except Exception as e:
            msg = f"Exception: {e}"
            print(f"[{symbol}] {msg}")
            summary.append((symbol, "ERROR", 0.0, msg))

    # Build Telegram lines
    lines = []
    for sym, action, conf, reason in summary:
        prev = state.get(sym) if on_change else None

        if action == "ERROR":
            lines.append(f"❗ {sym}: ERROR - {reason}")
            state[sym] = action
            continue

        # Optional confidence filter for BUY/SELL (useful when AI on later)
        if action in ("BUY", "SELL") and conf < conf_threshold:
            state[sym] = action
            continue

        if on_change and prev == action:
            continue

        if (not send_all) and action == "HOLD":
            state[sym] = action
            continue

        emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
        lines.append(f"{emoji} {sym}: {action} ({conf:.2f}) - {reason}")
        state[sym] = action

    if lines:
        header = f"📊 CryptoBot {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ({interval})"
        ok = telegram_send(header + "\n" + "\n".join(lines))
        print(f"Telegram sent: {ok} | lines={len(lines)}")
    else:
        print("Telegram: nothing to send (on-change and/or HOLD filtered)")

    if on_change:
        save_state(state)
        print(f"State saved: {os.getenv('STATE_FILE','/data/state.json')}")


if __name__ == "__main__":
    main()
