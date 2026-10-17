import os
import time
import requests
import logging
from datetime import datetime, timezone

# ==========================================
# AI OTC TRADER BOT - Initial Version
# Telegram Technical Analysis Signal Bot
# ==========================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "NZDUSD": "NZDUSD=X",
    "USDCHF": "USDCHF=X",
}


def send_message(chat_id, text):
    """Send a Telegram message."""
    try:
        response = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text
            },
            timeout=15
        )
        response.raise_for_status()
        return response.json().get("ok", False)
    except requests.RequestException as exc:
        logging.error("Telegram send error: %s", exc)
        return False


def get_updates(offset=None):
    """Read incoming Telegram messages."""
    params = {
        "timeout": 25,
        "allowed_updates": ["message"]
    }

    if offset is not None:
        params["offset"] = offset

    response = requests.get(
        f"{TELEGRAM_API}/getUpdates",
        params=params,
        timeout=35
    )
    response.raise_for_status()
    return response.json()


def get_market_data(pair):
    """
    Retrieve public FX data from Yahoo Finance.
    These prices are NOT Quotex OTC prices.
    """
    symbol = PAIRS.get(pair.upper())

    if not symbol:
        return None

    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}"
    )

    params = {
        "range": "5d",
        "interval": "1m"
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20
        )
        response.raise_for_status()

        result = response.json()["chart"]["result"]

        if not result:
            return None

        data = result[0]
        quotes = data["indicators"]["quote"][0]

        closes = [
            float(x)
            for x in quotes["close"]
            if x is not None
        ]

        highs = [
            float(x)
            for x in quotes["high"]
            if x is not None
        ]

        lows = [
            float(x)
            for x in quotes["low"]
            if x is not None
        ]

        if len(closes) < 50:
            return None

        return {
            "close": closes,
            "high": highs,
            "low": lows,
            "symbol": symbol
        }

    except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
        logging.error("Market data error: %s", exc)
        return None


def sma(values, period):
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def ema(values, period):
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)
    result = sum(values[:period]) / period

    for value in values[period:]:
        result = (value - result) * multiplier + result

    return result


def rsi(values, period=14):
    if len(values) < period + 1:
        return None

    changes = [
        values[i] - values[i - 1]
        for i in range(len(values) - period, len(values))
    ]

    gains = sum(max(change, 0) for change in changes) / period
    losses = sum(max(-change, 0) for change in changes) / period

    if losses == 0:
        return 100.0

    return 100 - (100 / (1 + gains / losses))


def calculate_signal(data):
    """
    Technical indicator voting model.
    This is a rule-based model, not a trained AI model.
    """
    closes = data["close"]

    price = closes[-1]
    score = 0
    reasons = []

    # 1. SMA 10 versus SMA 20
    s10 = sma(closes, 10)
    s20 = sma(closes, 20)

    if s10 is not None and s20 is not None:
        if s10 > s20:
            score += 1
            reasons.append("SMA: BUY")
        elif s10 < s20:
            score -= 1
            reasons.append("SMA: SELL")

    # 2. EMA 9 versus EMA 21
    e9 = ema(closes, 9)
    e21 = ema(closes, 21)

    if e9 is not None and e21 is not None:
        if e9 > e21:
            score += 1
            reasons.append("EMA: BUY")
        elif e9 < e21:
            score -= 1
            reasons.append("EMA: SELL")

    # 3. RSI
    r = rsi(closes)

    if r is not None:
        if 50 < r < 70:
            score += 1
            reasons.append(f"RSI: BUY ({r:.1f})")
        elif 30 < r < 50:
            score -= 1
            reasons.append(f"RSI: SELL ({r:.1f})")
        else:
            reasons.append(f"RSI: NEUTRAL/EXTREME ({r:.1f})")

    # 4. MACD
    macd_line = ema(closes, 12) - ema(closes, 26)
    signal_line = ema(closes, 9)

    if macd_line > 0:
        score += 1
        reasons.append("MACD direction: BUY")
    elif macd_line < 0:
        score -= 1
        reasons.append("MACD direction: SELL")

    # 5. Short-term momentum
    if len(closes) >= 6:
        momentum = closes[-1] - closes[-6]

        if momentum > 0:
            score += 1
            reasons.append("Momentum: BUY")
        elif momentum < 0:
            score -= 1
            reasons.append("Momentum: SELL")

    # 6. Recent price structure
    recent = closes[-10:]
    previous = closes[-20:-10]

    if len(previous) == 10:
        if max(recent) > max(previous) and min(recent) > min(previous):
            score += 1
            reasons.append("Structure: BULLISH")
        elif max(recent) < max(previous) and min(recent) < min(previous):
            score -= 1
            reasons.append("Structure: BEARISH")

    # Signal decision
    maximum_score = 6
    strength = abs(score) / maximum_score * 100

    if score >= 4:
        signal = "🟢 STRONG BUY"
    elif score >= 2:
        signal = "🟢 MODERATE BUY"
    elif score <= -4:
        signal = "🔴 STRONG SELL"
    elif score <= -2:
        signal = "🔴 MODERATE SELL"
    else:
        signal = "🟡 NO TRADE"

    return {
        "signal": signal,
        "score": score,
        "strength": round(strength, 1),
        "price": price,
        "rsi": r,
        "reasons": reasons
    }


def build_signal_message(pair, analysis):
    reasons = "\n".join(
        f"• {reason}" for reason in analysis["reasons"]
    )

    return (
        "🤖 AI OTC TRADER BOT\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"💱 Pair: {pair}\n"
        "⏱ Analysis: 1-minute public market data\n"
        "📡 Source: Yahoo Finance\n\n"
        f"🎯 Signal: {analysis['signal']}\n"
        f"📊 Model score: {analysis['score']}/6\n"
        f"📈 Directional agreement: {analysis['strength']}%\n"
        f"💰 Last price: {analysis['price']}\n"
        f"📉 RSI: {analysis['rsi']:.2f}\n\n"
        "🔎 Technical analysis:\n"
        f"{reasons}\n\n"
        "⚠️ This is not a win probability.\n"
        "⚠️ Public prices may differ from Quotex OTC.\n"
        "⚠️ No signal guarantees a profitable trade."
    )


def handle_message(chat_id, text):
    text = text.strip()
    command = text.split()[0].lower()

    if command in ("/start", "/help"):
        send_message(
            chat_id,
            "🤖 Welcome to AI OTC TRADER BOT\n\n"
            "Choose a pair:\n"
            "/signal EURUSD\n"
            "/signal GBPUSD\n"
            "/signal EURGBP\n"
            "/signal USDJPY\n"
            "/signal AUDUSD\n"
            "/pairs\n\n"
            "Signals use public market data, not Quotex OTC prices."
        )
        return

    if command == "/pairs":
        send_message(
            chat_id,
            "Available pairs:\n" +
            "\n".join(f"/signal {pair}" for pair in PAIRS)
        )
        return

    if command == "/signal":
        parts = text.split()

        if len(parts) != 2:
            send_message(
                chat_id,
                "Use this format:\n/signal EURUSD"
            )
            return

        pair = parts[1].upper()

        if pair not in PAIRS:
            send_message(
                chat_id,
                "Pair not supported.\nUse /pairs to see available pairs."
            )
            return

        send_message(chat_id, f"⏳ Analyzing {pair}...")

        data = get_market_data(pair)

        if not data:
            send_message(
                chat_id,
                "❌ Market data unavailable.\n"
                "Please try again later."
            )
            return

        analysis = calculate_signal(data)
        message = build_signal_message(pair, analysis)
        send_message(chat_id, message)
        return

    send_message(
        chat_id,
        "Unknown command. Send /start to begin."
    )


def main():
    if not BOT_TOKEN:
        raise RuntimeError("Set BOT_TOKEN in your hosting environment.")

    if not CHAT_ID:
        logging.warning(
            "CHAT_ID is not configured. "
            "The bot will reply to users who message it."
        )

    logging.info("Bot started.")

    offset = None

    while True:
        try:
            updates = get_updates(offset)

            for update in updates.get("result", []):
                offset = update["update_id"] + 1

                message = update.get("message")

                if not message or not message.get("text"):
                    continue

                chat_id = message["chat"]["id"]
                text = message["text"]

                if CHAT_ID and str(chat_id) != str(CHAT_ID):
                    continue

                handle_message(chat_id, text)

        except Exception as exc:
            logging.error("Polling error: %s", exc)
            time.sleep(5)


if __name__ == "__main__":
    main()
