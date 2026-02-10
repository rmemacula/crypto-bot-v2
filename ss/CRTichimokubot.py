import os
import json
import logging
import requests
import pandas as pd
import numpy as np
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID", "0"))

# ---------------- MANUAL TOP 50 COINS ----------------
MANUAL_TOP_50 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "TRXUSDT", "LINKUSDT", "MATICUSDT", "DOTUSDT", "LTCUSDT", "SHIBUSDT",
    "UNIUSDT", "NEARUSDT", "AAVEUSDT", "ATOMUSDT", "SUIUSDT", "PEPEUSDT", "FLOKIUSDT",
    "WIFUSDT", "SEIUSDT", "BONKUSDT", "ARBUSDT", "OPUSDT", "TIAUSDT", "ENSUSDT",
    "RUNEUSDT", "FTMUSDT", "GALAUSDT", "MEMEUSDT", "PYTHUSDT", "1000SATSUSDT",
    "JUPUSDT", "ORDIUSDT", "NOTUSDT", "WLDUSDT", "ETCUSDT", "HBARUSDT", "ICPUSDT",
    "BCHUSDT", "SANDUSDT", "MANAUSDT", "EGLDUSDT", "FILUSDT", "XLMUSDT", "ALGOUSDT",
    "IMXUSDT", "APTUSDT"
]

def fetch_top_volume_pairs(limit=20):
    """Fetch top N Binance USDT perpetual pairs by 24h quote volume."""
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        df = pd.DataFrame(data)
        df = df[df["symbol"].str.endswith("USDT")]
        df["quoteVolume"] = df["quoteVolume"].astype(float)
        top = df.sort_values("quoteVolume", ascending=False).head(limit)
        return top["symbol"].tolist()
    except Exception as e:
        logging.error(f"Error fetching top volume pairs: {e}")
        return []

# ---------------- FETCH BINANCE USDT PAIRS ----------------
def fetch_usdt_pairs():
    """Fetch all Binance USDT perpetual futures pairs."""
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        symbols = [
            s["symbol"]
            for s in data["symbols"]
            if s["quoteAsset"] == "USDT"
            and s["contractType"] == "PERPETUAL"
            and s["status"] == "TRADING"
        ]
        return sorted(symbols)
    except Exception as e:
        logging.error(f"Error fetching futures pairs: {e}")
        return []

# Load all current USDT pairs
def load_symbols():
    """Combine manual top 50 + top 20 by futures volume (unique)."""
    manual = set(MANUAL_TOP_50)
    top_vol = set(fetch_top_volume_pairs(20))
    combined = sorted(list(manual | top_vol))  # merge, no duplicates
    logging.info(f"✅ Loaded {len(combined)} symbols (Top 50 + Top 20 by volume).")
    return combined

SYMBOLS = load_symbols()

# ---------------- REFRESH ----------------
def refresh_pairs(context: CallbackContext):
    """Daily refresh: manual + top 20 high-volume Binance USDT futures pairs."""
    global SYMBOLS
    logging.info("🔄 Refreshing symbol list (Top 50 + Top 20)...")

    new_symbols = load_symbols()
    added = set(new_symbols) - set(SYMBOLS)
    removed = set(SYMBOLS) - set(new_symbols)
    SYMBOLS = new_symbols

    msg = f"♻️ Updated symbols: {len(SYMBOLS)} total."
    if added:
        msg += f"\n➕ Added: {', '.join(sorted(list(added))[:10])}..."
    if removed:
        msg += f"\n➖ Removed: {', '.join(sorted(list(removed))[:10])}..."

    logging.info(msg)
    try:
        context.bot.send_message(chat_id=CHAT_ID, text=msg)
    except Exception:
        pass

TIMEFRAMES = {"1h": "1h", "4h": "4h", "1d": "1d", "1w": "1w"}
DATA_FILE = "last_signals.json"

logging.basicConfig(level=logging.INFO)

# ---------------- STORAGE ----------------
def load_last_signals():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_last_signals(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

last_signals = load_last_signals()

def key_for(symbol, tf):
    return f"{symbol}_{tf}"

# ---------------- TRADINGVIEW LINK ----------------
def tradingview_link(symbol, tf_label):
    tf_map = {"1h": "60", "4h": "240", "1d": "1D", "1w": "1W"}  # ==== CRT: add 1w
    interval = tf_map.get(tf_label, "60")
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}&interval={interval}"

# ---------------- DATA FETCH ----------------
def fetch_ohlcv(symbol, interval, limit=150):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            logging.warning(f"Fetch fail {symbol} {interval}: HTTP {r.status_code}")
            return None
        data = r.json()
        if isinstance(data, dict) and data.get("code"):
            logging.warning(f"Binance API error for {symbol} {interval}: {data}")
            return None
        df = pd.DataFrame(data, columns=[
            "time", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"
        ])
        # ==== CRT: need 'o' too
        df["o"] = df["o"].astype(float)
        df["c"] = df["c"].astype(float)
        df["h"] = df["h"].astype(float)
        df["l"] = df["l"].astype(float)
        return df
    except Exception as e:
        logging.warning(f"Fetch exception {symbol} {interval}: {e}")
        return None

# ---------------- CRT DETECTION ----------------
def detect_crt_on_last_pair(df):
    """
    Detect Candle Range Theory on the last TWO *closed* candles:
      prev = df.iloc[-3], curr = df.iloc[-2]
    Conditions:
      Bullish CRT: prev red, curr sweeps prev low, curr closes back inside prev range
      Bearish CRT: prev green, curr sweeps prev high, curr closes back inside prev range
    """
    if df is None or len(df) < 3:
        return {"bullish_crt": False, "bearish_crt": False}

    prev = df.iloc[-3]
    curr = df.iloc[-2]

    prev_open = float(prev["o"])
    prev_close = float(prev["c"])
    prev_high = float(prev["h"])
    prev_low  = float(prev["l"])

    curr_close = float(curr["c"])
    curr_high  = float(curr["h"])
    curr_low   = float(curr["l"])

    prev_red   = prev_close < prev_open
    prev_green = prev_close > prev_open

    def inside_prev_range(x):
        return (prev_low < x < prev_high)

    bullish_crt = prev_red and (curr_low < prev_low) and inside_prev_range(curr_close)
    bearish_crt = prev_green and (curr_high > prev_high) and inside_prev_range(curr_close)

    return {
        "bullish_crt": bool(bullish_crt),
        "bearish_crt": bool(bearish_crt),
        "prev_high": prev_high,
        "prev_low": prev_low,
        "curr_high": curr_high,
        "curr_low": curr_low,
        "curr_close": curr_close,
    }

# ---------------- ANALYSIS ----------------
def analyze_df(df):
    if df is None or len(df) < 104:  # Need 52 + 26 + 26 for proper analysis
        return {"signal": "Neutral", "price": None, "rsi": None}

    close = df["c"]
    high = df["h"]
    low = df["l"]

    # ---- Ichimoku Components ----
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2

    # ---- RSI ----
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-2]) if not np.isnan(rsi.iloc[-2]) else None

    # ---- Last closed candle analysis ----
    last_idx = -2
    price = float(close.iloc[last_idx])
    tenkan_v = float(tenkan.iloc[last_idx])
    kijun_v = float(kijun.iloc[last_idx])

    # ---- Current Cloud Position (projected 26 back) ----
    cloud_idx = last_idx - 26
    if abs(cloud_idx) <= len(df) - 52:
        cloud_a_current = float(senkou_a.iloc[cloud_idx])
        cloud_b_current = float(senkou_b.iloc[cloud_idx])
    else:
        cloud_a_current = float(senkou_a.iloc[last_idx])
        cloud_b_current = float(senkou_b.iloc[last_idx])

    # ---- Chikou vs then-cloud ----
    chikou_above = chikou_below = False
    chikou_idx = last_idx - 26
    chikou_cloud_idx = chikou_idx - 26
    if abs(chikou_cloud_idx) <= len(df) - 52:
        ca = float(senkou_a.iloc[chikou_cloud_idx])
        cb = float(senkou_b.iloc[chikou_cloud_idx])
        if not np.isnan(ca) and not np.isnan(cb):
            chikou_above = price > max(ca, cb)
            chikou_below = price < min(ca, cb)

    # ---- Future Cloud (26 ahead projection) ----
    cloud_a_future = float(senkou_a.iloc[last_idx])
    cloud_b_future = float(senkou_b.iloc[last_idx])
    future_cloud_bullish = cloud_a_future > cloud_b_future
    future_cloud_bearish = cloud_a_future < cloud_b_future

    # ---- Ichimoku Checklist ----
    checklist_bull = [
        ("Price above cloud", price > max(cloud_a_current, cloud_b_current)),
        ("Tenkan > Kijun", tenkan_v > kijun_v),
        ("Chikou above cloud", chikou_above),
        ("Future cloud bullish", future_cloud_bullish),
    ]
    checklist_bear = [
        ("Price below cloud", price < min(cloud_a_current, cloud_b_current)),
        ("Tenkan < Kijun", tenkan_v < kijun_v),
        ("Chikou below cloud", chikou_below),
        ("Future cloud bearish", future_cloud_bearish),
    ]

    bullish_count = sum(c for _, c in checklist_bull)
    bearish_count = sum(c for _, c in checklist_bear)

    # ---- Signal ----
    signal = "Neutral"
    sl = tp = None
    if bullish_count >= 3:
        signal = "BUY"
        sl = min(cloud_a_current, cloud_b_current) * 0.995
        tp = price + 2 * (price - sl)
    elif bearish_count >= 3:
        signal = "SELL"
        sl = max(cloud_a_current, cloud_b_current) * 1.005
        tp = price - 2 * (sl - price)

    if sl is None or tp is None or np.isnan(sl) or np.isnan(tp):
        sl = tp = None

    # ==== CRT: detect on last two closed candles
    crt = detect_crt_on_last_pair(df)
    return {
        "price": price,
        "rsi": rsi_val,
        "signal": signal,
        "bull_count": bullish_count,
        "bear_count": bearish_count,
        "checklist_bull": checklist_bull,
        "checklist_bear": checklist_bear,
        "sl": sl,
        "tp": tp,
        "tenkan": tenkan_v,
        "kijun": kijun_v,
        "cloud_a": float(cloud_a_current),
        "cloud_b": float(cloud_b_current),
        "cloud_a_future": float(cloud_a_future),
        "cloud_b_future": float(cloud_b_future),
        # CRT fields
        "crt_bull": crt["bullish_crt"],
        "crt_bear": crt["bearish_crt"],
        "crt_prev_high": crt["prev_high"],
        "crt_prev_low": crt["prev_low"],
        "crt_curr_high": crt["curr_high"],
        "crt_curr_low": crt["curr_low"],
        "crt_curr_close": crt["curr_close"],
    }

# ---------------- CHECKLIST FORMATTER ----------------
def format_checklist(analysis):
    lines = []
    signal = analysis["signal"]
    for (bull_label, bull_val), (bear_label, bear_val) in zip(
        analysis["checklist_bull"], analysis["checklist_bear"]
    ):
        if signal == "BUY":
            lines.append(f"{'✅' if bull_val else '❌'} {bull_label}")
        elif signal == "SELL":
            lines.append(f"{'✅' if bear_val else '❌'} {bear_label}")
        else:
            if bull_val:
                lines.append("✅ " + bull_label)
            elif bear_val:
                lines.append("✅ " + bear_label)
            else:
                lines.append("❌ " + bull_label + " / " + bear_label)

    # CRT one-liner for visibility (not an alert by itself)
    if analysis.get("crt_bull"):
        lines.append("🕯️ CRT: Bullish (swept prior low, closed back inside)")
    elif analysis.get("crt_bear"):
        lines.append("🕯️ CRT: Bearish (swept prior high, closed back inside)")

    return "\n".join(lines)

# ---------------- COMMANDS ----------------
def test(update: Update, context: CallbackContext):
    update.message.reply_text("✅ Bot is working!")

def status(update: Update, context: CallbackContext):
    if not context.args:
        update.message.reply_text("Usage: /status BTC")
        return
    sym = context.args[0].upper() + "USDT"

    if sym not in SYMBOLS:
        update.message.reply_text("Unknown coin")
        return

    messages = []
    for tf_label, interval in TIMEFRAMES.items():
        df = fetch_ohlcv(sym, interval)
        if df is None:
            messages.append(f"❌ No data for {tf_label} (check symbol or API)")
            continue
        if len(df) < 104:
            messages.append(f"❌ Not enough data for {tf_label} (need 104+ candles, have {len[df]})")
            continue

        analysis = analyze_df(df)

        msg = (
            f"📊 {sym} ({tf_label})\n"
            f"Signal: {analysis['signal']}\n"
            f"Price: {analysis['price']:.2f} USDT\n"
            f"RSI: {analysis['rsi']:.2f}\n"
            f"📈 [View on TradingView]({tradingview_link(sym, tf_label)})\n"
        )
        if analysis["sl"] and analysis["tp"]:
            msg += f"SL: {analysis['sl']:.2f} | TP: {analysis['tp']:.2f}\n"

        # Show CRT details if present
        if analysis.get("crt_bull") or analysis.get("crt_bear"):
            side = "Bullish" if analysis.get("crt_bull") else "Bearish"
            msg += (
                f"🕯️ CRT: *{side}* "
                f"(prev H/L: {analysis['crt_prev_high']:.4f}/{analysis['crt_prev_low']:.4f}, "
                f"curr L/H/Close: {analysis['crt_curr_low']:.4f}/{analysis['crt_curr_high']:.4f}/{analysis['crt_curr_close']:.4f})\n"
            )

        msg += "\n" + format_checklist(analysis)
        messages.append(msg)

    update.message.reply_text("\n\n".join(messages), parse_mode="Markdown")

# ---------------- ALERT JOB ----------------
def check_and_alert(context: CallbackContext):
    global last_signals
    bot = context.bot

    for symbol in SYMBOLS:
        for tf_label, interval in TIMEFRAMES.items():
            df = fetch_ohlcv(symbol, interval)
            if df is None or len(df) < 104:
                continue

            analysis = analyze_df(df)
            sig = analysis["signal"]
            k = key_for(symbol, tf_label)

            # Include CRT tag to de-dup properly
            crt_tag = "CRT_BULL" if analysis.get("crt_bull") else ("CRT_BEAR" if analysis.get("crt_bear") else "CRT_NONE")
            sent_label = f"{sig}|{analysis['bull_count']}|{analysis['bear_count']}|{crt_tag}"
            prev = last_signals.get(k)

            # ✅ Ichimoku 4/4 alert (unchanged)
            if (sig == "BUY" and analysis["bull_count"] == 4) or (sig == "SELL" and analysis["bear_count"] == 4):
                if prev != sent_label:
                    tv_link = tradingview_link(symbol, tf_label)
                    safe_symbol = symbol.replace("_", "\\_").replace("-", "\\-")

                    msg = (
                        f"🚨 *{safe_symbol}* ({tf_label}) — *{sig} (4/4 confirmed)*\n\n"
                        f"💰 *Price:* {analysis['price']:.2f} USDT\n"
                        f"📊 *RSI:* {analysis['rsi']:.2f}\n"
                        f"🔗 [View on TradingView]({tv_link})\n\n"
                    )

                    if analysis["sl"] and analysis["tp"]:
                        msg += f"🎯 *SL:* {analysis['sl']:.2f} | *TP:* {analysis['tp']:.2f}\n\n"

                    msg += format_checklist(analysis)

                    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                    last_signals[k] = sent_label
                    save_last_signals(last_signals)

            # 🕯️ CRT alert ONLY if CRT aligns with Ichimoku **4/4** direction
            aligned_bullish_crt = analysis.get("crt_bull") and (sig == "BUY" and analysis["bull_count"] == 4)
            aligned_bearish_crt = analysis.get("crt_bear") and (sig == "SELL" and analysis["bear_count"] == 4)
            if (aligned_bullish_crt or aligned_bearish_crt) and prev != sent_label:
                tv_link = tradingview_link(symbol, tf_label)
                safe_symbol = symbol.replace("_", "\\_").replace("-", "\\-")
                side = "Bullish" if aligned_bullish_crt else "Bearish"

                ph, pl = analysis["crt_prev_high"], analysis["crt_prev_low"]
                ch, cl, cc = analysis["crt_curr_high"], analysis["crt_curr_low"], analysis["crt_curr_close"]

                msg = (
                    f"🕯️ *{safe_symbol}* ({tf_label}) — CRT {side} *aligned* with Ichimoku 4/4 {sig}\n\n"
                    f"• Prev H/L: {ph:.4f} / {pl:.4f}\n"
                    f"• Curr H/L/Close: {ch:.4f} / {cl:.4f} / {cc:.4f}\n"
                    f"🔗 [View on TradingView]({tv_link})\n\n"
                    f"{format_checklist(analysis)}"
                )

                bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                last_signals[k] = sent_label
                save_last_signals(last_signals)

            # ⚪ Exit from strong zone
            elif prev and ("BUY|4|0" in prev or "SELL|0|4" in prev):
                if not ((sig == "BUY" and analysis["bull_count"] == 4) or (sig == "SELL" and analysis["bear_count"] == 4)):
                    safe_symbol = symbol.replace("_", "\\_").replace("-", "\\-")
                    msg = f"⚪ *{safe_symbol}* ({tf_label}) — exited strong {prev.split('|')[0]} zone.\nNow: {sig} ({analysis['bull_count']}/4 bull, {analysis['bear_count']}/4 bear)"
                    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                    last_signals[k] = sent_label
                    save_last_signals(last_signals)

#---------------status1d--------------------
from datetime import datetime, timezone, timedelta

def status1d(update: Update, context: CallbackContext):
    tf_label = "1d"
    interval = TIMEFRAMES[tf_label]

    update.message.reply_text("⏳ Scanning 1D Ichimoku + CRT (showing Ichimoku 4/4 only; CRT noted if present)...")

    buy_msgs, sell_msgs = [], []
    manila_tz = timezone(timedelta(hours=8))  # Manila timezone

    for sym in SYMBOLS:
        try:
            df = fetch_ohlcv(sym, interval)
            if df is None or len(df) < 104:
                continue

            analysis = analyze_df(df)
            signal = analysis.get("signal", "Neutral")

            # Only show 4/4
            if not ((signal == "BUY" and analysis.get("bull_count", 0) == 4) or
                    (signal == "SELL" and analysis.get("bear_count", 0) == 4)):
                continue

            # Backtrack to first 4/4 candle (best-effort)
            trigger_time = None
            start_idx = max(103, len(df) - 100)
            for i in range(start_idx, len(df)):
                sub_df = df.iloc[: i + 1]
                if len(sub_df) < 104:
                    continue
                sub_analysis = analyze_df(sub_df)
                if signal == "BUY" and sub_analysis.get("bull_count", 0) == 4:
                    trigger_time = int(sub_df.iloc[-2]["time"])
                    break
                if signal == "SELL" and sub_analysis.get("bear_count", 0) == 4:
                    trigger_time = int(sub_df.iloc[-2]["time"])
                    break

            if trigger_time is None:
                trigger_time = int(df.iloc[-2]["time"])

            ts = datetime.fromtimestamp(trigger_time / 1000, tz=manila_tz).strftime("%Y-%m-%d %I:%M %p (Manila)")

            safe_symbol = sym.replace("_", "\\_").replace("-", "\\-")
            price_str = f"{analysis['price']:.2f} USDT" if analysis.get("price") is not None else "N/A"
            rsi_str = f"{analysis['rsi']:.2f}" if analysis.get("rsi") is not None else "N/A"

            msg = (
                f"{'🟩' if signal == 'BUY' else '🟥'} *{safe_symbol}* — STRONG {signal} (4/4)\n"
                f"🕒 Time: {ts}\n"
                f"💰 Price: {price_str}\n"
                f"📊 RSI: {rsi_str}\n"
                f"🔗 [TradingView]({tradingview_link(sym, tf_label)})\n"
            )

            if analysis.get("sl") is not None and analysis.get("tp") is not None:
                msg += f"🎯 SL: {analysis['sl']:.2f} | TP: {analysis['tp']:.2f}\n"

            if analysis.get("crt_bull") or analysis.get("crt_bear"):
                side = "Bullish" if analysis.get("crt_bull") else "Bearish"
                msg += (
                    f"🕯️ CRT: *{side}* "
                    f"(prev H/L: {analysis['crt_prev_high']:.4f}/{analysis['crt_prev_low']:.4f}, "
                    f"curr L/H/Close: {analysis['crt_curr_low']:.4f}/{analysis['crt_curr_high']:.4f}/{analysis['crt_curr_close']:.4f})\n"
                )

            msg += "\n" + format_checklist(analysis)

            if signal == "BUY":
                buy_msgs.append(msg)
            else:
                sell_msgs.append(msg)

        except Exception as e:
            logging.exception(f"status1d error for {sym}: {e}")
            continue

    if buy_msgs:
        update.message.reply_text(
            "🟩 *STRONG BUY signals (4/4 confirmed)*\n\n" + "\n\n".join(buy_msgs),
            parse_mode="Markdown",
            disable_web_page_preview=True
        )

    if sell_msgs:
        update.message.reply_text(
            "🟥 *STRONG SELL signals (4/4 confirmed)*\n\n" + "\n\n".join(sell_msgs),
            parse_mode="Markdown",
            disable_web_page_preview=True
        )

    if not buy_msgs and not sell_msgs:
        update.message.reply_text("⚪ No coins met all 4 Ichimoku checklist conditions (1D).")

    update.message.reply_text("✅ 1D scan complete.")

# ---------------- status1wk ----------------
def status1w(update: Update, context: CallbackContext):
    tf_label = "1w"
    interval = TIMEFRAMES[tf_label]

    update.message.reply_text("⏳ Scanning 1W Ichimoku + CRT (showing Ichimoku 4/4 only; CRT noted if present)...")

    buy_msgs, sell_msgs = [], []
    manila_tz = timezone(timedelta(hours=8))

    for sym in SYMBOLS:
        try:
            df = fetch_ohlcv(sym, interval)
            if df is None or len(df) < 104:
                continue

            analysis = analyze_df(df)
            signal = analysis.get("signal", "Neutral")

            if not ((signal == "BUY" and analysis.get("bull_count", 0) == 4) or
                    (signal == "SELL" and analysis.get("bear_count", 0) == 4)):
                continue

            trigger_time = None
            start_idx = max(103, len(df) - 100)
            for i in range(start_idx, len(df)):
                sub_df = df.iloc[: i + 1]
                if len(sub_df) < 104:
                    continue
                sub_analysis = analyze_df(sub_df)
                if signal == "BUY" and sub_analysis.get("bull_count", 0) == 4:
                    trigger_time = int(sub_df.iloc[-2]["time"])
                    break
                if signal == "SELL" and sub_analysis.get("bear_count", 0) == 4:
                    trigger_time = int(sub_df.iloc[-2]["time"])
                    break

            if trigger_time is None:
                trigger_time = int(df.iloc[-2]["time"])

            ts = datetime.fromtimestamp(trigger_time / 1000, tz=manila_tz).strftime("%Y-%m-%d %I:%M %p (Manila)")

            safe_symbol = sym.replace("_", "\\_").replace("-", "\\-")
            price_str = f"{analysis['price']:.2f} USDT" if analysis.get("price") is not None else "N/A"
            rsi_str = f"{analysis['rsi']:.2f}" if analysis.get("rsi") is not None else "N/A"

            msg = (
                f"{'🟩' if signal == 'BUY' else '🟥'} *{safe_symbol}* — STRONG {signal} (4/4)\n"
                f"🕒 Time: {ts}\n"
                f"💰 Price: {price_str}\n"
                f"📊 RSI: {rsi_str}\n"
                f"🔗 [TradingView]({tradingview_link(sym, tf_label)})\n"
            )

            if analysis.get("sl") is not None and analysis.get("tp") is not None:
                msg += f"🎯 SL: {analysis['sl']:.2f} | TP: {analysis['tp']:.2f}\n"

            if analysis.get("crt_bull") or analysis.get("crt_bear"):
                side = "Bullish" if analysis.get("crt_bull") else "Bearish"
                msg += (
                    f"🕯️ CRT: *{side}* "
                    f"(prev H/L: {analysis['crt_prev_high']:.4f}/{analysis['crt_prev_low']:.4f}, "
                    f"curr L/H/Close: {analysis['crt_curr_low']:.4f}/{analysis['crt_curr_high']:.4f}/{analysis['crt_curr_close']:.4f})\n"
                )

            msg += "\n" + format_checklist(analysis)

            if signal == "BUY":
                buy_msgs.append(msg)
            else:
                sell_msgs.append(msg)

        except Exception as e:
            logging.exception(f"status1w error for {sym}: {e}")
            continue

    if buy_msgs:
        update.message.reply_text(
            "🟩 *STRONG BUY signals (4/4 confirmed, 1W)*\n\n" + "\n\n".join(buy_msgs),
            parse_mode="Markdown",
            disable_web_page_preview=True
        )

    if sell_msgs:
        update.message.reply_text(
            "🟥 *STRONG SELL signals (4/4 confirmed, 1W)*\n\n" + "\n\n".join(sell_msgs),
            parse_mode="Markdown",
            disable_web_page_preview=True
        )

    if not buy_msgs and not sell_msgs:
        update.message.reply_text("⚪ No coins met all 4 Ichimoku checklist conditions (1W).")

    update.message.reply_text("✅ 1W scan complete.")

# ---------------- HEARTBEAT ----------------
def heartbeat(context: CallbackContext):
    context.bot.send_message(chat_id=CHAT_ID, text="💓 Bot is alive")

# ---------------- MAIN ----------------
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("test", test))
    dp.add_handler(CommandHandler("status", status))
    dp.add_handler(CommandHandler("status1d", status1d))
    dp.add_handler(CommandHandler("status1w", status1w))
    jq = updater.job_queue
    jq.run_repeating(check_and_alert, interval=300, first=10)
    jq.run_repeating(heartbeat, interval=14400, first=20)
    jq.run_repeating(refresh_pairs, interval=86400, first=60)  # refresh every 24 hours
    logging.info("Bot started")
    updater.start_polling()
    updater.idle()
    updater.bot.send_message(chat_id=CHAT_ID, text="🚀 Bot restarted and running!")

if __name__ == "__main__":
    main()
