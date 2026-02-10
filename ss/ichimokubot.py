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
    tf_map = {"1h": "60", "4h": "240", "1d": "1D"}
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
        df["c"] = df["c"].astype(float)
        df["h"] = df["h"].astype(float)
        df["l"] = df["l"].astype(float)
        return df
    except Exception as e:
        logging.warning(f"Fetch exception {symbol} {interval}: {e}")
        return None

# ---------------- ANALYSIS ----------------
def analyze_df(df):
    if df is None or len(df) < 104:  # Need 52 + 26 + 26 for proper analysis
        return {"signal": "Neutral", "price": None, "rsi": None}

    close = df["c"]
    high = df["h"]
    low = df["l"]

    # ---- Ichimoku Components ----
    # Tenkan-sen (Conversion Line): 9-period
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    
    # Kijun-sen (Base Line): 26-period
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    
    # Senkou Span A (Leading Span A): Average of Tenkan and Kijun
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): 52-period
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
    price = close.iloc[last_idx]
    tenkan_v = float(tenkan.iloc[last_idx])
    kijun_v = float(kijun.iloc[last_idx])
    
    # ---- Current Cloud Position (where price is NOW) ----
    # The cloud at the current candle was projected 26 periods ago
    # So we need to look back 26 periods in the Senkou calculations
    cloud_idx = last_idx - 26
    
    # Make sure we have enough historical data
    if abs(cloud_idx) <= len(df) - 52:
        cloud_a_current = float(senkou_a.iloc[cloud_idx])
        cloud_b_current = float(senkou_b.iloc[cloud_idx])
    else:
        # Fallback if not enough data
        cloud_a_current = float(senkou_a.iloc[last_idx])
        cloud_b_current = float(senkou_b.iloc[last_idx])

    # ---- Chikou Span (Lagging Span) Analysis ----
    # Chikou Span = Current close plotted 26 periods back
    # We compare current price to the cloud at the chikou position (26 periods ago)
    chikou_above = chikou_below = False
    
    # The chikou position is 26 periods back from current
    chikou_idx = last_idx - 26
    
    # The cloud at the chikou position was projected 26 periods before that
    # So we look back another 26 periods in the Senkou calculations
    chikou_cloud_idx = chikou_idx - 26
    
    # Make sure we have enough data
    if abs(chikou_cloud_idx) <= len(df) - 52:
        # Cloud values at the chikou position (what the cloud was 26 periods ago)
        cloud_a_at_chikou = float(senkou_a.iloc[chikou_cloud_idx])
        cloud_b_at_chikou = float(senkou_b.iloc[chikou_cloud_idx])
        
        # Current price (chikou span value) compared to the cloud at that historical position
        if not np.isnan(cloud_a_at_chikou) and not np.isnan(cloud_b_at_chikou):
            chikou_above = price > max(cloud_a_at_chikou, cloud_b_at_chikou)
            chikou_below = price < min(cloud_a_at_chikou, cloud_b_at_chikou)

    # ---- Future Cloud Analysis (26 periods ahead projection) ----
    # The Senkou values calculated NOW represent where the cloud WILL BE 26 periods ahead
    future_cloud_bullish = future_cloud_bearish = False
    
    cloud_a_future = float(senkou_a.iloc[last_idx])
    cloud_b_future = float(senkou_b.iloc[last_idx])
    
    if not np.isnan(cloud_a_future) and not np.isnan(cloud_b_future):
        # Future cloud is bullish when Senkou A > Senkou B (green/bullish cloud ahead)
        # Future cloud is bearish when Senkou A < Senkou B (red/bearish cloud ahead)
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

    # ---- Signal Generation and Risk Management ----
    signal = "Neutral"
    sl = tp = None
    
    if bullish_count >= 3:
        signal = "BUY"
        # Stop loss below the current cloud
        sl = min(cloud_a_current, cloud_b_current) * 0.995
        # Take profit at 2:1 reward-risk ratio
        tp = price + 2 * (price - sl)
    elif bearish_count >= 3:
        signal = "SELL"
        # Stop loss above the current cloud
        sl = max(cloud_a_current, cloud_b_current) * 1.005
        # Take profit at 2:1 reward-risk ratio
        tp = price - 2 * (sl - price)

    # Validate SL and TP
    if sl is None or tp is None or np.isnan(sl) or np.isnan(tp):
        sl = tp = None

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
        "cloud_a": cloud_a_current,
        "cloud_b": cloud_b_current,
        "cloud_a_future": cloud_a_future,
        "cloud_b_future": cloud_b_future,
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
            messages.append(f"❌ Not enough data for {tf_label} (need 104+ candles, have {len(df)})")
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
            if df is None:
                continue

            analysis = analyze_df(df)
            sig = analysis["signal"]
            k = key_for(symbol, tf_label)
            sent_label = f"{sig}|{analysis['bull_count']}|{analysis['bear_count']}"
            prev = last_signals.get(k)

            # ✅ Trigger new strong 4/4 BUY/SELL signals
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

            # ⚪ Detect EXIT from strong 4/4 condition (was BUY/SELL before, now not 4/4)
            elif prev and ("BUY|4|0" in prev or "SELL|0|4" in prev):
                # still same direction but not 4/4 anymore
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

    update.message.reply_text("⏳ Scanning 1D Ichimoku + RSI signals (4/4 confirmed only)...")

    buy_msgs, sell_msgs = [], []
    manila_tz = timezone(timedelta(hours=8))  # Manila timezone

    for sym in SYMBOLS:
        try:
            df = fetch_ohlcv(sym, interval)
            if df is None or len(df) < 104:
                continue

            analysis = analyze_df(df)
            signal = analysis.get("signal", "Neutral")

            # Skip coins that aren’t full 4/4 signals
            if not ((signal == "BUY" and analysis.get("bull_count", 0) == 4) or
                    (signal == "SELL" and analysis.get("bear_count", 0) == 4)):
                continue

            # 🔍 Backtrack to find the first candle when 4/4 became true
            trigger_time = None
            # start index ensures sub_df length >= 104; check the last ~100 candles
            start_idx = max(103, len(df) - 100)
            for i in range(start_idx, len(df) - 1):  # stop at last closed candle index -1
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

            # fallback to last closed candle if not found
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

            msg += "\n" + format_checklist(analysis)

            if signal == "BUY":
                buy_msgs.append(msg)
            else:
                sell_msgs.append(msg)

        except Exception as e:
            logging.exception(f"status1d error for {sym}: {e}")
            # continue scanning other symbols even if one fails
            continue

    # Send grouped results
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

    update.message.reply_text("⏳ Scanning 1W Ichimoku + RSI signals (4/4 confirmed only)...")

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
            for i in range(start_idx, len(df) - 1):
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