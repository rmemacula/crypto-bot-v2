import os
import json
import logging
import requests
import pandas as pd
import numpy as np
import time
import pagibig_scanner
from datetime import datetime, timezone, timedelta
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

# ---------------- FETCH TOP VOLUME ----------------
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

# ---------------- FETCH ALL USDT PAIRS ----------------
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

# ---------------- LOAD SYMBOLS ----------------
def load_symbols():
    """Combine manual top 50 + top 20 by futures volume (unique)."""
    manual = set(MANUAL_TOP_50)
    top_vol = set(fetch_top_volume_pairs(20))
    combined = sorted(list(manual | top_vol))
    logging.info(f"✅ Loaded {len(combined)} symbols (Top 50 + Top 20 by volume).")
    return combined

# 🔥 Global sets
TOP_VOLUME_SYMBOLS = set(fetch_top_volume_pairs(20))
SYMBOLS = load_symbols()

# ---------------- VOLUME TAG ----------------
def volume_tag(symbol):
    """Return 🔥 tag if symbol is high volume."""
    return " 🔥 High Volume" if symbol in TOP_VOLUME_SYMBOLS else ""

# ---------------- REFRESH ----------------
def refresh_pairs(context: CallbackContext):
    global SYMBOLS, TOP_VOLUME_SYMBOLS
    logging.info("🔄 Refreshing symbol list (Top 50 + Top 20)...")
    new_symbols = load_symbols()
    TOP_VOLUME_SYMBOLS = set(fetch_top_volume_pairs(20))
    added = set(new_symbols) - set(SYMBOLS)
    removed = set(SYMBOLS) - set(new_symbols)
    SYMBOLS = new_symbols

    msg = f"♻️ Updated symbols: {len(SYMBOLS)} total."
    if added:
        msg += f"\n➕ Added: {', '.join(sorted(list(added))[:10])}..."
    if removed:
        msg += f"\n➖ Removed: {', '.join(sorted(list(removed))[:10])}..."
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

def key_for(symbol, tf): return f"{symbol}_{tf}"

# ---------------- TRADINGVIEW ----------------
def tradingview_link(symbol, tf_label):
    """Return TradingView link for Binance perpetual pairs (all use USDT.P)."""
    tf_map = {"1h": "60", "4h": "240", "1d": "1D", "1w": "1W"}

    # Construct symbol directly using the consistent naming scheme
    tv_symbol = f"{symbol}.P"

    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{tv_symbol}&interval={tf_map.get(tf_label, '60')}"

# ---------------- FETCH DATA ----------------
def fetch_ohlcv(symbol, interval, limit=150):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        df = pd.DataFrame(data, columns=["time", "o", "h", "l", "c", "v", "ct", "qv", "n", "tb", "tq", "ig"])
        df["o"] = df["o"].astype(float)
        df["c"] = df["c"].astype(float)
        df["h"] = df["h"].astype(float)
        df["l"] = df["l"].astype(float)
        return df
    except Exception:
        return None

# ---------------- CRT DETECTION ----------------
def detect_crt_on_last_pair(df):
    if df is None or len(df) < 3:
        return {"bullish_crt": False, "bearish_crt": False}
    prev, curr = df.iloc[-3], df.iloc[-2]
    prev_open, prev_close, prev_high, prev_low = map(float, [prev["o"], prev["c"], prev["h"], prev["l"]])
    curr_close, curr_high, curr_low = map(float, [curr["c"], curr["h"], curr["l"]])
    bullish_crt = prev_close < prev_open and (curr_low < prev_low) and (prev_low < curr_close < prev_high)
    bearish_crt = prev_close > prev_open and (curr_high > prev_high) and (prev_low < curr_close < prev_high)
    return {
        "bullish_crt": bullish_crt, "bearish_crt": bearish_crt,
        "prev_high": prev_high, "prev_low": prev_low,
        "curr_high": curr_high, "curr_low": curr_low, "curr_close": curr_close
    }

# ---------------- ANALYSIS ----------------
def analyze_df(df):
    if df is None or len(df) < 104:
        return {"signal": "Neutral", "price": None, "rsi": None}
    close, high, low = df["c"], df["h"], df["l"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss)))
    rsi_val = float(rsi.iloc[-2]) if not np.isnan(rsi.iloc[-2]) else None
    last_idx = -2
    price = float(close.iloc[last_idx])
    tenkan_v, kijun_v = float(tenkan.iloc[last_idx]), float(kijun.iloc[last_idx])
    cloud_idx = last_idx - 26
    cloud_a_current = float(senkou_a.iloc[cloud_idx])
    cloud_b_current = float(senkou_b.iloc[cloud_idx])
    chikou_idx, chikou_cloud_idx = last_idx - 26, last_idx - 52
    ca, cb = float(senkou_a.iloc[chikou_cloud_idx]), float(senkou_b.iloc[chikou_cloud_idx])
    chikou_above = price > max(ca, cb)
    chikou_below = price < min(ca, cb)
    cloud_a_future, cloud_b_future = float(senkou_a.iloc[last_idx]), float(senkou_b.iloc[last_idx])
    checklist_bull = [
        ("Price above cloud", price > max(cloud_a_current, cloud_b_current)),
        ("Tenkan > Kijun", tenkan_v > kijun_v),
        ("Chikou above cloud", chikou_above),
        ("Future cloud bullish", cloud_a_future > cloud_b_future),
    ]
    checklist_bear = [
        ("Price below cloud", price < min(cloud_a_current, cloud_b_current)),
        ("Tenkan < Kijun", tenkan_v < kijun_v),
        ("Chikou below cloud", chikou_below),
        ("Future cloud bearish", cloud_a_future < cloud_b_future),
    ]
    bull_count, bear_count = sum(v for _, v in checklist_bull), sum(v for _, v in checklist_bear)
    signal, sl, tp = "Neutral", None, None
    if bull_count >= 3:
        signal, sl, tp = "BUY", min(cloud_a_current, cloud_b_current)*0.995, price + 2*(price - min(cloud_a_current, cloud_b_current)*0.995)
    elif bear_count >= 3:
        signal, sl, tp = "SELL", max(cloud_a_current, cloud_b_current)*1.005, price - 2*(max(cloud_a_current, cloud_b_current)*1.005 - price)
    crt = detect_crt_on_last_pair(df)
    return {
        "price": price, "rsi": rsi_val, "signal": signal,
        "bull_count": bull_count, "bear_count": bear_count,
        "checklist_bull": checklist_bull, "checklist_bear": checklist_bear,
        "sl": sl, "tp": tp, "crt_bull": crt["bullish_crt"], "crt_bear": crt["bearish_crt"],
        "crt_prev_high": crt["prev_high"], "crt_prev_low": crt["prev_low"],
        "crt_curr_high": crt["curr_high"], "crt_curr_low": crt["curr_low"], "crt_curr_close": crt["curr_close"]
    }

# ---------------- CHECKLIST FORMATTER ----------------
def format_checklist(analysis):
    lines, signal = [], analysis["signal"]
    for (bull_label, bull_val), (bear_label, bear_val) in zip(analysis["checklist_bull"], analysis["checklist_bear"]):
        if signal == "BUY": lines.append(f"{'✅' if bull_val else '❌'} {bull_label}")
        elif signal == "SELL": lines.append(f"{'✅' if bear_val else '❌'} {bear_label}")
    if analysis.get("crt_bull"): lines.append("🕯️ CRT: Bullish (swept prior low, closed back inside)")
    elif analysis.get("crt_bear"): lines.append("🕯️ CRT: Bearish (swept prior high, closed back inside)")
    return "\n".join(lines)

# ---------------- COMMANDS ----------------
def test(update, context): update.message.reply_text("✅ Bot is working!")

def status(update, context):
    if not context.args: return update.message.reply_text("Usage: /status BTC")
    sym = context.args[0].upper() + "USDT"
    if sym not in SYMBOLS: return update.message.reply_text("Unknown coin")
    messages = []
    for tf_label, interval in TIMEFRAMES.items():
        df = fetch_ohlcv(sym, interval)
        if df is None or len(df) < 104: continue
        analysis = analyze_df(df)
        msg = (f"📊 {sym} ({tf_label}){volume_tag(sym)}\n"
               f"Signal: {analysis['signal']}\n💰 Price: {analysis['price']:.2f} USDT\n"
               f"📊 RSI: {analysis['rsi']:.2f}\n"
               f"📈 [View on TradingView]({tradingview_link(sym, tf_label)})\n")
        if analysis["sl"] and analysis["tp"]: msg += f"🎯 SL: {analysis['sl']:.2f} | TP: {analysis['tp']:.2f}\n"
        if analysis.get("crt_bull") or analysis.get("crt_bear"):
            side = "Bullish" if analysis.get("crt_bull") else "Bearish"
            msg += (f"🕯️ CRT: *{side}*\n")
        msg += "\n" + format_checklist(analysis)
        messages.append(msg)
    update.message.reply_text("\n\n".join(messages), parse_mode="Markdown")

# ---------------- ALERT JOB ----------------
def check_and_alert(context):
    global last_signals
    bot = context.bot
    
    signals_to_save = {}  # ✅ Collect changes, don't save yet
    messages_sent = 0

    for symbol in SYMBOLS:
        for tf_label, interval in TIMEFRAMES.items():
            try:
                df = fetch_ohlcv(symbol, interval)
                if df is None or len(df) < 104:
                    continue

                analysis = analyze_df(df)
                sig = analysis["signal"]
                k = key_for(symbol, tf_label)

                crt_tag = "CRT_BULL" if analysis.get("crt_bull") else ("CRT_BEAR" if analysis.get("crt_bear") else "CRT_NONE")
                sent_label = f"{sig}|{analysis['bull_count']}|{analysis['bear_count']}|{crt_tag}"
                prev = last_signals.get(k)

                # Combined alert
                if (sig == "BUY" and analysis["bull_count"] == 4) or (sig == "SELL" and analysis["bear_count"] == 4):
                    if prev != sent_label:
                        tv = tradingview_link(symbol, tf_label)
                        
                        msg = f"🚨 *{symbol}* ({tf_label}) — *{sig} (4/4 confirmed)*{volume_tag(symbol)}\n\n"
                        
                        # ✅ FIXED: Only add CRT if it MATCHES the signal direction
                        if sig == "BUY" and analysis.get("crt_bull"):
                            msg += f"🕯️ *CRT Bullish ALIGNED!*\n\n"
                        elif sig == "SELL" and analysis.get("crt_bear"):
                            msg += f"🕯️ *CRT Bearish ALIGNED!*\n\n"
                        
                        msg += f"💰 *Price:* {analysis['price']:.2f}\n"
                        msg += f"📊 *RSI:* {analysis['rsi']:.2f}\n"
                        msg += f"🔗 [View on TradingView]({tv})\n\n"
                        msg += format_checklist(analysis)
                        
                        try:
                            bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                            messages_sent += 1
                            signals_to_save[k] = sent_label  # ✅ Queue for saving
                            
                            # ✅ Rate limit Telegram messages
                            if messages_sent % 10 == 0:
                                logging.info(f"Sent {messages_sent} messages, pausing 2s...")
                                time.sleep(2)
                                
                        except Exception as e:
                            logging.error(f"Failed to send message for {symbol} {tf_label}: {e}")

                # Exit alerts
                elif prev and ("BUY|4|0" in prev or "SELL|0|4" in prev):
                    if not ((sig == "BUY" and analysis["bull_count"] == 4) or (sig == "SELL" and analysis["bear_count"] == 4)):
                        msg = (
                            f"⚪ *{symbol}* ({tf_label}) — exited strong {prev.split('|')[0]} zone.\n"
                            f"Now: {sig} ({analysis['bull_count']}/4 bull, {analysis['bear_count']}/4 bear)"
                        )
                        try:
                            bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                            signals_to_save[k] = sent_label
                        except Exception as e:
                            logging.error(f"Failed to send exit message for {symbol} {tf_label}: {e}")

            except Exception as e:
                logging.error(f"Error processing {symbol} {tf_label}: {e}")
                continue

    # ✅ Save ONCE at the end
    if signals_to_save:
        last_signals.update(signals_to_save)
        save_last_signals(last_signals)
        logging.info(f"✅ Scan complete. Sent {messages_sent} alerts, saved {len(signals_to_save)} signals.")
# ---------------- STATUS 1D ----------------
def status1d(update, context):
    tf_label, interval = "1d", TIMEFRAMES["1d"]
    update.message.reply_text("⏳ Scanning 1D Ichimoku + CRT ...")
    buy_msgs, sell_msgs, manila_tz = [], [], timezone(timedelta(hours=8))
    
    for sym in SYMBOLS:
        try:
            df = fetch_ohlcv(sym, interval)
            if df is None or len(df) < 104: continue
            a = analyze_df(df)
            if not ((a["signal"]=="BUY" and a["bull_count"]==4) or (a["signal"]=="SELL" and a["bear_count"]==4)): continue
            ts = datetime.fromtimestamp(df.iloc[-2]["time"]/1000, tz=manila_tz).strftime("%Y-%m-%d %I:%M %p")
            msg = (f"{'🟩' if a['signal']=='BUY' else '🟥'} *{sym}* — STRONG {a['signal']} (4/4){volume_tag(sym)}\n"
                   f"🕒 Time: {ts}\n💰 Price: {a['price']:.2f}\n📊 RSI: {a['rsi']:.2f}\n"
                   f"🔗 [TradingView]({tradingview_link(sym, tf_label)})\n\n{format_checklist(a)}")
            (buy_msgs if a["signal"]=="BUY" else sell_msgs).append(msg)
        except Exception: continue
    
    # ✅ Send in batches of 5 coins per message
    def send_in_batches(msgs, title):
        if not msgs:
            return
        batch_size = 5
        for i in range(0, len(msgs), batch_size):
            batch = msgs[i:i+batch_size]
            header = f"{title} ({i+1}-{min(i+batch_size, len(msgs))} of {len(msgs)})\n\n"
            update.message.reply_text(
                header + "\n\n".join(batch), 
                parse_mode="Markdown",
                disable_web_page_preview=True
            )
    
    send_in_batches(buy_msgs, "🟩 *STRONG BUYs*")
    send_in_batches(sell_msgs, "🟥 *STRONG SELLs*")
    
    if not buy_msgs and not sell_msgs: 
        update.message.reply_text("⚪ No 1D 4/4 signals found.")
    update.message.reply_text("✅ 1D scan complete.")
# ---------------- STATUS 1W ----------------
def status1w(update, context):
    tf_label, interval = "1w", TIMEFRAMES["1w"]
    update.message.reply_text("⏳ Scanning 1W Ichimoku + CRT ...")
    buy_msgs, sell_msgs, manila_tz = [], [], timezone(timedelta(hours=8))
    
    for sym in SYMBOLS:
        try:
            df = fetch_ohlcv(sym, interval)
            if df is None or len(df) < 104: continue
            a = analyze_df(df)
            if not ((a["signal"]=="BUY" and a["bull_count"]==4) or (a["signal"]=="SELL" and a["bear_count"]==4)): continue
            ts = datetime.fromtimestamp(df.iloc[-2]["time"]/1000, tz=manila_tz).strftime("%Y-%m-%d %I:%M %p")
            msg = (f"{'🟩' if a['signal']=='BUY' else '🟥'} *{sym}* — STRONG {a['signal']} (4/4){volume_tag(sym)}\n"
                   f"🕒 Time: {ts}\n💰 Price: {a['price']:.2f}\n📊 RSI: {a['rsi']:.2f}\n"
                   f"🔗 [TradingView]({tradingview_link(sym, tf_label)})\n\n{format_checklist(a)}")
            (buy_msgs if a["signal"]=="BUY" else sell_msgs).append(msg)
        except Exception: continue
    
    # ✅ Send in batches
    def send_in_batches(msgs, title):
        if not msgs:
            return
        batch_size = 5
        for i in range(0, len(msgs), batch_size):
            batch = msgs[i:i+batch_size]
            header = f"{title} ({i+1}-{min(i+batch_size, len(msgs))} of {len(msgs)})\n\n"
            update.message.reply_text(
                header + "\n\n".join(batch), 
                parse_mode="Markdown",
                disable_web_page_preview=True
            )
    
    send_in_batches(buy_msgs, "🟩 *STRONG BUYs (1W)*")
    send_in_batches(sell_msgs, "🟥 *STRONG SELLs (1W)*")
    
    if not buy_msgs and not sell_msgs: 
        update.message.reply_text("⚪ No 1W 4/4 signals found.")
    update.message.reply_text("✅ 1W scan complete.")

# ---------------- STATUS ALIGNED (>=2 TFs) ----------------
def statusaligned(update, context):
    """
    List coins with at least 2 timeframes aligned as STRONG BUY or STRONG SELL.
    Scans: 1h, 4h, 1d
    """
    update.message.reply_text("⏳ Scanning aligned STRONG signals (1h, 4h, 1D)...")

    # Only these timeframes
    tf_scan = [("1h", TIMEFRAMES["1h"]), ("4h", TIMEFRAMES["4h"]), ("1d", TIMEFRAMES["1d"])]

    aligned_buys = []   # list of (symbol, [tfs])
    aligned_sells = []  # list of (symbol, [tfs])

    for sym in SYMBOLS:
        buy_tfs = []
        sell_tfs = []

        for tf_label, interval in tf_scan:
            try:
                df = fetch_ohlcv(sym, interval)
                if df is None or len(df) < 104:
                    continue

                a = analyze_df(df)

                # Strong BUY / SELL definition = 4/4 confirmed
                if a["signal"] == "BUY" and a.get("bull_count", 0) == 4:
                    buy_tfs.append(tf_label)
                elif a["signal"] == "SELL" and a.get("bear_count", 0) == 4:
                    sell_tfs.append(tf_label)

            except Exception:
                continue

        # Keep only if >=2 aligned in same direction
        if len(buy_tfs) >= 2:
            aligned_buys.append((sym, buy_tfs))
        if len(sell_tfs) >= 2:
            aligned_sells.append((sym, sell_tfs))

    # Sort: more aligned TFs first, then symbol
    aligned_buys.sort(key=lambda x: (-len(x[1]), x[0]))
    aligned_sells.sort(key=lambda x: (-len(x[1]), x[0]))

    # Build message lines
    buy_lines = [
        f"🟩 *{sym}* — STRONG BUY aligned: *{', '.join(tfs)}*{volume_tag(sym)}"
        for sym, tfs in aligned_buys
    ]
    sell_lines = [
        f"🟥 *{sym}* — STRONG SELL aligned: *{', '.join(tfs)}*{volume_tag(sym)}"
        for sym, tfs in aligned_sells
    ]

    # Send in batches to avoid Telegram limits
    def send_batches(lines, title):
        if not lines:
            return
        batch_size = 20  # adjust if you want shorter messages
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            header = f"{title} ({i+1}-{min(i+batch_size, len(lines))} of {len(lines)})\n\n"
            update.message.reply_text(
                header + "\n".join(batch),
                parse_mode="Markdown",
                disable_web_page_preview=True
            )

    if not buy_lines and not sell_lines:
        update.message.reply_text("⚪ No coins found with >=2 aligned STRONG signals (1h/4h/1D).")
        return

    send_batches(buy_lines, "🟩 *ALIGNED STRONG BUYs (>=2 TFs)*")
    send_batches(sell_lines, "🟥 *ALIGNED STRONG SELLs (>=2 TFs)*")
    update.message.reply_text("✅ /statusaligned scan complete.")

# ----------------STATUS VOL-----------------------
def statusvolume(update, context):
    """Show currently monitored top-volume coins."""
    if not TOP_VOLUME_SYMBOLS:
        return update.message.reply_text("⚠️ No top volume data available yet.")

    msg = "🔥 *Top 20 High-Volume Coins Currently Monitored*\n\n"
    sorted_syms = sorted(list(TOP_VOLUME_SYMBOLS))
    for i, sym in enumerate(sorted_syms, 1):
        msg += f"{i:02d}. {sym}\n"

    msg += "\n♻️ This list refreshes automatically every 6 hours."
    update.message.reply_text(msg, parse_mode="Markdown")

def pagibiglatest(update, context):
    update.message.reply_text("🔄 Fetching LIVE latest per location…")
    try:
        msg = pagibig_scanner.get_latest_properties_by_loc_live()
        update.message.reply_text(msg, disable_web_page_preview=True)
    except Exception as e:
        update.message.reply_text(f"❌ /pagibiglatest failed:\n{e}")

# ---------------- HEARTBEAT ----------------
def heartbeat(context): context.bot.send_message(chat_id=CHAT_ID, text="💓 Bot is alive")

# ---------------- MAIN ----------------
def main():
    updater = Updater(TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("test", test))
    dp.add_handler(CommandHandler("status", status))
    dp.add_handler(CommandHandler("status1d", status1d))
    dp.add_handler(CommandHandler("status1w", status1w))
    dp.add_handler(CommandHandler("statusvolume", statusvolume))
    dp.add_handler(CommandHandler("statusaligned", statusaligned))
    dp.add_handler(CommandHandler("pagibiglatest", pagibiglatest))
    jq = updater.job_queue
    jq.run_repeating(check_and_alert, interval=300, first=10)
    jq.run_repeating(heartbeat, interval=14400, first=20)
    jq.run_repeating(refresh_pairs, interval=14400, first=60)
    updater.start_polling()
    updater.bot.send_message(chat_id=CHAT_ID, text="🚀 Bot restarted and running!")
    updater.idle()

if __name__ == "__main__":
    main()
