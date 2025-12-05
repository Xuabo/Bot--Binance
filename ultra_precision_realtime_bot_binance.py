# ultra_precision_realtime_bot_binance.py
# Python 3.8+
# Requires: python-binance, websocket-client, numpy, pandas, joblib, python-dotenv

import os
import json
import time
import math
import threading
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from websocket import WebSocketApp
from binance.client import Client
from dotenv import load_dotenv

# ---------------------------------------------------------
# LOAD ENVIRONMENT
# ---------------------------------------------------------
ENV_PATH = "/opt/bot/.env"
print(f"[DEBUG] Loading environment from: {ENV_PATH}")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    print(f"[ERROR] .env NOT FOUND at {ENV_PATH}")

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")

print(f"[DEBUG] API_KEY loaded?: {'YES' if API_KEY else 'NO'}")
print(f"[DEBUG] API_SECRET loaded?: {'YES' if API_SECRET else 'NO'}")

if API_KEY:
    print(f"[SECURITY] BINANCE_API_KEY: {API_KEY[:4]}********")
if API_SECRET:
    print(f"[SECURITY] BINANCE_SECRET_KEY: {API_SECRET[:4]}********")

# ---------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------
SYMBOLS = ["BTCUSDT", "PENGUUSDT", "MYXUSDT", "ETCUSDT"]
DRY_RUN = True
LEVERAGE = 10
TRADE_PCT = 0.05
PROB_THRESHOLD = 0.60
MODEL_FILE = "model.joblib"
LOGFILE = "/opt/bot/ultra_precision_bot.log"

TP = 0.02
SL = 0.008
COOLDOWN_MIN = 5

BB_SHORT = 14
BB_LONG = 40
BB_STD = 2.0
EMA_TF = "30min"
EMA_LEN = 200

ULTRA_HOURS = [9, 13, 14, 15, 16, 17, 19]
ULTRA_DAYS  = [0, 1, 2, 3, 4]

INPUT_BAR_MINUTES = 1
BACKTEST_BAR_MINUTES = 3
AGG_BARS = int(BACKTEST_BAR_MINUTES / INPUT_BAR_MINUTES)

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
def log(msg):
    s = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(s)
    try:
        with open(LOGFILE, "a") as f:
            f.write(s + "\n")
    except:
        pass

# ---------------------------------------------------------
# BINANCE CLIENT
# ---------------------------------------------------------
print("[DEBUG] Initializing Binance client...")
client = Client(API_KEY, API_SECRET)

try:
    client.ping()
    print("[DEBUG] Binance connectivity: OK")
except Exception as e:
    print("[ERROR] Cannot reach Binance:", e)
    time.sleep(2)

# ---------------------------------------------------------
# MODEL LOAD
# ---------------------------------------------------------
print(f"[DEBUG] Checking model file: {MODEL_FILE}")
if not os.path.exists(MODEL_FILE):
    print("[FATAL] Model missing")
    raise SystemExit(1)

try:
    model = joblib.load(MODEL_FILE)
    print("[DEBUG] Model loaded successfully!")
except Exception as e:
    print("[FATAL] Error loading model:", e)
    raise SystemExit(1)

# ---------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------
def rolling_bbtrend(close_s, s_len, l_len, std):
    s_mean = close_s.rolling(s_len).mean()
    s_std = close_s.rolling(s_len).std().fillna(0)
    l_mean = close_s.rolling(l_len).mean()
    l_std = close_s.rolling(l_len).std().fillna(0)

    s_low = s_mean - std*s_std
    s_up  = s_mean + std*s_std
    l_low = l_mean - std*l_std
    l_up  = l_mean + std*l_std

    denom = s_mean.replace(0, 1e-10)
    return ((s_low - l_low).abs() - (s_up - l_up).abs()) / denom * 100

def compute_unshifted_kumo(df):
    high, low = df["high"], df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min())/2
    kijun  = (high.rolling(26).max() + low.rolling(26).min())/2
    senkouA = (tenkan + kijun)/2
    senkouB = (high.rolling(52).max() + low.rolling(52).min())/2
    top = pd.concat([senkouA, senkouB], axis=1).max(axis=1).ffill()
    bot = pd.concat([senkouA, senkouB], axis=1).min(axis=1).ffill()
    return top, bot

def compute_atr(df, length=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def compute_rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

# ---------------------------------------------------------
# FEATURE GENERATOR
# ---------------------------------------------------------
def build_features_from_df(df):
    log("[DEBUG] Building features...")
    close = df["close"]
    close_30 = close.resample("30min").last().ffill()

    bb_mtf = rolling_bbtrend(close_30, BB_SHORT, BB_LONG, BB_STD).reindex(df.index, method="ffill")
    ema_htf = close.resample(EMA_TF).last().ffill().ewm(span=EMA_LEN).mean()
    ema_diff = close - ema_htf.reindex(df.index).ffill()

    kumoTop, kumoBot = compute_unshifted_kumo(df)
    atr = compute_atr(df)
    rsi = compute_rsi(close)

    feat = pd.DataFrame(index=df.index)
    feat["close"] = close
    feat["ret1"] = close.pct_change().fillna(0)
    feat["bb_mtf"] = bb_mtf.fillna(0)
    feat["ema_diff"] = ema_diff.fillna(0)
    feat["kumo_top"] = kumoTop
    feat["kumo_bot"] = kumoBot
    feat["atr"] = atr
    feat["rsi"] = rsi
    feat["hour"] = feat.index.hour
    feat["dow"]  = feat.index.dayofweek

    log("[DEBUG] Features generated successfully.")
    return feat

# ---------------------------------------------------------
# ORDER EXECUTION
# ---------------------------------------------------------
def place_market_order(symbol, side, qty, reduce_only=False):
    if DRY_RUN:
        log(f"[DRY_RUN] MARKET {side} qty={qty}")
        return {"orderId": 0, "status": "DRY"}

    try:
        res = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=float(qty),
            reduceOnly=reduce_only
        )
        log(f"ORDER EXECUTED: {res}")
        return res
    except Exception as e:
        log(f"[ERROR] Order failed: {e}")
        return None

# ---------------------------------------------------------
# SYMBOL INFO + BALANCE
# ---------------------------------------------------------
def get_symbol_info(symbol):
    try:
        info = client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                return s
    except Exception as e:
        log(f"[ERROR] exchange info: {e}")
    return None

def qty_from_pct_balance(symbol, pct, leverage):
    try:
        bal = client.futures_account_balance()
        usdt = next(float(b["balance"]) for b in bal if b["asset"] == "USDT")
    except:
        usdt = 100

    try:
        price = float(client.futures_mark_price(symbol=symbol)["markPrice"])
    except:
        price = 1.0

    notional = usdt * pct * leverage
    qty = notional / price

    info = get_symbol_info(symbol)
    if info:
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
                qty = math.floor(qty / step) * step
                break

    return max(qty, 0.0)

# ---------------------------------------------------------
# FIXED: ROBUST PROBA EXTRACTION
# ---------------------------------------------------------
def extract_probs_from_model_output(model, proba_array):
    """
    Prevents error: cannot convert the series to <class 'int'>
    by normalizing class labels into scalar values.
    """
    classes = list(model.classes_)
    long_prob = 0.0
    short_prob = 0.0

    for c, p in zip(classes, proba_array):

        try:
            if isinstance(c, (list, tuple, np.ndarray)):
                c = c[0]
            if hasattr(c, "iloc"):
                c = c.iloc[0]
            if hasattr(c, "item"):
                c = c.item()
            ic = int(float(c))
        except:
            ic = None

        if ic == 1:
            long_prob = float(p)
        elif ic == -1:
            short_prob = float(p)
        elif ic == 0:
            short_prob = float(p)

    if long_prob == 0.0 and short_prob == 0.0 and len(proba_array) == 2:
        short_prob = float(proba_array[0])
        long_prob = float(proba_array[1])

    return long_prob, short_prob

# ---------------------------------------------------------
# WEBSOCKET CALLBACKS
# ---------------------------------------------------------
def on_open(ws):
    log("[WS] WebSocket opened.")

def on_close(ws, code, msg):
    log(f"[WS] WebSocket closed: {code}, {msg}")
    time.sleep(5)
    start_kline_ws(SYMBOL)

def on_error(ws, error):
    log(f"[WS ERROR] {error}")

# ---------------------------------------------------------
# KLINE HANDLER
# ---------------------------------------------------------
candles_1m = deque(maxlen=2000)
last_trade_time = None
in_position = False
current_position = None

def on_kline_message(msg):
    global last_trade_time, in_position, current_position

    try:
        data = json.loads(msg)
    except:
        return

    if "k" not in data:
        return

    k = data["k"]
    is_closed = k.get("x", False)

    candle = {
        "timestamp": pd.to_datetime(k["t"], unit="ms", utc=True),
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"])
    }

    if is_closed:
        candles_1m.append(candle)
        log("[DEBUG] 1m candle closed.")

        if len(candles_1m) < AGG_BARS:
            return

        df = pd.DataFrame(list(candles_1m)[-1000:]).set_index("timestamp")

        try:
            feat = build_features_from_df(df)
        except Exception as e:
            log(f"Feature build error: {e}")
            return

        t = feat.index[-1]
        X = feat.loc[[t]].select_dtypes(include=[np.number]).fillna(0)

        try:
            proba = model.predict_proba(X)[0]
            long_prob, short_prob = extract_probs_from_model_output(model, proba)
            log(f"[PREDICT] long_prob={long_prob:.3f} short_prob={short_prob:.3f}")
        except Exception as e:
            log(f"[ML ERROR] {e}")
            return

        # --------------------------
        # FIXED: safe hour/dow extraction
        # --------------------------
        try:
            hour = int(feat["hour"].iloc[-1])
            dow  = int(feat["dow"].iloc[-1])
        except:
            log("[ERROR] hour/dow extraction failed")
            return

        if hour not in ULTRA_HOURS or dow not in ULTRA_DAYS:
            log("[FILTER] Outside trading hours.")
            return

        now = pd.Timestamp.utcnow()
        close_price = candle["close"]

        if last_trade_time and (now - last_trade_time) < pd.Timedelta(minutes=COOLDOWN_MIN):
            log("Cooldown active.")

        # --------------------------
        # ENTRY LOGIC
        # --------------------------
        if not in_position:

            if long_prob >= PROB_THRESHOLD:
                qty = qty_from_pct_balance("BTCUSDT", TRADE_PCT, LEVERAGE)
                res = place_market_order("BTCUSDT", "BUY", qty)
                if res:
                    in_position = True
                    current_position = {
                        "side": "LONG",
                        "entry_price": close_price,
                        "qty": qty
                    }
                    last_trade_time = now
                    log(f"Entered LONG: {current_position}")

            elif short_prob >= PROB_THRESHOLD:
                qty = qty_from_pct_balance("BTCUSDT", TRADE_PCT, LEVERAGE)
                res = place_market_order("BTCUSDT", "SELL", qty)
                if res:
                    in_position = True
                    current_position = {
                        "side": "SHORT",
                        "entry_price": close_price,
                        "qty": qty
                    }
                    last_trade_time = now
                    log(f"Entered SHORT: {current_position}")

        # --------------------------
        # TP / SL MANAGEMENT
        # --------------------------
        if in_position:

            side = current_position["side"]
            entry = current_position["entry_price"]
            qty   = current_position["qty"]

            hi = candle["high"]
            lo = candle["low"]

            if side == "LONG":
                tp = entry * (1 + TP)
                sl = entry * (1 - SL)

                if hi >= tp:
                    place_market_order("BTCUSDT", "SELL", qty)
                    log("LONG TP hit.")
                    in_position = False
                    current_position = None
                    last_trade_time = now

                elif lo <= sl:
                    place_market_order("BTCUSDT", "SELL", qty)
                    log("LONG SL hit.")
                    in_position = False
                    current_position = None
                    last_trade_time = now

            else:  # SHORT
                tp = entry * (1 - TP)
                sl = entry * (1 + SL)

                if lo <= tp:
                    place_market_order("BTCUSDT", "BUY", qty)
                    log("SHORT TP hit.")
                    in_position = False
                    current_position = None
                    last_trade_time = now

                elif hi >= sl:
                    place_market_order("BTCUSDT", "BUY", qty)
                    log("SHORT SL hit.")
                    in_position = False
                    current_position = None
                    last_trade_time = now

# ---------------------------------------------------------
# START WEBSOCKET
# ---------------------------------------------------------
def start_kline_ws(symbol):
    stream = f"wss://fstream.binance.com/ws/{symbol.lower()}@kline_1m"
    log(f"[WS] Connecting to {stream}")

    ws = WebSocketApp(
        stream,
        on_open=on_open,
        on_message=lambda ws, msg: on_kline_message(msg),
        on_error=on_error,
        on_close=on_close
    )

    ws.run_forever()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    log("Starting MULTI-PAIR bot...")

    for sym in SYMBOLS:
        threading.Thread(target=start_kline_ws, args=(sym,), daemon=True).start()
        log(f"WebSocket iniciado para {sym}")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
