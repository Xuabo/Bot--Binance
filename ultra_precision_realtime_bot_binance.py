# ultra_precision_realtime_bot_binance.py
# Python 3.8+
# Requires: python-binance, websocket-client, numpy, pandas, joblib
#
# WARNING: Test first with DRY_RUN = True


import os
import json
import time
import math
from collections import deque
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from websocket import WebSocketApp
from binance.client import Client

# ---------------------------------------------------------
# LOAD ENVIRONMENT
# ---------------------------------------------------------
from dotenv import load_dotenv

ENV_PATH = "/home/ubuntu/binance-bot/.env"
print(f"[DEBUG] Loading environment from: {ENV_PATH}")
load_dotenv(ENV_PATH)

# -----------------------
# USER CONFIG
# -----------------------
API_KEY = os.getenv("BINANCE_KEY")
API_SECRET = os.getenv("BINANCE_SECRET")

print(f"[DEBUG] API_KEY loaded? {'YES' if API_KEY else 'NO'}")
print(f"[DEBUG] API_SECRET loaded? {'YES' if API_SECRET else 'NO'}")

if API_KEY:
    print(f"[SECURITY] BINANCE_KEY: {API_KEY[:4]}********")
if API_SECRET:
    print(f"[SECURITY] BINANCE_SECRET: {API_SECRET[:4]}********")

SYMBOL = "BTCUSDT"
DRY_RUN = True
LEVERAGE = 10
PROB_THRESHOLD = 0.60
MODEL_FILE = "model.joblib"
LOGFILE = "ultra_precision_bot.log"

BB_SHORT = 14
BB_LONG = 40
BB_STD = 2.0
EMA_TF = "30min"
EMA_LEN = 200

ULTRA_HOURS = [9, 13, 14, 15, 16, 17, 19]
ULTRA_DAYS = [0, 1, 2, 3, 4]

INPUT_BAR_MINUTES = 1
BACKTEST_BAR_MINUTES = 3
AGG_BARS = int(BACKTEST_BAR_MINUTES / INPUT_BAR_MINUTES)

FEATURE_COLUMNS = ['close', 'ret1', 'bb_mtf', 'ema_diff',
                   'kumo_top', 'kumo_bot', 'atr', 'rsi', 'hour', 'dow']


# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
def log(msg):
    s = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(s)
    with open(LOGFILE, "a") as f:
        f.write(s + "\n")


# ---------------------------------------------------------
# BINANCE CLIENT
# ---------------------------------------------------------
print("[DEBUG] Initializing Binance client...")
client = Client(API_KEY, API_SECRET)

# Quick connectivity test
try:
    ping = client.ping()
    print("[DEBUG] Binance connectivity: OK")
except Exception as e:
    print("[ERROR] Could NOT reach Binance API:", e)
    print("Check VPS internet or firewall!")
    time.sleep(2)


# ---------------------------------------------------------
# MODEL LOAD
# ---------------------------------------------------------
print(f"[DEBUG] Checking model file: {MODEL_FILE}")

if not os.path.exists(MODEL_FILE):
    print(f"[FATAL] Model file NOT FOUND: {MODEL_FILE}")
    raise SystemExit(1)

try:
    model = joblib.load(MODEL_FILE)
    print("[DEBUG] Model loaded successfully!")
except Exception as e:
    print("[FATAL] Error loading ML model:", e)
    raise SystemExit(1)


# ---------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------
def rolling_bbtrend(close_s, short_len, long_len, std=2.0):
    s_mean = close_s.rolling(short_len, min_periods=1).mean()
    s_std = close_s.rolling(short_len, min_periods=1).std().fillna(0.0)
    l_mean = close_s.rolling(long_len, min_periods=1).mean()
    l_std = close_s.rolling(long_len, min_periods=1).std().fillna(0.0)

    s_low = s_mean - std * s_std
    s_up = s_mean + std * s_std
    l_low = l_mean - std * l_std
    l_up = l_mean + std * l_std
    denom = s_mean.replace(0, 1e-10)

    return ((s_low - l_low).abs() - (s_up - l_up).abs()) / denom * 100.0


def compute_unshifted_kumo(df):
    high = df["high"]
    low = df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min())/2
    kijun = (high.rolling(26).max() + low.rolling(26).min())/2
    senkouA = (tenkan + kijun)/2
    senkouB = (high.rolling(52).max() + low.rolling(52).min())/2
    kumoTop = pd.concat([senkouA, senkouB], axis=1).max(axis=1).ffill().fillna(0.0)
    kumoBot = pd.concat([senkouA, senkouB], axis=1).min(axis=1).ffill().fillna(0.0)
    return kumoTop, kumoBot


def compute_atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()


def compute_rsi(close, length=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length, min_periods=1).mean()
    avg_loss = loss.rolling(length, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


# ---------------------------------------------------------
# FEATURE GENERATOR
# ---------------------------------------------------------
def build_features_from_df(df):
    print("[DEBUG] Building features...")

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
    feat["dow"] = feat.index.dayofweek

    print("[DEBUG] Features generated successfully.")
    return feat


# ---------------------------------------------------------
# ORDER EXECUTION
# ---------------------------------------------------------
def place_market_order(symbol, side, qty):
    if DRY_RUN:
        log(f"[DRY_RUN] Would execute {side} {qty}")
        return {"orderId": 0, "status": "TEST"}

    try:
        res = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=float(qty)
        )
        log(f"ORDER EXECUTED: {res}")
        return res
    except Exception as e:
        log(f"[ERROR] Order failed: {e}")
        return None


# ---------------------------------------------------------
# WEBSOCKET CALLBACKS
# ---------------------------------------------------------
def on_open(ws):
    log("[WS] WebSocket opened.")


def on_close(ws, code, msg):
    log(f"[WS] WebSocket closed — code={code}, msg={msg}")
    log("[WS] Attempting reconnection in 5 seconds...")
    time.sleep(5)
    start_kline_ws(SYMBOL)


def on_error(ws, error):
    log(f"[WS ERROR] {error}")


# ---------------------------------------------------------
# MAIN KLINE HANDLER
# ---------------------------------------------------------
candles_1m = deque(maxlen=2000)


def on_kline_message(msg):
    try:
        data = json.loads(msg)
    except:
        return

    if "k" not in data:
        return

    k = data["k"]
    closed = k["x"]

    candle = {
        "timestamp": pd.to_datetime(k["t"], unit="ms", utc=True),
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"])
    }

    if closed:
        candles_1m.append(candle)
        log("[DEBUG] 1m candle closed.")

        if len(candles_1m) < AGG_BARS:
            return

        df = pd.DataFrame(candles_1m).set_index("timestamp")

        feat = build_features_from_df(df)
        last_index = feat.index[-1]

        X = feat.loc[[last_index]].select_dtypes(include=[np.number]).fillna(0)

        try:
            proba = model.predict_proba(X)[0]
            log(f"[PREDICT] Proba = {proba}")
        except Exception as e:
            log(f"[ML ERROR] {e}")
            return


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
    print("\n==================== BOT STARTING ====================")
    print(f" SYMBOL     : {SYMBOL}")
    print(f" DRY_RUN    : {DRY_RUN}")
    print(f" LEVERAGE   : {LEVERAGE}")
    print(f" MODEL      : {MODEL_FILE}")
    print("=====================================================\n")

    log("Bot started.")

    try:
        start_kline_ws(SYMBOL)
    except KeyboardInterrupt:
        log("Bot manually stopped.")
    except Exception as e:
        log(f"[FATAL] Crash: {e}")


if __name__ == "__main__":
    main()
