# ultra_precision_realtime_bot_binance.py
# Python 3.8+
# Uses python-binance (already installed in venv). Reads USE_TESTNET from /opt/bot/.env
# WARNING: Keep DRY_RUN=True while testing.

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
from dotenv import load_dotenv

# Try import python-binance client (installed in venv)
try:
    from binance.client import Client
except Exception as e:
    raise SystemExit(f"Missing python-binance in venv. Install it: /opt/bot/venv/bin/pip install python-binance\nError: {e}")

# --------------------- load .env ---------------------
ENV_PATH = "/opt/bot/.env"
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    print(f"[FATAL] .env not found at {ENV_PATH}")
    raise SystemExit(1)

USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

TESTNET_API_KEY = os.getenv("TESTNET_API_KEY", "")
TESTNET_SECRET_KEY = os.getenv("TESTNET_SECRET_KEY", "")
MAINNET_API_KEY = os.getenv("MAINNET_API_KEY", "")
MAINNET_SECRET_KEY = os.getenv("MAINNET_SECRET_KEY", "")

# --------------------- config -------------------------
SYMBOLS = ["BTCUSDT", "PENGUUSDT", "MYXUSDT", "ETCUSDT"]
MODEL_FILE = "model.joblib"
LOGFILE = "/opt/bot/ultra_precision_bot.log"

LEVERAGE = 10
TRADE_PCT = 0.05
PROB_THRESHOLD = 0.60
TP = 0.02
SL = 0.008
COOLDOWN_MIN = 5

BB_SHORT = 14
BB_LONG = 40
BB_STD = 2.0
EMA_TF = "30min"
EMA_LEN = 200

ULTRA_HOURS = [9, 13, 14, 15, 16, 17, 19]
ULTRA_DAYS  = [0,1,2,3,4]

INPUT_BAR_MINUTES = 1
BACKTEST_BAR_MINUTES = 3
AGG_BARS = int(BACKTEST_BAR_MINUTES / INPUT_BAR_MINUTES)

# --------------------- logging ------------------------
def log(msg):
    s = f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}"
    print(s)
    try:
        with open(LOGFILE, "a") as f:
            f.write(s + "\n")
    except Exception:
        pass

# --------------------- Binance client -----------------
if USE_TESTNET:
    API_KEY = TESTNET_API_KEY
    API_SECRET = TESTNET_SECRET_KEY
else:
    API_KEY = MAINNET_API_KEY
    API_SECRET = MAINNET_SECRET_KEY

if not API_KEY or not API_SECRET:
    log("[FATAL] API keys not set. Check /opt/bot/.env")
    raise SystemExit(1)

# instantiate client
log(f"[INFO] USE_TESTNET={USE_TESTNET} | DRY_RUN={DRY_RUN}")
client = Client(API_KEY, API_SECRET)

# Try to point python-binance to futures testnet endpoints if requested
if USE_TESTNET:
    try:
        # python-binance sometimes supports switching base urls via attributes:
        # set futures base url to testnet (common testnet endpoint)
        client.API_URL = "https://testnet.binancefuture.com"             # best-effort
        client.FUTURES_URL = "https://testnet.binancefuture.com"
        log("[INFO] Set client to Binance Futures TESTNET endpoints (best-effort).")
    except Exception as e:
        log(f"[WARN] Could not set testnet urls on client: {e}")

# quick connectivity check (best-effort)
try:
    client.ping()
    log("[DEBUG] Binance connectivity OK")
except Exception as e:
    log(f"[WARN] Binance ping failed (still may work): {e}")

# --------------------- load model ---------------------
if not os.path.exists(MODEL_FILE):
    log(f"[FATAL] Model file not found: {MODEL_FILE}")
    raise SystemExit(1)

try:
    model = joblib.load(MODEL_FILE)
    log("[DEBUG] Model loaded successfully")
except Exception as e:
    log(f"[FATAL] Could not load model: {e}")
    raise SystemExit(1)

# --------------------- indicators & features ----------
def rolling_bbtrend(close_s, s_len, l_len, std):
    s_mean = close_s.rolling(s_len, min_periods=1).mean()
    s_std = close_s.rolling(s_len, min_periods=1).std().fillna(0.0)
    l_mean = close_s.rolling(l_len, min_periods=1).mean()
    l_std = close_s.rolling(l_len, min_periods=1).std().fillna(0.0)
    s_low = s_mean - std*s_std
    s_up  = s_mean + std*s_std
    l_low = l_mean - std*l_std
    l_up  = l_mean + std*l_std
    denom = s_mean.replace(0, 1e-10)
    return ((s_low - l_low).abs() - (s_up - l_up).abs()) / denom * 100.0

def compute_unshifted_kumo(df):
    high = df["high"]; low = df["low"]
    tenkan = (high.rolling(9).max() + low.rolling(9).min())/2
    kijun  = (high.rolling(26).max() + low.rolling(26).min())/2
    senkouA = (tenkan + kijun)/2
    senkouB = (high.rolling(52).max() + low.rolling(52).min())/2
    kumoTop = pd.concat([senkouA, senkouB], axis=1).max(axis=1).ffill().fillna(0.0)
    kumoBot = pd.concat([senkouA, senkouB], axis=1).min(axis=1).ffill().fillna(0.0)
    return kumoTop, kumoBot

def compute_atr(df, length=14):
    high = df["high"]; low = df["low"]; close = df["close"]
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
    feat["dow"] = feat.index.dayofweek
    log("[DEBUG] Features generated successfully.")
    return feat

# --------------------- utils: symbol info, qty ----------------
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
        usdt_bal = 0.0
        for b in bal:
            if b["asset"] == "USDT":
                usdt_bal = float(b["balance"]); break
    except Exception as e:
        log(f"[WARN] Error fetching balance: {e}")
        usdt_bal = 100.0
    try:
        price = float(client.futures_mark_price(symbol=symbol)["markPrice"])
    except Exception as e:
        log(f"[WARN] Error fetching mark price: {e}")
        price = 1.0
    usdt_alloc = usdt_bal * pct
    notional = usdt_alloc * leverage
    qty = notional / price
    info = get_symbol_info(symbol)
    if info:
        step = None
        for f in info["filters"]:
            if f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"]); break
        if step:
            qty = math.floor(qty / step) * step
    return max(qty, 0.0)

# --------------------- robust proba extraction ----------------
def extract_probs_from_model_output(model, proba_array):
    classes = list(model.classes_)
    long_prob = 0.0; short_prob = 0.0
    for c, p in zip(classes, proba_array):
        try:
            if isinstance(c, (list, tuple, np.ndarray)): c = c[0]
            if hasattr(c, "iloc"): c = c.iloc[0]
            if hasattr(c, "item"): c = c.item()
            ic = int(float(c))
        except Exception:
            ic = None
        if ic == 1:
            long_prob = float(p)
        elif ic == -1:
            short_prob = float(p)
        elif ic == 0:
            short_prob = float(p)
    if long_prob == 0.0 and short_prob == 0.0 and len(proba_array) == 2:
        short_prob = float(proba_array[0]); long_prob = float(proba_array[1])
    return long_prob, short_prob

# --------------------- order execution ------------------------
def place_market_order(symbol, side, qty, reduce_only=False):
    if DRY_RUN:
        log(f"[DRY_RUN] Would execute MARKET {side} {symbol} qty={qty}")
        return {"orderId": 0, "status": "DRY"}
    try:
        res = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=float(qty),
            reduceOnly=reduce_only
        )
        log(f"[ORDER EXECUTED] {res}")
        return res
    except Exception as e:
        log(f"[ERROR] Order failed: {e}")
        return None

# --------------------- per-symbol state ------------------------
CANDLES_BUFFER_MAX = 2000
candles_map = {s: deque(maxlen=CANDLES_BUFFER_MAX) for s in SYMBOLS}
last_trade_time = {s: None for s in SYMBOLS}
in_position = {s: False for s in SYMBOLS}
current_position = {s: None for s in SYMBOLS}

# --------------------- websocket callbacks --------------------
def on_open(ws, symbol=None):
    log(f"[WS {symbol}] opened.")

def on_close(ws, code, msg, symbol=None):
    log(f"[WS {symbol}] closed — code={code}, msg={msg}. reconnect in 5s")
    time.sleep(5)
    # restart thread for that symbol
    threading.Thread(target=start_kline_ws, args=(symbol,), daemon=True).start()

def on_error(ws, error, symbol=None):
    log(f"[WS {symbol}] ERROR: {error}")

def on_kline_message_factory(symbol):
    # return a handler bound to this symbol
    def on_kline_message(msg):
        global last_trade_time, in_position, current_position
        try:
            data = json.loads(msg)
        except Exception:
            return
        if not isinstance(data, dict) or "k" not in data:
            return
        k = data["k"]
        is_candle_closed = k.get("x", False)
        candle = {
            "timestamp": pd.to_datetime(k["t"], unit="ms", utc=True),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"])
        }
        if is_candle_closed:
            candles_map[symbol].append(candle)
            log(f"[DEBUG {symbol}] 1m candle closed.")
            if len(candles_map[symbol]) < AGG_BARS:
                return
            df1m = pd.DataFrame(list(candles_map[symbol])[-1000:]).set_index("timestamp")
            try:
                feat = build_features_from_df(df1m)
            except Exception as e:
                log(f"[ERROR {symbol}] Feature build error: {e}")
                return
            t_index = feat.index[-1]
            X = feat.loc[[t_index]].select_dtypes(include=[np.number]).fillna(0)
            try:
                proba = model.predict_proba(X)[0]
                long_prob, short_prob = extract_probs_from_model_output(model, proba)
                log(f"[PREDICT {symbol}] long_prob={long_prob:.3f} short_prob={short_prob:.3f}")
            except Exception as e:
                log(f"[ML ERROR {symbol}] {e}")
                return
            # time filter
            try:
                hour = int(feat["hour"].iloc[-1]); dow = int(feat["dow"].iloc[-1])
            except Exception as e:
                log(f"[WS ERROR {symbol}] time extraction failed: {e}")
                return
            if hour not in ULTRA_HOURS or dow not in ULTRA_DAYS:
                log(f"[FILTER {symbol}] Outside trading hours. skip.")
                return
            now = pd.Timestamp.utcnow()
            if last_trade_time[symbol] is not None and (now - last_trade_time[symbol]) < pd.Timedelta(minutes=COOLDOWN_MIN):
                log(f"[{symbol}] In cooldown.")
            entry_price = candle["close"]
            # try open
            if not in_position[symbol]:
                if long_prob >= PROB_THRESHOLD:
                    qty = qty_from_pct_balance(symbol, TRADE_PCT, LEVERAGE)
                    if qty <= 0:
                        log(f"[{symbol}] Computed qty <= 0; skipping")
                    else:
                        res = place_market_order(symbol, "BUY", qty)
                        if res is not None:
                            in_position[symbol] = True
                            current_position[symbol] = {"side": "LONG", "entry_price": float(entry_price), "qty": qty, "entry_time": t_index}
                            last_trade_time[symbol] = pd.Timestamp.utcnow()
                            log(f"[{symbol}] Entered LONG: {current_position[symbol]}")
                elif short_prob >= PROB_THRESHOLD:
                    qty = qty_from_pct_balance(symbol, TRADE_PCT, LEVERAGE)
                    if qty <= 0:
                        log(f"[{symbol}] Computed qty <= 0; skipping")
                    else:
                        res = place_market_order(symbol, "SELL", qty)
                        if res is not None:
                            in_position[symbol] = True
                            current_position[symbol] = {"side": "SHORT", "entry_price": float(entry_price), "qty": qty, "entry_time": t_index}
                            last_trade_time[symbol] = pd.Timestamp.utcnow()
                            log(f"[{symbol}] Entered SHORT: {current_position[symbol]}")
            # if in position check TP/SL
            if in_position[symbol] and current_position[symbol] is not None:
                hi = candle["high"]; lo = candle["low"]
                entry = current_position[symbol]["entry_price"]; qty = current_position[symbol]["qty"]; side = current_position[symbol]["side"]
                if side == "LONG":
                    tp_price = entry * (1 + TP); sl_price = entry * (1 - SL)
                    if hi >= tp_price:
                        res = place_market_order(symbol, "SELL", qty)
                        log(f"[{symbol}] LONG TP hit. closed. res={res}")
                        in_position[symbol] = False; current_position[symbol] = None; last_trade_time[symbol] = pd.Timestamp.utcnow()
                    elif lo <= sl_price:
                        res = place_market_order(symbol, "SELL", qty)
                        log(f"[{symbol}] LONG SL hit. closed. res={res}")
                        in_position[symbol] = False; current_position[symbol] = None; last_trade_time[symbol] = pd.Timestamp.utcnow()
                else:
                    tp_price = entry * (1 - TP); sl_price = entry * (1 + SL)
                    if lo <= tp_price:
                        res = place_market_order(symbol, "BUY", qty)
                        log(f"[{symbol}] SHORT TP hit. closed. res={res}")
                        in_position[symbol] = False; current_position[symbol] = None; last_trade_time[symbol] = pd.Timestamp.utcnow()
                    elif hi >= sl_price:
                        res = place_market_order(symbol, "BUY", qty)
                        log(f"[{symbol}] SHORT SL hit. closed. res={res}")
                        in_position[symbol] = False; current_position[symbol] = None; last_trade_time[symbol] = pd.Timestamp.utcnow()
    return on_kline_message

# --------------------- start ws per symbol -------------------
def start_kline_ws(symbol):
    stream = f"wss://fstream.binance.com/ws/{symbol.lower()}@kline_1m"
    # If testnet, use testnet websocket base (best-effort)
    if USE_TESTNET:
        stream = f"wss://stream.binancefuture.com/ws/{symbol.lower()}@kline_1m"
    log(f"[WS] Connecting to {stream}")
    handler = on_kline_message_factory(symbol)
    ws = WebSocketApp(stream,
                      on_open=lambda ws: on_open(ws, symbol),
                      on_message=lambda ws, msg: handler(msg),
                      on_error=lambda ws, err: on_error(ws, err, symbol),
                      on_close=lambda ws, code, msg: on_close(ws, code, msg, symbol))
    try:
        ws.run_forever()
    except Exception as e:
        log(f"[WS {symbol}] run_forever error: {e}")
        time.sleep(5)
        # try restart
        threading.Thread(target=start_kline_ws, args=(symbol,), daemon=True).start()

# --------------------- main -------------------------------
def main():
    log("Starting multi-pair bot...")
    for sym in SYMBOLS:
        threading.Thread(target=start_kline_ws, args=(sym,), daemon=True).start()
        log(f"WebSocket iniciado para {sym}")
    # keep main alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()


