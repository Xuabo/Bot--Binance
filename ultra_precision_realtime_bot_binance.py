#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ultra_precision_realtime_bot_binance.py
Full single-file bot (Testnet default). Copy-paste to /opt/bot/ultra_precision_realtime_bot_binance.py
Requirements (inside venv): python-binance, websocket-client, numpy, pandas, joblib, python-dotenv, requests
"""

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
import requests

from websocket import WebSocketApp
from dotenv import load_dotenv

# Try import python-binance
try:
    from binance.client import Client
except Exception as e:
    raise SystemExit(
        f"[FATAL] python-binance not found in venv.\n"
        f"Install using: /opt/bot/venv/bin/pip install python-binance\nError: {e}"
    )

# -------------------- LOAD .env --------------------
ENV_PATH = "/opt/bot/.env"
if not os.path.exists(ENV_PATH):
    print(f"[FATAL] .env not found at {ENV_PATH}")
    raise SystemExit(1)

load_dotenv(ENV_PATH)

# Modes & keys
USE_TESTNET = os.getenv("USE_TESTNET", "True").lower() == "true"   # default True (safe)
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"          # don't send real orders by default

TESTNET_API_KEY = os.getenv("TESTNET_API_KEY", "").strip()
TESTNET_SECRET_KEY = os.getenv("TESTNET_SECRET_KEY", "").strip()
MAINNET_API_KEY = os.getenv("MAINNET_API_KEY", "").strip()
MAINNET_SECRET_KEY = os.getenv("MAINNET_SECRET_KEY", "").strip()

# Config from env with safe defaults
SYMBOLS = [s.strip().upper() for s in os.getenv("SYMBOLS", "BTCUSDT").split(",") if s.strip()]
MODEL_FILE = os.getenv("MODEL_FILE", "model.joblib")
LOGFILE = os.getenv("LOGFILE", "/opt/bot/ultra_precision_bot.log")

LEVERAGE = float(os.getenv("LEVERAGE", 10))
TRADE_PCT = float(os.getenv("TRADE_PCT", 0.05))
PROB_THRESHOLD = float(os.getenv("PROB_THRESHOLD", 0.60))
TP = float(os.getenv("TP", 0.02))
SL = float(os.getenv("SL", 0.008))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", 5))

BB_SHORT = int(os.getenv("BB_SHORT", 14))
BB_LONG = int(os.getenv("BB_LONG", 40))
BB_STD = float(os.getenv("BB_STD", 2.0))
EMA_TF = os.getenv("EMA_TF", "30min")
EMA_LEN = int(os.getenv("EMA_LEN", 200))

ULTRA_HOURS = [int(x) for x in os.getenv("ULTRA_HOURS", "9,13,14,15,16,17,19").split(",") if x.strip()!=""]
ULTRA_DAYS = [int(x) for x in os.getenv("ULTRA_DAYS", "0,1,2,3,4").split(",") if x.strip()!=""]

INPUT_BAR_MINUTES = int(os.getenv("INPUT_BAR_MINUTES", 1))
BACKTEST_BAR_MINUTES = int(os.getenv("BACKTEST_BAR_MINUTES", 3))
AGG_BARS = max(1, int(BACKTEST_BAR_MINUTES / INPUT_BAR_MINUTES))
CANDLES_BUFFER_MAX = int(os.getenv("CANDLES_BUFFER_MAX", 2000))

# -------------------- LOGGING --------------------
def log(msg):
    s = f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}"
    print(s)
    try:
        with open(LOGFILE, "a") as f:
            f.write(s + "\n")
    except Exception:
        pass

# -------------------- CHOOSE API KEYS --------------------
if USE_TESTNET:
    API_KEY = TESTNET_API_KEY
    API_SECRET = TESTNET_SECRET_KEY
else:
    API_KEY = MAINNET_API_KEY
    API_SECRET = MAINNET_SECRET_KEY

if not API_KEY or not API_SECRET:
    log("[FATAL] API keys missing for the selected environment. Check /opt/bot/.env")
    raise SystemExit(1)

log(f"[INFO] USE_TESTNET={USE_TESTNET} DRY_RUN={DRY_RUN} SYMBOLS={SYMBOLS}")

# -------------------- INIT BINANCE CLIENT --------------------
client = Client(API_KEY, API_SECRET)

# Best-effort: adjust python-binance urls for futures testnet
if USE_TESTNET:
    try:
        # some python-binance versions allow changing base URLs
        client.API_URL = "https://testnet.binancefuture.com"
        client.FUTURES_URL = "https://testnet.binancefuture.com"
        log("[INFO] Attempted to set client to Binance Futures TESTNET endpoints.")
    except Exception as e:
        log(f"[WARN] Could not change client base URLs: {e}")

# Quick connectivity check
try:
    client.ping()
    log("[DEBUG] Binance ping OK")
except Exception as e:
    log(f"[WARN] Binance ping failed (may still work): {e}")

# -------------------- LOAD MODEL --------------------
if not os.path.exists(MODEL_FILE):
    log(f"[FATAL] Model file not found: {MODEL_FILE}")
    raise SystemExit(1)

try:
    model = joblib.load(MODEL_FILE)
    log("[DEBUG] ML model loaded successfully.")
except Exception as e:
    log(f"[FATAL] Could not load ML model: {e}")
    raise SystemExit(1)

# If the model was created with newer scikit-learn, warn (already printed by sklearn)
# -------------------- INDICATORS & FEATURES --------------------
def rolling_bbtrend(close_s, short_len, long_len, std=2.0):
    try:
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
    except Exception as e:
        log(f"[INDICATOR] rolling_bbtrend error: {e}")
        return pd.Series(0.0, index=close_s.index)

def compute_unshifted_kumo(df):
    try:
        high = df["high"]; low = df["low"]
        tenkan = (high.rolling(9).max() + low.rolling(9).min())/2
        kijun = (high.rolling(26).max() + low.rolling(26).min())/2
        senkouA = (tenkan + kijun)/2
        senkouB = (high.rolling(52).max() + low.rolling(52).min())/2
        kumoTop = pd.concat([senkouA, senkouB], axis=1).max(axis=1).ffill().fillna(0.0)
        kumoBot = pd.concat([senkouA, senkouB], axis=1).min(axis=1).ffill().fillna(0.0)
        return kumoTop, kumoBot
    except Exception as e:
        log(f"[INDICATOR] compute_unshifted_kumo error: {e}")
        return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)

def compute_atr(df, length=14):
    try:
        high = df["high"]; low = df["low"]; close = df["close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(length, min_periods=1).mean()
    except Exception as e:
        log(f"[INDICATOR] compute_atr error: {e}")
        return pd.Series(0.0, index=df.index)

def compute_rsi(close, length=14):
    try:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(length, min_periods=1).mean()
        avg_loss = loss.rolling(length, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))
    except Exception as e:
        log(f"[INDICATOR] compute_rsi error: {e}")
        return pd.Series(50.0, index=close.index)

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

# -------------------- UTILITIES: exchange info fallback --------------------
def _futures_exchange_info_fallback_testnet():
    """
    Best-effort fallback to fetch exchangeInfo from testnet REST endpoint if python-binance fails.
    """
    try:
        url = "https://testnet.binancefuture.com/fapi/v1/exchangeInfo"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log(f"[WARN] testnet exchangeInfo fallback failed: {e}")
    return None

def get_symbol_info(symbol):
    """
    Try client.futures_exchange_info(); if response structure unexpected, try fallback for testnet.
    """
    try:
        info = client.futures_exchange_info()
        # ensure it looks like expected
        if isinstance(info, dict) and "symbols" in info:
            for s in info.get("symbols", []):
                if s.get("symbol") == symbol:
                    return s
        else:
            log("[WARN] futures_exchange_info returned unexpected structure; trying fallback.")
    except Exception as e:
        log(f"[WARN] futures_exchange_info error: {e}")

    # fallback for testnet
    if USE_TESTNET:
        info = _futures_exchange_info_fallback_testnet()
        if info and isinstance(info, dict) and "symbols" in info:
            for s in info.get("symbols", []):
                if s.get("symbol") == symbol:
                    return s
        else:
            log("[WARN] fallback testnet exchangeInfo didn't return expected 'symbols' key")
    return None

def qty_from_pct_balance(symbol, pct, leverage):
    # compute usable qty for futures with step size
    try:
        bal = client.futures_account_balance()
        usdt_bal = 0.0
        for b in bal:
            if b.get("asset") == "USDT":
                usdt_bal = float(b.get("balance", 0.0))
                break
    except Exception as e:
        log(f"[WARN] Error fetching balance: {e}")
        usdt_bal = 100.0  # fallback for safety in tests

    try:
        price = float(client.futures_mark_price(symbol=symbol)["markPrice"])
    except Exception as e:
        log(f"[WARN] Error fetching mark price: {e}")
        price = 1.0

    usdt_alloc = usdt_bal * pct
    notional = usdt_alloc * leverage
    raw_qty = notional / price

    info = get_symbol_info(symbol)
    if info:
        step = None
        min_qty = None
        for f in info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                step = float(f.get("stepSize", 0.0))
                min_qty = float(f.get("minQty", 0.0))
                break
        if step and step > 0:
            qty = math.floor(raw_qty / step) * step
        else:
            qty = math.floor(raw_qty * 1e6) / 1e6
        if min_qty and qty < min_qty:
            log(f"[QTY] computed {qty} < minQty {min_qty} for {symbol} -> returning 0")
            return 0.0
        return max(qty, 0.0)
    else:
        return max(math.floor(raw_qty * 1e6) / 1e6, 0.0)

# -------------------- PROB EXTRACTION --------------------
def extract_probs_from_model_output(model, proba_array):
    long_prob = 0.0
    short_prob = 0.0
    try:
        classes = list(model.classes_)
        for c, p in zip(classes, proba_array):
            try:
                if isinstance(c, (list, tuple, np.ndarray)):
                    c2 = c[0]
                else:
                    c2 = c
                if hasattr(c2, "item"):
                    c2 = c2.item()
                ic = int(float(c2))
            except Exception:
                ic = None
            if ic == 1:
                long_prob = float(p)
            elif ic == -1:
                short_prob = float(p)
            elif ic == 0:
                short_prob = float(p)
    except Exception as e:
        log(f"[PROB] extract_probs error: {e}")
    if long_prob == 0.0 and short_prob == 0.0 and len(proba_array) == 2:
        try:
            short_prob = float(proba_array[0])
            long_prob = float(proba_array[1])
        except Exception:
            pass
    return long_prob, short_prob

# -------------------- ORDERS --------------------
def place_market_order(symbol, side, qty, reduce_only=False):
    if qty is None or qty <= 0:
        log(f"[ORDER] {symbol} qty invalid ({qty}) - skipping")
        return None
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
        log(f"[ORDER EXECUTED] {symbol} {side} qty={qty} -> {res}")
        return res
    except Exception as e:
        log(f"[ERROR] Order failed for {symbol}: {e}")
        return None

# -------------------- PER-SYMBOL STATE --------------------
candles_map = {s: deque(maxlen=CANDLES_BUFFER_MAX) for s in SYMBOLS}
last_trade_time = {s: None for s in SYMBOLS}
in_position = {s: False for s in SYMBOLS}
current_position = {s: None for s in SYMBOLS}
state_lock = threading.Lock()
log(f"[STATE] per-symbol state initialized for: {', '.join(SYMBOLS)}")

# -------------------- WEBSOCKET HANDLERS FACTORY --------------------
def on_kline_message_factory(symbol):
    """
    Returns a handler function bound to 'symbol' that processes incoming kline messages.
    """
    def on_kline_message(msg):
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
            # Time filter
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
            # Try open
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
            # If in position check TP/SL using intrabar high/low
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

# -------------------- WEBSOCKET START + CALLBACKS --------------------
def on_open(ws, symbol):
    log(f"[WS {symbol}] opened.")

def on_close(ws, code, msg, symbol):
    log(f"[WS {symbol}] closed — code={code}, msg={msg}. reconnect in 5s")
    try:
        ws.close()
    except Exception:
        pass
    time.sleep(5)
    threading.Thread(target=start_kline_ws, args=(symbol,), daemon=True).start()

def on_error(ws, error, symbol):
    log(f"[WS {symbol}] error: {error}")

def start_kline_ws(symbol):
    symbol = symbol.strip().upper()
    if USE_TESTNET:
        stream = f"wss://stream.binancefuture.com/ws/{symbol.lower()}@kline_1m"
    else:
        stream = f"wss://fstream.binance.com/ws/{symbol.lower()}@kline_1m"

    log(f"[WS {symbol}] Connecting to {stream}")
    handler = on_kline_message_factory(symbol)
    ws = WebSocketApp(
        stream,
        on_open=lambda ws: on_open(ws, symbol),
        on_message=lambda ws, msg: handler(msg),
        on_error=lambda ws, err: on_error(ws, err, symbol),
        on_close=lambda ws, code, msg: on_close(ws, code, msg, symbol)
    )
    # keep trying
    while True:
        try:
            ws.run_forever()
        except Exception as e:
            log(f"[WS {symbol}] run_forever exception: {e}")
        log(f"[WS {symbol}] Sleeping 3s before reconnect...")
        time.sleep(3)

# -------------------- HEALTH REPORT --------------------
def health_report():
    try:
        thread_count = len(threading.enumerate())
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "symbols": SYMBOLS,
            "threads": thread_count,
            "in_position": {s: in_position.get(s) for s in SYMBOLS}
        }
        return report
    except Exception as e:
        return {"error": str(e)}

# -------------------- MAIN --------------------
def main():
    log("=== Starting multi-pair ultra_precision bot (final) ===")
    log(f"CONFIG -> USE_TESTNET={USE_TESTNET} DRY_RUN={DRY_RUN} SYMBOLS={SYMBOLS}")
    started = []
    for sym in SYMBOLS:
        s = sym.strip().upper()
        if not s:
            continue
        t = threading.Thread(target=start_kline_ws, args=(s,), daemon=True)
        t.start()
        started.append(s)
        log(f"Thread started for {s}")
    log(f"All threads started: {', '.join(started)}")
    try:
        while True:
            time.sleep(60)
            report = health_report()
            log(f"[HEALTH] {report}")
    except KeyboardInterrupt:
        log("KeyboardInterrupt received — shutting down.")
    except Exception as e:
        log(f"Main loop exception: {e}")

if __name__ == "__main__":
    main()
