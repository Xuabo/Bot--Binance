#!/usr/bin/env python3
# ultra_precision_realtime_bot_render.py
# Versão do teu script adaptada para Render (mantém 100% da lógica original)
# Referência do original: :contentReference[oaicite:1]{index=1}

import os
import json
import time
import math
import traceback
from collections import deque
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib
import requests

from websocket import WebSocketApp
from binance.client import Client

# -----------------------
# CONFIG via ENV (Render)
# -----------------------
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
DRY_RUN = os.getenv("DRY_RUN", "true").lower() not in ["0", "false", "no"]
LEVERAGE = int(os.getenv("LEVERAGE", "5"))
POSITION_SIDE_HEDGE = os.getenv("POSITION_SIDE_HEDGE", "false").lower() in ["1","true","yes"]
PROB_THRESHOLD = float(os.getenv("PROB_THRESHOLD", "0.60"))
MODEL_FILE = os.getenv("MODEL_FILE", "/tmp/model.joblib")
MODEL_URL = os.getenv("MODEL_URL", "")  # se fornecido, será descarregado no startup

# Trading params
TP = float(os.getenv("TP", "0.02"))
SL = float(os.getenv("SL", "0.008"))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "5"))

# Indicator params
BB_SHORT = int(os.getenv("BB_SHORT", "14"))
BB_LONG  = int(os.getenv("BB_LONG", "40"))
BB_STD   = float(os.getenv("BB_STD", "2.0"))
EMA_TF   = os.getenv("EMA_TF", "30min")
EMA_LEN  = int(os.getenv("EMA_LEN", "200"))

# Ultra filters
ULTRA_HOURS = [int(x) for x in os.getenv("ULTRA_HOURS", "9,13,14,15,16,17,19").split(",") if x!=""]
ULTRA_DAYS  = [int(x) for x in os.getenv("ULTRA_DAYS", "0,1,2,3,4").split(",") if x!=""]

# Aggregation
INPUT_BAR_MINUTES = int(os.getenv("INPUT_BAR_MINUTES", "1"))
BACKTEST_BAR_MINUTES = int(os.getenv("BACKTEST_BAR_MINUTES", "3"))
AGG_BARS = max(1, int(BACKTEST_BAR_MINUTES / INPUT_BAR_MINUTES))

# Feature expectation
FEATURE_COLUMNS = ['close', 'ret1', 'bb_mtf', 'ema_diff', 'kumo_top', 'kumo_bot', 'atr', 'rsi', 'hour', 'dow']

# Logs & status (use /tmp in Render)
LOGFILE = os.getenv("LOGFILE", "/tmp/ultra_precision_bot.log")
STATUSFILE = os.getenv("STATUSFILE", "/tmp/bot_status.json")

# -----------------------
# Setup client
# -----------------------
client = Client(API_KEY, API_SECRET)

# -----------------------
# Utilities
# -----------------------
def log(msg):
    s = f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}"
    print(s, flush=True)
    try:
        with open(LOGFILE, "a") as f:
            f.write(s + "\n")
    except Exception:
        pass

def save_status(status: dict):
    try:
        status['last_update'] = datetime.utcnow().isoformat() + "Z"
        with open(STATUSFILE, "w") as f:
            json.dump(status, f)
    except Exception as e:
        log(f"Failed saving status: {e}")

def read_status():
    try:
        if os.path.exists(STATUSFILE):
            with open(STATUSFILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"running": True, "signals": [], "equity_history": []}

def round_step_size(qty, step):
    return math.floor(qty / step) * step

def get_symbol_info(symbol):
    info = client.futures_exchange_info()
    for s in info['symbols']:
        if s['symbol'] == symbol:
            return s
    return None

def qty_from_pct_balance(symbol, pct, leverage):
    try:
        bal = client.futures_account_balance()
        usdt_bal = 0.0
        for b in bal:
            if b['asset'] == 'USDT':
                usdt_bal = float(b['balance'])
                break
    except Exception as e:
        log(f"Error fetching balance: {e}")
        usdt_bal = 100.0
    price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
    usdt_alloc = usdt_bal * pct
    notional = usdt_alloc * leverage
    qty = notional / price
    info = get_symbol_info(symbol)
    if info:
        step = None
        for f in info['filters']:
            if f['filterType'] == 'LOT_SIZE':
                step = float(f['stepSize'])
                break
        if step:
            qty = round_step_size(qty, step)
    return max(qty, 0.0)

# -----------------------
# Indicators (identical lógica)
# -----------------------
def rolling_bbtrend(close_s, short_len, long_len, std=2.0):
    s_mean = close_s.rolling(short_len, min_periods=1).mean()
    s_std  = close_s.rolling(short_len, min_periods=1).std().fillna(0.0)
    l_mean = close_s.rolling(long_len, min_periods=1).mean()
    l_std  = close_s.rolling(long_len, min_periods=1).std().fillna(0.0)
    s_low = s_mean - std * s_std
    s_up  = s_mean + std * s_std
    l_low = l_mean - std * l_std
    l_up  = l_mean + std * l_std
    denom = s_mean.replace(0, 1e-10)
    return ((s_low - l_low).abs() - (s_up - l_up).abs()) / denom * 100.0

def compute_unshifted_kumo(df):
    high = df['high']
    low  = df['low']
    tenkan = (high.rolling(9).max() + low.rolling(9).min())/2
    kijun  = (high.rolling(26).max() + low.rolling(26).min())/2
    senkouA = (tenkan + kijun)/2
    senkouB = (high.rolling(52).max() + low.rolling(52).min())/2
    kumoTop = pd.concat([senkouA, senkouB], axis=1).max(axis=1).ffill().fillna(0.0)
    kumoBot = pd.concat([senkouA, senkouB], axis=1).min(axis=1).ffill().fillna(0.0)
    return kumoTop, kumoBot

def compute_atr(df, length=14):
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(length, min_periods=1).mean()
    return atr

def compute_rsi(close, length=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.rolling(length, min_periods=1).mean()
    ma_down = down.rolling(length, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -----------------------
# Buffers & state
# -----------------------
CANDLES_BUFFER_MAX = 2000
candles_1m = deque(maxlen=CANDLES_BUFFER_MAX)

last_trade_time = None
in_position = False
current_position = None

# -----------------------
# Model loading (download if MODEL_URL provided)
# -----------------------
def download_model_if_needed():
    try:
        if os.path.exists(MODEL_FILE):
            log(f"Model found at {MODEL_FILE}")
            return True
        if MODEL_URL:
            log(f"Downloading model from MODEL_URL...")
            r = requests.get(MODEL_URL, timeout=60)
            r.raise_for_status()
            with open(MODEL_FILE, "wb") as f:
                f.write(r.content)
            log("Model downloaded to " + MODEL_FILE)
            return True
        log("No MODEL_FILE found and no MODEL_URL provided.")
        return False
    except Exception as e:
        log("Error downloading model: " + str(e))
        return False

def load_model():
    try:
        if not download_model_if_needed():
            return None
        m = joblib.load(MODEL_FILE)
        log("Model loaded successfully.")
        return m
    except Exception as e:
        log("Failed to load model: " + str(e))
        log(traceback.format_exc())
        return None

model = load_model()
if model is None:
    log("Model not loaded — running in passive mode (no predictions).")

# -----------------------
# Feature builder (igual à tua versão)
# -----------------------
def build_features_from_df(df, bb_s=BB_SHORT, bb_l=BB_LONG, bb_std=BB_STD, ema_tf=EMA_TF, ema_len=EMA_LEN):
    close = df['close']
    close_30 = close.resample("30min").last().ffill()
    bb_mtf = rolling_bbtrend(close_30, bb_s, bb_l, bb_std).reindex(df.index, method='ffill').fillna(0.0)
    ema_htf = close.resample(ema_tf).last().ffill().ewm(span=ema_len, adjust=False).mean()
    ema_htf_ff = ema_htf.reindex(df.index).ffill().fillna(close)
    kumoTop, kumoBot = compute_unshifted_kumo(df)
    atr = compute_atr(df, 14).fillna(0.0)
    rsi = compute_rsi(close, 14).fillna(50.0)
    feat = pd.DataFrame(index=df.index)
    feat['close'] = close
    feat['ret1'] = close.pct_change().fillna(0.0)
    feat['bb_mtf'] = bb_mtf
    feat['ema_diff'] = (close - ema_htf_ff)
    feat['kumo_top'] = kumoTop
    feat['kumo_bot'] = kumoBot
    feat['atr'] = atr
    feat['rsi'] = rsi
    feat['hour'] = df.index.hour
    feat['dow'] = df.index.dayofweek
    return feat

# -----------------------
# Prediction & execution logic (igual à tua versão)
# -----------------------
def check_filters_and_predict(df_full, feat_full, t_index, prob_threshold=PROB_THRESHOLD):
    if model is None:
        return 0, 0.0, "no_model"
    if t_index not in feat_full.index:
        return 0, 0.0, "no_feat"
    Xrow = feat_full.loc[[t_index]].select_dtypes(include=[np.number]).fillna(0.0)
    try:
        proba = model.predict_proba(Xrow)[0]
        classes = model.classes_
        prob_map = dict(zip(classes, proba))
        best_class = classes[np.argmax(proba)]
        best_prob = prob_map[best_class]
    except Exception as e:
        log(f"Model predict_proba error: {e}")
        return 0, 0.0, "model_err"
    if best_prob < prob_threshold:
        return 0, best_prob, "prob_below"
    price = float(df_full.loc[t_index, 'close'])
    bbv = float(feat_full.loc[t_index, 'bb_mtf'])
    kumo_top = float(feat_full.loc[t_index, 'kumo_top'])
    kumo_bot = float(feat_full.loc[t_index, 'kumo_bot'])
    ema_diff = float(feat_full.loc[t_index, 'ema_diff'])
    if int(feat_full.loc[t_index, 'hour']) not in ULTRA_HOURS or int(feat_full.loc[t_index, 'dow']) not in ULTRA_DAYS:
        return 0, best_prob, "time_filter"
    if int(best_class) == 1:
        if (bbv > 0) and (price > kumo_top) and (ema_diff > 0):
            return 1, best_prob, f"long_ok bbv={bbv:.3f}"
        else:
            return 0, best_prob, f"long_filters_fail bbv={bbv:.3f} ema_diff={ema_diff:.3f}"
    elif int(best_class) == -1:
        if (bbv < 0) and (price < kumo_bot) and (ema_diff < 0):
            return -1, best_prob, f"short_ok bbv={bbv:.3f}"
        else:
            return 0, best_prob, f"short_filters_fail bbv={bbv:.3f} ema_diff={ema_diff:.3f}"
    else:
        return 0, best_prob, "flat"

def place_market_order(symbol, side, qty, reduce_only=False):
    if DRY_RUN:
        log(f"[DRY_RUN] Market order: {symbol} {side} qty={qty} reduce_only={reduce_only}")
        return {"orderId": 0, "status": "DRY"}
    try:
        res = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=float(qty),
            reduceOnly=reduce_only
        )
        log(f"Market order placed: {res}")
        return res
    except Exception as e:
        log(f"Error placing market order: {e}")
        return None

def close_position_market(symbol, side, qty):
    return place_market_order(symbol, side, qty, reduce_only=False)

# -----------------------
# WebSocket handler (mantém comportamento original)
# -----------------------
def on_kline_message(msg):
    global last_trade_time, in_position, current_position
    try:
        data = json.loads(msg)
    except Exception:
        data = msg
    if not isinstance(data, dict):
        return
    if 'k' not in data:
        return
    k = data['k']
    is_candle_closed = k.get('x', False)
    candle = {
        'timestamp': pd.to_datetime(k['t'], unit='ms', utc=True),
        'open': float(k['o']),
        'high': float(k['h']),
        'low': float(k['l']),
        'close': float(k['c']),
        'volume': float(k['v'])
    }
    if is_candle_closed:
        candles_1m.append(candle)
        if len(candles_1m) >= AGG_BARS:
            to_agg = list(candles_1m)[-AGG_BARS:]
            recs = list(candles_1m)[-1000:]
            df1m = pd.DataFrame(recs).set_index('timestamp')
            try:
                feat = build_features_from_df(df1m)
            except Exception as e:
                log(f"Feature build error: {e}")
                return
            t_index = feat.index[-1]
            action, prob, reason = check_filters_and_predict(df1m, feat, t_index)
            log(f"Signal check at {t_index.tz_convert(None)} => action={action}, prob={prob:.3f}, reason={reason}")
            now = pd.Timestamp.utcnow()
            if last_trade_time is not None:
                if (now - last_trade_time) < pd.Timedelta(minutes=COOLDOWN_MIN):
                    log("In cooldown. skipping entry.")
                    action = 0
            if (not in_position) and (action != 0):
                side = "BUY" if action == 1 else "SELL"
                pct = float(os.getenv("ORDER_PCT", "0.01"))
                qty = qty_from_pct_balance(SYMBOL, pct, LEVERAGE)
                if qty <= 0:
                    log("Computed qty <= 0; skipping")
                else:
                    res = place_market_order(SYMBOL, side, qty)
                    if res is not None:
                        in_position = True
                        current_position = {
                            'side': 'LONG' if action == 1 else 'SHORT',
                            'entry_price': float(candle['close']),
                            'qty': qty,
                            'entry_time': t_index.isoformat()
                        }
                        last_trade_time = pd.Timestamp.utcnow()
                        log(f"Entered position: {current_position}")
                        # update status file for dashboard
                        st = read_status()
                        st.setdefault('signals',[]).append({'t': t_index.isoformat(), 's': int(action), 'p': float(prob)})
                        save_status(st)
            elif in_position:
                hi = candle['high']
                lo = candle['low']
                entry = current_position['entry_price']
                side = current_position['side']
                qty = current_position['qty']
                if side == 'LONG':
                    tp_price = entry * (1 + TP)
                    sl_price = entry * (1 - SL)
                    if hi >= tp_price:
                        close_res = place_market_order(SYMBOL, "SELL", qty)
                        log(f"LONG TP hit. closed: {close_res}")
                        in_position = False
                        current_position = None
                        last_trade_time = pd.Timestamp.utcnow()
                    elif lo <= sl_price:
                        close_res = place_market_order(SYMBOL, "SELL", qty)
                        log(f"LONG SL hit. closed: {close_res}")
                        in_position = False
                        current_position = None
                        last_trade_time = pd.Timestamp.utcnow()
                else:
                    tp_price = entry * (1 - TP)
                    sl_price = entry * (1 + SL)
                    if lo <= tp_price:
                        close_res = place_market_order(SYMBOL, "BUY", qty)
                        log(f"SHORT TP hit. closed: {close_res}")
                        in_position = False
                        current_position = None
                        last_trade_time = pd.Timestamp.utcnow()
                    elif hi >= sl_price:
                        close_res = place_market_order(SYMBOL, "BUY", qty)
                        log(f"SHORT SL hit. closed: {close_res}")
                        in_position = False
                        current_position = None
                        last_trade_time = pd.Timestamp.utcnow()

def on_error(ws, error):
    log(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    log("### WebSocket closed ###")

def on_open(ws):
    log("WebSocket connection opened.")

# -----------------------
# Start WebSocket with reconnection loop (Render-friendly)
# -----------------------
def start_kline_ws(symbol):
    stream = f"wss://fstream.binance.com/ws/{symbol.lower()}@kline_1m"

    def _on_message(ws, message):
        try:
            on_kline_message(json.loads(message))
        except Exception as e:
            log(f"on_message handler error: {e}")
            log(traceback.format_exc())

    # reconnect loop
    while True:
        try:
            ws = WebSocketApp(stream,
                              on_open=lambda ws: on_open(ws),
                              on_message=lambda ws, msg: _on_message(ws, msg),
                              on_error=lambda ws, err: on_error(ws, err),
                              on_close=lambda ws, code, msg: on_close(ws, code, msg))
            log("Connecting to WebSocket...")
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except Exception as e:
            log(f"Websocket crashed: {e}")
            log(traceback.format_exc())
        log("Websocket disconnected. Reconnecting in 5s...")
        time.sleep(5)

# -----------------------
# Entrypoint
# -----------------------
def main():
    log(f"Starting Ultra-Precision Realtime Bot — DRY_RUN={DRY_RUN}")
    # download model if needed and reload
    global model
    if MODEL_URL and not os.path.exists(MODEL_FILE):
        download_model_if_needed()
        model = load_model()
    # basic check symbol info
    try:
        s_info = get_symbol_info(SYMBOL)
        if not s_info:
            log(f"Symbol {SYMBOL} not found in exchange info. Exiting.")
            return
        log("Symbol info OK.")
    except Exception as e:
        log(f"Error fetching symbol info: {e}")
    try:
        start_kline_ws(SYMBOL)
    except KeyboardInterrupt:
        log("KeyboardInterrupt received — shutting down.")
    except Exception as e:
        log(f"Fatal error in main: {e}")
        log(traceback.format_exc())

if __name__ == "__main__":
    main()
