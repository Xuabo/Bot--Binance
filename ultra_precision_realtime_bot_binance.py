# ultra_precision_realtime_bot_binance.py
# Python 3.8+
# Requires: python-binance, websocket-client, numpy, pandas, joblib, ta (optional)
#
# Usage: python ultra_precision_realtime_bot_binance.py
#
# WARNING: Test in DRY_RUN mode first (DRY_RUN = True)

import os
import json
import time
import math
import threading
from collections import deque
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib

from websocket import WebSocketApp
from binance.client import Client

# -----------------------
# USER CONFIG
# -----------------------
API_KEY = os.getenv("BINANCE_KEY")
API_SECRET = os.getenv("BINANCE_SECRET")
SYMBOL = "BTCUSDT"                 # instrument to trade
DRY_RUN = True                     # True => don't send live orders
LEVERAGE = 5                       # futures leverage used to compute qty
POSITION_SIDE_HEDGE = False        # If you need HEDGE mode, set True and adapt orders
PROB_THRESHOLD = 0.60              # minimal model probability to act
MODEL_FILE = "model.joblib"        # path to joblib model

# Trading params (can override with best found)
TP = 0.02                          # take profit (fraction) e.g. 0.02 => 2%
SL = 0.008                         # stop loss (fraction)
COOLDOWN_MIN = 5                   # cooldown between trades (minutes)

# Indicator params
BB_SHORT = 14
BB_LONG = 40
BB_STD = 2.0
EMA_TF = "30min"                   # resample alias (pandas)
EMA_LEN = 200

# Ultra filters
ULTRA_HOURS = [9, 13, 14, 15, 16, 17, 19]
ULTRA_DAYS = [0, 1, 2, 3, 4]       # Mon-Fri

# Aggregation
INPUT_BAR_MINUTES = 1
BACKTEST_BAR_MINUTES = 3
AGG_BARS = int(BACKTEST_BAR_MINUTES / INPUT_BAR_MINUTES)  # e.g., 3

# Model features expectation: the model must accept the same columns as built here
FEATURE_COLUMNS = ['close', 'ret1', 'bb_mtf', 'ema_diff', 'kumo_top', 'kumo_bot', 'atr', 'rsi', 'hour', 'dow']

LOGFILE = "ultra_precision_bot.log"

# -----------------------
# Setup clients
# -----------------------
client = Client(API_KEY, API_SECRET)
# Ensure using futures endpoints for account info and orders
# The python-binance Client uses same object but futures methods are under .futures_*
# e.g. client.futures_account_balance(), client.futures_create_order(...)

# -----------------------
# Utilities
# -----------------------
def log(msg):
    s = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(s)
    with open(LOGFILE, "a") as f:
        f.write(s + "\n")

def round_step_size(qty, step):
    # round down to step
    return math.floor(qty / step) * step

def get_symbol_info(symbol):
    info = client.futures_exchange_info()
    for s in info['symbols']:
        if s['symbol'] == symbol:
            return s
    return None

def qty_from_pct_balance(symbol, pct, leverage):
    """
    Compute quantity for futures given pct of USDT balance (cross wallet),
    using current price and leverage.
    """
    try:
        bal = client.futures_account_balance()
        usdt_bal = 0.0
        for b in bal:
            if b['asset'] == 'USDT':
                usdt_bal = float(b['balance'])
                break
    except Exception as e:
        log(f"Error fetching balance: {e}")
        usdt_bal = 100.0  # fallback fake

    price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
    usdt_alloc = usdt_bal * pct
    notional = usdt_alloc * leverage
    qty = notional / price
    # apply symbol precision/step
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
# Indicator functions
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
# Data buffers
# -----------------------
# store last N 1m candles (as dicts)
CANDLES_BUFFER_MAX = 2000
candles_1m = deque(maxlen=CANDLES_BUFFER_MAX)  # each element: dict with ts, open, high, low, close, volume

# last trade timestamp for cooldown
last_trade_time = None
in_position = False
current_position = None  # dict with side, entry_price, qty, entry_time

# Load model
try:
    model = joblib.load(MODEL_FILE)
    log(f"Loaded model from {MODEL_FILE}")
except Exception as e:
    log(f"ERROR loading model: {e}. Exiting.")
    raise SystemExit(1)

# -----------------------
# Feature builder (matches training)
# -----------------------
def build_features_from_df(df, bb_s=BB_SHORT, bb_l=BB_LONG, bb_std=BB_STD, ema_tf=EMA_TF, ema_len=EMA_LEN):
    close = df['close']
    # MTF bbtrend on 30min
    close_30 = close.resample("30min").last().ffill()
    bb_mtf = rolling_bbtrend(close_30, bb_s, bb_l, bb_std).reindex(df.index, method='ffill').fillna(0.0)

    # EMA HTF
    ema_htf = close.resample(ema_tf).last().ffill().ewm(span=ema_len, adjust=False).mean()
    ema_htf_ff = ema_htf.reindex(df.index).ffill().fillna(close)


    # kumo
    kumoTop, kumoBot = compute_unshifted_kumo(df)

    atr = compute_atr(df, 14).fillna(0.0)
    rsi = compute_rsi(close, 14).fillna(50.0)

    feat = pd.DataFrame(index=df.index)
    feat['close'] = close
    feat['ret1'] = close.pct_change().fillna(0.0)
    feat['bb_mtf'] = bb_mtf
    feat['ema_diff'] = (close - ema_htf_ff)
    # For compatibility with some models, keep both names: training may expect kumo_top/kumo_bot or kumo_top_diff/kumo_bot_diff
    feat['kumo_top'] = kumoTop
    feat['kumo_bot'] = kumoBot
    feat['atr'] = atr
    feat['rsi'] = rsi
    feat['hour'] = df.index.hour
    feat['dow'] = df.index.dayofweek
    return feat

# -----------------------
# Simulation / Execution helpers
# -----------------------
def check_filters_and_predict(df_full, feat_full, t_index, prob_threshold=PROB_THRESHOLD):
    """
    At timestamp t_index (a pd.Timestamp in feat_full.index) check ML pred and filters.
    Returns (action, prob, details) where action in {1,-1,0}
    """
    # if timestamp not in feats (rare), return 0
    if t_index not in feat_full.index:
        return 0, 0.0, "no_feat"

    Xrow = feat_full.loc[[t_index]].select_dtypes(include=[np.number]).fillna(0.0)
    try:
        proba = model.predict_proba(Xrow)[0]
        classes = model.classes_
        # map class->prob
        prob_map = dict(zip(classes, proba))
        # choose best non-zero class
        best_class = classes[np.argmax(proba)]
        best_prob = prob_map[best_class]
    except Exception as e:
        log(f"Model predict_proba error: {e}")
        return 0, 0.0, "model_err"

    if best_prob < prob_threshold:
        return 0, best_prob, "prob_below"

    # Execution filters (recompute quick values)
    price = float(df_full.loc[t_index, 'close'])
    bbv = float(feat_full.loc[t_index, 'bb_mtf'])
    kumo_top = float(feat_full.loc[t_index, 'kumo_top'])
    kumo_bot = float(feat_full.loc[t_index, 'kumo_bot'])
    ema_diff = float(feat_full.loc[t_index, 'ema_diff'])

    if int(feat_full.loc[t_index, 'hour']) not in ULTRA_HOURS or int(feat_full.loc[t_index, 'dow']) not in ULTRA_DAYS:
        return 0, best_prob, "time_filter"

    if int(best_class) == 1:
        # long: require bbv>0, price above kumo, price > ema
        if (bbv > 0) and (price > kumo_top) and (ema_diff > 0):
            return 1, best_prob, f"long_ok bbv={bbv:.3f}"
        else:
            return 0, best_prob, f"long_filters_fail bbv={bbv:.3f} ema_diff={ema_diff:.3f}"
    elif int(best_class) == -1:
        # short: require bbv<0, price below kumo, price < ema
        if (bbv < 0) and (price < kumo_bot) and (ema_diff < 0):
            return -1, best_prob, f"short_ok bbv={bbv:.3f}"
        else:
            return 0, best_prob, f"short_filters_fail bbv={bbv:.3f} ema_diff={ema_diff:.3f}"
    else:
        return 0, best_prob, "flat"

def place_market_order(symbol, side, qty, reduce_only=False):
    """
    Place market order on futures. side: "BUY" or "SELL"
    reduce_only param included for compatibility if using HEDGE mode.
    """
    if DRY_RUN:
        log(f"[DRY_RUN] Market order: {symbol} {side} qty={qty} reduce_only={reduce_only}")
        return {"orderId": 0, "status": "DRY"}
    try:
        # For hedge mode you must pass positionSide if needed (omitted here for simplicity)
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
    # side should close inverse (if we had long, close with SELL)
    return place_market_order(symbol, side, qty, reduce_only=False)

# -----------------------
# Candle handling
# -----------------------
def on_kline_message(msg):
    """
    WebSocket message handler for kline stream.
    Each message is a JSON (dict) from Binance.
    We extract finished 1m candle and append to buffer.
    When buffer has AGG_BARS multiples we compute 3m candle and run logic.
    """
    global last_trade_time, in_position, current_position
    try:
        data = json.loads(msg)
    except Exception:
        # sometimes msg already dict
        data = msg

    # filter event format: expects 'k' inside
    if not isinstance(data, dict):
        return
    if 'k' not in data:
        return

    k = data['k']
    is_candle_closed = k.get('x', False)
    # Build candle from k
    candle = {
        'timestamp': pd.to_datetime(k['t'], unit='ms', utc=True),  # open time
        'open': float(k['o']),
        'high': float(k['h']),
        'low': float(k['l']),
        'close': float(k['c']),
        'volume': float(k['v'])
    }

    # Always append last 1m candle (we keep both ongoing and closed; will use closed)
    # Only act on closed candles
    if is_candle_closed:
        candles_1m.append(candle)
        # if we have enough 1m candles to form a 3m candle, aggregate last AGG_BARS
        if len(candles_1m) >= AGG_BARS:
            # aggregate last AGG_BARS to form 3m candle
            to_agg = list(candles_1m)[-AGG_BARS:]
            ts0 = to_agg[0]['timestamp']
            open_v = to_agg[0]['open']
            high_v = max(x['high'] for x in to_agg)
            low_v = min(x['low'] for x in to_agg)
            close_v = to_agg[-1]['close']
            vol_v = sum(x['volume'] for x in to_agg)
            # Create a DataFrame from recent candles (we will take last N for indicators)
            # Use last 500 1m candles to build features (ok for perf)
            recs = list(candles_1m)[-1000:]
            df1m = pd.DataFrame(recs).set_index('timestamp')
            # Resample/aggregate to 3m index for features: we will compute features on 1m series but use MTF resample inside functions
            # For simplicity compute features on 1m DF directly (functions handle resample internally)
            try:
                feat = build_features_from_df(df1m)
            except Exception as e:
                log(f"Feature build error: {e}")
                return

            # Use last (most recent) feature timestamp
            t_index = feat.index[-1]
            action, prob, reason = check_filters_and_predict(df1m, feat, t_index)
            log(f"Signal check at {t_index.tz_convert(None)} => action={action}, prob={prob:.3f}, reason={reason}")

            # Cooldown check
            now = pd.Timestamp.utcnow()
            if last_trade_time is not None:
                if (now - last_trade_time) < pd.Timedelta(minutes=COOLDOWN_MIN):
                    log("In cooldown. skipping entry.")
                    action = 0

            # If not in position and action requested => open
            if (not in_position) and (action != 0):
                side = "BUY" if action == 1 else "SELL"
                # compute qty based on percent of balance; choose percent (example 1%): adjust to desired
                pct = 0.01  # 1% of balance by default; change as desired or compute dynamically
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
                            'entry_time': t_index
                        }
                        last_trade_time = pd.Timestamp.utcnow()
                        log(f"Entered position: {current_position}")
            # If in position: check TP/SL intrabar
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
                        # close with market sell
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
                    # SHORT
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
# Websocket runner
# -----------------------
def start_kline_ws(symbol):
    # stream: wss://fstream.binance.com/ws/{symbol_lower}@kline_1m
    stream = f"wss://fstream.binance.com/ws/{symbol.lower()}@kline_1m"

    def _on_message(ws, message):
        try:
            on_kline_message(json.loads(message))
        except Exception as e:
            log(f"on_message handler error: {e}")

    ws = WebSocketApp(stream,
                      on_open=lambda ws: on_open(ws),
                      on_message=lambda ws, msg: _on_message(ws, msg),
                      on_error=lambda ws, err: on_error(ws, err),
                      on_close=lambda ws, code, msg: on_close(ws, code, msg))
    # run forever
    ws.run_forever()

# -----------------------
# Entrypoint
# -----------------------
def main():
    log("Starting Ultra-Precision Realtime Bot (Binance Futures) — DRY_RUN=%s" % DRY_RUN)
    # pre-check symbol info
    s_info = get_symbol_info(SYMBOL)
    if not s_info:
        log(f"Symbol {SYMBOL} not found in exchange info. Exiting.")
        return
    log(f"Symbol info OK. Starting websocket for {SYMBOL} ...")
    # start websocket in main thread (blocking)
    try:
        start_kline_ws(SYMBOL)
    except KeyboardInterrupt:
        log("KeyboardInterrupt received — shutting down.")
    except Exception as e:
        log(f"Fatal error in main: {e}")

if __name__ == "__main__":
    main()
