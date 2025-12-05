# ultra_precision_realtime_bot_binance.py
# Python 3.8+
# Requires: python-binance, websocket-client, numpy, pandas, joblib, python-dotenv
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
# LOAD ENVIRONMENT (.env stored on server)
# ---------------------------------------------------------
from dotenv import load_dotenv

ENV_PATH = "/opt/bot/.env"
print(f"[DEBUG] Loading environment from: {ENV_PATH}")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    print(f"[ERROR] .env file NOT FOUND at {ENV_PATH}. Binance API keys missing!")

# -----------------------
# USER CONFIG
# -----------------------
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_SECRET_KEY")

print(f"[DEBUG] API_KEY loaded?: {'YES' if API_KEY else 'NO'}")
print(f"[DEBUG] API_SECRET loaded?: {'YES' if API_SECRET else 'NO'}")

if API_KEY:
    print(f"[SECURITY] BINANCE_API_KEY: {API_KEY[:4]}********")
if API_SECRET:
    print(f"[SECURITY] BINANCE_SECRET_KEY: {API_SECRET[:4]}********")

# === MULTI-PAIR SUPPORT ===
SYMBOLS = ["BTCUSDT", "PENGUUSDT", "MYXUSDT", "ETCUSDT"]  # adicione os pares
SYMBOL = None  # compatibilidade, não usado diretamente
DRY_RUN = True
LEVERAGE = 10                # updated to 10x
TRADE_PCT = 0.05             # 5% do saldo por trade (risco alto)
PROB_THRESHOLD = 0.60
MODEL_FILE = "model.joblib"
LOGFILE = "/opt/bot/ultra_precision_bot.log"

# Risk management
TP = 0.02   # take profit: 2%
SL = 0.008  # stop loss: 0.8%
COOLDOWN_MIN = 5

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

FEATURE_COLUMNS = [
    'close', 'ret1', 'bb_mtf', 'ema_diff',
    'kumo_top', 'kumo_bot', 'atr', 'rsi', 'hour', 'dow'
]

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
def log(msg):
    s = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(s)
    try:
        with open(LOGFILE, "a") as f:
            f.write(s + "\n")
    except Exception:
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
    print("[ERROR] Could NOT reach Binance API:", e)
    print("Check internet connection!")
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
    # log short message instead of print to keep consistent logs
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

# ---------------------------------------------------------
# ORDER EXECUTION
# ---------------------------------------------------------
def place_market_order(symbol, side, qty, reduce_only=False):
    if DRY_RUN:
        log(f"[DRY_RUN] Would execute MARKET {side} qty={qty}")
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
# UTILITIES: symbol info, qty calculation
# ---------------------------------------------------------

def get_symbol_info(symbol):
    try:
        info = client.futures_exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                return s
    except Exception as e:
        log(f"[ERROR] exchange info: {e}")
    return None


def qty_from_pct_balance(symbol, pct, leverage):
    """
    Compute quantity for futures given pct of USDT balance (cross wallet),
    using current mark price and leverage.
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

    try:
        price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
    except Exception as e:
        log(f"Error fetching mark price: {e}")
        price = 1.0

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
            qty = math.floor(qty / step) * step
    return max(qty, 0.0)

# ---------------------------------------------------------
# PREDICTION HELPERS
# ---------------------------------------------------------
def extract_probs_from_model_output(model, proba_array):
    """Try to robustly get long/short probabilities from model classes/proba."""
    classes = list(model.classes_)
    long_prob = 0.0
    short_prob = 0.0
    for c, p in zip(classes, proba_array):
        try:
            ic = int(float(c))
        except Exception:
            ic = None
        if ic == 1:
            long_prob = float(p)
        elif ic == -1:
            short_prob = float(p)
        elif ic == 0:
            # if classes are [0,1] treat 1 as long
            short_prob = float(p)
    # Fallback: if still zero and proba_array length == 2 assume index 1 = long
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
    log(f"[WS] WebSocket closed — code={code}, msg={msg}")
    log("[WS] Reconnecting in 5 seconds...")
    time.sleep(5)
    start_kline_ws(SYMBOL)


def on_error(ws, error):
    log(f"[WS ERROR] {error}")

# ---------------------------------------------------------
# MAIN KLINE HANDLER + POSITION MANAGEMENT
# ---------------------------------------------------------
CANDLES_BUFFER_MAX = 2000
candles_1m = deque(maxlen=CANDLES_BUFFER_MAX)

last_trade_time = None
in_position = False
current_position = None  # dict with side, entry_price, qty, entry_time


def on_kline_message(msg):
    global last_trade_time, in_position, current_position
    try:
        data = json.loads(msg)
    except Exception:
        data = msg

    if not isinstance(data, dict) or 'k' not in data:
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
        log("[DEBUG] 1m candle closed.")

        if len(candles_1m) < AGG_BARS:
            return

        # Build dataframe from recent candles
        recs = list(candles_1m)[-1000:]
        df1m = pd.DataFrame(recs).set_index('timestamp')

        try:
            feat = build_features_from_df(df1m)
        except Exception as e:
            log(f"Feature build error: {e}")
            return

        t_index = feat.index[-1]
        X = feat.loc[[t_index]].select_dtypes(include=[np.number]).fillna(0)

        try:
            proba = model.predict_proba(X)[0]
            long_prob, short_prob = extract_probs_from_model_output(model, proba)
            log(f"[PREDICT] long_prob={long_prob:.3f} short_prob={short_prob:.3f}")
        except Exception as e:
            log(f"[ML ERROR] {e}")
            return

        # Time filter
        hour = int(feat.loc[t_index,'hour'])
        dow = int(feat.loc[t_index,'dow'])
        if hour not in ULTRA_HOURS or dow not in ULTRA_DAYS:
            log("[FILTER] Outside allowed trading hours/days. Skipping.")
            return

        # Cooldown
        now = pd.Timestamp.utcnow()
        if last_trade_time is not None and (now - last_trade_time) < pd.Timedelta(minutes=COOLDOWN_MIN):
            log("In cooldown. skipping new entries.")
            # still allow TP/SL checks below
        entry_price = candle['close']

        # If not in position, try to open
        if (not in_position):
            # LONG
            if long_prob >= PROB_THRESHOLD:
                qty = qty_from_pct_balance(SYMBOL, TRADE_PCT, LEVERAGE)
                if qty <= 0:
                    log("Computed qty <= 0; skipping")
                else:
                    res = place_market_order(SYMBOL, "BUY", qty)
                    if res is not None:
                        in_position = True
                        current_position = {
                            'side': 'LONG',
                            'entry_price': float(entry_price),
                            'qty': qty,
                            'entry_time': t_index
                        }
                        last_trade_time = pd.Timestamp.utcnow()
                        log(f"Entered LONG: {current_position}")
            # SHORT
            elif short_prob >= PROB_THRESHOLD:
                qty = qty_from_pct_balance(SYMBOL, TRADE_PCT, LEVERAGE)
                if qty <= 0:
                    log("Computed qty <= 0; skipping")
                else:
                    res = place_market_order(SYMBOL, "SELL", qty)
                    if res is not None:
                        in_position = True
                        current_position = {
                            'side': 'SHORT',
                            'entry_price': float(entry_price),
                            'qty': qty,
                            'entry_time': t_index
                        }
                        last_trade_time = pd.Timestamp.utcnow()
                        log(f"Entered SHORT: {current_position}")

        # If in position, check TP/SL using intrabar high/low
        if in_position and current_position is not None:
            hi = candle['high']
            lo = candle['low']
            entry = current_position['entry_price']
            qty = current_position['qty']
            side = current_position['side']

            if side == 'LONG':
                tp_price = entry * (1 + TP)
                sl_price = entry * (1 - SL)
                if hi >= tp_price:
                    res = place_market_order(SYMBOL, "SELL", qty)
                    log(f"LONG TP hit. closed. res={res}")
                    in_position = False
                    current_position = None
                    last_trade_time = pd.Timestamp.utcnow()
                elif lo <= sl_price:
                    res = place_market_order(SYMBOL, "SELL", qty)
                    log(f"LONG SL hit. closed. res={res}")
                    in_position = False
                    current_position = None
                    last_trade_time = pd.Timestamp.utcnow()
            else:
                # SHORT
                tp_price = entry * (1 - TP)
                sl_price = entry * (1 + SL)
                if lo <= tp_price:
                    res = place_market_order(SYMBOL, "BUY", qty)
                    log(f"SHORT TP hit. closed. res={res}")
                    in_position = False
                    current_position = None
                    last_trade_time = pd.Timestamp.utcnow()
                elif hi >= sl_price:
                    res = place_market_order(SYMBOL, "BUY", qty)
                    log(f"SHORT SL hit. closed. res={res}")
                    in_position = False
                    current_position = None
                    last_trade_time = pd.Timestamp.utcnow()

# ---------------------------------------------------------
# START WEBSOCKET
# ---------------------------------------------------------
def start_kline_ws(symbol):  # atualizado para multi-par
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
    log("Iniciando bot multi-par (modelo BTC para todos os pares)...")

    for sym in SYMBOLS:
        threading.Thread(target=start_kline_ws, args=(sym,), daemon=True).start()
        log(f"WebSocket iniciado para {sym}")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
