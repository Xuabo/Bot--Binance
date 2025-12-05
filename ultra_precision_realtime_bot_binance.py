# ============================================================
# ultra_precision_realtime_bot_binance.py  (PARTE 1/3)
# ============================================================
# Python 3.8+
# Suporta Testnet/Mainnet automaticamente via .env
# Suporte multi-par, TP/SL reais, logs avançados
# ============================================================

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

# ------------------------------------------------------------
# Attempt import python-binance
# ------------------------------------------------------------
try:
    from binance.client import Client
except Exception as e:
    raise SystemExit(
        f"[FATAL] python-binance not found in venv.\n"
        f"Install using:\n"
        f"  /opt/bot/venv/bin/pip install python-binance\n"
        f"Error: {e}"
    )

# ------------------------------------------------------------
# LOAD .ENV
# ------------------------------------------------------------
ENV_PATH = "/opt/bot/.env"
if not os.path.exists(ENV_PATH):
    print(f"[FATAL] .env not found at {ENV_PATH}")
    raise SystemExit(1)

load_dotenv(ENV_PATH)

# Mode
USE_TESTNET = os.getenv("USE_TESTNET", "False").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "True").lower() == "true"

# API keys
TESTNET_API_KEY = os.getenv("TESTNET_API_KEY", "")
TESTNET_SECRET_KEY = os.getenv("TESTNET_SECRET_KEY", "")
MAINNET_API_KEY = os.getenv("MAINNET_API_KEY", "")
MAINNET_SECRET_KEY = os.getenv("MAINNET_SECRET_KEY", "")

# Symbols list from .env
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT").split(",")

# Files
MODEL_FILE = os.getenv("MODEL_FILE", "model.joblib")
LOGFILE = os.getenv("LOGFILE", "/opt/bot/ultra_precision_bot.log")

# Strategy settings
LEVERAGE = float(os.getenv("LEVERAGE", 10))
TRADE_PCT = float(os.getenv("TRADE_PCT", 0.05))
PROB_THRESHOLD = float(os.getenv("PROB_THRESHOLD", 0.60))
TP = float(os.getenv("TP", 0.02))
SL = float(os.getenv("SL", 0.008))
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", 5))

# Indicator settings
BB_SHORT = int(os.getenv("BB_SHORT", 14))
BB_LONG = int(os.getenv("BB_LONG", 40))
BB_STD = float(os.getenv("BB_STD", 2.0))
EMA_TF = os.getenv("EMA_TF", "30min")
EMA_LEN = int(os.getenv("EMA_LEN", 200))

# Trading time filters
ULTRA_HOURS = [int(x) for x in os.getenv("ULTRA_HOURS", "9,13,14,15,16,17,19").split(",")]
ULTRA_DAYS = [int(x) for x in os.getenv("ULTRA_DAYS", "0,1,2,3,4").split(",")]

# Candle settings
INPUT_BAR_MINUTES = int(os.getenv("INPUT_BAR_MINUTES", 1))
BACKTEST_BAR_MINUTES = int(os.getenv("BACKTEST_BAR_MINUTES", 3))
AGG_BARS = int(BACKTEST_BAR_MINUTES / INPUT_BAR_MINUTES)
CANDLES_BUFFER_MAX = int(os.getenv("CANDLES_BUFFER_MAX", 2000))


# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------
def log(msg):
    s = f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}"
    print(s)
    try:
        with open(LOGFILE, "a") as f:
            f.write(s + "\n")
    except Exception:
        pass


# ------------------------------------------------------------
# SELECT API KEYS BASED ON TESTNET OR MAINNET
# ------------------------------------------------------------
if USE_TESTNET:
    API_KEY = TESTNET_API_KEY
    API_SECRET = TESTNET_SECRET_KEY
else:
    API_KEY = MAINNET_API_KEY
    API_SECRET = MAINNET_SECRET_KEY

if not API_KEY or not API_SECRET:
    log("[FATAL] Missing API keys. Configure .env properly.")
    raise SystemExit(1)


# ------------------------------------------------------------
# INITIALIZE BINANCE CLIENT
# ------------------------------------------------------------
log(f"[INFO] USE_TESTNET={USE_TESTNET} | DRY_RUN={DRY_RUN}")

client = Client(API_KEY, API_SECRET)

# Adjust python-binance base URLs for TESTNET
if USE_TESTNET:
    try:
        client.API_URL = "https://testnet.binancefuture.com"
        client.FUTURES_URL = "https://testnet.binancefuture.com"
        log("[INFO] Switched client to Binance FUTURES TESTNET.")
    except Exception as e:
        log(f"[WARN] Could not adjust URLs for testnet: {e}")

# Connection check
try:
    client.ping()
    log("[DEBUG] Binance ping OK.")
except Exception as e:
    log(f"[WARN] Binance ping failed (may still work): {e}")


# ------------------------------------------------------------
# LOAD ML MODEL
# ------------------------------------------------------------
if not os.path.exists(MODEL_FILE):
    log(f"[FATAL] ML model not found: {MODEL_FILE}")
    raise SystemExit(1)

try:
    model = joblib.load(MODEL_FILE)
    log("[DEBUG] ML model loaded successfully.")
except Exception as e:
    log(f"[FATAL] Could not load model: {e}")
    raise SystemExit(1)
# ============================================================
# ultra_precision_realtime_bot_binance.py  (PARTE 2/3)
# ============================================================

# ----------------------------
# INDICATORS & FEATURE BUILDER
# ----------------------------
def rolling_bbtrend(close_s, short_len, long_len, std=2.0):
    """
    Compute a normalized difference between short and long Bollinger bands (multi-timeframe).
    Returns a series aligned with close_s index.
    """
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
    """
    Compute ichimoku kumo top / bot without shifting (useful as features).
    """
    try:
        high = df["high"]
        low = df["low"]
        tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkouA = (tenkan + kijun) / 2
        senkouB = (high.rolling(52).max() + low.rolling(52).min()) / 2
        kumoTop = pd.concat([senkouA, senkouB], axis=1).max(axis=1).ffill().fillna(0.0)
        kumoBot = pd.concat([senkouA, senkouB], axis=1).min(axis=1).ffill().fillna(0.0)
        return kumoTop, kumoBot
    except Exception as e:
        log(f"[INDICATOR] compute_unshifted_kumo error: {e}")
        idx = df.index
        return pd.Series(0.0, index=idx), pd.Series(0.0, index=idx)


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
    """
    Build features expected by the ML model from 1m dataframe with index==timestamp.
    """
    try:
        log("[DEBUG] Building features...")
        close = df["close"]
        # HTF resample for 30min
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
    except Exception as e:
        log(f"[FEATURE] build_features_from_df failed: {e}")
        raise


# ----------------------------
# UTILITIES: Exchange info & qty calc
# ----------------------------
def get_symbol_info(symbol):
    """
    Return exchange symbol info for futures (LOT_SIZE filter etc)
    """
    try:
        info = client.futures_exchange_info()
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                return s
    except Exception as e:
        log(f"[ERROR] get_symbol_info: {e}")
    return None


def qty_from_pct_balance(symbol, pct, leverage):
    """
    Compute quantity for futures given pct of USDT balance (cross wallet),
    using current mark price and leverage. Respects stepSize/minQty if available.
    Returns float qty (0.0 if too small).
    """
    try:
        bal = client.futures_account_balance()
        usdt_bal = 0.0
        for b in bal:
            if b.get("asset") == "USDT":
                usdt_bal = float(b.get("balance", 0.0))
                break
    except Exception as e:
        log(f"[WARN] Error fetching balance: {e}")
        usdt_bal = 100.0  # sensible fallback for testing

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


# ----------------------------
# PROBABILITY EXTRACTION (robust)
# ----------------------------
def extract_probs_from_model_output(model, proba_array):
    """
    Robustly map model.classes_ and predict_proba output to (long_prob, short_prob).
    Handles classes like [1, -1], [0,1], strings, numpy types, etc.
    """
    long_prob = 0.0
    short_prob = 0.0
    try:
        classes = list(model.classes_)
        for c, p in zip(classes, proba_array):
            try:
                # coerce class to int if possible
                if isinstance(c, (list, tuple, np.ndarray)):
                    c2 = c[0]
                else:
                    c2 = c
                # pandas/numpy scalars
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
                # interpret 0 as short in some encodings
                short_prob = float(p)
    except Exception as e:
        log(f"[PROB] extract_probs error: {e}")

    # fallback when nothing extracted
    if long_prob == 0.0 and short_prob == 0.0 and len(proba_array) == 2:
        try:
            short_prob = float(proba_array[0])
            long_prob = float(proba_array[1])
        except Exception:
            pass
    return long_prob, short_prob


# ----------------------------
# ORDER EXECUTION (safe)
# ----------------------------
def place_market_order(symbol, side, qty, reduce_only=False):
    """
    Place market order on futures. In DRY_RUN mode just log.
    Returns response dict or None.
    """
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


# ----------------------------
# PER-SYMBOL STATE
# ----------------------------
# Keep buffers and states per-symbol to allow multi-par operation
candles_map = {s.strip(): deque(maxlen=CANDLES_BUFFER_MAX) for s in SYMBOLS}
last_trade_time = {s.strip(): None for s in SYMBOLS}
in_position = {s.strip(): False for s in SYMBOLS}
current_position = {s.strip(): None for s in SYMBOLS}
state_lock = threading.Lock()

log(f"[STATE] Initialized per-symbol state for: {', '.join(list(candles_map.keys()))}")
# ============================================================
# ultra_precision_realtime_bot_binance.py  (PARTE 3/3)
# ============================================================

# ----------------------------
# WEBSOCKET: per-symbol handlers (factory already in PART 2)
# ----------------------------
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

# ----------------------------
# START WEBSOCKET (per symbol)
# ----------------------------
def start_kline_ws(symbol):
    symbol = symbol.strip().upper()
    # choose stream base depending on testnet/mainnet
    if USE_TESTNET:
        # testnet futures streaming endpoint (best-effort)
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

    # run forever with simple retry loop
    while True:
        try:
            ws.run_forever()
        except Exception as e:
            log(f"[WS {symbol}] run_forever exception: {e}")
        log(f"[WS {symbol}] Sleeping 3s before reconnect...")
        time.sleep(3)


# ----------------------------
# HEALTH CHECK helpers (optional)
# ----------------------------
def health_report():
    """
    Return a short diagnostic summary useful to print in logs or use in an external health-check.
    """
    try:
        threads = threading.enumerate()
        thread_count = len(threads)
        sockets = []
        try:
            # best-effort: count non-daemon threads beyond main
            pass
        except Exception:
            pass
        active_symbols = [s for s, d in candles_map.items()]
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "symbols": active_symbols,
            "threads": thread_count,
            "in_position": {s: in_position.get(s) for s in active_symbols}
        }
        return report
    except Exception as e:
        return {"error": str(e)}


# ----------------------------
# MAIN
# ----------------------------
def main():
    log("=== Starting multi-pair ultra_precision bot (final) ===")
    log(f"CONFIG -> USE_TESTNET={USE_TESTNET} DRY_RUN={DRY_RUN} SYMBOLS={SYMBOLS}")

    # start one websocket thread per symbol
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
    # main loop: periodic health log
    try:
        while True:
            # every 60s print lightweight health
            time.sleep(60)
            report = health_report()
            log(f"[HEALTH] {report}")
    except KeyboardInterrupt:
        log("KeyboardInterrupt received — shutting down.")
    except Exception as e:
        log(f"Main loop exception: {e}")

if __name__ == "__main__":
    main()
