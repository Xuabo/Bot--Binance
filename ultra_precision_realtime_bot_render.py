#!/usr/bin/env python3
# Ultra Precision Realtime Bot - Render Version
import os, json, time, traceback
import numpy as np
import pandas as pd
from joblib import load
import requests

from binance.websocket.futures.websocket_client import FuturesWebsocketClient
from binance.um_futures import UMFutures

# ==========================================================
# ENVIRONMENT VARIABLES (Render)
# ==========================================================
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# URL where your trained ML model (.joblib) is stored (Google Drive RAW link, S3, Dropbox direct, etc.)
MODEL_URL = os.getenv("MODEL_URL", "")

# Render allows writing to /tmp only
MODEL_FILE = "/tmp/model.joblib"


# ==========================================================
# LOAD MODEL
# ==========================================================
def load_model():
    if os.path.exists(MODEL_FILE):
        print("Model found locally, loading...")
        return load(MODEL_FILE)

    if MODEL_URL:
        print("Downloading model from:", MODEL_URL)
        r = requests.get(MODEL_URL)
        with open(MODEL_FILE, "wb") as f:
            f.write(r.content)
        print("Model saved to /tmp")
        return load(MODEL_FILE)

    raise Exception("MODEL_URL not specified and no local model found")


model = load_model()


# ==========================================================
# Binance Client
# ==========================================================
client = UMFutures(key=API_KEY, secret=API_SECRET)


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator


window_prices = []  # store last 500 candles


def extract_features(prices):
    if len(prices) < 200:
        return None

    close = np.array([p["close"] for p in prices], dtype=float)
    s = pd.Series(close)

    # Indicators
    bb = BollingerBands(s, window=20, window_dev=2)
    ema = EMAIndicator(s, window=200).ema_indicator()
    rsi = RSIIndicator(s, window=14).rsi()

    features = np.array([
        close[-1],
        bb.bollinger_hband().iloc[-1],
        bb.bollinger_lband().iloc[-1],
        ema.iloc[-1],
        rsi.iloc[-1]
    ]).reshape(1, -1)

    return features


# ==========================================================
# PLACE ORDER
# ==========================================================
def place_order(signal):
    side = "BUY" if signal == 1 else "SELL"
    qty = 0.001  # you can modify this

    try:
        print(f"Placing order {side} ...")
        resp = client.new_order(
            symbol="BTCUSDT",
            side=side,
            type="MARKET",
            quantity=qty
        )
        print("Order response:", resp)

    except Exception as e:
        print("Order error:", e)
        traceback.print_exc()


# ==========================================================
# WEBSOCKET HANDLER
# ==========================================================
def start_ws():
    ws = FuturesWebsocketClient()
    ws.start()

    print("Websocket started.")

    def handle_message(msg):
        global window_prices

        try:
            if "data" not in msg:
                return

            k = msg["data"]["k"]  # candle

            if k["x"] is False:
                # Candle not closed yet
                return

            close_price = float(k["c"])
            window_prices.append({"close": close_price})

            # keep last 500 candles
            if len(window_prices) > 500:
                window_prices = window_prices[-500:]

            features = extract_features(window_prices)
            if features is None:
                return

            # prediction
            pred = model.predict(features)[0]
            print(f"[Prediction] {pred}")

            if pred in [1, -1]:
                place_order(pred)

        except Exception as e:
            print("Error in message:", e)
            traceback.print_exc()

    ws.kline(
        symbol="BTCUSDT",
        interval="1m",
        id=1,
        callback=handle_message
    )

    # Keep websocket alive
    while True:
        time.sleep(5)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    print("🚀 Ultra Precision Bot running on Render...")
    print("Waiting for market data...\n")
    start_ws()
