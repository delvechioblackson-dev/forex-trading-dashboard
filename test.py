import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FASTFOREX_API_KEY = os.getenv("FASTFOREX_API_KEY", "")
TELEGRAM_TOKEN_DEFAULT = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID_DEFAULT = os.getenv("TELEGRAM_CHAT_ID", "")


def send_telegram_alert(message: str):
    """Send a simple Telegram alert (optional).

    Resolution order for configuration:
    1. Values from the Streamlit UI (session_state)
    2. Environment variables
    3. Streamlit secrets (if available)
    """
    ui_token = st.session_state.get("TELEGRAM_TOKEN_UI")
    ui_chat_id = st.session_state.get("TELEGRAM_CHAT_ID_UI")

    token = ui_token or os.getenv("TELEGRAM_TOKEN")
    chat_id = ui_chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if (not token) or (not chat_id):
        try:
            secrets = getattr(st, "secrets", None)
            if secrets is not None:
                if not token and "TELEGRAM_TOKEN" in secrets:
                    token = secrets["TELEGRAM_TOKEN"]
                if not chat_id and "TELEGRAM_CHAT_ID" in secrets:
                    chat_id = secrets["TELEGRAM_CHAT_ID"]
        except Exception:
            pass

    if not token and TELEGRAM_TOKEN_DEFAULT:
        token = TELEGRAM_TOKEN_DEFAULT
    if not chat_id and TELEGRAM_CHAT_ID_DEFAULT:
        chat_id = TELEGRAM_CHAT_ID_DEFAULT

    if not token or not chat_id:
        st.warning("Telegram is not configured (token or chat ID missing). No alert sent.")
        return

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=15)
        if resp.status_code != 200:
            try:
                info = resp.json()
                desc = info.get("description", "unknown error")
            except Exception:
                desc = resp.text[:200]
            st.warning(f"Telegram API error ({resp.status_code}): {desc}")
    except Exception as exc:
        st.warning(f"Could not send Telegram alert: {exc}")


def build_news_query(instrument_type, base_currency=None, target_currency=None, index_choice=None):
    """Build a simple news query based on the instrument."""
    if instrument_type == "Forex" and base_currency and target_currency:
        pair = f"{base_currency}{target_currency}"
        return f"({base_currency} OR {target_currency} OR {pair}) AND (forex OR currency OR FX OR ECB OR FED OR central bank OR interest rates)"

    if index_choice:
        if index_choice.startswith("US30"):
            return "(Dow Jones OR \"US30\" OR \"Dow\" OR \"DJIA\") AND (stocks OR index OR futures)"
        if index_choice.startswith("NAS100"):
            return "(Nasdaq 100 OR \"Nasdaq\" OR \"NAS100\") AND (stocks OR index OR futures)"

    return "(forex OR stocks OR indices) AND (market OR economy OR inflation OR interest rates)"


def fetch_news_articles(query, api_key, language="en", max_articles=8):
    """Fetch news articles via NewsData.io (metadata only)."""
    if not api_key:
        return []

    url = "https://newsdata.io/api/1/latest"
    params = {
        "apikey": api_key,
        "q": query,
        "language": language,
        "page": 1,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if not data or data.get("status") not in ("success", True):
            st.warning(
                f"News API error: {data.get('message', 'unknown error') if isinstance(data, dict) else 'unknown response'}"
            )
            return []

        articles = []
        for art in data.get("results", [])[:max_articles]:
            articles.append(
                {
                    "title": art.get("title", ""),
                    "description": art.get("description", ""),
                    "url": art.get("link") or art.get("source_url"),
                    "source": art.get("source_id", ""),
                    "publishedAt": art.get("pubDate", ""),
                }
            )
        return articles
    except Exception as exc:
        st.warning(f"Could not fetch news: {exc}")
        return []


def analyze_news_sentiment(articles):
    """Simple sentiment score based on words in title + description."""
    if not articles:
        return {"score": 0.0, "label": "Neutral (no news)"}

    positive_words = [
        "rises",
        "rise",
        "surge",
        "rally",
        "record high",
        "growth",
        "strong",
        "bullish",
        "optimistic",
        "rebound",
        "recovery",
        "beats expectations",
    ]
    negative_words = [
        "falls",
        "fall",
        "plunge",
        "selloff",
        "recession",
        "weak",
        "bearish",
        "pessimistic",
        "crisis",
        "war",
        "inflation",
        "rate hike",
        "cuts forecast",
    ]

    total_score = 0
    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')}".lower()
        score = 0
        for word in positive_words:
            if word in text:
                score += 1
        for word in negative_words:
            if word in text:
                score -= 1
        total_score += score

    avg = total_score / max(len(articles), 1)

    if avg > 0.5:
        label = "Bullish"
    elif avg < -0.5:
        label = "Bearish"
    else:
        label = "Neutral"

    return {"score": float(avg), "label": label}


def filter_signals_by_news(signals, news_sentiment):
    """Attach news score to signals and filter extreme conflicts."""
    if not signals or not news_sentiment:
        return signals

    score = float(news_sentiment.get("score", 0.0))
    label = news_sentiment.get("label", "")

    def _estimate_success(direction, s):
        strength = min(abs(s), 2.0) / 2.0
        base = 0.5

        if direction == "Buy":
            if s > 0:
                prob = base + 0.3 * strength
            elif s < 0:
                prob = base - 0.3 * strength
            else:
                prob = base
        elif direction == "Sell":
            if s < 0:
                prob = base + 0.3 * strength
            elif s > 0:
                prob = base - 0.3 * strength
            else:
                prob = base
        else:
            prob = base

        prob = max(0.1, min(0.9, prob))
        return prob * 100.0

    filtered = []
    for sig in signals:
        out = dict(sig)
        direction = str(out.get("signal", "")).capitalize()

        out["news_sentiment_score"] = score
        out["news_sentiment_label"] = label
        out["news_success_score"] = _estimate_success(direction, score)

        if score > 0.5 and direction == "Sell" and out["news_success_score"] < 30:
            continue
        if score < -0.5 and direction == "Buy" and out["news_success_score"] < 30:
            continue

        filtered.append(out)

    return filtered


# Fetch single pair exchange rate (no caching so price can update live)
def fetch_single_pair(from_currency="USD", to_currency="EUR"):
    if not FASTFOREX_API_KEY:
        st.warning("FASTFOREX_API_KEY is not set.")
        return None

    url = (
        "https://api.fastforex.io/fetch-one"
        f"?from={from_currency}&to={to_currency}&api_key={FASTFOREX_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "result" in data:
            return data["result"].get(to_currency, None)
        st.error(f"API Error: {data.get('error', 'Unknown error')}")
        return None
    except Exception as exc:
        st.warning(f"Error fetching exchange rate: {exc}")
        return None


def fetch_fx_history_alpha_vantage(from_currency, to_currency, freq, periods, api_key):
    """Fetch intraday forex data via Alpha Vantage."""
    if not api_key:
        return pd.DataFrame()

    interval = freq
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": from_currency,
        "to_symbol": to_currency,
        "interval": interval,
        "outputsize": "compact",
        "apikey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if "Error Message" in data or "Note" in data:
            msg = data.get("Error Message") or data.get("Note", "Unknown Alpha Vantage error")
            st.warning(f"Alpha Vantage FX error: {msg}")
            return pd.DataFrame()

        ts_key = f"Time Series FX ({interval})"
        ts = data.get(ts_key)
        if not ts:
            return pd.DataFrame()

        records = []
        for ts_str, values in ts.items():
            records.append(
                {
                    "Datetime": pd.to_datetime(ts_str),
                    "Open": float(values.get("1. open", 0.0)),
                    "High": float(values.get("2. high", 0.0)),
                    "Low": float(values.get("3. low", 0.0)),
                    "Close": float(values.get("4. close", 0.0)),
                }
            )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values("Datetime")
        df = df.tail(periods)

        volume_multiplier = {
            "1min": 1,
            "5min": 5,
            "15min": 15,
            "30min": 30,
        }
        multiplier = volume_multiplier.get(freq, 1)
        df["Volume"] = np.random.randint(100 * multiplier, 1000 * multiplier, len(df))

        return df
    except Exception as exc:
        st.warning(f"Error fetching FX data: {exc}")
        return pd.DataFrame()


# Generate historical data for forex
def generate_historical_data(rate, periods, freq, base_currency=None, target_currency=None, fx_api_key=None):
    """Generate historical data aligned to timeframe."""
    if rate is None:
        st.error("Rate is unavailable. Cannot generate historical data.")
        return pd.DataFrame()

    if fx_api_key and base_currency and target_currency:
        real_df = fetch_fx_history_alpha_vantage(base_currency, target_currency, freq, periods, fx_api_key)
        if not real_df.empty:
            return real_df

    base_price = rate
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq=freq)

    volatility_map = {
        "1min": 0.005,
        "5min": 0.008,
        "15min": 0.012,
        "30min": 0.015,
    }
    volatility = volatility_map.get(freq, 0.005)

    closes = np.random.normal(base_price, base_price * volatility, len(dates))
    opens = closes + np.random.normal(0, base_price * volatility * 0.6, len(dates))
    highs = closes + abs(np.random.normal(0, base_price * volatility * 0.8, len(dates)))
    lows = closes - abs(np.random.normal(0, base_price * volatility * 0.8, len(dates)))

    volume_multiplier = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "30min": 30,
    }
    multiplier = volume_multiplier.get(freq, 1)
    volumes = np.random.randint(100 * multiplier, 1000 * multiplier, len(dates))

    return pd.DataFrame(
        {
            "Datetime": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    )


# Generate order flow data (bid/ask volumes, delta)
def generate_order_flow_data(df):
    """Simulate order flow metrics."""
    df = df.copy()

    price_change = df["Close"] - df["Open"]
    bullish_bars = price_change > 0

    df["Buy_Volume"] = np.where(
        bullish_bars,
        df["Volume"] * np.random.uniform(0.55, 0.75, len(df)),
        df["Volume"] * np.random.uniform(0.25, 0.45, len(df)),
    )

    df["Sell_Volume"] = df["Volume"] - df["Buy_Volume"]
    df["Delta"] = df["Buy_Volume"] - df["Sell_Volume"]
    df["Cumulative_Delta"] = df["Delta"].cumsum()

    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    return df


def apply_pip_limits(entry_price, stop_loss, take_profit, direction, pip_size, max_sl_pips=50, max_tp_pips=100):
    """Cap SL/TP distances to configured pip limits."""
    if stop_loss is None or take_profit is None or pip_size is None:
        return stop_loss, take_profit

    max_sl = max_sl_pips * pip_size
    max_tp = max_tp_pips * pip_size

    sl_dist = abs(stop_loss - entry_price)
    tp_dist = abs(take_profit - entry_price)

    if max_sl > 0 and sl_dist > max_sl:
        sl_dist = max_sl
        if direction == "Buy":
            stop_loss = entry_price - sl_dist
        else:
            stop_loss = entry_price + sl_dist

    if max_tp > 0 and tp_dist > max_tp:
        tp_dist = max_tp
        if direction == "Buy":
            take_profit = entry_price + tp_dist
        else:
            take_profit = entry_price - tp_dist

    return stop_loss, take_profit


# Identify Supply and Demand Zones
def identify_supply_demand_zones(df, lookback=50, threshold=0.002):
    """Identify supply/demand zones using wicks and volume."""
    zones = []

    for i in range(lookback, len(df) - 10):
        body = abs(df.iloc[i]["Close"] - df.iloc[i]["Open"])
        upper_wick = df.iloc[i]["High"] - max(df.iloc[i]["Close"], df.iloc[i]["Open"])
        lower_wick = min(df.iloc[i]["Close"], df.iloc[i]["Open"]) - df.iloc[i]["Low"]
        total_range = df.iloc[i]["High"] - df.iloc[i]["Low"]

        if total_range == 0:
            continue

        if upper_wick > body * 2 and upper_wick / total_range > 0.5:
            avg_volume = df.iloc[i - lookback : i]["Volume"].mean()
            if df.iloc[i]["Volume"] > avg_volume * 1.3:
                zone_top = df.iloc[i]["High"]
                zone_bottom = df.iloc[i]["High"] - (total_range * 0.3)

                future_prices = df.iloc[i + 1 : i + 11]["High"]
                if len(future_prices) > 0 and future_prices.max() < zone_top * 1.001:
                    zones.append(
                        {
                            "type": "Supply",
                            "top": zone_top,
                            "bottom": zone_bottom,
                            "start_idx": i,
                            "strength": "High" if df.iloc[i]["Volume"] > avg_volume * 1.5 else "Medium",
                            "touches": 1,
                        }
                    )

        if lower_wick > body * 2 and lower_wick / total_range > 0.5:
            avg_volume = df.iloc[i - lookback : i]["Volume"].mean()
            if df.iloc[i]["Volume"] > avg_volume * 1.3:
                zone_bottom = df.iloc[i]["Low"]
                zone_top = df.iloc[i]["Low"] + (total_range * 0.3)

                future_prices = df.iloc[i + 1 : i + 11]["Low"]
                if len(future_prices) > 0 and future_prices.min() > zone_bottom * 0.999:
                    zones.append(
                        {
                            "type": "Demand",
                            "top": zone_top,
                            "bottom": zone_bottom,
                            "start_idx": i,
                            "strength": "High" if df.iloc[i]["Volume"] > avg_volume * 1.5 else "Medium",
                            "touches": 1,
                        }
                    )

    filtered_zones = []
    for zone in zones:
        overlap = False
        for existing in filtered_zones:
            if zone["type"] == existing["type"]:
                if not (zone["top"] < existing["bottom"] or zone["bottom"] > existing["top"]):
                    overlap = True
                    if zone["strength"] == "High" and existing["strength"] == "Medium":
                        filtered_zones.remove(existing)
                        overlap = False
                    break
        if not overlap:
            filtered_zones.append(zone)

    return filtered_zones


# Generate Supply/Demand Zone Signals
def generate_supply_demand_signals(df, zones, pip_size=0.0001):
    """Generate trading signals when price enters supply/demand zones."""
    signals = []

    for i in range(1, len(df)):
        previous_price = df.iloc[i - 1]["Close"]
        current_high = df.iloc[i]["High"]
        current_low = df.iloc[i]["Low"]
        timestamp = df.iloc[i]["Datetime"]

        for zone in zones:
            zone_top = zone["top"]
            zone_bottom = zone["bottom"]

            if zone["type"] == "Supply":
                if previous_price < zone_bottom and current_high >= zone_bottom:
                    entry_price = float(zone_bottom)
                    stop_loss = float(zone_top * 1.0005)
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (2.5 * risk)

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Sell", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Sell",
                            "type": "Supply Zone Rejection",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "1m",
                            "zone_strength": zone["strength"],
                            "risk_reward": "2.5:1",
                        }
                    )

            elif zone["type"] == "Demand":
                if previous_price > zone_top and current_low <= zone_top:
                    entry_price = float(zone_top)
                    stop_loss = float(zone_bottom * 0.9995)
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (2.5 * risk)

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Buy", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Buy",
                            "type": "Demand Zone Bounce",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "1m",
                            "zone_strength": zone["strength"],
                            "risk_reward": "2.5:1",
                        }
                    )

    return signals


def generate_m15_market_structure_signals(df_15m, zones, pip_size=0.0001):
    """Generate 15m market structure signals."""
    signals = []

    if df_15m.empty or not zones:
        return signals

    for i in range(1, len(df_15m)):
        row = df_15m.iloc[i]
        prev = df_15m.iloc[i - 1]
        timestamp = row["Datetime"]

        vol_start = max(0, i - 10)
        avg_vol = df_15m["Volume"].iloc[vol_start:i].mean() if i > 0 else df_15m["Volume"].iloc[:1].mean()

        for zone in zones:
            zone_top = zone["top"]
            zone_bottom = zone["bottom"]

            if zone["type"] == "Supply":
                if prev["Close"] <= zone_top and row["Close"] > zone_top:
                    entry_price = float(row["Close"])
                    stop_loss = float(zone_top * 0.9995)
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price + 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Buy", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Buy",
                            "type": "M15 Supply Breakout",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "15m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

                if row["High"] > zone_top and row["Close"] < zone_top:
                    entry_price = float(row["Close"])
                    stop_loss = float(max(row["High"], zone_top) * 1.0005)
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price - 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Sell", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Sell",
                            "type": "M15 Supply Reversal",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "15m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

            elif zone["type"] == "Demand":
                if prev["Close"] >= zone_bottom and row["Close"] < zone_bottom:
                    entry_price = float(row["Close"])
                    stop_loss = float(zone_bottom * 1.0005)
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price - 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Sell", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Sell",
                            "type": "M15 Demand Breakout",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "15m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

                if row["Low"] < zone_bottom and row["Close"] > zone_bottom:
                    entry_price = float(row["Close"])
                    stop_loss = float(min(row["Low"], zone_bottom) * 0.9995)
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price + 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Buy", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Buy",
                            "type": "M15 Demand Reversal",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "15m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

    return signals


def generate_m5_market_structure_signals(df_5m, zones, pip_size=0.0001):
    """Generate 5m market structure signals around higher-timeframe zones."""
    signals = []

    if df_5m.empty or not zones:
        return signals

    for i in range(1, len(df_5m)):
        row = df_5m.iloc[i]
        prev = df_5m.iloc[i - 1]
        timestamp = row["Datetime"]

        vol_start = max(0, i - 10)
        avg_vol = df_5m["Volume"].iloc[vol_start:i].mean() if i > 0 else df_5m["Volume"].iloc[:1].mean()

        for zone in zones:
            zone_top = zone["top"]
            zone_bottom = zone["bottom"]

            if zone["type"] == "Supply":
                if prev["Close"] <= zone_top and row["Close"] > zone_top:
                    entry_price = float(row["Close"])
                    stop_loss = float(zone_top * 0.9995)
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price + 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Buy", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Buy",
                            "type": "M5 Supply Breakout",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "5m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

                if row["High"] > zone_top and row["Close"] < zone_top:
                    entry_price = float(row["Close"])
                    stop_loss = float(max(row["High"], zone_top) * 1.0005)
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price - 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Sell", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Sell",
                            "type": "M5 Supply Reversal",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "5m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

            elif zone["type"] == "Demand":
                if prev["Close"] >= zone_bottom and row["Close"] < zone_bottom:
                    entry_price = float(row["Close"])
                    stop_loss = float(zone_bottom * 1.0005)
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price - 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Sell", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Sell",
                            "type": "M5 Demand Breakout",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "5m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

                if row["Low"] < zone_bottom and row["Close"] > zone_bottom:
                    entry_price = float(row["Close"])
                    stop_loss = float(min(row["Low"], zone_bottom) * 0.9995)
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price + 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Buy", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Buy",
                            "type": "M5 Demand Reversal",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "5m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

    return signals


def generate_m30_market_structure_signals(df_30m, zones, pip_size=0.0001):
    """Generate 30m market structure signals around zones."""
    signals = []

    if df_30m.empty or not zones:
        return signals

    for i in range(1, len(df_30m)):
        row = df_30m.iloc[i]
        prev = df_30m.iloc[i - 1]
        timestamp = row["Datetime"]

        vol_start = max(0, i - 10)
        avg_vol = df_30m["Volume"].iloc[vol_start:i].mean() if i > 0 else df_30m["Volume"].iloc[:1].mean()

        for zone in zones:
            zone_top = zone["top"]
            zone_bottom = zone["bottom"]

            if zone["type"] == "Supply":
                if prev["Close"] <= zone_top and row["Close"] > zone_top:
                    entry_price = float(row["Close"])
                    stop_loss = float(zone_top * 0.9995)
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price + 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Buy", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Buy",
                            "type": "M30 Supply Breakout",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "30m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

                if row["High"] > zone_top and row["Close"] < zone_top:
                    entry_price = float(row["Close"])
                    stop_loss = float(max(row["High"], zone_top) * 1.0005)
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price - 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Sell", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Sell",
                            "type": "M30 Supply Reversal",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "30m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

            elif zone["type"] == "Demand":
                if prev["Close"] >= zone_bottom and row["Close"] < zone_bottom:
                    entry_price = float(row["Close"])
                    stop_loss = float(zone_bottom * 1.0005)
                    risk = stop_loss - entry_price
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price - 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Sell", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Sell",
                            "type": "M30 Demand Breakout",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "30m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

                if row["Low"] < zone_bottom and row["Close"] > zone_bottom:
                    entry_price = float(row["Close"])
                    stop_loss = float(min(row["Low"], zone_bottom) * 0.9995)
                    risk = entry_price - stop_loss
                    if risk <= 0:
                        risk = abs(entry_price) * 0.001
                    take_profit = entry_price + 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        entry_price, stop_loss, take_profit, "Buy", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Buy",
                            "type": "M30 Demand Reversal",
                            "price": float(round(entry_price, 5)),
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                            "timeframe": "30m",
                            "zone_strength": zone.get("strength", ""),
                            "volume": float(row["Volume"]),
                        }
                    )

    return signals


# Generate footprint chart data (price levels with buy/sell volume)
def generate_footprint_data(df, bar_index):
    """Create footprint data for a specific candlestick bar."""
    if bar_index >= len(df):
        return pd.DataFrame()

    row = df.iloc[bar_index]
    low_price = row["Low"]
    high_price = row["High"]
    close_price = row["Close"]
    total_volume = row["Volume"]
    buy_vol = row["Buy_Volume"]
    sell_vol = row["Sell_Volume"]

    num_levels = max(5, int((high_price - low_price) / (low_price * 0.0001)))
    num_levels = min(num_levels, 20)

    price_levels = np.linspace(low_price, high_price, num_levels)

    weights = np.exp(-((price_levels - close_price) ** 2) / (2 * ((high_price - low_price) / 4) ** 2))
    weights = weights / weights.sum()

    buy_volumes = buy_vol * weights
    sell_volumes = sell_vol * weights

    footprint_df = pd.DataFrame(
        {
            "Price": price_levels,
            "Buy_Volume": buy_volumes,
            "Sell_Volume": sell_volumes,
            "Delta": buy_volumes - sell_volumes,
            "Total_Volume": buy_volumes + sell_volumes,
        }
    )

    return footprint_df


# Detect order flow signals
def detect_order_flow_signals(df, timeframe_label="1m", pip_size=0.0001):
    """Detect signals based on order flow heuristics."""
    signals = []

    for i in range(20, len(df)):
        current_price = df.iloc[i]["Close"]
        current_delta = df.iloc[i]["Delta"]
        current_cum_delta = df.iloc[i]["Cumulative_Delta"]
        current_volume = df.iloc[i]["Volume"]
        timestamp = df.iloc[i]["Datetime"]

        lookback = df.iloc[i - 10 : i]
        avg_volume = lookback["Volume"].mean()

        swing_start = max(0, i - 5)
        recent_low = float(df["Low"].iloc[swing_start : i + 1].min())
        recent_high = float(df["High"].iloc[swing_start : i + 1].max())

        if i >= 20:
            recent_price_low = df.iloc[i - 10 : i]["Low"].min()
            recent_delta_at_low_idx = df.iloc[i - 10 : i]["Low"].idxmin()

            if df.iloc[i]["Low"] < recent_price_low:
                if df.iloc[i]["Delta"] > df.loc[recent_delta_at_low_idx, "Delta"]:
                    stop_loss = min(recent_low, current_price)
                    risk = current_price - stop_loss
                    if risk <= 0:
                        risk = abs(current_price) * 0.001
                    take_profit = current_price + 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        current_price, stop_loss, take_profit, "Buy", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Buy",
                            "type": "Delta Divergence (Bullish)",
                            "price": float(current_price),
                            "delta": float(current_delta),
                            "cum_delta": float(current_cum_delta),
                            "strength": "High",
                            "timeframe": timeframe_label,
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                        }
                    )

        if i >= 20:
            recent_price_high = df.iloc[i - 10 : i]["High"].max()
            recent_delta_at_high_idx = df.iloc[i - 10 : i]["High"].idxmax()

            if df.iloc[i]["High"] > recent_price_high:
                if df.iloc[i]["Delta"] < df.loc[recent_delta_at_high_idx, "Delta"]:
                    stop_loss = max(recent_high, current_price)
                    risk = stop_loss - current_price
                    if risk <= 0:
                        risk = abs(current_price) * 0.001
                    take_profit = current_price - 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        current_price, stop_loss, take_profit, "Sell", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Sell",
                            "type": "Delta Divergence (Bearish)",
                            "price": float(current_price),
                            "delta": float(current_delta),
                            "cum_delta": float(current_cum_delta),
                            "strength": "High",
                            "timeframe": timeframe_label,
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                        }
                    )

        if current_volume > avg_volume * 1.5 and current_delta < -avg_volume * 0.3:
            price_change_pct = (df.iloc[i]["Close"] - df.iloc[i]["Open"]) / df.iloc[i]["Open"]
            if price_change_pct > -0.001:
                stop_loss = min(recent_low, current_price)
                risk = current_price - stop_loss
                if risk <= 0:
                    risk = abs(current_price) * 0.001
                take_profit = current_price + 2 * risk

                stop_loss, take_profit = apply_pip_limits(
                    current_price, stop_loss, take_profit, "Buy", pip_size
                )

                signals.append(
                    {
                        "timestamp": timestamp,
                        "signal": "Buy",
                        "type": "Selling Exhaustion",
                        "price": float(current_price),
                        "delta": float(current_delta),
                        "cum_delta": float(current_cum_delta),
                        "strength": "Medium",
                        "timeframe": timeframe_label,
                        "stop_loss": float(round(stop_loss, 5)),
                        "take_profit": float(round(take_profit, 5)),
                    }
                )

        if current_volume > avg_volume * 1.5 and current_delta > avg_volume * 0.3:
            price_change_pct = (df.iloc[i]["Close"] - df.iloc[i]["Open"]) / df.iloc[i]["Open"]
            if price_change_pct < 0.001:
                stop_loss = max(recent_high, current_price)
                risk = stop_loss - current_price
                if risk <= 0:
                    risk = abs(current_price) * 0.001
                take_profit = current_price - 2 * risk

                stop_loss, take_profit = apply_pip_limits(
                    current_price, stop_loss, take_profit, "Sell", pip_size
                )

                signals.append(
                    {
                        "timestamp": timestamp,
                        "signal": "Sell",
                        "type": "Buying Exhaustion",
                        "price": float(current_price),
                        "delta": float(current_delta),
                        "cum_delta": float(current_cum_delta),
                        "strength": "Medium",
                        "timeframe": timeframe_label,
                        "stop_loss": float(round(stop_loss, 5)),
                        "take_profit": float(round(take_profit, 5)),
                    }
                )

        if i >= 10:
            delta_slope = (df.iloc[i]["Cumulative_Delta"] - df.iloc[i - 5]["Cumulative_Delta"]) / 5
            price_slope = (df.iloc[i]["Close"] - df.iloc[i - 5]["Close"]) / 5

            if delta_slope > 0 and price_slope > 0 and current_delta > 0:
                if i % 15 == 0:
                    stop_loss = min(recent_low, current_price)
                    risk = current_price - stop_loss
                    if risk <= 0:
                        risk = abs(current_price) * 0.001
                    take_profit = current_price + 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        current_price, stop_loss, take_profit, "Buy", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Buy",
                            "type": "Strong Buying Flow",
                            "price": float(current_price),
                            "delta": float(current_delta),
                            "cum_delta": float(current_cum_delta),
                            "strength": "Medium",
                            "timeframe": timeframe_label,
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                        }
                    )

            elif delta_slope < 0 and price_slope < 0 and current_delta < 0:
                if i % 15 == 0:
                    stop_loss = max(recent_high, current_price)
                    risk = stop_loss - current_price
                    if risk <= 0:
                        risk = abs(current_price) * 0.001
                    take_profit = current_price - 2 * risk

                    stop_loss, take_profit = apply_pip_limits(
                        current_price, stop_loss, take_profit, "Sell", pip_size
                    )

                    signals.append(
                        {
                            "timestamp": timestamp,
                            "signal": "Sell",
                            "type": "Strong Selling Flow",
                            "price": float(current_price),
                            "delta": float(current_delta),
                            "cum_delta": float(current_cum_delta),
                            "strength": "Medium",
                            "timeframe": timeframe_label,
                            "stop_loss": float(round(stop_loss, 5)),
                            "take_profit": float(round(take_profit, 5)),
                        }
                    )

    return signals


# Add lower highs and lower lows detection and technical indicators
def add_technical_indicators(df):
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=1).mean()
    df["Lower_High"] = (df["High"].diff() < 0) & (df["High"].shift(-1) < df["High"])
    df["Lower_Low"] = (df["Low"].diff() < 0) & (df["Low"].shift(-1) < df["Low"])
    df["Higher_High"] = (df["High"].diff() > 0) & (df["High"].shift(-1) > df["High"])
    df["Higher_Low"] = (df["Low"].diff() > 0) & (df["Low"].shift(-1) > df["Low"])
    return df


# Generate trading signals based on lower highs/lows and SMAs
def generate_sell_signals(df, pip_size=0.0001):
    signals = []
    for i in range(1, len(df)):
        entry_price = float(df.iloc[i]["Close"])

        if df["Lower_High"][i] and df["Lower_Low"][i]:
            if df["SMA_50"][i] < df["SMA_200"][i] and df["Close"][i] < df["SMA_50"][i]:
                start_idx = max(0, i - 5)
                recent_high = float(df["High"].iloc[start_idx : i + 1].max())
                stop_loss = max(recent_high, entry_price)
                risk = stop_loss - entry_price
                if risk <= 0:
                    risk = entry_price * 0.001
                take_profit = entry_price - 2 * risk

                stop_loss, take_profit = apply_pip_limits(
                    entry_price, stop_loss, take_profit, "Sell", pip_size
                )
                signals.append(
                    {
                        "timestamp": df.iloc[i]["Datetime"],
                        "signal": "Sell",
                        "type": "Technical Pattern",
                        "price": entry_price,
                        "timeframe": "1m",
                        "stop_loss": float(round(stop_loss, 5)),
                        "take_profit": float(round(take_profit, 5)),
                    }
                )

        if df["Higher_High"][i] and df["Higher_Low"][i]:
            if df["SMA_50"][i] > df["SMA_200"][i] and df["Close"][i] > df["SMA_50"][i]:
                start_idx = max(0, i - 5)
                recent_low = float(df["Low"].iloc[start_idx : i + 1].min())
                stop_loss = min(recent_low, entry_price)
                risk = entry_price - stop_loss
                if risk <= 0:
                    risk = entry_price * 0.001
                take_profit = entry_price + 2 * risk

                stop_loss, take_profit = apply_pip_limits(
                    entry_price, stop_loss, take_profit, "Buy", pip_size
                )
                signals.append(
                    {
                        "timestamp": df.iloc[i]["Datetime"],
                        "signal": "Buy",
                        "type": "Technical Pattern",
                        "price": entry_price,
                        "timeframe": "1m",
                        "stop_loss": float(round(stop_loss, 5)),
                        "take_profit": float(round(take_profit, 5)),
                    }
                )
    return signals


# Plot footprint chart for a specific bar
def plot_footprint_chart(footprint_df, bar_datetime):
    """Create a footprint chart visualization."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=footprint_df["Price"],
            x=footprint_df["Buy_Volume"],
            orientation="h",
            name="Buy Volume",
            marker=dict(color="green"),
            text=footprint_df["Buy_Volume"].round(0),
            textposition="auto",
        )
    )

    fig.add_trace(
        go.Bar(
            y=footprint_df["Price"],
            x=-footprint_df["Sell_Volume"],
            orientation="h",
            name="Sell Volume",
            marker=dict(color="red"),
            text=footprint_df["Sell_Volume"].round(0),
            textposition="auto",
        )
    )

    fig.update_layout(
        title=f"Footprint Chart - {bar_datetime}",
        xaxis_title="Volume (Buy-> / <-Sell)",
        yaxis_title="Price Level",
        barmode="overlay",
        template="plotly_dark",
        height=500,
    )

    return fig


# Streamlit app main function
def main():
    st.set_page_config(page_title="Supply/Demand Trading Dashboard", page_icon="chart", layout="wide")
    st.title("Advanced Supply/Demand Zone Trading Dashboard")

    st.markdown(
        """
    Features:
    - Real-time Forex Data
    - Supply and Demand Zone Detection
    - Order Flow Analysis (Delta, Cumulative Delta, VWAP)
    - Footprint Charts
    - Multiple Trading Signal Types
    """
    )

    st.sidebar.title("Settings")

    st.sidebar.subheader("Primary Chart Timeframe")
    primary_timeframe_choice = st.sidebar.selectbox(
        "Select Primary Chart",
        ["M1 (1 minute)", "M5 (5 minutes)", "M15 (15 minutes)", "M30 (30 minutes)"],
        index=0,
        help="This timeframe will be shown in the main chart",
    )

    timeframe_config = {
        "M1 (1 minute)": {"freq": "1min", "periods": 1500, "label": "1m", "lookback": 50},
        "M5 (5 minutes)": {"freq": "5min", "periods": 500, "label": "5m", "lookback": 40},
        "M15 (15 minutes)": {"freq": "15min", "periods": 300, "label": "15m", "lookback": 30},
        "M30 (30 minutes)": {"freq": "30min", "periods": 200, "label": "30m", "lookback": 20},
    }

    tf_config = timeframe_config[primary_timeframe_choice]
    primary_freq = tf_config["freq"]
    primary_periods = tf_config["periods"]
    primary_label = tf_config["label"]
    zone_lookback = tf_config["lookback"]

    instrument_type = st.sidebar.selectbox(
        "Instrument Type",
        ["Forex", "Indices (US30, NAS100)"],
        index=0,
    )

    base_currency = None
    target_currency = None
    index_choice = None

    if instrument_type == "Forex":
        base_currency = st.sidebar.selectbox("Base Currency", ["USD", "EUR", "GBP"], index=0)
        target_currency = st.sidebar.selectbox("Target Currency", ["EUR", "USD", "GBP"], index=1)
    else:
        index_choice = st.sidebar.selectbox(
            "Index",
            ["US30 (Dow Jones)", "NAS100 (Nasdaq 100)"],
            index=0,
        )

    refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 15)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Options")
    show_supply_demand = st.sidebar.checkbox("Show Supply/Demand Zones", value=True)
    show_orderflow = st.sidebar.checkbox("Show Order Flow Signals", value=True)
    show_technical = st.sidebar.checkbox("Show Technical Signals", value=False)
    show_footprint = st.sidebar.checkbox("Show Footprint Chart", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("News and Sentiment")
    show_news = st.sidebar.checkbox("Analyze market news", value=True)
    default_news_key = os.getenv("NEWSDATA_API_KEY", os.getenv("NEWS_API_KEY", ""))
    news_api_key = st.sidebar.text_input(
        "NewsData.io API key (optional)", value=default_news_key, type="password"
    )
    news_language = st.sidebar.selectbox(
        "News language",
        ["en", "de", "fr", "es", "nl"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Real Forex Data (Alpha Vantage)")
    default_fx_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    fx_api_key_input = st.sidebar.text_input(
        "Alpha Vantage API key (optional)",
        value="" if default_fx_key else "",
        type="password",
    )

    fx_api_key = fx_api_key_input or default_fx_key

    if default_fx_key:
        st.sidebar.caption("Alpha Vantage key loaded from environment.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Test Zone Settings")
    enable_test_zone = st.sidebar.checkbox("Enable Test Zone (Fake Trading)", value=True)
    starting_balance = st.sidebar.number_input(
        "Starting Balance (fake money)",
        min_value=1000.0,
        max_value=1000000.0,
        value=10000.0,
        step=1000.0,
    )
    pip_value_money = st.sidebar.number_input(
        "Value per Pip (fake money)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Alert Settings")
    default_tg_token = os.getenv("TELEGRAM_TOKEN", TELEGRAM_TOKEN_DEFAULT)
    default_tg_chat = os.getenv("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID_DEFAULT)

    tg_token_input = st.sidebar.text_input(
        "Telegram bot token (optional)",
        value=default_tg_token,
        type="password",
    )
    tg_chat_input = st.sidebar.text_input(
        "Telegram chat ID (optional)",
        value=default_tg_chat,
    )

    st.session_state["TELEGRAM_TOKEN_UI"] = tg_token_input
    st.session_state["TELEGRAM_CHAT_ID_UI"] = tg_chat_input

    enable_alerts = st.sidebar.checkbox("Enable Telegram alerts for new signals", value=False)

    if st.sidebar.button("Send test message to Telegram"):
        send_telegram_alert("Test alert from your Streamlit dashboard")
        st.sidebar.success("Test message sent (if token/chat ID are correct).")

    placeholder = st.empty()

    with placeholder.container():
        if instrument_type == "Forex":
            instrument_label = f"{base_currency}/{target_currency}"
            pip_size = 0.0001
            rate = fetch_single_pair(base_currency, target_currency)
        else:
            if index_choice.startswith("US30"):
                instrument_label = "US30"
                pip_size = 1.0
                rate = 38000.0
            else:
                instrument_label = "NAS100"
                pip_size = 1.0
                rate = 17000.0

        news_articles = []
        news_sentiment = None
        if show_news and news_api_key:
            query = build_news_query(
                instrument_type,
                base_currency=base_currency,
                target_currency=target_currency,
                index_choice=index_choice,
            )
            news_articles = fetch_news_articles(query, news_api_key, language=news_language)
            news_sentiment = analyze_news_sentiment(news_articles)

        col1, col2, col3 = st.columns(3)

        if rate:
            col1.metric(label=f"{instrument_label}", value=f"{rate:.5f}")
        else:
            st.error("Failed to fetch price.")
            return

        df_primary = generate_historical_data(
            rate,
            primary_periods,
            primary_freq,
            base_currency=base_currency,
            target_currency=target_currency,
            fx_api_key=fx_api_key,
        )
        if df_primary.empty:
            st.warning("Unable to generate historical data.")
            return

        df_primary = add_technical_indicators(df_primary)
        df_primary = generate_order_flow_data(df_primary)

        df_idx = df_primary.set_index("Datetime")

        def _resample_ohlcv(frame, rule):
            res = frame.resample(rule).agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            ).dropna()
            res = res.reset_index()
            res = add_technical_indicators(res)
            res = generate_order_flow_data(res)
            return res

        higher_tfs = []
        df_5m = None
        df_15m = None
        df_30m = None

        if primary_label == "1m":
            df_5m = _resample_ohlcv(df_idx, "5min")
            df_15m = _resample_ohlcv(df_idx, "15min")
            df_30m = _resample_ohlcv(df_idx, "30min")
            higher_tfs = [("5m", df_5m), ("15m", df_15m), ("30m", df_30m)]
            df_for_zones = df_15m
        elif primary_label == "5m":
            df_15m = _resample_ohlcv(df_idx, "15min")
            df_30m = _resample_ohlcv(df_idx, "30min")
            higher_tfs = [("15m", df_15m), ("30m", df_30m)]
            df_for_zones = df_15m
        elif primary_label == "15m":
            df_30m = _resample_ohlcv(df_idx, "30min")
            higher_tfs = [("30m", df_30m)]
            df_for_zones = df_primary
        else:
            df_for_zones = df_primary

        supply_demand_zones = []
        if show_supply_demand:
            supply_demand_zones = identify_supply_demand_zones(df_for_zones, lookback=zone_lookback)

        current_delta = df_primary.iloc[-1]["Delta"]
        current_cum_delta = df_primary.iloc[-1]["Cumulative_Delta"]

        col2.metric(
            label="Current Delta",
            value=f"{current_delta:.0f}",
            delta=f"{'Buying' if current_delta > 0 else 'Selling'} Pressure",
        )
        col3.metric(
            label="Cumulative Delta",
            value=f"{current_cum_delta:.0f}",
            delta=f"{'Bullish' if current_cum_delta > 0 else 'Bearish'} Trend",
        )

        if show_news:
            if news_api_key and news_sentiment is not None:
                st.metric(
                    label="News Sentiment",
                    value=news_sentiment.get("label", "Neutral"),
                    delta=f"Score: {news_sentiment.get('score', 0.0):.2f}",
                )
            elif not news_api_key:
                st.info("Enter your NewsData.io key in the sidebar to analyze news.")

        all_signals = []
        m5_ms_signals = []
        m15_ms_signals = []
        m30_ms_signals = []

        if show_supply_demand and supply_demand_zones:
            sd_signals = generate_supply_demand_signals(
                df_primary, supply_demand_zones, pip_size=pip_size
            )
            for sig in sd_signals:
                sig["timeframe"] = primary_label
            all_signals.extend(sd_signals)

        if show_orderflow:
            orderflow_primary = detect_order_flow_signals(
                df_primary, timeframe_label=primary_label, pip_size=pip_size
            )
            all_signals.extend(orderflow_primary)

        for tf_label, tf_df in higher_tfs:
            if show_orderflow:
                orderflow_htf = detect_order_flow_signals(tf_df, timeframe_label=tf_label, pip_size=pip_size)
                all_signals.extend(orderflow_htf)

            if show_supply_demand and supply_demand_zones:
                if tf_label == "5m" and df_5m is not None:
                    ms_htf = generate_m5_market_structure_signals(
                        tf_df, supply_demand_zones, pip_size=pip_size
                    )
                elif tf_label == "15m" and df_15m is not None:
                    ms_htf = generate_m15_market_structure_signals(
                        tf_df, supply_demand_zones, pip_size=pip_size
                    )
                elif tf_label == "30m" and df_30m is not None:
                    ms_htf = generate_m30_market_structure_signals(
                        tf_df, supply_demand_zones, pip_size=pip_size
                    )
                else:
                    ms_htf = []

                if tf_label == "5m":
                    m5_ms_signals = ms_htf
                elif tf_label == "15m":
                    m15_ms_signals = ms_htf
                elif tf_label == "30m":
                    m30_ms_signals = ms_htf

                all_signals.extend(ms_htf)

        if show_technical:
            technical_signals = generate_sell_signals(df_primary, pip_size=pip_size)
            for sig in technical_signals:
                sig["timeframe"] = primary_label
            all_signals.extend(technical_signals)

        if show_news and news_sentiment is not None and all_signals:
            before_n = len(all_signals)
            all_signals = filter_signals_by_news(all_signals, news_sentiment)
            after_n = len(all_signals)
            removed = before_n - after_n
            if removed > 0:
                st.info(
                    f"{removed} signals filtered by news sentiment: {news_sentiment.get('label', '')}"
                )

        if enable_alerts and all_signals:
            signal_df_alert = pd.DataFrame(all_signals).copy()
            if "timestamp" in signal_df_alert.columns:
                signal_df_alert["timestamp"] = pd.to_datetime(signal_df_alert["timestamp"])

            alert_key = f"last_alert_ts::{instrument_label}"

            if alert_key not in st.session_state:
                if "timestamp" in signal_df_alert.columns and not signal_df_alert["timestamp"].empty:
                    st.session_state[alert_key] = signal_df_alert["timestamp"].max()
                else:
                    st.session_state[alert_key] = pd.Timestamp.utcnow()
                st.info("Telegram alerts enabled: only new signals will be sent.")
            else:
                last_ts = st.session_state[alert_key]

                new_mask = signal_df_alert["timestamp"] > last_ts
                new_signals = signal_df_alert[new_mask]

                if not new_signals.empty:
                    for _, sig in new_signals.sort_values("timestamp").iterrows():
                        tf = sig.get("timeframe", primary_label)
                        direction = sig.get("signal", "")
                        sig_type = sig.get("type", "")
                        price = sig.get("price", np.nan)
                        ts_str = sig.get("timestamp")

                        msg = f"{instrument_label} | {tf} {direction} @ {price:.5f} | {sig_type} | {ts_str}"
                        send_telegram_alert(msg)

                    st.success(f"{len(new_signals)} new signals sent as alerts.")

                    st.session_state[alert_key] = new_signals["timestamp"].max()

        if show_news:
            st.subheader("Latest market news for this instrument")
            if news_api_key and news_articles:
                for art in news_articles:
                    title = art.get("title", "(no title)")
                    desc = art.get("description") or ""
                    src = art.get("source") or ""
                    url = art.get("url") or ""
                    when = art.get("publishedAt") or ""

                    st.markdown(
                        f"**{title}**  \n"
                        f"{desc}  \n"
                        f"Source: {src} | {when}  \n"
                        f"[Open article]({url})"
                    )
            elif news_api_key and not news_articles:
                st.info("No relevant news articles found for this instrument.")
            elif not news_api_key:
                st.info("No NewsData.io key provided; news is not fetched.")

        if show_supply_demand and supply_demand_zones:
            st.subheader("Supply and Demand Zones")
            zone_col1, zone_col2 = st.columns(2)

            supply_zones = [z for z in supply_demand_zones if z["type"] == "Supply"]
            demand_zones = [z for z in supply_demand_zones if z["type"] == "Demand"]

            with zone_col1:
                st.markdown("### Supply Zones (Resistance)")
                if supply_zones:
                    for idx, zone in enumerate(supply_zones[:3]):
                        st.info(
                            f"Zone {idx + 1}: {zone['bottom']:.5f} - {zone['top']:.5f} | Strength: {zone['strength']}"
                        )
                else:
                    st.write("No supply zones detected")

            with zone_col2:
                st.markdown("### Demand Zones (Support)")
                if demand_zones:
                    for idx, zone in enumerate(demand_zones[:3]):
                        st.success(
                            f"Zone {idx + 1}: {zone['bottom']:.5f} - {zone['top']:.5f} | Strength: {zone['strength']}"
                        )
                else:
                    st.write("No demand zones detected")

        st.subheader("Trading Signals")
        if all_signals:
            signal_df = pd.DataFrame(all_signals)

            if "timeframe" not in signal_df.columns:
                signal_df["timeframe"] = "1m"

            preferred_order = ["1m", "5m", "15m", "30m"]
            unique_tfs = set(signal_df["timeframe"].dropna().unique()) | set(preferred_order)
            unique_tfs_sorted = sorted(
                unique_tfs,
                key=lambda x: preferred_order.index(x) if x in preferred_order else len(preferred_order),
            )

            tabs = st.tabs([f"{tf} Signals" for tf in unique_tfs_sorted])

            for tf, tab in zip(unique_tfs_sorted, tabs):
                with tab:
                    tf_df = signal_df[signal_df["timeframe"] == tf]

                    col1, col2 = st.columns(2)

                    buy_signals = tf_df[tf_df["signal"] == "Buy"]
                    sell_signals = tf_df[tf_df["signal"] == "Sell"]

                    with col1:
                        st.markdown("### Buy Signals")
                        if not buy_signals.empty:
                            st.dataframe(buy_signals, use_container_width=True)
                        else:
                            st.info("No buy signals")

                    with col2:
                        st.markdown("### Sell Signals")
                        if not sell_signals.empty:
                            st.dataframe(sell_signals, use_container_width=True)
                        else:
                            st.info("No sell signals")
        else:
            st.info("No signals generated based on current data.")

        any_ms = False
        if m5_ms_signals:
            any_ms = True
            st.subheader("M5 Market Structure / Supply-Demand Signals")
            ms5_df = pd.DataFrame(m5_ms_signals)
            st.dataframe(ms5_df, use_container_width=True)

        if m15_ms_signals:
            any_ms = True
            st.subheader("M15 Market Structure / Supply-Demand Signals")
            ms_df = pd.DataFrame(m15_ms_signals)
            st.dataframe(ms_df, use_container_width=True)

        if m30_ms_signals:
            any_ms = True
            st.subheader("M30 Market Structure / Supply-Demand Signals")
            ms30_df = pd.DataFrame(m30_ms_signals)
            st.dataframe(ms30_df, use_container_width=True)

        if not any_ms and show_supply_demand and supply_demand_zones:
            st.info("No M5/M15/M30 market-structure signals for the current data.")

        if enable_test_zone and all_signals:
            st.subheader("Test Zone: Backtest Signals with Fake Money")

            signal_df_full = pd.DataFrame(all_signals).copy()
            if "timestamp" in signal_df_full.columns:
                signal_df_full["timestamp"] = pd.to_datetime(signal_df_full["timestamp"])

            timeframe_to_df = {primary_label: df_primary}
            for tf_label, tf_df in higher_tfs:
                timeframe_to_df[tf_label] = tf_df

            results = []
            equity = starting_balance

            for _, sig in signal_df_full.iterrows():
                direction = sig.get("signal")
                entry_price = sig.get("price")
                ts = sig.get("timestamp")
                sl = sig.get("stop_loss")
                tp = sig.get("take_profit")

                result = "Open"
                exit_price = np.nan
                exit_time = pd.NaT
                pips = 0.0
                pnl = 0.0

                if pd.notna(entry_price) and pd.notna(ts) and pd.notna(sl) and pd.notna(tp):
                    timeframe_label = sig.get("timeframe", primary_label)
                    price_df = timeframe_to_df.get(timeframe_label, df_primary).sort_values("Datetime")

                    after_mask = price_df["Datetime"] >= ts
                    if after_mask.any():
                        idx_start = price_df.index[after_mask][0]

                        for j in range(idx_start + 1, len(price_df)):
                            bar = price_df.iloc[j]
                            bar_low = bar["Low"]
                            bar_high = bar["High"]
                            bar_time = bar["Datetime"]

                            if direction == "Buy":
                                sl_hit = bar_low <= sl
                                tp_hit = bar_high >= tp
                                if sl_hit:
                                    result = "Loss"
                                    exit_price = sl
                                    exit_time = bar_time
                                    break
                                if tp_hit:
                                    result = "Win"
                                    exit_price = tp
                                    exit_time = bar_time
                                    break
                            elif direction == "Sell":
                                sl_hit = bar_high >= sl
                                tp_hit = bar_low <= tp
                                if sl_hit:
                                    result = "Loss"
                                    exit_price = sl
                                    exit_time = bar_time
                                    break
                                if tp_hit:
                                    result = "Win"
                                    exit_price = tp
                                    exit_time = bar_time
                                    break

                        if result in ("Win", "Loss"):
                            move = (exit_price - entry_price) if direction == "Buy" else (entry_price - exit_price)
                            pips = move / pip_size
                            pnl = pips * pip_value_money
                            equity += pnl

                results.append(
                    {
                        "timestamp": ts,
                        "signal": direction,
                        "type": sig.get("type"),
                        "timeframe": sig.get("timeframe", "1m"),
                        "price": entry_price,
                        "stop_loss": sl,
                        "take_profit": tp,
                        "exit_time": exit_time,
                        "exit_price": exit_price,
                        "result": result,
                        "pips": pips,
                        "pnl": pnl,
                        "equity_after": equity,
                    }
                )

            results_df = pd.DataFrame(results)

            if not results_df.empty:
                wins = (results_df["result"] == "Win").sum()
                losses = (results_df["result"] == "Loss").sum()
                opens = (results_df["result"] == "Open").sum()
                total_pips = results_df["pips"].sum()

                st.markdown(
                    f"**Wins:** {wins} | **Losses:** {losses} | **Open:** {opens} | **Total Pips:** {total_pips:.1f}"
                )
                st.markdown(
                    f"**Starting Balance:** {starting_balance:.2f} -> **Final Balance:** {equity:.2f}"
                )

                def highlight_result(row):
                    if row["result"] == "Win":
                        color = "background-color: rgba(0, 150, 0, 0.6); color: white;"
                    elif row["result"] == "Loss":
                        color = "background-color: rgba(200, 0, 0, 0.7); color: white;"
                    else:
                        color = ""
                    return [color] * len(row)

                preferred_order = ["1m", "5m", "15m", "30m"]
                unique_tfs = set(results_df["timeframe"].dropna().unique()) | set(preferred_order)
                unique_tfs_sorted = sorted(
                    unique_tfs,
                    key=lambda x: preferred_order.index(x) if x in preferred_order else len(preferred_order),
                )

                tabs = st.tabs([f"{tf} Results" for tf in unique_tfs_sorted])

                for tf, tab in zip(unique_tfs_sorted, tabs):
                    with tab:
                        tf_df = results_df[results_df["timeframe"] == tf]

                        if tf_df.empty:
                            st.info(f"No trades for {tf} timeframe.")
                        else:
                            wins_tf = (tf_df["result"] == "Win").sum()
                            losses_tf = (tf_df["result"] == "Loss").sum()
                            opens_tf = (tf_df["result"] == "Open").sum()
                            total_pips_tf = tf_df["pips"].sum()

                            st.markdown(
                                f"**{tf} Wins:** {wins_tf} | **Losses:** {losses_tf} | **Open:** {opens_tf} | **Total Pips:** {total_pips_tf:.1f}"
                            )

                            st.dataframe(
                                tf_df.style.apply(highlight_result, axis=1),
                                use_container_width=True,
                            )

        st.subheader(f"{instrument_label} - {primary_label.upper()} Chart with Multi-Timeframe Analysis")

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("Price Action with Supply/Demand Zones", "Volume and Delta", "Cumulative Delta"),
        )

        fig.add_trace(
            go.Candlestick(
                x=df_primary["Datetime"],
                open=df_primary["Open"],
                high=df_primary["High"],
                low=df_primary["Low"],
                close=df_primary["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        if show_supply_demand and supply_demand_zones:
            for zone in supply_demand_zones:
                zone_start = df_primary["Datetime"].iloc[0]
                zone_end = df_primary["Datetime"].iloc[-1]

                color = "rgba(255, 0, 0, 0.2)" if zone["type"] == "Supply" else "rgba(0, 255, 0, 0.2)"

                fig.add_shape(
                    type="rect",
                    x0=zone_start,
                    x1=zone_end,
                    y0=zone["bottom"],
                    y1=zone["top"],
                    fillcolor=color,
                    line=dict(color=color.replace("0.2", "0.5"), width=1),
                    layer="below",
                    row=1,
                    col=1,
                )

        fig.add_trace(
            go.Scatter(
                x=df_primary["Datetime"],
                y=df_primary["SMA_50"],
                mode="lines",
                name="SMA-50",
                line=dict(color="blue", width=1),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_primary["Datetime"],
                y=df_primary["VWAP"],
                mode="lines",
                name="VWAP",
                line=dict(color="purple", width=1, dash="dash"),
            ),
            row=1,
            col=1,
        )

        if all_signals:
            signal_df = pd.DataFrame(all_signals)
            buy_sigs = signal_df[signal_df["signal"] == "Buy"]
            sell_sigs = signal_df[signal_df["signal"] == "Sell"]

            if not buy_sigs.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_sigs["timestamp"],
                        y=buy_sigs["price"],
                        mode="markers",
                        name="Buy Signals",
                        marker=dict(color="lime", size=12, symbol="triangle-up"),
                        text=buy_sigs["type"],
                        hovertemplate="<b>%{text}</b><br>Price: %{y}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

            if not sell_sigs.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_sigs["timestamp"],
                        y=sell_sigs["price"],
                        mode="markers",
                        name="Sell Signals",
                        marker=dict(color="red", size=12, symbol="triangle-down"),
                        text=sell_sigs["type"],
                        hovertemplate="<b>%{text}</b><br>Price: %{y}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

            for _, row_sig in signal_df.iterrows():
                ts = row_sig["timestamp"]
                entry = row_sig["price"]
                tp = row_sig.get("take_profit")
                sl = row_sig.get("stop_loss")
                color = "lime" if row_sig["signal"] == "Buy" else "red"

                if tp is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[ts, ts],
                            y=[entry, tp],
                            mode="lines",
                            line=dict(color=color, width=2, dash="dash"),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=1,
                    )

                if sl is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[ts, ts],
                            y=[entry, sl],
                            mode="lines",
                            line=dict(color="gray", width=2, dash="dot"),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=1,
                        col=1,
                    )

        fig.add_trace(
            go.Bar(
                x=df_primary["Datetime"],
                y=df_primary["Buy_Volume"],
                name="Buy Volume",
                marker=dict(color="green"),
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=df_primary["Datetime"],
                y=df_primary["Sell_Volume"],
                name="Sell Volume",
                marker=dict(color="red"),
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=df_primary["Datetime"],
                y=df_primary["Delta"],
                mode="lines",
                name="Delta",
                line=dict(color="yellow", width=2),
                yaxis="y2",
            ),
            row=2,
            col=1,
        )

        colors = ["green" if val > 0 else "red" for val in df_primary["Cumulative_Delta"]]
        fig.add_trace(
            go.Bar(
                x=df_primary["Datetime"],
                y=df_primary["Cumulative_Delta"],
                name="Cumulative Delta",
                marker=dict(color=colors),
            ),
            row=3,
            col=1,
        )

        fig.update_layout(
            height=900,
            template="plotly_dark",
            showlegend=True,
            xaxis_rangeslider_visible=False,
        )

        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1, tickformat=".5f")
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Delta", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        if show_footprint:
            st.subheader("Footprint Chart (Latest Bar)")
            latest_bar_idx = len(df_primary) - 1
            footprint_df = generate_footprint_data(df_primary, latest_bar_idx)

            if not footprint_df.empty:
                footprint_fig = plot_footprint_chart(footprint_df, df_primary.iloc[latest_bar_idx]["Datetime"])
                st.plotly_chart(footprint_fig, use_container_width=True)

                st.markdown("**Volume Profile Analysis:**")
                col1, col2, col3 = st.columns(3)

                max_buy_idx = footprint_df["Buy_Volume"].idxmax()
                max_sell_idx = footprint_df["Sell_Volume"].idxmax()
                poc_idx = footprint_df["Total_Volume"].idxmax()

                col1.metric("Max Buy Level", f"{footprint_df.loc[max_buy_idx, 'Price']:.5f}")
                col2.metric("Max Sell Level", f"{footprint_df.loc[max_sell_idx, 'Price']:.5f}")
                col3.metric("POC (Point of Control)", f"{footprint_df.loc[poc_idx, 'Price']:.5f}")


if __name__ == "__main__":
    main()
