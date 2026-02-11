# Forex Trading Dashboard (Streamlit)

This repo contains a Streamlit app for multi-timeframe supply/demand analysis, order flow signals, and optional news sentiment.

## Run locally

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Start the app:

```
streamlit run test.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and create a new app.
3. Select this repo and set the entrypoint to `test.py`.
4. Set secrets (recommended) in Streamlit Cloud:

```
FASTFOREX_API_KEY = "your_fastforex_key"
ALPHAVANTAGE_API_KEY = "your_alpha_vantage_key"
NEWSDATA_API_KEY = "your_newsdata_key"
TELEGRAM_TOKEN = "your_telegram_bot_token"
TELEGRAM_CHAT_ID = "your_telegram_chat_id"
```

You can also set these as environment variables in your deployment settings.

## Notes

- If `FASTFOREX_API_KEY` is not set, live FX prices will not load.
- Alpha Vantage is optional. If missing, the app uses simulated candles.
- Telegram alerts are optional and only sent when enabled in the sidebar.
