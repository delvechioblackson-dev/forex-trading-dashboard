# Forex Trading Dashboard (Streamlit)

This repo contains a Streamlit app for multi-timeframe supply/demand analysis, order flow signals, and optional news sentiment.

It also includes an **Energie Leads Dashboard** (`energie_leads.py`) — a Dutch-language tool to help reach potential customers for home batteries (thuisbatterijen), solar panels (zonnepanelen), and EV charging stations (laadpalen).

## Run locally

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Start the Forex Trading Dashboard:

```
streamlit run test.py
```

3. Start the Energie Leads Dashboard:

```
streamlit run energie_leads.py
```

## Energie Leads Dashboard

`energie_leads.py` answers the question: **"Hoe kan ik het snelste mensen bereiken voor het plaatsen van thuisbatterijen, zonnepanelen en laadpalen?"**

It provides four pages:

| Pagina | Beschrijving |
|--------|-------------|
| 📡 **Marketingkanalen** | Vergelijking van 8 kanalen op snelheid, kosten, conversie en bereik — gesorteerd op snelheid van eerste leads |
| 💶 **ROI-calculator** | Berekening van verwachte leads, opdrachten en ROI per kanaal op basis van uw eigen budget en marges |
| 📋 **Lead Management** | Vastleggen en bijhouden van leads (naam, telefoon, product, herkomstkanaal, status) |
| 🚀 **Actieplan** | Week-voor-week actieplan om zo snel mogelijk te starten met leadgeneratie |

No external API keys are required to run this module.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and create a new app.
3. Select this repo and set the entrypoint to `test.py` (Forex dashboard) or `energie_leads.py` (Energie Leads Dashboard).
4. For the Forex dashboard, set secrets (recommended) in Streamlit Cloud:

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
- The Energie Leads Dashboard requires no API keys and works fully offline.
