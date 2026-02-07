# 📊 Forex Trading Dashboard

Advanced Supply/Demand Zone Trading Dashboard with Order Flow Analysis

## 🚀 Features

- 📈 Real-time Forex Data (USD/EUR/GBP pairs)
- 🎯 Supply & Demand Zone Detection
- 📊 Order Flow Analysis (Delta, Cumulative Delta, VWAP)
- 👣 Footprint Charts
- 💡 Multiple Trading Signal Types (1m, 5m, 15m timeframes)
- 🧪 Backtest Zone with Fake Money

## 🌐 Live Demo

Access the app at: `https://forex-trading-dashboard.streamlit.app` (after deployment)

## 🛠️ Local Setup

### Prerequisites
- Python 3.8 or higher

### Installation

1. Clone the repository:
```bash
git clone https://github.com/delvechioblackson-dev/forex-trading-dashboard.git
cd forex-trading-dashboard
```

2. Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run test.py
```

5. Open your browser to `http://localhost:8501`

## 📱 Access from Phone

Once deployed to Streamlit Cloud, you can access the dashboard from any device using the deployment URL.

## ⚙️ Configuration

- **Instrument Types**: Forex (USD/EUR/GBP) or Indices (US30, NAS100)
- **Timeframes**: 1m, 5m, 15m analysis
- **Analysis Options**: Supply/Demand Zones, Order Flow Signals, Technical Signals
- **Test Zone**: Backtest signals with fake money

## 🔑 API Key

The app uses FastForex API for live forex data. Current API key is included for demo purposes.

## 📄 License

MIT License

## 👤 Author

delvechioblackson-dev
