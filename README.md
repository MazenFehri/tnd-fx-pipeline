# USD/TND Daily FX Prediction Pipeline

This project maintains a SQLite database of historical FX rates and BCT USD/TND fixings, runs a **basket OLS regression with rolling 90-day weights + Kalman filter on the IB-fixing spread**, and generates daily predictions with Excel reports, optional Telegram notifications, and a Streamlit dashboard.

**Tech Stack (100% Free):**
- **Data Sources**: exchangerate.host, frankfurter.app, ExchangeRate-API v6 (optional), BCT website scraping
- **Storage**: SQLite database (`tnd.db`)
- **Libraries**: NumPy, Pandas, Requests, OpenPyXL, Streamlit, Plotly, SciPy, BeautifulSoup4
- **APIs**: Telegram Bot API (optional)

## Project Overview

The pipeline consists of 12 Python files that work together to:
1. Fetch daily FX rates from free APIs
2. Clean and compute log-returns
3. Train OLS models with rolling weights
4. Apply Kalman filter to interbank spreads
5. Generate predictions and export to Excel
6. Send notifications via Telegram
7. Display results in an interactive dashboard

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize Database
```bash
python init_db.py
```

### 3. Seed Historical Data
The model requires at least 90 days of historical data with BCT fixings.

```bash
python seed_db.py
```

This imports FX rates and fixings from `data/FX-CLEAN-Data.csv` and optionally IB rates from `data/ib.csv` (if available).

### 5. Run the Pipeline
```bash
python run_pipeline.py
```

This automatically fetches today's BCT USD/TND fixing from the BCT website. Optional environment variables:
- `EXCHANGERATE_API_KEY`: For premium FX API access
- `BCT_FIX_MID`: Override today's BCT fixing (manual)
- `TELEGRAM_BOT_TOKEN`: For notifications
- `TELEGRAM_CHAT_ID`: Telegram chat ID

### 6. Launch Dashboard
```bash
streamlit run dashboard.py
```

## Pipeline Execution Flow

When you run `python run_pipeline.py`, here's what happens step-by-step:

### 1. **Scrape BCT Website** → Get Latest Fixing
- Automatically fetches today's official USD/TND fixing from the BCT website
- Uses BeautifulSoup to parse the exchange rates table
- Extracts the USD rate (currently 2.9442 TND = 1 USD)
- Falls back to environment variable `BCT_FIX_MID` if scraping fails

### 2. **Fetch FX Rates** → EUR/USD, GBP/USD, USD/JPY
- Queries multiple free APIs in priority order:
  1. ExchangeRate-API v6 (with optional API key)
  2. exchangerate.host
  3. frankfurter.app
- Gets latest rates for EUR/USD, GBP/USD, and USD/JPY
- Stores all data in SQLite database

### 3. **Generate Predictions** → Basket Model + Kalman Filter
- **Data Preparation**: Calculates log-returns and spreads from historical data
- **Basket Model**: OLS regression with rolling 90-day weights on EUR/USD, GBP/USD, USD/JPY
- **Kalman Filter**: Applied to interbank spreads for additional adjustment
- **Predictions**: 
  - `intrinsic_v1`: Basket model prediction
  - `intrinsic_v2`: Basket + Kalman adjustment
- Uses previous day's BCT fixing as the base for exponential growth calculation

### 4. **Create Excel Report** → Full Analysis
- Generates `reports/tnd_report_YYYY-MM-DD.xlsx` with:
  - Today's predictions and metrics
  - Historical data and trends
  - Model weights and R-squared
  - Kalman spread analysis
- Includes charts and formatted tables

### 5. **Optional Notifications** → Telegram
- Sends prediction summary via Telegram (if configured)
- Requires `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` environment variables

## Automatic BCT Fixing

The pipeline automatically scrapes today's USD/TND fixing from the BCT (Banque Centrale de Tunisie) website at https://www.bct.gov.tn/bct/siteprod/cours.jsp?lang=en. This ensures the model uses the latest official fixing without manual intervention.

- **How it works**: The `fetch_daily.py` script uses BeautifulSoup to parse the BCT's exchange rates table and extract the USD value.
- **Fallback**: If scraping fails, it falls back to the `BCT_FIX_MID` environment variable (if set).
- **Override**: You can manually set `BCT_FIX_MID` to override the scraped value for testing or corrections.

## File Descriptions

| File | Description |
|------|-------------|
| `init_db.py` | Creates SQLite tables for fx_rates and predictions |
| `seed_db.py` | Imports historical FX, BCT fixings, and IB rates from CSV files |
| `fetch_daily.py` | Fetches EUR/USD, GBP/USD, USD/JPY from APIs; scrapes BCT USD/TND fixing |
| `clean_returns.py` | Computes log-returns and spreads |
| `model.py` | OLS regression, rolling weights, Kalman filter |
| `predict.py` | Generates daily predictions |
| `export_excel.py` | Creates formatted Excel reports |
| `notify_telegram.py` | Sends prediction alerts |
| `run_pipeline.py` | Orchestrates the full daily process |
| `dashboard.py` | Streamlit web app for visualization |
| `requirements.txt` | Python dependencies |
| `README.md` | This documentation |

## Data Files

- `data/tnd.db`: SQLite database with historical data and predictions
- `data/FX-CLEAN-Data.csv`: Historical FX rates and BCT fixings
- `reports/`: Generated Excel reports
- `data/ib.csv`: Interbank rates (optional)

## Usage

### Daily Operation
Run `python run_pipeline.py` daily to:
- Fetch latest FX rates
- Update database
- Generate predictions
- Export Excel report
- Send Telegram notification (if configured)

### Dashboard Features
- Metric cards: Today's prediction, previous fixing, change %
- Line chart: BCT fix vs intrinsic predictions
- Rolling weights plot
- Spread analysis
- Data table with latest predictions

## Cost

**$0** — All components are free:
- Public APIs (exchangerate.host, frankfurter.app)
- SQLite (no server costs)
- Streamlit (local dashboard)
- Telegram Bot API
- Python libraries (open-source)

## Troubleshooting

- **No predictions**: Ensure at least 90 days of historical data with fix_mid
- **Empty plots**: Run backfill if needed (historical predictions)
- **API failures**: Pipeline tries multiple sources automatically
- **Kalman spread = 0**: IB rates not imported; model falls back to basket-only

## License

This project is for educational and research purposes. Use responsibly.
|------|------|
| `init_db.py` | Create `fx_rates` and `predictions` tables |
| `fetch_daily.py` | Pull USD/EUR, USD/GBP, USD/JPY; upsert SQLite |
| `clean_returns.py` | Build modeling frame from DB |
| `model.py` | OLS, rolling regression, Kalman (numpy only) |
| `predict.py` | Daily prediction + insert into `predictions` |
| `export_excel.py` | `reports/tnd_report_YYYY-MM-DD.xlsx` |
| `notify_telegram.py` | Optional Telegram message |
| `run_pipeline.py` | Orchestrator for CI and local runs |
| `dashboard.py` | Streamlit + Plotly UI |
