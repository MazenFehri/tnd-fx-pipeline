# USD/TND — Real-Time Intrinsic Value Model

A research-grade Python pipeline that computes a **real-time intrinsic value for the USD/TND exchange rate**, combining a fundamental basket of global currency moves with a stochastic adjustment for local Tunisian liquidity conditions.

Produced for **IN 22-21 — Time Series Analysis** (Dr Eymen Errais).

---

## What it does

For every minute of the trading day the pipeline computes:

```
intrinsic = (BCT_anchor_fix) · exp(w₁·ΔEUR/USD + w₂·ΔGBP/USD + w₃·ΔUSD/JPY)
            + Kalman_filtered_spread(IB_rate − BCT_fix)
```

- **Basket baseline** — rolling 90-day OLS on log-returns of EUR/USD, GBP/USD, USD/JPY, with Newey-West HAC standard errors.
- **Liquidity adjustment** — AR(1) Kalman filter on the interbank-vs-fixing spread, parameters `(c, φ, Q, R)` jointly calibrated by maximum likelihood. A 2-regime Markov-switching extension is also fit.
- **Macro overlay** — augments the spread regression with Brent, broad-USD (DXY) and VIX; finds DXY significant at p<0.001.

The output is a continuously updating **fair-value estimate** for USD/TND, plus a **premium/discount in basis points** that flags how rich or cheap the dinar trades versus the model.

---

## At a glance

| Layer | Tech |
|---|---|
| Language | Python 3.11 |
| Store | SQLite (`data/tnd.db`) — 5 tables |
| Modelling | NumPy, SciPy, statsmodels (optional), python-docx |
| Backend | FastAPI + Uvicorn |
| Frontend | Single-file SPA: Tailwind CDN + Alpine.js + Plotly — no build step |
| AI assistant | Groq (Llama 3.3 70B) or Anthropic (Claude Haiku 4.5) |
| Data sources | ExchangeRate-API v6, exchangerate.host, frankfurter.app, yfinance (intraday), FRED + Yahoo Finance fallback (macro) |
| Notifications | Telegram Bot API (optional) |

100% free / zero-cost stack — works without any paid API.

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │  Daily pipeline (run_pipeline)  │
                        │    fetch → predict → Excel      │
                        │      → Telegram → backtest      │
                        └─────────────┬───────────────────┘
                                      │
                                      ▼
              ┌─────────────────────────────────────────────┐
              │      SQLite (data/tnd.db) — 5 tables        │
              │ fx_rates · predictions · fx_intraday        │
              │ intrinsic_intraday · macro                  │
              └────────────┬──────────────────┬─────────────┘
                           │                  │
                  ┌────────▼──────┐   ┌───────▼────────────┐
                  │  Realtime     │   │  FastAPI (serve)   │
                  │  tick loop    │   │  JSON API + SPA    │
                  │  yfinance 1m  │   │  + LLM assistant   │
                  └───────────────┘   └────────────────────┘
```

---

## Quick start

```powershell
# 1. Install deps
pip install -r requirements.txt
pip install python-docx fastapi uvicorn pydantic yfinance statsmodels

# 2. Initialize the database (idempotent)
python init_db.py

# 3. Seed historical data (one-off)
python seed_db.py

# 4. Optional: configure secrets
copy .env.example .env
# edit .env — paste your Groq API key, Telegram tokens, etc.

# 5. Run the daily pipeline
python run_pipeline.py

# 6. Run the backtest
python backtest.py

# 7. Generate the Word-doc report
python build_report.py

# 8. Launch the dashboard
python serve.py
# → http://127.0.0.1:8000
```

For minute-resolution intraday operation:

```powershell
python run_realtime.py --interval 60      # tick every minute
python run_realtime.py --once             # single tick (cron mode)
```

---

## Project structure

```
tnd-fx-pipeline/
├── run_pipeline.py        # daily orchestrator: fetch → predict → Excel → notify → backtest
├── run_realtime.py        # intraday loop runner
├── realtime.py            # tick engine — yfinance + Kalman forward-propagation
├── fetch_daily.py         # FX APIs (3-source fallback) + BCT AM/PM fixing scrape
├── fetch_macro.py         # FRED + Yahoo fallback (Brent / DXY / VIX)
├── clean_returns.py       # log-returns + IB-Fix spread (PM-preferred)
├── model.py               # OLS w/ Newey-West HAC, rolling weights, MLE Kalman
├── msk.py                 # 2-regime Markov-switching Kalman (Kim filter + MLE)
├── macro_overlay.py       # Nested OLS leaderboard with macro covariates
├── predict.py             # daily prediction; writes to predictions table
├── backtest.py            # walk-forward MAE/RMSE/MAPE/DA, DM, Ljung-Box, ADF/KPSS
├── export_excel.py        # 5-sheet workbook with charts
├── build_report.py        # python-docx generator → reports/*.docx
├── notify_telegram.py     # optional alert at end of daily run
├── serve.py               # FastAPI backend — JSON API + LLM chat + SPA mount
├── static/index.html      # Perplexity-style dashboard, no build step
├── dashboard.py           # legacy Streamlit dashboard (superseded)
├── init_db.py             # idempotent schema + forward-only migrations
├── seed_db.py             # CSV bootstrap (FX-CLEAN-Data.csv + ib.csv)
├── data/
│   ├── tnd.db
│   ├── FX-CLEAN-Data.csv
│   └── ib.csv
├── reports/
│   ├── TND_Intrinsic_Value_Report.docx
│   ├── tnd_report_YYYY-MM-DD.xlsx
│   ├── backtest_metrics.json
│   ├── backtest_trace.csv
│   └── macro_overlay.json
├── CONTEXT.md             # AI-handoff context file
├── Project - TSA FV.pdf   # course specification
├── .env.example           # template — copy to .env
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Database schema

```sql
CREATE TABLE fx_rates (
    date TEXT PRIMARY KEY,
    eurusd REAL, gbpusd REAL, usdjpy REAL,
    fix_am REAL,           -- BCT morning fixing
    fix_pm REAL,           -- BCT evening fixing (closing reference)
    fix_mid REAL,          -- avg(am, pm) or whichever exists
    ib_rate REAL,          -- interbank USD/TND (T-1 lag)
    created_at TEXT
);

CREATE TABLE predictions (
    date TEXT PRIMARY KEY,
    intrinsic_v1 REAL,     -- basket baseline (anchor_fix · exp(basket_ret))
    intrinsic_v2 REAL,     -- v1 + Kalman spread (the headline number)
    w_eurusd REAL, w_gbpusd REAL, w_usdjpy REAL,
    kf_spread REAL,
    created_at TEXT
);

CREATE TABLE fx_intraday (
    ts TEXT PRIMARY KEY,   -- ISO8601 UTC, minute resolution
    eurusd REAL, gbpusd REAL, usdjpy REAL,
    source TEXT, created_at TEXT
);

CREATE TABLE intrinsic_intraday (
    ts TEXT PRIMARY KEY,
    anchor_date TEXT, anchor_fix REAL,
    basket_ret REAL, intrinsic_v1 REAL,
    kf_state REAL, kf_sigma REAL, intrinsic_v2 REAL,
    created_at TEXT
);

CREATE TABLE macro (
    date TEXT PRIMARY KEY,
    brent REAL, dxy REAL, vix REAL,
    source TEXT, created_at TEXT
);
```

---

## API reference (FastAPI)

| Endpoint | Returns |
|---|---|
| `GET /api/snapshot` | latest fix, IB, intrinsic, premium bps, freshness pill |
| `GET /api/timeseries?days=N` | daily history merged with predictions |
| `GET /api/intraday?hours=N` | minute-resolution intrinsic values |
| `GET /api/weights?days=N` | rolling basket weights history |
| `GET /api/backtest` | walk-forward metrics + spread stationarity tests |
| `GET /api/residuals?days=N` | `fix_mid − intrinsic_v2` over time |
| `POST /api/chat` | LLM analyst (Groq or Anthropic, grounded on snapshot) |
| `GET /` | the SPA |

All endpoints are cached server-side for 30 seconds.

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and fill in:

```ini
# Provide ONE of these to enable the on-board LLM analyst:
GROQ_API_KEY=gsk_...
# ANTHROPIC_API_KEY=sk-ant-...

# Optional — daily Telegram push notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Optional — manual BCT fixings (skips the live scrape)
BCT_FIX_AM=
BCT_FIX_PM=

# Optional — your own ExchangeRate-API v6 key
EXCHANGERATE_API_KEY=
```

The `.env` file is git-ignored; never commit it.

---

## Key empirical findings

Backtested over 1262 out-of-sample predictions (≈5 years):

| Metric | Model V2 | Random Walk |
|---|---:|---:|
| MAE | 0.01630 | **0.00730** |
| RMSE | 0.02436 | **0.01030** |
| MAPE | 0.54% | — |
| Directional Accuracy | 51.3% | — |
| Out-of-sample R² | **0.971** | — |

- The model achieves a high R² in *levels* but is **beaten by a one-day random walk** in absolute error (DM stat = −12.3, p ≈ 1e-35).
- Residuals are strongly autocorrelated (Ljung-Box Q(10) = 6964, p ≈ 0) — significant structure remains uncaptured.
- The spread is borderline non-stationary (ADF p = 0.052; KPSS rejects).
- None of the basket coefficients is statistically significant after Newey-West correction — consistent with the BCT actively managing the fixing rather than freely floating against a basket.
- **Macro overlay** finds the broad-dollar index (DXY) significant at p<0.001 as a driver of the IB-Fix spread (M3 adds +1.83% R² over M2).
- **Markov-switching Kalman** improves log-likelihood by 184 over single-regime with 6 extra parameters — strongly rejects the single-regime null.

The full quantitative tables, mathematical derivations, and discussion live in `reports/TND_Intrinsic_Value_Report.docx`.

---

## Features

- ✅ AM/PM BCT fixing schema with UTC-hour slot detection
- ✅ Real-time minute-resolution intraday engine (yfinance)
- ✅ Forward-propagated Kalman state with ±2σ confidence band
- ✅ Joint MLE calibration of `(c, φ, Q, R)`
- ✅ Newey-West HAC inference on basket coefficients
- ✅ Walk-forward backtest with DM, Ljung-Box, ADF/KPSS
- ✅ Two-regime Markov-switching Kalman extension
- ✅ Macro-overlay regression (Brent, DXY, VIX) with incremental R²
- ✅ FastAPI dashboard with Perplexity-style dark UI
- ✅ Embedded LLM analyst grounded on live snapshot
- ✅ 5-sheet Excel report (Daily, History, OLS Diagnostics, Rolling weights, Backtest)
- ✅ Auto-generated Word report (12 sections, fully populated from current DB)
- ✅ Optional Telegram push notifications

---

## Honest limitations

- Historical rows pre-date the AM/PM split — `fix_am` / `fix_pm` are populated only going forward.
- The intraday engine depends on yfinance, which has rate limits and weekend gaps.
- FRED is blocked on some networks; the fetcher falls back to Yahoo Finance automatically.
- The IB rate is published with a one-day lag — the entire reason the model exists. The nowcasting handles this via forward-propagation; no exact intraday IB is observable.
- BCT's managed-float behaviour means linear basket coefficients are mostly insignificant; the high OOS R² comes from slow-moving levels, not predictive power on returns.

---

## License

This project is educational coursework. Free to read, fork, and adapt for non-commercial purposes.

---

## Acknowledgements

- **Dr Eymen Errais** — project supervisor, IN 22-21
- **BCT (Banque Centrale de Tunisie)** — fixing publications
- **FRED (St Louis Fed)** — macro covariates
- **ExchangeRate-API, exchangerate.host, frankfurter.app, Yahoo Finance, Stooq** — free FX data
- **Anthropic & Groq** — LLM inference for the on-board analyst
