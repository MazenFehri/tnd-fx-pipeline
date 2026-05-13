# USD/TND Real-Time Intrinsic Value Model — AI Handoff Context

> Paste this entire file into a fresh AI session to get full project context. No prior conversation needed.

---

## 1. What this project is

A Python pipeline that estimates a **real-time intrinsic value** for the USD/TND (Tunisian Dinar) exchange rate, blending:
1. A **basket-based baseline** built from EUR/USD, GBP/USD, USD/JPY moves.
2. A **stochastic local-liquidity adjustment** that captures the gap between the official BCT fixing and the interbank (IB) market rate.

It is the deliverable for **IN 22-21 — Time Series Analysis** (Instructor: Dr Eymen Errais), titled *"Real-Time Intrinsic TND valuation model"*.

---

## 2. Why it matters (problem statement)

- The Central Bank of Tunisia (**BCT**) publishes an **official USD/TND fixing twice per day** (morning + evening).
- The actual interbank (**IB**) USD/TND rate — the true local market clearing price — is published **with a 1-day lag**.
- The market deviates from the basket-implied "fair" value due to **local FX liquidity pressure** (USD scarcity → premium; surplus → discount).
- We need a model that **(a)** continuously prices what TND *should* trade at given global FX moves, and **(b)** nowcasts the local liquidity adjustment despite the IB data lag.

Headline equation from the spec:

```
USD/TND = (Latest fixing) × (w1·%ΔEUR/USD + w2·%ΔGBP/USD + w3·%ΔUSD/JPY)  +  liquidity_adjustment
```

`w_i` and the functional form are to be estimated; the liquidity adjustment is a stochastic process (mean-reverting / Kalman / ECM).

---

## 3. Project goals (from the PDF spec)

The professor explicitly asks for **a Word doc + Excel sheet (and/or Python code)** delivering:

1. **Model framework design** — written design document, math formulation, real-time operation plan.
2. **Basket weight estimation** — fit `f(EUR/USD, GBP/USD, USD/JPY) → USD/TND`, justify methodology, report goodness-of-fit.
3. **Interbank rate nowcasting** — algorithm that infers today's IB rate / liquidity deviation despite the 1-day lag.
4. **Stochastic adjustment integration** — wire the nowcast into the baseline; handle volatility / sudden jumps.
5. **Real-time intrinsic value output** — prototype that ingests live data and emits the intrinsic rate.
6. **Performance evaluation & backtesting** — out-of-sample tests, error stats, correlation, charts, failure-mode discussion.

Two research questions called out in the spec:
- Which fixing (today AM, today PM, yesterday PM) anchors the basket and the spread?
- How does the model handle **intraday regime shifts** (AM vs PM volatility)?

---

## 4. Tech stack

- **Python 3.11**
- numpy, pandas, scipy (only scipy is in requirements; current code is numpy-only for the model)
- requests, beautifulsoup4 — FX APIs + BCT scrape
- sqlite3 — local store (`data/tnd.db`)
- openpyxl — Excel report
- streamlit + plotly — read-only dashboard
- Telegram Bot API — optional alerts
- **Free / zero-cost** sources only: `exchangerate-api.com` v6, `exchangerate.host`, `frankfurter.app`, BCT website scrape

---

## 5. Repo layout

```
tnd-fx-pipeline/
├── run_pipeline.py        # daily orchestrator: init → fetch → predict → Excel → notify
├── run_realtime.py        # intraday loop runner (calls realtime.tick on a cadence)
├── realtime.py            # intraday tick engine (1-min yfinance bars + Kalman forward-prop)
├── fetch_daily.py         # FX APIs (3-source fallback) + BCT AM/PM fixing scrape/env
├── fetch_macro.py         # FRED CSV: Brent (DCOILBRENTEU) + DXY (DTWEXBGS) + VIX (VIXCLS)
├── clean_returns.py       # log-returns; spread = ib_rate − fix_pm (PM-preferred, Fix_Mid fallback)
├── model.py               # OLS w/ Newey-West HAC, rolling 90d OLS, AR(1) Kalman, MLE calibration
├── msk.py                 # 2-regime Markov-switching Kalman (Kim filter + joint MLE)
├── macro_overlay.py       # Nested OLS leaderboard M0..M3 with macro covariates + incremental R²
├── predict.py             # predict_today() / predict_for_date(); writes predictions
├── backtest.py            # walk-forward MAE/RMSE/MAPE/DA, DM-test, Ljung-Box, ADF/KPSS
├── export_excel.py        # workbook: Daily / History / OLS Diagnostics / Rolling weights / Backtest
├── dashboard.py           # Streamlit dashboard (legacy — superseded by FastAPI app)
├── serve.py               # FastAPI backend serving JSON API + SPA at http://127.0.0.1:8000
├── static/index.html      # SPA: Tailwind CDN + Alpine.js + Plotly (no build step)
├── build_report.py        # python-docx generator → reports/TND_Intrinsic_Value_Report.docx
├── notify_telegram.py     # optional alert
├── init_db.py             # idempotent schema + forward-only ALTER TABLE migrations
├── seed_db.py             # CSV bootstrap (FX-CLEAN-Data.csv + ib.csv)
├── data/
│   ├── tnd.db             # SQLite store (fx_rates, predictions, fx_intraday, intrinsic_intraday, macro)
│   ├── FX-CLEAN-Data.csv  # historical FX + BCT fix
│   └── ib.csv             # historical IB rates
├── reports/               # generated *.xlsx + .docx + backtest_metrics.json + macro_overlay.json
├── run_pipeline.{bat,ps1,vbs}  # EMPTY — scheduling not wired yet
├── requirements.txt
├── README.md
├── CONTEXT.md             # this file
└── Project - TSA FV.pdf   # the spec
```

---

## 6. Database schema (`data/tnd.db`)

```sql
CREATE TABLE fx_rates (
    date    TEXT PRIMARY KEY,   -- 'YYYY-MM-DD'
    eurusd  REAL,                -- USD per 1 EUR
    gbpusd  REAL,                -- USD per 1 GBP
    usdjpy  REAL,                -- JPY per 1 USD
    fix_mid REAL,                -- BCT official mid fixing (TND per USD)
    ib_rate REAL,                -- interbank USD/TND (T-1 lag in practice)
    created_at TEXT
);

CREATE TABLE predictions (
    date         TEXT PRIMARY KEY,
    intrinsic_v1 REAL,           -- basket-only
    intrinsic_v2 REAL,           -- basket + Kalman spread (headline)
    w_eurusd     REAL,           -- last rolling-OLS weights
    w_gbpusd     REAL,
    w_usdjpy     REAL,
    kf_spread    REAL,           -- Kalman one-step state
    created_at   TEXT
);
```

Note: schema has **only `fix_mid`** — no AM/PM split yet. The PDF asks us to address AM vs PM explicitly; this is a known gap.

---

## 7. Model details (current implementation)

### 7.1 Basket OLS (`model.fit_ols`)
- `y = ret_Fix = log(fix_mid_t / fix_mid_{t-1})`
- `X = [1, ret_EURUSD, ret_GBPUSD, ret_USDJPY]` (log-returns)
- Solved via `numpy.linalg.lstsq` → intercept + 3 weights + R².

### 7.2 Rolling OLS (`model.rolling_weights`, window=90)
- Refits every step. `predict_today()` uses the **last** row's weights.

### 7.3 Spread Kalman (`model.kalman_filter_spread`)
- Spread series: `spread_pub = ib_rate − fix_mid`.
- AR(1) fit via OLS on `(spread_t, spread_{t-1})` → `(c, φ, Q)`. Q from residual variance.
- Observation noise hard-coded: `R = 0.25 · Var(spread)` — arbitrary; should be MLE-calibrated.
- State: `x_t = c + φ·x_{t-1} + w_t`, observation: `y_t = x_t + v_t`.

### 7.4 Intrinsic value (`predict.predict_today`)
```
basket_ret  = w_const + w_EUR·r_EURUSD + w_GBP·r_GBPUSD + w_JPY·r_USDJPY
intrinsic_v1 = prev_fix · exp(basket_ret)
intrinsic_v2 = intrinsic_v1 + kf_spread_today
```

---

## 8. Status — what is DONE ✅

- Daily pipeline end-to-end (fetch → predict → Excel → optional Telegram).
- SQLite store seeded with historical FX + BCT fix + IB rates.
- Basket OLS — full-sample + 90-day rolling, with **Newey-West HAC** standard errors, t-stats, and p-values; significance flags surfaced in the Excel "OLS Diagnostics" sheet.
- AR(1) + Kalman filter on the spread — **jointly MLE-calibrated** `(c, φ, Q, R)` with multi-start Nelder-Mead, `tanh`/`exp` parameterizations for stability; OLS fallback if SciPy missing.
- **Markov-switching Kalman** (`msk.py`): two-regime AR(1) state-space with Hamilton filter + Kim collapsed-Kalman + joint MLE of 10 parameters. AIC/BIC reported.
- **Macro-overlay** regression (`macro_overlay.py`): nested OLS leaderboard M0/M1/M2/M3 with Brent + DXY + VIX, incremental R² per block, Newey-West inference on each coefficient.
- **AM/PM fixing schema** + slot-detection scraper (UTC-hour heuristic) + env-var overrides `BCT_FIX_AM` / `BCT_FIX_PM`.
- **Intraday engine** (`realtime.py` + `run_realtime.py`): minute-resolution yfinance ticks, anchor on latest fix, forward-propagate Kalman state with σ for ±2σ band, cached MLE params.
- **Walk-forward backtest** with MAE/RMSE/MAPE/DA, Diebold-Mariano vs random walk, Ljung-Box residuals, ADF/KPSS on the spread. Results in JSON + CSV + Excel sheets.
- **FastAPI dashboard** (`serve.py` + `static/index.html`) — Perplexity-style SPA, dark mode, all KPIs from real SQLite. Replaces the older Streamlit dashboard.
- **Word-doc report generator** (`build_report.py`) — auto-populates §1, §5.3, §6.2, §9.2, §10.3, §10.4 from current DB and JSON artifacts.
- 3-source FX fallback (ExchangeRate-API v6 → exchangerate.host → frankfurter).
- BCT fixing scraper + env-var override.

---

## 9. Status — what is MISSING ❌ (remaining open items)

| # | Gap | Severity |
|---|---|---|
| 1 | **Live AM/PM fixings** — schema and scraper are in place but the BCT page only exposes the latest fixing; need a few weeks of forward AM+PM collection before the AM/PM research question can be answered empirically | Medium |
| 2 | **`run_pipeline.{bat,ps1,vbs}` scheduling** files are empty | Low |
| 3 | **Premium/discount alerting** with hit-ratio backtest as a tradable signal — code stub not yet written | Low |
| 4 | Author name on cover page is still the explicit placeholder `« Replace with author names »` — edit `AUTHORS` in `build_report.py` | Trivial |
| 5 | **Macro data** — `fetch_macro.py` is implemented but FRED is unreachable from some networks; re-run when external access is available to populate §10.4 | Trivial |

### Closed (all previously open gaps now addressed)
- ✅ AM/PM schema + slot-detection scraper + spread-vs-fix_pm convention
- ✅ Real-time / intraday operation (`realtime.py` + `run_realtime.py`)
- ✅ IB nowcasting under 1-day lag (forward-propagated Kalman state)
- ✅ Intraday regime shifts (Markov-switching Kalman in `msk.py`)
- ✅ Backtesting deliverable (walk-forward + DM-test + Ljung-Box + ADF/KPSS)
- ✅ Design document (`build_report.py` → `.docx`)
- ✅ MLE-calibrated `(c, φ, Q, R)`
- ✅ OLS Newey-West HAC inference (p-values, t-stats)
- ✅ Kalman σ surfaced + ±2σ confidence band
- ✅ FastAPI endpoint (`serve.py`)

---

## 10. How to "distinguish ourselves" (above-the-brief ideas)

Things that turn this into a top-grade submission:

- **VECM** between IB and Fix as an alternative to AR(1) Kalman; benchmark side-by-side.
- **Markov-switching Kalman** for AM vs PM volatility regimes (directly addresses spec's intraday-regime ask).
- **Bayesian / particle filter** with stochastic vol on the spread → probabilistic premium/discount bands.
- **Macro overlay**: Brent oil, DXY, Tunisian sovereign yield, BCT FX-reserves change. Report incremental R².
- **Premium/Discount signal** with hit-ratio backtest — answers the spec's framing about identifying liquidity pressure.
- **Walk-forward leaderboard**: (a) basket-only, (b) +AR(1), (c) +Kalman, (d) +VECM, (e) GBM/LSTM on residuals. Diebold-Mariano p-values.
- **FastAPI `/intrinsic` endpoint** alongside Streamlit — true real-time prototype.
- **`make report`** — one command produces both the Word doc (with embedded plots) and the Excel.
- **Stress periods** — explicit discussion of episodes the model misses (spec asks for "scenarios where the model struggled").

---

## 11. Run instructions

```powershell
# Windows / PowerShell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# One-time bootstrap
python init_db.py
python seed_db.py            # imports data/FX-CLEAN-Data.csv + data/ib.csv

# Daily run
python run_pipeline.py       # fetch + predict + Excel + (optional) Telegram

# Dashboard
streamlit run dashboard.py
```

Optional environment variables:
- `BCT_FIX_MID` — manual override of today's BCT fixing (bypasses scraping).
- `EXCHANGERATE_API_KEY` — override the built-in v6 key.
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — for alerts.

---

## 12. Conventions / things future AI sessions should know

- All FX rates are stored as **per-USD conventions** consistent with `_parse_rates_usd_base` in `fetch_daily.py`: `EURUSD = USD per EUR`, `GBPUSD = USD per GBP`, `USDJPY = JPY per USD`, `fix_mid / ib_rate = TND per USD`.
- Returns are **log-returns**.
- Spread is `ib_rate − fix_mid` (TND, not bps).
- Premium = IB richer than intrinsic = local USD scarcity (red in the redesigned dashboard).
- Pure numpy is preferred for the model (no statsmodels/scikit) — kept lightweight. Adding scipy/statsmodels is acceptable if a feature genuinely needs it.
- Memory store at `~/.claude/projects/C--Users-faouz-tnd-fx-pipeline/memory/` holds long-term project facts for Claude Code sessions; ignore if you are a different AI.
- The user works on Windows; PowerShell-compatible commands preferred.

---

## 13. Suggested next-task prompts (copy/paste ready)

Pick whichever the user asks for:

- *"Add `fix_am` and `fix_pm` columns to `fx_rates`, migrate existing `fix_mid` to the average, update `fetch_daily.scrape_bct_fixing` to extract both, and decide which fixing anchors the basket vs the spread — document the choice in a comment block."*
- *"Write `nowcast_ib.py` that takes the previous day's observed spread as the morning anchor, then updates the Kalman state intraday from each new global-FX tick."*
- *"Build `backtest.py` — walk-forward expanding window, output MAE/RMSE/MAPE/DA, DM test vs random-walk, ADF/KPSS on spread, Ljung-Box on residuals; save a 4th sheet to the Excel."*
- *"Jointly MLE-calibrate `(c, φ, Q, R)` for the Kalman; replace the hard-coded `R = 0.25·Var`."*
- *"Draft `docs/model_design.md` with full math: state/observation equations, A/H matrices, derivation of the basket return, choice of fixing, regime treatment, limitations."*
- *"Add a Markov-switching Kalman with two regimes (AM/PM) and benchmark vs the single-regime version."*

---

## 14. Reference: PDF spec key quotes

- *"The intrinsic value of USD/TND is treated as one driven by a weighted basket of these major currencies' movements."*
- *"The Central Bank of Tunisia publishes the average interbank TND exchange rates with a one-day lag."*
- *"One research question is which fixing to use when modeling the adjustment — today's morning, evening, or yesterday's. The appropriate approach is left for you to determine."*
- *"The model might allow the volatility or bias of the adjustment factor to change between, say, morning and afternoon trading sessions."*
- *"Backtest the intrinsic value estimates against actual market outcomes ... include error statistics, correlation analysis, and visual charts."*
- *"Discuss scenarios where the model struggled and any adjustments made to improve it."*

Kalman state-space (from the spec appendix):
```
x_t = A · x_{t-1} + w_t,   w_t ~ N(0, Q)
y_t = H · x_t + v_t,        v_t ~ N(0, R)
```
