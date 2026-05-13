"""
Real-time / intraday intrinsic value engine.

One tick = (1) fetch latest EUR/USD, GBP/USD, USD/JPY,
          (2) compute basket return vs the anchor fixing,
          (3) advance Kalman spread state forward (no IB observation intraday),
          (4) emit intrinsic_v1 and intrinsic_v2, persist to fx_intraday + intrinsic_intraday.

Source: yfinance (1-minute FX bars, free, no API key).
Fallback: last row of fx_rates if yfinance is unavailable (degraded daily mode).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from clean_returns import load_and_clean
from model import rolling_weights, kalman_filter_spread, mle_kalman_ar1

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "tnd.db"


# ---------------------------------------------------------------------------
# Quote fetcher
# ---------------------------------------------------------------------------

def fetch_intraday_quote() -> Optional[Dict[str, Any]]:
    """
    Latest minute bar for EURUSD, GBPUSD, USDJPY via yfinance.
    Returns None if yfinance is unavailable or quotes are stale.
    """
    try:
        import yfinance as yf  # local import — optional dependency
    except ImportError:
        return None

    tickers = {"EURUSD=X": "eurusd", "GBPUSD=X": "gbpusd", "JPY=X": "usdjpy"}
    out: Dict[str, Any] = {}
    latest_ts = None
    try:
        data = yf.download(
            tickers=list(tickers.keys()),
            period="1d",
            interval="1m",
            progress=False,
            threads=True,
            auto_adjust=False,
        )
    except Exception:
        return None
    if data is None or len(data) == 0:
        return None

    # yfinance returns multi-index columns when multiple tickers are passed.
    closes = data["Close"] if "Close" in data.columns.get_level_values(0) else None
    if closes is None:
        return None

    last_row = closes.dropna().tail(1)
    if last_row.empty:
        return None
    latest_ts = last_row.index[-1].to_pydatetime()
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.replace(tzinfo=timezone.utc)
    else:
        latest_ts = latest_ts.astimezone(timezone.utc)

    for tk, key in tickers.items():
        try:
            v = float(last_row[tk].iloc[0])
            if not np.isfinite(v):
                return None
            out[key] = v
        except Exception:
            return None

    out["ts"] = latest_ts.isoformat(timespec="seconds")
    out["source"] = "yfinance"
    return out


# ---------------------------------------------------------------------------
# Anchor + Kalman state
# ---------------------------------------------------------------------------

def _latest_anchor(conn: sqlite3.Connection) -> Optional[Tuple[str, float, float, float, float]]:
    """
    Anchor = most recent fx_rates row that has a non-null fix_mid AND non-null FX quotes.
    Returns (date, fix_mid, eurusd, gbpusd, usdjpy) or None.
    """
    row = conn.execute(
        """
        SELECT date, fix_mid, eurusd, gbpusd, usdjpy
        FROM fx_rates
        WHERE fix_mid IS NOT NULL
          AND eurusd IS NOT NULL AND gbpusd IS NOT NULL AND usdjpy IS NOT NULL
        ORDER BY date DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    return str(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])


# Module-level cache for the daily-refit Kalman parameters. Refitting the
# AR(1) + Kalman on every intraday tick is wasteful — the daily spread series
# only changes once per day. We invalidate when the latest fx_rates row date
# changes (proxy for "new daily observation arrived").
_KF_CACHE: Dict[str, Any] = {
    "anchor_date": None,
    "c": 0.0,
    "phi": 0.0,
    "Q": 0.0,
    "R": 0.0,
    "last_state": 0.0,
    "last_P": 0.0,
}


def _invalidate_kf_cache() -> None:
    _KF_CACHE["anchor_date"] = None


def _last_kf_state(conn: sqlite3.Connection, anchor_date: str) -> Tuple[float, float, float, float, float]:
    """
    Returns (last_filtered_state, c, phi, Q, last_P). Cached by anchor_date —
    the daily spread series only ticks once per day, so refitting per
    intraday tick is wasteful.

    Caller advances the state forward intraday:
        x_pred = c + phi * last_state
        P_pred = phi^2 * last_P + Q
        sigma  = sqrt(P_pred)
    No observation update — IB is T-1 lagged so no intraday y_t exists.
    """
    if _KF_CACHE["anchor_date"] == anchor_date:
        return (
            _KF_CACHE["last_state"],
            _KF_CACHE["c"],
            _KF_CACHE["phi"],
            _KF_CACHE["Q"],
            _KF_CACHE["last_P"],
        )

    df = load_and_clean(conn, lookback_days=500)
    if df.empty or "spread_pub" not in df.columns or df["spread_pub"].dropna().empty:
        _KF_CACHE.update({"anchor_date": anchor_date, "last_state": 0.0, "c": 0.0, "phi": 0.0, "Q": 0.0, "last_P": 0.0})
        return 0.0, 0.0, 0.0, 0.0, 0.0

    kf = kalman_filter_spread(df["spread_pub"], use_mle=True)
    last_state = float(kf.iloc[-1]) if not kf.empty else 0.0

    s = df["spread_pub"].dropna().values.astype(float)
    if len(s) < 3:
        _KF_CACHE.update({"anchor_date": anchor_date, "last_state": last_state, "c": 0.0, "phi": 0.0, "Q": 0.0, "last_P": 0.0})
        return last_state, 0.0, 0.0, 0.0, 0.0

    # Joint MLE of (c, phi, Q, R). Falls back to OLS-seed if SciPy missing.
    mle = mle_kalman_ar1(df["spread_pub"])
    c, phi, Q, R = mle["c"], mle["phi"], mle["Q"], mle["R"]
    # Steady-state P from Riccati: P = (phi² P + Q) − (phi² P + Q)² / (phi² P + Q + R)
    # ⇒ P satisfies  P · R = (phi² P + Q) · R / (phi² P + Q + R) · R … use fixed-point.
    P = max(Q / (1.0 - phi * phi), Q) if abs(phi) < 0.999 else max(Q, 1e-12)
    for _ in range(200):
        P_pred = phi * phi * P + Q
        K = P_pred / (P_pred + R) if (P_pred + R) > 0 else 0.0
        P_new = (1.0 - K) * P_pred
        if abs(P_new - P) < 1e-14:
            break
        P = P_new
    last_P = max(P, 1e-12)

    _KF_CACHE.update({
        "anchor_date": anchor_date,
        "c": c, "phi": phi, "Q": Q,
        "last_state": last_state, "last_P": last_P,
    })
    return last_state, c, phi, Q, last_P


def _last_weights(conn: sqlite3.Connection) -> Optional[Dict[str, float]]:
    df = load_and_clean(conn, lookback_days=500)
    if df.empty or len(df) < 90:
        return None
    roll = rolling_weights(df, 90)
    w = roll.iloc[-1]
    if pd.isna(w["w_EURUSD"]):
        return None
    return {
        "w_const":  float(w["w_const"]),
        "w_EURUSD": float(w["w_EURUSD"]),
        "w_GBPUSD": float(w["w_GBPUSD"]),
        "w_USDJPY": float(w["w_USDJPY"]),
    }


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

def tick(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Single intraday tick. Persists fx_intraday + intrinsic_intraday rows.
    Returns a summary dict (always — sets ok=False with a reason on failure).
    """
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    quote = fetch_intraday_quote()
    if quote is None:
        return {"ok": False, "ts": now_iso, "reason": "no intraday quote (yfinance unavailable or empty)"}

    anchor = _latest_anchor(conn)
    if anchor is None:
        return {"ok": False, "ts": quote["ts"], "reason": "no anchor row in fx_rates"}
    anchor_date, fix_anchor, e_anchor, g_anchor, j_anchor = anchor

    weights = _last_weights(conn)
    if weights is None:
        return {"ok": False, "ts": quote["ts"], "reason": "rolling weights unavailable (need ≥90d)"}

    # Persist raw quote (idempotent — INSERT OR REPLACE on minute key)
    conn.execute(
        """
        INSERT OR REPLACE INTO fx_intraday (ts, eurusd, gbpusd, usdjpy, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (quote["ts"], quote["eurusd"], quote["gbpusd"], quote["usdjpy"], quote["source"], now_iso),
    )

    # Basket log-return from anchor → now
    r1 = float(np.log(quote["eurusd"] / e_anchor))
    r2 = float(np.log(quote["gbpusd"] / g_anchor))
    r3 = float(np.log(quote["usdjpy"] / j_anchor))
    basket_ret = (
        weights["w_const"]
        + weights["w_EURUSD"] * r1
        + weights["w_GBPUSD"] * r2
        + weights["w_USDJPY"] * r3
    )
    intrinsic_v1 = float(fix_anchor * np.exp(basket_ret))

    # Kalman: forward propagate one step from last filtered state (no obs update).
    # Persist sigma so the dashboard can render a ±2σ confidence band.
    kf_last, c, phi, Q, last_P = _last_kf_state(conn, anchor_date)
    kf_state = float(c + phi * kf_last)
    P_pred = float(phi * phi * last_P + Q)
    kf_sigma = float(np.sqrt(max(P_pred, 0.0)))
    intrinsic_v2 = intrinsic_v1 + kf_state

    conn.execute(
        """
        INSERT OR REPLACE INTO intrinsic_intraday
        (ts, anchor_date, anchor_fix, basket_ret, intrinsic_v1, kf_state, kf_sigma, intrinsic_v2, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            quote["ts"], anchor_date, float(fix_anchor),
            float(basket_ret), intrinsic_v1, kf_state, kf_sigma, intrinsic_v2, now_iso,
        ),
    )
    conn.commit()

    return {
        "ok": True,
        "ts": quote["ts"],
        "anchor_date": anchor_date,
        "anchor_fix": fix_anchor,
        "eurusd": quote["eurusd"],
        "gbpusd": quote["gbpusd"],
        "usdjpy": quote["usdjpy"],
        "basket_ret_pct": basket_ret * 100.0,
        "intrinsic_v1": intrinsic_v1,
        "kf_state": kf_state,
        "kf_sigma": kf_sigma,
        "intrinsic_v2": intrinsic_v2,
    }


if __name__ == "__main__":
    import json
    from init_db import init_db

    init_db(DEFAULT_DB)
    with sqlite3.connect(str(DEFAULT_DB)) as conn:
        result = tick(conn)
    print(json.dumps(result, indent=2, default=str))
