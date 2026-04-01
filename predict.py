"""
Daily prediction: rolling basket + Kalman spread, persist to SQLite.
"""
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from clean_returns import load_and_clean
from model import fit_ols, rolling_weights, kalman_filter_spread

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "tnd.db"


def _latest_fx_pair(conn: sqlite3.Connection) -> Optional[pd.DataFrame]:
    """Last two rows of fx_rates with FX quotes (eurusd not null)."""
    q = """
    SELECT date, eurusd, gbpusd, usdjpy, fix_mid
    FROM fx_rates
    WHERE eurusd IS NOT NULL AND gbpusd IS NOT NULL AND usdjpy IS NOT NULL
    ORDER BY date DESC
    LIMIT 2
    """
    df = pd.read_sql_query(q, conn)
    if len(df) < 2:
        return None
    return df


def predict_today(conn: sqlite3.Connection, db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Train on history, apply latest rolling weights to today's FX returns,
    write one row to predictions, return summary dict.
    """
    df_clean = load_and_clean(conn, lookback_days=500)

    if df_clean.empty or len(df_clean) < 90:
        return {
            "ok": False,
            "reason": "need at least 90 rows with non-null fix_mid and returns",
            "date": date.today().isoformat(),
            "intrinsic_v1": None,
            "intrinsic_v2": None,
            "w_eurusd": None,
            "w_gbpusd": None,
            "w_usdjpy": None,
            "kf_spread": None,
            "r_squared": None,
            "prev_fix": None,
            "basket_ret_pct": None,
        }

    ols = fit_ols(df_clean)
    roll = rolling_weights(df_clean, 90)
    kf = kalman_filter_spread(df_clean["spread_pub"])

    w_last = roll.iloc[-1]
    if np.isnan(w_last["w_EURUSD"]):
        return {
            "ok": False,
            "reason": "rolling weights not available",
            "date": date.today().isoformat(),
            "intrinsic_v1": None,
            "intrinsic_v2": None,
            "w_eurusd": None,
            "w_gbpusd": None,
            "w_usdjpy": None,
            "kf_spread": None,
            "r_squared": float(ols["r_squared"]) if ols else None,
            "prev_fix": None,
            "basket_ret_pct": None,
        }

    pair = _latest_fx_pair(conn)
    if pair is None:
        return {
            "ok": False,
            "reason": "not enough FX rows in fx_rates",
            "date": date.today().isoformat(),
            "intrinsic_v1": None,
            "intrinsic_v2": None,
            "w_eurusd": float(w_last["w_EURUSD"]),
            "w_gbpusd": float(w_last["w_GBPUSD"]),
            "w_usdjpy": float(w_last["w_USDJPY"]),
            "kf_spread": None,
            "r_squared": float(ols["r_squared"]),
            "prev_fix": None,
            "basket_ret_pct": None,
        }

    d0 = pair.iloc[0]
    d1 = pair.iloc[1]
    e0, e1 = float(d0["eurusd"]), float(d1["eurusd"])
    g0, g1 = float(d0["gbpusd"]), float(d1["gbpusd"])
    j0, j1 = float(d0["usdjpy"]), float(d1["usdjpy"])

    r1 = np.log(e0 / e1)
    r2 = np.log(g0 / g1)
    r3 = np.log(j0 / j1)

    wc = float(w_last["w_const"])
    basket_ret = (
        wc
        + float(w_last["w_EURUSD"]) * r1
        + float(w_last["w_GBPUSD"]) * r2
        + float(w_last["w_USDJPY"]) * r3
    )

    # Previous BCT fixing: prefer fix on the older FX row (day before latest quote)
    pf = d1.get("fix_mid")
    if pf is not None and not (isinstance(pf, float) and np.isnan(pf)):
        prev_fix = float(pf)
    else:
        prev_fix_row = conn.execute(
            """
            SELECT fix_mid FROM fx_rates
            WHERE date < ? AND fix_mid IS NOT NULL
            ORDER BY date DESC LIMIT 1
            """,
            (str(d0["date"]),),
        ).fetchone()
        if prev_fix_row is None or prev_fix_row[0] is None:
            return {
                "ok": False,
                "reason": "no previous BCT fixing available",
                "date": str(d0["date"]),
                "intrinsic_v1": None,
                "intrinsic_v2": None,
                "w_eurusd": float(w_last["w_EURUSD"]),
                "w_gbpusd": float(w_last["w_GBPUSD"]),
                "w_usdjpy": float(w_last["w_USDJPY"]),
                "kf_spread": float(kf.iloc[-1]),
                "r_squared": float(ols["r_squared"]),
                "prev_fix": None,
                "basket_ret_pct": float(basket_ret * 100.0),
            }
        prev_fix = float(prev_fix_row[0])

    intrinsic_v1 = prev_fix * np.exp(basket_ret)
    kf_today = float(kf.iloc[-1])
    intrinsic_v2 = intrinsic_v1 + kf_today

    pred_date = str(d0["date"])
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    conn.execute(
        """
        INSERT OR REPLACE INTO predictions
        (date, intrinsic_v1, intrinsic_v2, w_eurusd, w_gbpusd, w_usdjpy, kf_spread, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pred_date,
            float(intrinsic_v1),
            float(intrinsic_v2),
            float(w_last["w_EURUSD"]),
            float(w_last["w_GBPUSD"]),
            float(w_last["w_USDJPY"]),
            kf_today,
            now,
        ),
    )
    conn.commit()

    return {
        "ok": True,
        "date": pred_date,
        "intrinsic_v1": float(intrinsic_v1),
        "intrinsic_v2": float(intrinsic_v2),
        "w_eurusd": float(w_last["w_EURUSD"]),
        "w_gbpusd": float(w_last["w_GBPUSD"]),
        "w_usdjpy": float(w_last["w_USDJPY"]),
        "kf_spread": kf_today,
        "r_squared": float(ols["r_squared"]),
        "prev_fix": prev_fix,
        "basket_ret_pct": float(basket_ret * 100.0),
    }


def predict_for_date(conn: sqlite3.Connection, target_date: str) -> Dict[str, Any]:
    """
    Predict for a specific date using all history up to that date.
    """
    # Only use data up to and including target_date
    df_clean = load_and_clean(conn, lookback_days=500)
    df_clean = df_clean[df_clean["date"] <= pd.to_datetime(target_date)]
    if df_clean.empty or len(df_clean) < 90:
        return {"ok": False, "date": target_date}
    ols = fit_ols(df_clean)
    roll = rolling_weights(df_clean, 90)
    kf = kalman_filter_spread(df_clean["spread_pub"])
    w_last = roll.iloc[-1]
    # Find the two most recent FX rows up to target_date
    q = """
    SELECT date, eurusd, gbpusd, usdjpy, fix_mid FROM fx_rates
    WHERE eurusd IS NOT NULL AND gbpusd IS NOT NULL AND usdjpy IS NOT NULL AND date <= ?
    ORDER BY date DESC LIMIT 2
    """
    pair = pd.read_sql_query(q, conn, params=(target_date,))
    if len(pair) < 2:
        return {"ok": False, "date": target_date}
    d0, d1 = pair.iloc[0], pair.iloc[1]
    e0, e1 = float(d0["eurusd"]), float(d1["eurusd"])
    g0, g1 = float(d0["gbpusd"]), float(d1["gbpusd"])
    j0, j1 = float(d0["usdjpy"]), float(d1["usdjpy"])
    r1 = np.log(e0 / e1)
    r2 = np.log(g0 / g1)
    r3 = np.log(j0 / j1)
    wc = float(w_last["w_const"])
    basket_ret = wc + float(w_last["w_EURUSD"]) * r1 + float(w_last["w_GBPUSD"]) * r2 + float(w_last["w_USDJPY"]) * r3
    pf = d1.get("fix_mid")
    prev_fix = float(pf) if pf is not None and not (isinstance(pf, float) and np.isnan(pf)) else None
    if prev_fix is None:
        prev_fix_row = conn.execute(
            "SELECT fix_mid FROM fx_rates WHERE date < ? AND fix_mid IS NOT NULL ORDER BY date DESC LIMIT 1",
            (str(d0["date"]),),
        ).fetchone()
        prev_fix = float(prev_fix_row[0]) if prev_fix_row and prev_fix_row[0] is not None else None
    if prev_fix is None:
        return {"ok": False, "date": target_date}
    intrinsic_v1 = prev_fix * np.exp(basket_ret)
    kf_today = float(kf.iloc[-1]) if not kf.empty else 0.0
    intrinsic_v2 = intrinsic_v1 + kf_today
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn.execute(
        "INSERT OR REPLACE INTO predictions (date, intrinsic_v1, intrinsic_v2, w_eurusd, w_gbpusd, w_usdjpy, kf_spread, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (target_date, float(intrinsic_v1), float(intrinsic_v2), float(w_last["w_EURUSD"]), float(w_last["w_GBPUSD"]), float(w_last["w_USDJPY"]), kf_today, now),
    )
    conn.commit()
    return {"ok": True, "date": target_date, "intrinsic_v2": float(intrinsic_v2)}
