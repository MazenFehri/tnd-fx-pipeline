"""
Load FX history from SQLite and compute log-returns for modeling.
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "tnd.db"


def load_and_clean(
    conn: sqlite3.Connection,
    lookback_days: int = 500,
) -> pd.DataFrame:
    """
    SELECT last N days from fx_rates where fix_mid IS NOT NULL.
    Compute log-returns and spread_pub; drop rows with incomplete returns.
    """
    # Try AM/PM schema first; fall back if columns don't exist (very old DBs).
    try:
        q = """
        SELECT date, eurusd, gbpusd, usdjpy, fix_am, fix_pm, fix_mid, ib_rate
        FROM fx_rates
        WHERE fix_mid IS NOT NULL
        ORDER BY date ASC
        """
        df = pd.read_sql_query(q, conn)
    except Exception:
        q = """
        SELECT date, eurusd, gbpusd, usdjpy, fix_mid, ib_rate
        FROM fx_rates
        WHERE fix_mid IS NOT NULL
        ORDER BY date ASC
        """
        df = pd.read_sql_query(q, conn)
        df["fix_am"] = np.nan
        df["fix_pm"] = np.nan
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(lookback_days).reset_index(drop=True)

    df = df.rename(
        columns={
            "eurusd": "EURUSD",
            "gbpusd": "GBPUSD",
            "usdjpy": "USDJPY",
            "fix_am": "Fix_AM",
            "fix_pm": "Fix_PM",
            "fix_mid": "Fix_Mid",
            "ib_rate": "IB_USD_TND",
        }
    )

    for col in ("EURUSD", "GBPUSD", "USDJPY", "Fix_AM", "Fix_PM", "Fix_Mid"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["IB_USD_TND"] = pd.to_numeric(df["IB_USD_TND"], errors="coerce")

    df["ret_EURUSD"] = np.log(df["EURUSD"] / df["EURUSD"].shift(1))
    df["ret_GBPUSD"] = np.log(df["GBPUSD"] / df["GBPUSD"].shift(1))
    df["ret_USDJPY"] = np.log(df["USDJPY"] / df["USDJPY"].shift(1))
    df["ret_Fix"] = np.log(df["Fix_Mid"] / df["Fix_Mid"].shift(1))

    # Spread reference: prefer the PM (closing) fixing, since IB is the day's
    # average interbank rate — closest to the closing reference. Falls back to
    # Fix_Mid for historical rows that pre-date the AM/PM split.
    fix_ref = df["Fix_PM"].where(df["Fix_PM"].notna(), df["Fix_Mid"])
    df["spread_pub"] = df["IB_USD_TND"] - fix_ref

    req = ["ret_Fix", "ret_EURUSD", "ret_GBPUSD", "ret_USDJPY"]
    df_clean = df.dropna(subset=req).copy()
    return df_clean
