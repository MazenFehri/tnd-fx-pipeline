"""
Create SQLite schema for USD/TND pipeline (idempotent).
"""
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "tnd.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS fx_rates (
    date TEXT PRIMARY KEY,
    eurusd REAL,
    gbpusd REAL,
    usdjpy REAL,
    fix_am REAL,                 -- BCT morning fixing (USD/TND)
    fix_pm REAL,                 -- BCT evening fixing (USD/TND) — closing reference
    fix_mid REAL,                -- AVG(fix_am, fix_pm) if both, else whichever present
    ib_rate REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    date TEXT PRIMARY KEY,
    intrinsic_v1 REAL,
    intrinsic_v2 REAL,
    w_eurusd REAL,
    w_gbpusd REAL,
    w_usdjpy REAL,
    kf_spread REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS fx_intraday (
    ts TEXT PRIMARY KEY,         -- ISO8601 UTC, minute resolution
    eurusd REAL,
    gbpusd REAL,
    usdjpy REAL,
    source TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS intrinsic_intraday (
    ts TEXT PRIMARY KEY,         -- ISO8601 UTC
    anchor_date TEXT,            -- date of the BCT fix used as anchor
    anchor_fix REAL,             -- fix_mid value at anchor
    basket_ret REAL,             -- log-return basket × weights
    intrinsic_v1 REAL,           -- anchor_fix * exp(basket_ret)
    kf_state REAL,               -- Kalman predicted spread (forward from last obs)
    kf_sigma REAL,               -- Kalman state std-dev (for ±2σ band)
    intrinsic_v2 REAL,           -- v1 + kf_state
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_fx_intraday_ts ON fx_intraday(ts);
CREATE INDEX IF NOT EXISTS idx_intrinsic_intraday_ts ON intrinsic_intraday(ts);

CREATE TABLE IF NOT EXISTS macro (
    date TEXT PRIMARY KEY,
    brent REAL,                  -- Brent crude (USD/bbl)
    dxy REAL,                    -- US Dollar Index
    vix REAL,                    -- CBOE volatility index
    source TEXT,
    created_at TEXT
);
"""


def init_db(db_path=None):
    """Create tables if they do not exist; run forward-only column migrations."""
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA)
        # Forward-only migrations: add columns to pre-existing tables. SQLite
        # has no IF NOT EXISTS for ADD COLUMN; ignore "duplicate column" errors.
        for stmt in (
            "ALTER TABLE intrinsic_intraday ADD COLUMN kf_sigma REAL",
            "ALTER TABLE fx_rates ADD COLUMN fix_am REAL",
            "ALTER TABLE fx_rates ADD COLUMN fix_pm REAL",
        ):
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DEFAULT_DB}")
