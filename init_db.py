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
    fix_mid REAL,
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
"""


def init_db(db_path=None):
    """Create tables if they do not exist."""
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DEFAULT_DB}")
