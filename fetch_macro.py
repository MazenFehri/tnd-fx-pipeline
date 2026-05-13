"""
Macro covariates fetcher — Brent, DXY, VIX from FRED (St. Louis Fed).

FRED publishes daily history as plain CSV with no API key:
  Brent crude:   DCOILBRENTEU   (USD per barrel)
  Broad dollar:  DTWEXBGS       (Fed trade-weighted broad dollar index)
  VIX:           VIXCLS

Persists to the `macro` table in data/tnd.db with one row per date.

Run:  python fetch_macro.py
"""
from __future__ import annotations

import csv
import io
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict

import requests

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "tnd.db"

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"
SERIES = {
    "brent": "DCOILBRENTEU",
    "dxy":   "DTWEXBGS",
    "vix":   "VIXCLS",
}


def _fetch_fred(series_id: str) -> Dict[str, float]:
    """
    Download FRED CSV for `series_id` and return {date_iso: value}.
    FRED uses '.' to denote missing observations — those are dropped.
    """
    url = f"{FRED_BASE}?id={series_id}"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=60, headers=headers)
            r.raise_for_status()
            break
        except Exception as e:
            last_err = e
    else:
        print(f"[fetch_macro] {series_id} download failed after 3 attempts: {last_err}")
        return {}

    text = r.text.strip()
    if not text:
        return {}

    out: Dict[str, float] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        # FRED columns: 'observation_date' (or 'DATE'), then the series_id value column
        d = row.get("observation_date") or row.get("DATE")
        if not d:
            continue
        v_raw = None
        for k, val in row.items():
            if k in ("observation_date", "DATE"):
                continue
            v_raw = val
            break
        if v_raw is None or v_raw.strip() in ("", "."):
            continue
        try:
            out[d] = float(v_raw)
        except ValueError:
            continue
    return out


YAHOO_TICKERS = {
    "brent": "BZ=F",        # ICE Brent front-month
    "dxy":   "DX-Y.NYB",    # NYBOT US Dollar Index
    "vix":   "%5EVIX",      # ^VIX (URL-encoded)
}


def _fetch_yahoo(ticker: str) -> Dict[str, float]:
    """
    Pull daily close history from Yahoo Finance v8 chart API.
    Returns {YYYY-MM-DD: close}. No API key needed.
    """
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range=10y&interval=1d&events=history")
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    last_err = None
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30, headers=headers)
            r.raise_for_status()
            break
        except Exception as e:
            last_err = e
    else:
        print(f"[fetch_macro] yahoo {ticker} failed: {last_err}")
        return {}

    try:
        payload = r.json()
        result = payload["chart"]["result"][0]
        ts_arr = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        print(f"[fetch_macro] yahoo {ticker} parse error: {e}")
        return {}

    out: Dict[str, float] = {}
    from datetime import date as _date
    for ts, c in zip(ts_arr, closes):
        if c is None:
            continue
        d = _date.fromtimestamp(int(ts)).isoformat()
        try:
            out[d] = float(c)
        except (TypeError, ValueError):
            continue
    return out


def fetch_all() -> Dict[str, Dict[str, float]]:
    """
    Pull all three series. Tries FRED first; falls back to Yahoo per-series
    if FRED is unreachable from the current network (common on corporate / ISP
    firewalls that reset TLS to fred.stlouisfed.org).
    """
    out: Dict[str, Dict[str, float]] = {}
    for name, sid in SERIES.items():
        data = _fetch_fred(sid)
        if not data:
            print(f"[fetch_macro] falling back to Yahoo for {name}")
            data = _fetch_yahoo(YAHOO_TICKERS[name])
        out[name] = data
    return out


def upsert_macro(conn: sqlite3.Connection, series: Dict[str, Dict[str, float]]) -> int:
    """
    Merge the three series by date and upsert into `macro`.
    Returns the number of rows touched.
    """
    all_dates = sorted({d for s in series.values() for d in s})
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    n = 0
    for d in all_dates:
        brent = series.get("brent", {}).get(d)
        dxy   = series.get("dxy",   {}).get(d)
        vix   = series.get("vix",   {}).get(d)
        if brent is None and dxy is None and vix is None:
            continue
        conn.execute(
            """
            INSERT OR REPLACE INTO macro (date, brent, dxy, vix, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (d, brent, dxy, vix, "stooq", now),
        )
        n += 1
    conn.commit()
    return n


def main() -> int:
    from init_db import init_db
    init_db(DEFAULT_DB)
    series = fetch_all()
    counts = {k: len(v) for k, v in series.items()}
    print(f"[fetch_macro] downloaded: {counts}")
    if not any(counts.values()):
        print("[fetch_macro] no series available — aborting.")
        return 1
    with sqlite3.connect(str(DEFAULT_DB)) as conn:
        n = upsert_macro(conn, series)
    print(f"[fetch_macro] upserted {n} rows into macro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
