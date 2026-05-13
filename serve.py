"""
USD/TND intrinsic-value dashboard — FastAPI backend.

Run with:  python serve.py
Serves the SPA at /  and JSON API under /api.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests

ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "tnd.db"
STATIC_DIR = ROOT / "static"
REPORTS_DIR = ROOT / "reports"


def _load_dotenv() -> None:
    """
    Auto-load .env at repo root (no python-dotenv dependency). Lines that look
    like KEY=VALUE are pushed into os.environ unless already set. Lines starting
    with # or blank are ignored. Surrounding quotes are stripped. This is enough
    for a single-key file like:
        ANTHROPIC_API_KEY=sk-ant-...
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()

app = FastAPI(title="USD/TND Intrinsic Value", version="2.0")

# ----------------------------------------------------------------------------
# DB helpers
# ----------------------------------------------------------------------------

@contextmanager
def db():
    if not DB_PATH.exists():
        # Allow API to respond with empty arrays rather than 500 if db missing.
        yield None
        return
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def rows_to_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows] if rows else []


# ----------------------------------------------------------------------------
# Cache (30s server-side)
# ----------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, Any]] = {}
_TTL = 30.0


def cached(key: str, builder):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    val = builder()
    _CACHE[key] = (now, val)
    return val


# ----------------------------------------------------------------------------
# Pydantic response models
# ----------------------------------------------------------------------------

class Snapshot(BaseModel):
    fix_date: Optional[str]
    fix_am: Optional[float]
    fix_pm: Optional[float]
    fix_mid: Optional[float]
    ib_date: Optional[str]
    ib_rate: Optional[float]
    intrinsic_v1: Optional[float]
    intrinsic_v2: Optional[float]
    kf_state: Optional[float]
    kf_sigma: Optional[float]
    premium_bps: Optional[float]
    business_days_lag: int
    freshness: str  # "fresh" | "amber" | "stale"
    anchor_date: Optional[str]
    anchor_fix: Optional[float]
    weights: dict[str, Optional[float]]


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------

def _business_days(a: str, b: str) -> int:
    try:
        da = datetime.fromisoformat(a).date()
        db_ = datetime.fromisoformat(b).date()
    except Exception:
        return 0
    days = 0
    cur = da
    while cur < db_:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


@app.get("/api/snapshot", response_model=Snapshot)
def snapshot():
    def build():
        with db() as conn:
            if conn is None:
                return Snapshot(
                    fix_date=None, fix_am=None, fix_pm=None, fix_mid=None,
                    ib_date=None, ib_rate=None,
                    intrinsic_v1=None, intrinsic_v2=None,
                    kf_state=None, kf_sigma=None, premium_bps=None,
                    business_days_lag=0, freshness="stale",
                    anchor_date=None, anchor_fix=None,
                    weights={"eurusd": None, "gbpusd": None, "usdjpy": None},
                ).model_dump()

            fx = conn.execute(
                "SELECT date, fix_am, fix_pm, fix_mid, ib_rate "
                "FROM fx_rates ORDER BY date DESC LIMIT 1"
            ).fetchone()
            ib_row = conn.execute(
                "SELECT date, ib_rate FROM fx_rates "
                "WHERE ib_rate IS NOT NULL ORDER BY date DESC LIMIT 1"
            ).fetchone()
            pred = conn.execute(
                "SELECT date, intrinsic_v1, intrinsic_v2, kf_spread, "
                "w_eurusd, w_gbpusd, w_usdjpy "
                "FROM predictions ORDER BY date DESC LIMIT 1"
            ).fetchone()
            intra = conn.execute(
                "SELECT ts, anchor_date, anchor_fix, kf_state, kf_sigma, "
                "intrinsic_v1, intrinsic_v2 "
                "FROM intrinsic_intraday ORDER BY ts DESC LIMIT 1"
            ).fetchone()

            today = datetime.utcnow().date().isoformat()
            ib_date = ib_row["date"] if ib_row else None
            lag = _business_days(ib_date, today) if ib_date else 0
            freshness = "fresh" if lag <= 1 else ("amber" if lag <= 2 else "stale")

            v1 = intra["intrinsic_v1"] if intra else (pred["intrinsic_v1"] if pred else None)
            v2 = intra["intrinsic_v2"] if intra else (pred["intrinsic_v2"] if pred else None)
            ib = ib_row["ib_rate"] if ib_row else None
            premium_bps = None
            if v2 and ib:
                premium_bps = (ib - v2) / v2 * 10_000

            return Snapshot(
                fix_date=fx["date"] if fx else None,
                fix_am=fx["fix_am"] if fx else None,
                fix_pm=fx["fix_pm"] if fx else None,
                fix_mid=fx["fix_mid"] if fx else None,
                ib_date=ib_date,
                ib_rate=ib,
                intrinsic_v1=v1,
                intrinsic_v2=v2,
                kf_state=intra["kf_state"] if intra else None,
                kf_sigma=intra["kf_sigma"] if intra else None,
                premium_bps=premium_bps,
                business_days_lag=lag,
                freshness=freshness,
                anchor_date=intra["anchor_date"] if intra else None,
                anchor_fix=intra["anchor_fix"] if intra else None,
                weights={
                    "eurusd": pred["w_eurusd"] if pred else None,
                    "gbpusd": pred["w_gbpusd"] if pred else None,
                    "usdjpy": pred["w_usdjpy"] if pred else None,
                },
            ).model_dump()

    return JSONResponse(cached("snapshot", build))


@app.get("/api/timeseries")
def timeseries(days: int = Query(90, ge=1, le=3650)):
    def build():
        with db() as conn:
            if conn is None:
                return []
            cutoff = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
            sql = """
                SELECT f.date AS date, f.fix_mid, f.fix_am, f.fix_pm, f.ib_rate,
                       p.intrinsic_v1 AS v1, p.intrinsic_v2 AS v2
                FROM fx_rates f
                LEFT JOIN predictions p ON p.date = f.date
                WHERE f.date >= ?
                ORDER BY f.date ASC
            """
            rows = rows_to_dicts(conn.execute(sql, (cutoff,)).fetchall())
            for r in rows:
                r["spread"] = (
                    (r["ib_rate"] - r["fix_mid"]) if r["ib_rate"] and r["fix_mid"] else None
                )
            return rows

    return JSONResponse(cached(f"timeseries:{days}", build))


@app.get("/api/intraday")
def intraday(hours: int = Query(24, ge=1, le=720)):
    def build():
        with db() as conn:
            if conn is None:
                return []
            cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
            sql = """
                SELECT ts, anchor_date, anchor_fix, basket_ret,
                       intrinsic_v1 AS v1, intrinsic_v2 AS v2,
                       kf_state, kf_sigma
                FROM intrinsic_intraday
                WHERE ts >= ?
                ORDER BY ts ASC
            """
            return rows_to_dicts(conn.execute(sql, (cutoff,)).fetchall())

    return JSONResponse(cached(f"intraday:{hours}", build))


@app.get("/api/weights")
def weights(days: int = Query(180, ge=1, le=3650)):
    def build():
        with db() as conn:
            if conn is None:
                return []
            cutoff = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
            sql = """
                SELECT date, w_eurusd, w_gbpusd, w_usdjpy
                FROM predictions
                WHERE date >= ?
                ORDER BY date ASC
            """
            rows = rows_to_dicts(conn.execute(sql, (cutoff,)).fetchall())
            for r in rows:
                ws = [r.get("w_eurusd"), r.get("w_gbpusd"), r.get("w_usdjpy")]
                r["sum"] = sum(w for w in ws if w is not None) if any(w is not None for w in ws) else None
            return rows

    return JSONResponse(cached(f"weights:{days}", build))


@app.get("/api/backtest")
def backtest():
    def build():
        path = REPORTS_DIR / "backtest_metrics.json"
        if not path.exists():
            return {"summary": {}, "spread_stationarity": {}}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {"summary": {}, "spread_stationarity": {}}

    return JSONResponse(cached("backtest", build))


@app.get("/api/residuals")
def residuals(days: int = Query(365, ge=1, le=3650)):
    def build():
        with db() as conn:
            if conn is None:
                return []
            cutoff = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
            sql = """
                SELECT f.date AS date,
                       (f.fix_mid - p.intrinsic_v2) AS residual
                FROM fx_rates f
                JOIN predictions p ON p.date = f.date
                WHERE f.date >= ? AND f.fix_mid IS NOT NULL
                  AND p.intrinsic_v2 IS NOT NULL
                ORDER BY f.date ASC
            """
            return rows_to_dicts(conn.execute(sql, (cutoff,)).fetchall())

    return JSONResponse(cached(f"residuals:{days}", build))


# ----------------------------------------------------------------------------
# LLM chat (Claude API)
# ----------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user|assistant", "content": "..."}]


_SYSTEM_PROMPT = """You are the on-board analyst for the USD/TND Intrinsic Value Model — a
quantitative pipeline that estimates a real-time fair value for the Tunisian Dinar
against the US Dollar. You speak as a knowledgeable guide to a user exploring the
dashboard. Keep answers concise and substantive; avoid generic AI disclaimers.

Model architecture you must know:
- Basket baseline: rolling 90-day OLS regression of log(USD/TND fix) on log-returns
  of EUR/USD, GBP/USD, USD/JPY. Newey-West HAC standard errors are used.
- Stochastic liquidity adjustment: AR(1) Kalman filter on the spread between the
  interbank rate (IB, published with a one-day lag) and the BCT fixing, with
  parameters (c, φ, Q, R) calibrated by joint maximum likelihood.
- Intrinsic value = Basket baseline + Kalman-filtered spread.
- Premium/discount = (IB − Intrinsic) / Intrinsic in basis points. Positive = local
  USD scarcity. Negative = USD surplus.
- A 2-regime Markov-switching extension is also fit (quiet vs stressed liquidity).
- Walk-forward backtest with MAE/RMSE/MAPE/DA, DM test vs random walk, Ljung-Box
  on residuals, ADF/KPSS on the spread.

When the user asks a question, ground your answer in the current snapshot data
provided to you, and explain in plain language what is happening on the market and
in the model. If the user asks about model design, cite the relevant section."""


def _resolve_llm_provider() -> tuple[str, str]:
    """
    Picks the first available LLM provider from env vars. Returns
    (provider_name, api_key). Order: Groq, Anthropic. Env var names are
    matched case-insensitively to tolerate edits like `groq_API_KEY`.
    """
    env = {k.upper(): v for k, v in os.environ.items()}
    for name, key_env in (("groq", "GROQ_API_KEY"),
                          ("anthropic", "ANTHROPIC_API_KEY")):
        v = (env.get(key_env) or "").strip()
        if v:
            return name, v
    return "", ""


def _build_snapshot_text() -> str:
    """Re-read the snapshot endpoint output and stringify it for the system prompt."""
    try:
        resp = snapshot()
        return json.dumps(json.loads(resp.body), default=str, indent=2)
    except Exception:
        return ""


def _call_groq(api_key: str, system: str, messages: list[dict]) -> str:
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        timeout=60,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            # Groq's fastest capable model at the time of writing
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 600,
            "messages": [{"role": "system", "content": system}, *messages],
        },
    )
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()


def _call_anthropic(api_key: str, system: str, messages: list[dict]) -> str:
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        timeout=60,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 600,
            "system": system,
            "messages": messages,
        },
    )
    r.raise_for_status()
    j = r.json()
    return "".join(
        b.get("text", "") for b in j.get("content", []) if b.get("type") == "text"
    ).strip()


@app.post("/api/chat")
def chat(req: ChatRequest):
    provider, api_key = _resolve_llm_provider()
    if not provider:
        return JSONResponse({
            "reply": "⚠ No LLM key found. Add GROQ_API_KEY or ANTHROPIC_API_KEY to your .env "
                     "(see .env.example) and restart the server."
        }, status_code=200)

    sys = _SYSTEM_PROMPT
    snap_text = _build_snapshot_text()
    if snap_text:
        sys += "\n\nCURRENT SNAPSHOT (use this to ground answers):\n" + snap_text

    history = (req.history or [])[-16:]
    msgs = [*history, {"role": "user", "content": req.message}]

    try:
        if provider == "groq":
            text = _call_groq(api_key, sys, msgs)
        else:
            text = _call_anthropic(api_key, sys, msgs)
        return {"reply": text or "(empty response)", "provider": provider}
    except Exception as e:
        return JSONResponse(
            {"reply": f"⚠ Chat error ({provider}): {type(e).__name__}: {e}"},
            status_code=200,
        )


# ----------------------------------------------------------------------------
# Static SPA mount
# ----------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(idx)
    raise HTTPException(404, "static/index.html not found")


if __name__ == "__main__":
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=False)
