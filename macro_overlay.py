"""
Macro-overlay regression for the IB-Fix spread.

Question: does adding global macro covariates (Brent oil log-return, DXY
log-return, VIX level) materially improve the explanatory power of a linear
model for the spread or for ret_Fix beyond the basket alone?

We fit four nested models on the joined dataset:

    M0:  spread_t ~ const                          (baseline)
    M1:  spread_t ~ const + spread_{t-1}            (AR(1) only)
    M2:  spread_t ~ const + spread_{t-1} + basket_returns
    M3:  spread_t ~ const + spread_{t-1} + basket_returns + macros

For each model we report R², adjusted R², AIC, and the incremental R² over
the previous nested specification. Each coefficient is reported with
Newey-West HAC inference (lags chosen by NW 1994 rule).

Run:  python macro_overlay.py [--db data/tnd.db]
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from clean_returns import load_and_clean
from model import newey_west_cov, _norm_sf_two_sided

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "tnd.db"
OUT_JSON = ROOT / "reports" / "macro_overlay.json"


# ---------------------------------------------------------------------------
# Data load
# ---------------------------------------------------------------------------

def _load_macro(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        df = pd.read_sql_query(
            "SELECT date, brent, dxy, vix FROM macro ORDER BY date ASC", conn
        )
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    for c in ("brent", "dxy", "vix"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_joined(conn: sqlite3.Connection) -> pd.DataFrame:
    """Inner-join clean FX/spread with macro covariates and derive features."""
    base = load_and_clean(conn, lookback_days=10_000)
    if base.empty:
        return base
    macro = _load_macro(conn)
    if macro.empty:
        return pd.DataFrame()

    df = base.merge(macro, on="date", how="inner")
    # Forward-fill macros over short gaps (holidays in one market, not the other)
    df[["brent", "dxy", "vix"]] = df[["brent", "dxy", "vix"]].ffill().bfill()

    df["ret_brent"] = np.log(df["brent"] / df["brent"].shift(1))
    df["ret_dxy"]   = np.log(df["dxy"]   / df["dxy"].shift(1))
    df["d_vix"]     = df["vix"].diff()
    df["spread_lag"] = df["spread_pub"].shift(1)

    keep = ["date", "spread_pub", "spread_lag", "ret_EURUSD", "ret_GBPUSD",
            "ret_USDJPY", "ret_brent", "ret_dxy", "d_vix"]
    return df[keep].dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# OLS with Newey-West
# ---------------------------------------------------------------------------

def _ols_hac(y: np.ndarray, X: np.ndarray, names: List[str]) -> Dict[str, Any]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n, k = X.shape
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)
    sigma2 = ss_res / max(n - k, 1)
    aic = n * math.log(ss_res / n) + 2 * k if ss_res > 0 else float("nan")
    bic = n * math.log(ss_res / n) + math.log(n) * k if ss_res > 0 else float("nan")

    try:
        cov = newey_west_cov(X, resid)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    t = np.where(se > 0, beta / se, np.nan)
    p = np.array([_norm_sf_two_sided(float(z)) if np.isfinite(z) else np.nan for z in t])

    coefs = []
    for i, nm in enumerate(names):
        coefs.append({
            "name": nm,
            "est": float(beta[i]),
            "se":  float(se[i]) if np.isfinite(se[i]) else None,
            "t":   float(t[i])  if np.isfinite(t[i])  else None,
            "p":   float(p[i])  if np.isfinite(p[i])  else None,
        })
    return {
        "n": int(n), "k": int(k), "R2": float(r2), "adj_R2": float(adj_r2),
        "AIC": float(aic) if aic == aic else None,
        "BIC": float(bic) if bic == bic else None,
        "sigma2": float(sigma2),
        "coefs": coefs,
    }


def _design(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    y = df["spread_pub"].values.astype(float)
    blocks = [np.ones(len(df))]
    names = ["const"] + cols
    for c in cols:
        blocks.append(df[c].values.astype(float))
    X = np.column_stack(blocks)
    return y, X, names


# ---------------------------------------------------------------------------
# Run the nested-model leaderboard
# ---------------------------------------------------------------------------

def run() -> Dict[str, Any]:
    conn = sqlite3.connect(str(DEFAULT_DB))
    df = build_joined(conn)
    conn.close()
    if df.empty:
        return {"ok": False,
                "reason": "no overlap between fx_rates+predictions+macro — run fetch_macro.py first"}

    M: Dict[str, Any] = {}
    specs = {
        "M0_const":     [],
        "M1_ar1":       ["spread_lag"],
        "M2_basket":    ["spread_lag", "ret_EURUSD", "ret_GBPUSD", "ret_USDJPY"],
        "M3_macro":     ["spread_lag", "ret_EURUSD", "ret_GBPUSD", "ret_USDJPY",
                         "ret_brent", "ret_dxy", "d_vix"],
    }
    last_r2 = 0.0
    for name, cols in specs.items():
        y, X, nm = _design(df, cols)
        fit = _ols_hac(y, X, nm)
        fit["incremental_R2"] = fit["R2"] - last_r2
        last_r2 = fit["R2"]
        M[name] = fit

    return {"ok": True, "n_obs": int(len(df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "models": M}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default=str(DEFAULT_DB))
    args = ap.parse_args()

    res = run()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, default=str))
    print(f"[macro_overlay] wrote {OUT_JSON}")

    if not res.get("ok"):
        print(f"[macro_overlay] {res.get('reason')}")
        return 1

    print(f"\nN={res['n_obs']}  range={res['date_min']} -> {res['date_max']}")
    print(f"{'Model':14s}  {'k':>3s}  {'R2':>8s}  {'adjR2':>8s}  {'AIC':>10s}  {'dR2 vs prev':>12s}")
    for name, fit in res["models"].items():
        print(f"{name:14s}  {fit['k']:>3d}  {fit['R2']:>8.4f}  {fit['adj_R2']:>8.4f}  "
              f"{(fit['AIC'] or float('nan')):>10.2f}  {fit['incremental_R2']:>+12.4f}")

    print("\nM3 — coefficients (Newey-West HAC):")
    for c in res["models"]["M3_macro"]["coefs"]:
        sig = "***" if (c["p"] or 1) < 0.001 else ("**" if (c["p"] or 1) < 0.01
              else ("*" if (c["p"] or 1) < 0.05 else ("." if (c["p"] or 1) < 0.10 else "")))
        se = f"{c['se']:.5f}" if c["se"] is not None else "—"
        t  = f"{c['t']:+.3f}" if c["t"]  is not None else "—"
        p  = f"{c['p']:.4f}"  if c["p"]  is not None else "—"
        print(f"  {c['name']:14s}  est={c['est']:+.6f}  SE={se}  t={t}  p={p}  {sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
