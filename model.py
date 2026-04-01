"""
Basket OLS, rolling regression, Kalman filter on spread — pure numpy only.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd


def _lstsq_ols(y: np.ndarray, X: np.ndarray):
    """Return beta, r_squared, y_hat."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    sse = float(np.sum((y - y_hat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan
    return beta, r2, y_hat


def fit_ols(df_clean: pd.DataFrame) -> Dict[str, Any]:
    """
    y = ret_Fix, X = [1, ret_EURUSD, ret_GBPUSD, ret_USDJPY].
    Returns intercept, weights, r_squared.
    """
    y = df_clean["ret_Fix"].values
    X = np.column_stack(
        [
            np.ones(len(df_clean)),
            df_clean["ret_EURUSD"].values,
            df_clean["ret_GBPUSD"].values,
            df_clean["ret_USDJPY"].values,
        ]
    )
    beta, r2, _ = _lstsq_ols(y, X)
    return {
        "intercept": float(beta[0]),
        "w_EURUSD": float(beta[1]),
        "w_GBPUSD": float(beta[2]),
        "w_USDJPY": float(beta[3]),
        "r_squared": float(r2) if np.isfinite(r2) else np.nan,
    }


def rolling_weights(df_clean: pd.DataFrame, window: int = 90) -> pd.DataFrame:
    """
    Rolling OLS per window ending at each row. Columns: w_const, w_EURUSD,
    w_GBPUSD, w_USDJPY, R2.
    """
    y = df_clean["ret_Fix"].values
    Xn = df_clean[["ret_EURUSD", "ret_GBPUSD", "ret_USDJPY"]].values
    n = len(y)
    out = np.full((n, 5), np.nan)
    for i in range(window - 1, n):
        sl = slice(i - window + 1, i + 1)
        y_w = y[sl]
        X_w = np.column_stack([np.ones(window), Xn[sl]])
        beta, r2, _ = _lstsq_ols(y_w, X_w)
        out[i, :4] = beta
        out[i, 4] = r2
    cols = ["w_const", "w_EURUSD", "w_GBPUSD", "w_USDJPY", "R2"]
    return pd.DataFrame(out, columns=cols, index=df_clean.index)


def _ar1_fit_numpy(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 3:
        return 0.0, 0.0, 1.0
    yt = y[1:]
    yl = y[:-1]
    X = np.column_stack([np.ones(len(yt)), yl])
    beta, _, y_hat = _lstsq_ols(yt, X)
    c, phi = float(beta[0]), float(beta[1])
    e = yt - y_hat
    sig = float(np.std(e, ddof=1)) if len(e) > 1 else 1.0
    return c, phi, max(sig**2, 1e-12)


def _kalman_ar1_obs(
    y: np.ndarray,
    c: float,
    phi: float,
    Q: float,
    R: float,
) -> np.ndarray:
    """State x_t = c + phi*x_{t-1} + w; observe y_t = x_t + v."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    x_f = np.zeros(n)
    P = np.zeros(n)
    x_f[0] = y[0] if np.isfinite(y[0]) else 0.0
    P[0] = R
    for t in range(1, n):
        x_pred = c + phi * x_f[t - 1]
        P_pred = phi * phi * P[t - 1] + Q
        if not np.isfinite(y[t]):
            x_f[t] = x_pred
            P[t] = P_pred
            continue
        S = P_pred + R
        K = P_pred / S if S > 0 else 0.0
        x_f[t] = x_pred + K * (y[t] - x_pred)
        P[t] = (1.0 - K) * P_pred
    return x_f


def kalman_filter_spread(spread_series: pd.Series) -> pd.Series:
    """
    AR(1) on spread for phi and Q; Kalman with R = 0.25 * var(spread).
    """
    y = spread_series.values.astype(float)
    c, phi, Q = _ar1_fit_numpy(y)
    v = float(np.nanvar(y))
    R = max(0.25 * v, 1e-12)
    x_f = _kalman_ar1_obs(y, c, phi, Q, R)
    return pd.Series(x_f, index=spread_series.index)


def compute_intrinsic(
    df_clean: pd.DataFrame,
    ols_weights: Dict[str, float] | pd.DataFrame,
    kf_series: pd.Series,
) -> pd.DataFrame:
    """
    basket_ret = intercept + w1*r1 + w2*r2 + w3*r3
    intrinsic_v1 = Fix_Mid.shift(1) * exp(basket_ret)
    intrinsic_v2 = intrinsic_v1 + kf_spread
    ols_weights: either full-sample dict (broadcast) or rolling DataFrame aligned by index.
    """
    df = df_clean.copy()
    r1 = df["ret_EURUSD"].values
    r2 = df["ret_GBPUSD"].values
    r3 = df["ret_USDJPY"].values

    if isinstance(ols_weights, dict):
        br = (
            ols_weights["intercept"]
            + ols_weights["w_EURUSD"] * r1
            + ols_weights["w_GBPUSD"] * r2
            + ols_weights["w_USDJPY"] * r3
        )
    else:
        ow = ols_weights.reindex(df.index)
        br = (
            ow["w_const"].values
            + ow["w_EURUSD"].values * r1
            + ow["w_GBPUSD"].values * r2
            + ow["w_USDJPY"].values * r3
        )

    df["basket_ret"] = br
    df["intrinsic_v1"] = df["Fix_Mid"].shift(1) * np.exp(br)
    kf = kf_series.reindex(df.index)
    df["kf_spread"] = kf.values
    df["intrinsic_v2"] = df["intrinsic_v1"] + df["kf_spread"]
    return df
