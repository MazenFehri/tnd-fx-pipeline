"""
Basket OLS, rolling regression, Kalman filter on spread — pure numpy only.
"""
import math
from typing import Any, Dict

import numpy as np
import pandas as pd


def _norm_sf_two_sided(z: float) -> float:
    """Two-sided p-value for standard normal: 2 · (1 − Φ(|z|))."""
    return math.erfc(abs(z) / math.sqrt(2))


def newey_west_cov(X: np.ndarray, residuals: np.ndarray, lags: int | None = None) -> np.ndarray:
    """
    Newey-West (1987) HAC covariance for OLS β.

    Var(β̂) = (X'X)^{-1} · S · (X'X)^{-1}

        S = Γ₀ + Σ_{l=1..L} w(l) · (Γ_l + Γ_l')        w(l) = 1 − l/(L+1)
        Γ_l = Σ_t X_t X_{t-l}' · e_t e_{t-l}

    Automatic bandwidth (Newey-West 1994): L = ⌊4·(T/100)^(2/9)⌋.
    """
    X = np.asarray(X, dtype=float)
    e = np.asarray(residuals, dtype=float).ravel()
    T, k = X.shape
    if lags is None:
        lags = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    lags = max(0, int(lags))

    XX_inv = np.linalg.inv(X.T @ X)
    # Γ₀
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        Xt   = X[l:]
        Xtl  = X[:-l]
        et   = e[l:]
        etl  = e[:-l]
        gamma_l = (Xt * et[:, None]).T @ (Xtl * etl[:, None])
        S += w * (gamma_l + gamma_l.T)
    return XX_inv @ S @ XX_inv


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

    Returns: intercept, weights, r_squared, plus Newey-West HAC standard
    errors, t-stats, and two-sided p-values for each coefficient. The HAC
    correction is appropriate here because the daily basket-residual is
    likely to be autocorrelated and heteroskedastic.
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
    beta, r2, y_hat = _lstsq_ols(y, X)
    resid = y - y_hat

    try:
        cov_nw = newey_west_cov(X, resid)
        se = np.sqrt(np.maximum(np.diag(cov_nw), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)

    t_stat = np.where(se > 0, beta / se, np.nan)
    p_val = np.array([_norm_sf_two_sided(float(t)) if np.isfinite(t) else np.nan for t in t_stat])

    keys = ["intercept", "w_EURUSD", "w_GBPUSD", "w_USDJPY"]
    out: Dict[str, Any] = {k: float(beta[i]) for i, k in enumerate(keys)}
    out["r_squared"] = float(r2) if np.isfinite(r2) else np.nan
    out["n_obs"] = int(len(df_clean))
    out["nw_lags"] = int(np.floor(4.0 * (len(df_clean) / 100.0) ** (2.0 / 9.0)))
    for i, k in enumerate(keys):
        out[f"se_{k}"] = float(se[i]) if np.isfinite(se[i]) else np.nan
        out[f"t_{k}"]  = float(t_stat[i]) if np.isfinite(t_stat[i]) else np.nan
        out[f"p_{k}"]  = float(p_val[i]) if np.isfinite(p_val[i]) else np.nan
    return out


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


def kalman_ar1_loglik(y: np.ndarray, c: float, phi: float, Q: float, R: float) -> float:
    """
    Exact Gaussian log-likelihood for the AR(1) + observation-noise state-space:
        x_t = c + φ · x_{t-1} + w_t,   w_t ~ N(0, Q)
        y_t = x_t + v_t,                v_t ~ N(0, R)

    Diffuse-ish initialization: x_0 = mean(y), P_0 = Var(y). Innovations form.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 5 or Q <= 0 or R <= 0:
        return -np.inf
    x = float(np.mean(y))
    P = float(np.var(y, ddof=1)) if n > 1 else 1.0
    ll = 0.0
    for t in range(n):
        # predict
        x_pred = c + phi * x
        P_pred = phi * phi * P + Q
        # innovation
        v = y[t] - x_pred
        S = P_pred + R
        if S <= 0:
            return -np.inf
        ll += -0.5 * (math.log(2.0 * math.pi * S) + v * v / S)
        # update
        K = P_pred / S
        x = x_pred + K * v
        P = (1.0 - K) * P_pred
    return ll


def mle_kalman_ar1(spread_series: pd.Series, restarts: int = 4) -> Dict[str, float]:
    """
    Joint MLE of (c, φ, Q, R) for the AR(1) state-space on the spread series.

    Strategy: Nelder-Mead on log-Q and log-R (positivity) with a logit-style
    transform on φ to keep |φ| < 1 for stationarity. Multiple restarts from
    different OLS-based seeds; returns the highest-likelihood fit.

    Returns dict with c, phi, Q, R, loglik, n_obs. Falls back to OLS-based
    estimates if scipy is unavailable.
    """
    s = np.asarray(spread_series.dropna().values, dtype=float)
    n = len(s)
    if n < 10:
        return {"c": 0.0, "phi": 0.0, "Q": 1e-8, "R": 1e-8, "loglik": float("nan"), "n_obs": n, "method": "fallback"}

    # OLS seed
    yt, yl = s[1:], s[:-1]
    X = np.column_stack([np.ones(len(yt)), yl])
    beta, *_ = np.linalg.lstsq(X, yt, rcond=None)
    c0, phi0 = float(beta[0]), float(beta[1])
    resid = yt - X @ beta
    Q0 = max(float(np.var(resid, ddof=1)), 1e-12)
    var_s = max(float(np.var(s, ddof=1)), 1e-12)

    try:
        from scipy.optimize import minimize
    except ImportError:
        # Pure-numpy fallback: just return the OLS estimate with R = 0.25 * Var(s)
        return {"c": c0, "phi": phi0, "Q": Q0, "R": 0.25 * var_s,
                "loglik": kalman_ar1_loglik(s, c0, phi0, Q0, 0.25 * var_s),
                "n_obs": n, "method": "ols-fallback (no scipy)"}

    # Parameterization: phi via tanh(z) to enforce |φ| < 1; Q, R via exp.
    def unpack(theta):
        c, z, logQ, logR = theta
        phi = math.tanh(z)
        return float(c), float(phi), float(math.exp(logQ)), float(math.exp(logR))

    def neg_ll(theta):
        c, phi, Q, R = unpack(theta)
        return -kalman_ar1_loglik(s, c, phi, Q, R)

    seeds = [
        [c0, math.atanh(np.clip(phi0, -0.99, 0.99)), math.log(max(Q0, 1e-12)), math.log(max(0.25 * var_s, 1e-12))],
        [c0, 0.0, math.log(var_s / 2), math.log(var_s / 2)],
        [0.0, math.atanh(0.5), math.log(var_s / 4), math.log(var_s / 4)],
        [c0, math.atanh(np.clip(phi0, -0.99, 0.99)), math.log(max(Q0, 1e-12) * 2), math.log(max(Q0, 1e-12) * 0.5)],
    ][:max(1, restarts)]

    best = None
    for x0 in seeds:
        try:
            res = minimize(neg_ll, x0, method="Nelder-Mead",
                           options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 5000})
            if not np.isfinite(res.fun):
                continue
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue

    if best is None:
        return {"c": c0, "phi": phi0, "Q": Q0, "R": 0.25 * var_s,
                "loglik": kalman_ar1_loglik(s, c0, phi0, Q0, 0.25 * var_s),
                "n_obs": n, "method": "ols-fallback (optimizer failed)"}

    c, phi, Q, R = unpack(best.x)
    return {"c": c, "phi": phi, "Q": Q, "R": R,
            "loglik": float(-best.fun), "n_obs": n, "method": "mle"}


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


def kalman_filter_spread(spread_series: pd.Series, use_mle: bool = True) -> pd.Series:
    """
    Filtered Kalman state for the spread. Uses joint MLE of (c, φ, Q, R) when
    SciPy is available; falls back to the OLS-based heuristic otherwise.
    """
    y = spread_series.values.astype(float)
    if use_mle:
        mle = mle_kalman_ar1(spread_series)
        c, phi, Q, R = mle["c"], mle["phi"], mle["Q"], mle["R"]
    else:
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
