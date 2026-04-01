"""
Send daily prediction summary via Telegram Bot API (no cost).
"""
import os
from typing import Any, Dict

import requests


def send_telegram(prediction_dict: Dict[str, Any]) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print(
            "[notify_telegram] Warning: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing; skip."
        )
        return

    d = prediction_dict.get("date", "")
    iv2 = prediction_dict.get("intrinsic_v2")
    iv1 = prediction_dict.get("intrinsic_v1")
    pf = prediction_dict.get("prev_fix")
    pct = prediction_dict.get("basket_ret_pct")
    we = prediction_dict.get("w_eurusd")
    wg = prediction_dict.get("w_gbpusd")
    wj = prediction_dict.get("w_usdjpy")
    kf = prediction_dict.get("kf_spread")
    r2 = prediction_dict.get("r_squared")

    def fmt(x, nd=4):
        if x is None:
            return "N/A"
        if isinstance(x, (int, float)):
            return f"{x:.{nd}f}"
        return str(x)

    text = f"""USD/TND Daily Prediction — {d}

Predicted rate (full model): {fmt(iv2)} TND
Basket only:                 {fmt(iv1)} TND
Previous BCT fixing:         {fmt(pf)} TND
Est. change:                 {f"{pct:+.3f}%" if pct is not None else "N/A"}

Weights: EUR {fmt(we,3)} | GBP {fmt(wg,3)} | JPY {fmt(wj,3)}
Kalman spread: {fmt(kf)} | R^2: {fmt(r2,3)}
"""

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": chat_id, "text": text},
            timeout=30,
        )
        r.raise_for_status()
        print("[notify_telegram] Message sent.")
    except Exception as e:
        print(f"[notify_telegram] Warning: failed to send: {e}")
