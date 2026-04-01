"""
Master orchestrator: fetch FX, optional BCT fix, predict, Excel, Telegram.
"""
import sqlite3
import traceback
from pathlib import Path

from fetch_daily import (
    DEFAULT_DB,
    fetch_fx_rates,
    load_bct_fixing,
    upsert_fx_rates,
)
from init_db import init_db
from export_excel import write_excel_report
from notify_telegram import send_telegram
from predict import predict_today


ROOT = Path(__file__).resolve().parent


def _merge_fx_row(conn, date_str, fetched, fix_from_env):
    """Preserve existing fix_mid / ib_rate when new values are None."""
    cur = conn.execute(
        "SELECT fix_mid, ib_rate FROM fx_rates WHERE date = ?",
        (date_str,),
    )
    ex = cur.fetchone()
    fix_mid = fix_from_env if fix_from_env is not None else (ex[0] if ex else None)
    ib_rate = ex[1] if ex else None
    return {
        "date": date_str,
        "eurusd": fetched["eurusd"],
        "gbpusd": fetched["gbpusd"],
        "usdjpy": fetched["usdjpy"],
        "fix_mid": fix_mid,
        "ib_rate": ib_rate,
    }


def main():
    db_path = ROOT / "data" / "tnd.db"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))

    fetched = fetch_fx_rates()
    d = fetched["date"]
    fix_env = load_bct_fixing()
    row = _merge_fx_row(conn, d, fetched, fix_env)
    upsert_fx_rates(conn, row)
    conn.commit()
    print(f"[run_pipeline] Upserted fx_rates for {d} (fix_mid={row['fix_mid']})")

    pred = predict_today(conn)
    conn.close()

    conn2 = sqlite3.connect(str(db_path))
    out = write_excel_report(conn2, pred)
    conn2.close()
    print(f"[run_pipeline] Excel: {out}")

    send_telegram(pred)

    print("\n--- Summary ---")
    print(f"ok:          {pred.get('ok')}")
    print(f"date:        {pred.get('date')}")
    print(f"intrinsic_v2:{pred.get('intrinsic_v2')}")
    print(f"intrinsic_v1:{pred.get('intrinsic_v1')}")
    print(f"prev_fix:    {pred.get('prev_fix')}")
    print(f"r_squared:   {pred.get('r_squared')}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    raise SystemExit(0)
