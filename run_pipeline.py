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


def _merge_fx_row(conn, date_str, fetched, fix_payload):
    """
    Merge a freshly fetched FX row with existing AM/PM fixings.

    fix_payload is the {'fix_am','fix_pm','fix_mid'} dict from load_bct_fixing();
    None slots preserve whatever is already stored for date_str. Slot-level merge
    means a morning scrape doesn't wipe an existing PM fixing and vice versa.
    """
    cur = conn.execute(
        "SELECT fix_am, fix_pm, fix_mid, ib_rate FROM fx_rates WHERE date = ?",
        (date_str,),
    )
    ex = cur.fetchone()
    ex_am, ex_pm, ex_mid, ex_ib = (ex if ex else (None, None, None, None))

    new_am = fix_payload.get("fix_am") if fix_payload else None
    new_pm = fix_payload.get("fix_pm") if fix_payload else None

    fix_am = new_am if new_am is not None else ex_am
    fix_pm = new_pm if new_pm is not None else ex_pm

    # Recompute mid from the merged AM/PM. Fall back to provided mid, then existing.
    if fix_am is not None and fix_pm is not None:
        fix_mid = (fix_am + fix_pm) / 2.0
    elif fix_am is not None or fix_pm is not None:
        fix_mid = fix_am if fix_am is not None else fix_pm
    else:
        fix_mid = (fix_payload or {}).get("fix_mid") or ex_mid

    return {
        "date": date_str,
        "eurusd": fetched["eurusd"],
        "gbpusd": fetched["gbpusd"],
        "usdjpy": fetched["usdjpy"],
        "fix_am": fix_am,
        "fix_pm": fix_pm,
        "fix_mid": fix_mid,
        "ib_rate": ex_ib,
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

    # Optional: refresh macro covariates and re-run the overlay regression.
    # Both are best-effort — failures (network outage, FRED unreachable) do not
    # break the daily pipeline.
    try:
        from fetch_macro import fetch_all, upsert_macro
        series = fetch_all()
        if any(series.values()):
            with sqlite3.connect(str(db_path)) as c3:
                n = upsert_macro(c3, series)
            print(f"[run_pipeline] macro: upserted {n} rows")
            from macro_overlay import run as run_overlay
            ov = run_overlay()
            if ov.get("ok"):
                import json as _json
                (ROOT / "reports" / "macro_overlay.json").write_text(
                    _json.dumps(ov, indent=2, default=str))
                print("[run_pipeline] macro overlay: wrote reports/macro_overlay.json")
    except Exception as _e:
        print(f"[run_pipeline] macro step skipped: {_e}")

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
