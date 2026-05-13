"""
Intraday loop runner. Calls realtime.tick() every INTERVAL seconds.

Usage:
    python run_realtime.py                # default 60s tick
    python run_realtime.py --interval 30  # tick every 30 seconds
    python run_realtime.py --once         # single tick then exit (cron mode)

Stop with Ctrl-C. Robust to transient errors — logs and continues.
"""
from __future__ import annotations

import argparse
import sqlite3
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from init_db import init_db
from realtime import DEFAULT_DB, tick


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=60, help="seconds between ticks (default 60)")
    p.add_argument("--once", action="store_true", help="single tick then exit")
    p.add_argument("--db", type=str, default=str(DEFAULT_DB))
    args = p.parse_args()

    init_db(Path(args.db))
    print(f"[{_stamp()}] realtime loop starting · db={args.db} · interval={args.interval}s")

    while True:
        try:
            with sqlite3.connect(args.db) as conn:
                r = tick(conn)
            if r.get("ok"):
                print(
                    f"[{_stamp()}] tick ok · ts={r['ts']} · "
                    f"v1={r['intrinsic_v1']:.5f} · v2={r['intrinsic_v2']:.5f} · "
                    f"basket={r['basket_ret_pct']:+.4f}% · kf={r['kf_state']:+.5f}"
                )
            else:
                print(f"[{_stamp()}] tick skipped · {r.get('reason')}")
        except KeyboardInterrupt:
            print(f"\n[{_stamp()}] interrupted — exiting")
            return 0
        except Exception:
            print(f"[{_stamp()}] tick error:")
            traceback.print_exc()

        if args.once:
            return 0
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
