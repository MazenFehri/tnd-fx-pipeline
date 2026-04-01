"""
Fetch daily FX rates from free APIs.

Priority:
  1) ExchangeRate-API v6 — https://www.exchangerate-api.com/ (key from env or built-in default)
  2) exchangerate.host (no key)
  3) frankfurter.app (no key)

Correct v6 usage for USD crosses: use base **USD**, not TND.
  OK:    GET .../v6/{API_KEY}/latest/USD
  Wrong: GET .../latest/TND  → rates are "per 1 TND", not EURUSD/GBPUSD/USDJPY.

Env var EXCHANGERATE_API_KEY overrides the built-in default below (use that for CI).
"""
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "tnd.db"

# ExchangeRate-API v6 — used when env EXCHANGERATE_API_KEY is unset
_EXCHANGERATE_API_KEY_DEFAULT = "ef454c6d974d063ae67a1c9a"


def _parse_rates_usd_base(data: dict) -> tuple[float, float, float]:
    """
    rates/conversion_rates: EUR, GBP, JPY = units per 1 USD
    -> EURUSD, GBPUSD (USD per unit currency), USDJPY (JPY per USD).
    """
    rates = data.get("rates") or data.get("conversion_rates") or {}
    if not all(k in rates for k in ("EUR", "GBP", "JPY")):
        raise ValueError("Missing EUR, GBP, or JPY in rates")
    eur_per_usd = float(rates["EUR"])
    gbp_per_usd = float(rates["GBP"])
    jpy_per_usd = float(rates["JPY"])
    eurusd = 1.0 / eur_per_usd
    gbpusd = 1.0 / gbp_per_usd
    usdjpy = jpy_per_usd
    return eurusd, gbpusd, usdjpy


def _fetch_exchangerate_api_v6(date_str: str) -> Optional[Dict[str, Any]]:
    """
    ExchangeRate-API v6: latest (today) or historical by calendar day.
    Uses EXCHANGERATE_API_KEY env if set, else the module default. Base must be USD.

    Docs: https://www.exchangerate-api.com/docs/
    History path: /v6/{key}/history/USD/YEAR/MONTH/DAY (month/day without leading zeros).
    Historical endpoint may require a paid plan; on failure callers fall back to other APIs.
    """
    key = (os.environ.get("EXCHANGERATE_API_KEY") or "").strip() or _EXCHANGERATE_API_KEY_DEFAULT

    today = date.today().isoformat()
    if date_str == today:
        url = f"https://v6.exchangerate-api.com/v6/{key}/latest/USD"
    else:
        y, m, d = (int(x) for x in date_str.split("-"))
        url = f"https://v6.exchangerate-api.com/v6/{key}/history/USD/{y}/{m}/{d}"

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("result") == "error":
        raise ValueError(payload.get("error-type") or "exchangerate-api.com error")
    if "conversion_rates" not in payload:
        raise ValueError("exchangerate-api.com: no conversion_rates")
    eurusd, gbpusd, usdjpy = _parse_rates_usd_base(payload)
    return {
        "date": date_str,
        "eurusd": eurusd,
        "gbpusd": gbpusd,
        "usdjpy": usdjpy,
        "source": "exchangerate-api.com",
    }


def fetch_fx_rates(date_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch EUR/USD, GBP/USD, USD/JPY for a calendar day (YYYY-MM-DD).
    Tries ExchangeRate-API v6 (if key set), then exchangerate.host, then frankfurter.
    """
    if date_str is None:
        date_str = date.today().isoformat()

    last_error = None

    try:
        out = _fetch_exchangerate_api_v6(date_str)
        if out is not None:
            print(f"[fetch_fx_rates] Source: exchangerate-api.com (v6, USD base) for {date_str}")
            return out
    except Exception as e:
        last_error = e
        print(f"[fetch_fx_rates] ExchangeRate-API v6 failed: {e}")

    url1 = f"https://api.exchangerate.host/{date_str}?base=USD&symbols=EUR,GBP,JPY"
    try:
        r = requests.get(url1, timeout=30)
        r.raise_for_status()
        payload = r.json()
        if "rates" not in payload or not payload["rates"]:
            raise ValueError("exchangerate.host: no rates in response")
        eurusd, gbpusd, usdjpy = _parse_rates_usd_base(payload)
        print(f"[fetch_fx_rates] Source: exchangerate.host for {date_str}")
        return {
            "date": date_str,
            "eurusd": eurusd,
            "gbpusd": gbpusd,
            "usdjpy": usdjpy,
            "source": "exchangerate.host",
        }
    except Exception as e:
        last_error = e
        print(f"[fetch_fx_rates] exchangerate.host failed: {e}")

    url2 = f"https://api.frankfurter.app/{date_str}?from=USD&to=EUR,GBP,JPY"
    try:
        r = requests.get(url2, timeout=30)
        r.raise_for_status()
        payload = r.json()
        eurusd, gbpusd, usdjpy = _parse_rates_usd_base(payload)
        print(f"[fetch_fx_rates] Source: frankfurter.app (fallback) for {date_str}")
        return {
            "date": date_str,
            "eurusd": eurusd,
            "gbpusd": gbpusd,
            "usdjpy": usdjpy,
            "source": "frankfurter.app",
        }
    except Exception as e:
        last_error = e
        raise RuntimeError(
            "All FX sources failed (v6 if configured, exchangerate.host, frankfurter). "
            f"Last error: {last_error}"
        ) from last_error


def load_bct_fixing_from_env() -> Optional[float]:
    """
    BCT USD/TND mid fixing from BCT_FIX_MID environment variable (e.g. GitHub secret).
    Returns None if unset or invalid — pipeline may use previous fixes only.
    """
    raw = os.environ.get("BCT_FIX_MID")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(str(raw).strip().replace(",", "."))
    except ValueError:
        print("[load_bct_fixing_from_env] Warning: BCT_FIX_MID is not a valid float; ignoring.")
        return None


def scrape_bct_fixing() -> Optional[float]:
    """
    Scrape today's USD/TND fixing from BCT website.
    Returns the mid fixing if found, else None.
    """
    url = "https://www.bct.gov.tn/bct/siteprod/cours.jsp?lang=en"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table with fixings
        table = soup.find('table')
        if not table:
            return None
        
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 4 and 'USD' in cells[1].get_text().upper():
                # Assuming columns: Currency Name, Sigle, Unit, Value
                try:
                    mid = float(cells[3].get_text().replace(',', '.'))
                    return mid
                except (IndexError, ValueError):
                    continue
        return None
    except Exception as e:
        print(f"[scrape_bct_fixing] Failed: {e}")
        return None


def load_bct_fixing() -> Optional[float]:
    """
    Get BCT fixing: first try env var, then scrape from BCT website.
    """
    fixing = load_bct_fixing_from_env()
    if fixing is not None:
        return fixing
    return scrape_bct_fixing()


def upsert_fx_rates(conn: sqlite3.Connection, row_dict: Dict[str, Any]) -> None:
    """INSERT OR REPLACE into fx_rates (full row each time)."""
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    conn.execute(
        """
        INSERT OR REPLACE INTO fx_rates
        (date, eurusd, gbpusd, usdjpy, fix_mid, ib_rate, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row_dict["date"],
            row_dict.get("eurusd"),
            row_dict.get("gbpusd"),
            row_dict.get("usdjpy"),
            row_dict.get("fix_mid"),
            row_dict.get("ib_rate"),
            now,
        ),
    )


if __name__ == "__main__":
    d = fetch_fx_rates()
    print(d)
