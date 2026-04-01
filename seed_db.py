#!/usr/bin/env python3
"""
Seed the database with historical data.

This script imports:
1. Historical FX rates and BCT fixings from FX-CLEAN-Data.csv
2. Interbank USD/TND rates from ib.csv (if available)

Usage:
    python seed_db.py
"""

import sqlite3
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "tnd.db"
FX_CSV_PATH = ROOT / "data" / "FX-CLEAN-Data.csv"
IB_CSV_PATH = ROOT / "data" / "ib.csv"


def import_fx_data():
    """Import historical FX rates and BCT fixings from CSV."""
    print("Importing FX data from FX-CLEAN-Data.csv...")

    # Connect to the database
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Open the CSV file
    with open(FX_CSV_PATH, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        imported_count = 0
        for row in reader:
            # Parse the row
            date_str = row[8]  # Exchange Date
            if not date_str:
                continue
            try:
                # Convert DD/MM/YYYY to YYYY-MM-DD
                date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                date = date_obj.strftime('%Y-%m-%d')
                eurusd = float(row[1]) if row[1] else None
                gbpusd = float(row[2]) if row[2] else None
                usdjpy = float(row[3]) if row[3] else None
                fix_mid = float(row[9]) if row[9] else None
                # Insert or replace
                cursor.execute('''
                    INSERT OR REPLACE INTO fx_rates (date, eurusd, gbpusd, usdjpy, fix_mid, created_at)
                    VALUES (?, ?, ?, ?, ?, datetime('now'))
                ''', (date, eurusd, gbpusd, usdjpy, fix_mid))
                imported_count += 1
            except ValueError:
                continue

    conn.commit()
    conn.close()
    print(f"Successfully imported {imported_count} FX records")


def import_ib_rates():
    """Import IB rates from CSV into SQLite database."""
    if not IB_CSV_PATH.exists():
        print("IB rates file not found, skipping IB import.")
        return True

    print("Importing IB rates from ib.csv...")

    try:
        # Read CSV
        df = pd.read_csv(IB_CSV_PATH)
        print(f"Loaded {len(df)} rows from {IB_CSV_PATH}")

        # Validate required columns
        if 'Date' not in df.columns or 'USD' not in df.columns:
            print("Warning: IB CSV must have 'Date' and 'USD' columns, skipping IB import")
            return True

        # Format dates
        df['date_fmt'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        # Connect to database
        conn = sqlite3.connect(str(DB_PATH))

        # Import rates
        imported_count = 0
        for _, row in df.iterrows():
            usd_rate = row['USD']
            date_str = row['date_fmt']

            # Update existing row with IB rate
            cursor = conn.execute(
                "UPDATE fx_rates SET ib_rate = ? WHERE date = ?",
                (usd_rate, date_str)
            )

            if cursor.rowcount > 0:
                imported_count += 1

        conn.commit()
        conn.close()

        print(f"Successfully imported {imported_count} IB rates")
        return True

    except Exception as e:
        print(f"Error importing IB rates: {e}")
        return False


def main():
    """Main seeding function."""
    print("Starting database seeding...")

    # Import FX data first
    import_fx_data()

    # Then import IB rates
    import_ib_rates()

    print("Database seeding completed successfully!")


if __name__ == "__main__":
    main()