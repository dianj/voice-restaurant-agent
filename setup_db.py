"""Rebuild the demo database and the CSV export.

Usage: python setup_db.py
The data is deterministic (fixed seed). Existing reservations are deleted.
"""
from restaurant import config
from restaurant.data import build_database

if __name__ == "__main__":
    counts = build_database(config.DB_PATH, config.CSV_PATH)
    print(f"Created {config.DB_PATH}: " + ", ".join(f"{n} {table}" for table, n in counts.items()))
    print(f"Exported {config.CSV_PATH}")
