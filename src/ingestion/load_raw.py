"""
Charge les 9 CSV Olist dans le schéma raw de DuckDB.
Lancer depuis la racine du projet : python src/ingestion/load_raw.py
"""
import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH      = PROJECT_ROOT / "data" / "duckdb" / "olist.db"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

TABLES = {
    "orders":               "olist_orders_dataset.csv",
    "order_items":          "olist_order_items_dataset.csv",
    "order_payments":       "olist_order_payments_dataset.csv",
    "order_reviews":        "olist_order_reviews_dataset.csv",
    "customers":            "olist_customers_dataset.csv",
    "products":             "olist_products_dataset.csv",
    "sellers":              "olist_sellers_dataset.csv",
    "geolocation":          "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}


def load(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    total = 0
    for table, csv_file in TABLES.items():
        csv_path = (RAW_DIR / csv_file).as_posix()
        con.execute(f"DROP TABLE IF EXISTS raw.{table}")
        con.execute(
            f"CREATE TABLE raw.{table} AS "
            f"SELECT * FROM read_csv_auto('{csv_path}')"
        )
        n = con.execute(f"SELECT COUNT(*) FROM raw.{table}").fetchone()[0]
        total += n
        print(f"  raw.{table:<24}: {n:>9,} lignes")
    print(f"\n  Total : {total:,} lignes  →  {DB_PATH}")


if __name__ == "__main__":
    print(f"Base   : {DB_PATH}")
    print(f"Source : {RAW_DIR}\n")
    with duckdb.connect(str(DB_PATH)) as con:
        load(con)
    print("\nIngestion terminée.")
