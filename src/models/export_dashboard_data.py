"""
Export Parquet pour le dashboard déployé (Streamlit Community Cloud).

app/app.py lit normalement data/duckdb/olist.db (gitignored, 93 Mo, généré). Un clone
du dépôt sur Streamlit Cloud n'a pas ce fichier — ce script exporte, en colonnes
strictement nécessaires à l'app (vérifié en lisant app/app.py, pas deviné), les 7
tables dont il a besoin vers data/dashboard_export/*.parquet, versionné dans git
(pas couvert par les règles .gitignore actuelles, qui ne couvrent que data/raw/,
data/duckdb/ et les motifs *.db/*.duckdb).

app/app.py lit ces fichiers via des VUES DuckDB schema-qualifiées quand olist.db est
absent (voir _connect() dans app/app.py) — même SQL dans les deux modes, seule la
source de données change.

Séquencement obligatoire : lancer APRÈS `python src/models/predict.py` (a besoin de
main.order_risk_scores / main.order_risk_drivers). Lancer depuis la racine du projet :
python src/models/export_dashboard_data.py
"""
from datetime import datetime, timezone
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "olist.db"
EXPORT_DIR = PROJECT_ROOT / "data" / "dashboard_export"

# Une entrée = (nom de fichier, requête). order_risk_drivers est restreint à
# driver_rank=1 : c'est le SEUL rang que app/app.py lit jamais (vérifié par lecture
# du code) — exporter les rangs 2/3 gonflerait le fichier x3 pour des lignes jamais
# requêtées. La colonne driver_rank elle-même n'est pas exportée (toujours 1) ; la vue
# créée côté app.py la resynthétise pour garder le SQL partagé identique dans les deux
# modes (clause `where dr.driver_rank = 1` inchangée).
EXPORTS: dict[str, str] = {
    "fct_orders": """
        select order_id, customer_id, product_key, date_key, is_late, total_price
        from marts.fct_orders
    """,
    "dim_customer": """
        select distinct c.customer_id, c.customer_state
        from marts.dim_customer c
        join marts.fct_orders o on o.customer_id = c.customer_id
    """,
    "dim_product": """
        select distinct p.product_key, p.product_category_name
        from marts.dim_product p
        join marts.fct_orders o on o.product_key = p.product_key
    """,
    "dim_date": """
        select distinct d.date_key, d.annee, d.mois
        from marts.dim_date d
        join marts.fct_orders o on o.date_key = d.date_key
    """,
    "int_orders_enriched": """
        select order_id, review_score, order_purchase_timestamp,
               order_estimated_delivery_date, order_delivered_customer_date
        from intermediate.int_orders_enriched
    """,
    "order_risk_scores": """
        select order_id, risk_probability, risk_tier, is_in_sample
        from main.order_risk_scores
    """,
    "order_risk_drivers": """
        select order_id, feature_name
        from main.order_risk_drivers
        where driver_rank = 1
    """,
}


def export_all(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """
    Exécute chaque export en COPY TO Parquet (compression ZSTD — le format binaire
    en colonnes de DuckDB, pas un CSV : plus petit et directement relisible par
    read_parquet() côté app.py sans reparser). Retourne les tailles de fichiers pour
    que main() puisse imprimer un résumé vérifiable (jamais un total supposé).
    """
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for name, query in EXPORTS.items():
        path = EXPORT_DIR / f"{name}.parquet"
        con.execute(f"COPY ({query}) TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        sizes[name] = path.stat().st_size
    return sizes


def write_manifest(sizes: dict[str, int]) -> None:
    """
    Petit fichier texte horodaté, versionné avec les .parquet : seule façon de savoir,
    en regardant le dépôt GitHub, à quelle date la démo publique a été figée — sans
    lui, "la démo est peut-être périmée" n'est vérifiable par personne.
    """
    manifest = EXPORT_DIR / "MANIFEST.txt"
    lines = [
        f"Export généré le {datetime.now(timezone.utc).isoformat()} par "
        "src/models/export_dashboard_data.py",
        "Instantané figé pour la démo Streamlit Cloud — PAS mis à jour automatiquement.",
        "Pour rafraîchir : relancer predict.py puis ce script, puis commit + push.",
        "",
    ]
    for name, size in sizes.items():
        lines.append(f"  {name}.parquet — {size / 1024:.1f} Ko")
    manifest.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        sizes = export_all(con)

    write_manifest(sizes)

    total = sum(sizes.values())
    print(f"Export écrit dans {EXPORT_DIR}")
    for name, size in sizes.items():
        print(f"  {name}.parquet — {size / 1024:>8.1f} Ko")
    print(f"TOTAL : {total / 1024:.1f} Ko = {total / 1024 / 1024:.2f} Mo")
    if total > 50 * 1024 * 1024:
        raise AssertionError(
            f"Export total ({total / 1024 / 1024:.1f} Mo) dépasse le seuil de 50 Mo "
            "fixé pour un fichier versionné dans git — ne pas committer tel quel."
        )


if __name__ == "__main__":
    main()
