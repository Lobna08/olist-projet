"""
Construction des features point-in-time pour la prédiction du retard de livraison.
Écrit la table main.features_orders dans DuckDB.
Lancer depuis la racine du projet : python src/features/build_features.py
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "olist.db"

# Paramètre de lissage bayésien (credibility weighting) du taux de retard vendeur.
# Il fixe le nombre de commandes "virtuelles" attribuées au prior global dans la moyenne
# pondérée : un vendeur avec SELLER_SMOOTHING_K commandes passées pèse autant que le prior.
#   - K petit  → on fait confiance vite à l'historique propre du vendeur (bruyant si peu de data)
#   - K grand  → on tire fort vers le taux global tant que l'historique du vendeur est court
SELLER_SMOOTHING_K = 10


def compute_seller_late_rate(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Pour chaque commande, calcule le taux de retard lissé du PIRE vendeur associé,
    strictement à partir de l'historique connu au moment de l'achat (order_purchase_timestamp).

    Retourne un DataFrame (order_id, seller_late_rate_max) — une ligne par order_id.
    """
    # 1. Historique brut : une ligne par commande "delivered", avec son issue (is_late) et
    #    la date à laquelle cette issue devient connaissable (order_delivered_customer_date).
    #    Même population que stg_orders (delivered + delivered_customer_date non nul).
    history = con.execute("""
        select
            order_id,
            order_delivered_customer_date,
            (order_delivered_customer_date > order_estimated_delivery_date)::int as is_late
        from staging.stg_orders
    """).fetchdf()

    # 2. Association commande → vendeur(s), dédupliquée : un vendeur livrant plusieurs
    #    articles sur la même commande ne doit compter qu'une fois dans son propre historique.
    order_sellers = con.execute("""
        select distinct order_id, seller_id
        from staging.stg_order_items
    """).fetchdf()

    # 3. t = instant de la décision = order_purchase_timestamp de la commande à featuriser.
    purchase_ts = con.execute("""
        select order_id, order_purchase_timestamp
        from staging.stg_orders
    """).fetchdf()

    # -- Historique vendeur, avec compteurs cumulés triés par date de résultat connu --
    seller_history = order_sellers.merge(history, on="order_id", how="inner")
    seller_history = seller_history.sort_values("order_delivered_customer_date")
    seller_history["cum_n"] = seller_history.groupby("seller_id").cumcount() + 1
    seller_history["cum_late"] = seller_history.groupby("seller_id")["is_late"].cumsum()
    seller_history = seller_history.rename(
        columns={"order_delivered_customer_date": "seller_last_delivered_date"}
    )

    # -- Historique global (prior), même logique sans regroupement par vendeur --
    global_history = history.sort_values("order_delivered_customer_date").reset_index(drop=True)
    global_history["cum_n_global"] = np.arange(1, len(global_history) + 1)
    global_history["cum_late_global"] = global_history["is_late"].cumsum()
    global_history = global_history.rename(
        columns={"order_delivered_customer_date": "global_last_delivered_date"}
    )

    # 4. Grille (order_id à featuriser, seller_id) x t, triée par t pour l'as-of join.
    #    inner join : stg_order_items couvre TOUTES les commandes (y compris non-delivered),
    #    alors que purchase_ts ne couvre que la population delivered. Un "left" laisserait
    #    des order_purchase_timestamp NULL pour les commandes hors population.
    targets = order_sellers.merge(purchase_ts, on="order_id", how="inner")
    targets = targets.sort_values("order_purchase_timestamp")

    # -- As-of join VENDEUR : pour chaque (order_id, seller_id), on prend la dernière ligne
    #    de l'historique DE CE VENDEUR dont seller_last_delivered_date < t.
    #    allow_exact_matches=False = le garde-fou anti-fuite : une commande dont le résultat
    #    est connu exactement à t n'est pas considérée comme "connue" à t (inégalité stricte).
    targets = pd.merge_asof(
        targets,
        seller_history[["seller_id", "seller_last_delivered_date", "cum_n", "cum_late"]]
            .sort_values("seller_last_delivered_date"),
        left_on="order_purchase_timestamp",
        right_on="seller_last_delivered_date",
        by="seller_id",
        direction="backward",
        allow_exact_matches=False,
    )
    targets["cum_n"] = targets["cum_n"].fillna(0)
    targets["cum_late"] = targets["cum_late"].fillna(0)

    # -- As-of join GLOBAL (prior) : même principe, sans regroupement par vendeur --
    targets = targets.sort_values("order_purchase_timestamp")
    targets = pd.merge_asof(
        targets,
        global_history[["global_last_delivered_date", "cum_n_global", "cum_late_global"]],
        left_on="order_purchase_timestamp",
        right_on="global_last_delivered_date",
        direction="backward",
        allow_exact_matches=False,
    )
    # Cas marginal : aucune commande livrée avant t nulle part dans le dataset (les tout
    # premiers achats chronologiques). Prior neutre 0.5 (incertitude maximale) — documenté
    # en limite, affecte un nombre négligeable de lignes.
    global_rate = (targets["cum_late_global"] / targets["cum_n_global"]).fillna(0.5)

    # 5. Lissage bayésien : (retards observés vendeur + k * prior global) / (n observé + k)
    targets["seller_late_rate_smoothed"] = (
        targets["cum_late"] + SELLER_SMOOTHING_K * global_rate
    ) / (targets["cum_n"] + SELLER_SMOOTHING_K)

    # -- ASSERTIONS ANTI-FUITE (filet de sécurité exécutable) --
    # Pour chaque paire (order, seller) ayant matché un historique VENDEUR, vérifie que la
    # date de résultat retenue est strictement antérieure à t. Lève une exception si violation.
    seller_matched = targets["seller_last_delivered_date"].notna()
    seller_violations = (
        targets.loc[seller_matched, "seller_last_delivered_date"]
        >= targets.loc[seller_matched, "order_purchase_timestamp"]
    )
    if seller_violations.any():
        raise AssertionError(
            f"Fuite temporelle détectée (vendeur) : {seller_violations.sum()} lignes utilisent "
            "un historique vendeur connu après ou au moment de l'achat (order_purchase_timestamp)."
        )

    # Même vérification pour le prior GLOBAL (2e merge_asof) — un trou ici serait une fuite
    # silencieuse : le seller-level peut être propre pendant que le prior triche.
    global_matched = targets["global_last_delivered_date"].notna()
    global_violations = (
        targets.loc[global_matched, "global_last_delivered_date"]
        >= targets.loc[global_matched, "order_purchase_timestamp"]
    )
    if global_violations.any():
        raise AssertionError(
            f"Fuite temporelle détectée (prior global) : {global_violations.sum()} lignes "
            "utilisent un historique global connu après ou au moment de l'achat."
        )

    # Cas marginal documenté : commandes sans AUCUN historique global disponible (les tout
    # premiers achats chronologiques, avant que la première livraison du dataset ne soit
    # connue). Vérifié empiriquement le 2026-07-06 : 266 commandes (0.28 %), toutes achetées
    # avant 2016-10-11 13:46:32 (1re order_delivered_customer_date de tout le dataset).
    # Ces lignes reçoivent le prior neutre 0.5 — à documenter dans le README (section limites).
    # dédoublonnage par order_id : global_matched est identique pour toutes les lignes
    # (order_id, seller_id) d'une même commande (t ne dépend que de l'order_id), donc compter
    # les lignes brutes surcompterait les commandes multi-vendeurs.
    n_no_global_history = targets.loc[~global_matched, "order_id"].nunique()
    print(f"  Commandes sans historique global (prior neutre 0.5) : {n_no_global_history}")

    # 6. Agrégation au grain commande : le PIRE vendeur (max du taux lissé) pilote le risque.
    result = (
        targets.groupby("order_id")["seller_late_rate_smoothed"]
        .max()
        .reset_index()
        .rename(columns={"seller_late_rate_smoothed": "seller_late_rate_max"})
    )
    return result
