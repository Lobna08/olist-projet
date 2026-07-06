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

# Colonnes qui ne doivent JAMAIS apparaitre dans la table de features : postérieures à
# l'achat (fuite directe de la cible) ou signal de satisfaction (post-livraison).
FORBIDDEN_FEATURE_COLUMNS = {
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "review_score",
    "is_negative",
    "is_positive",
}

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


def compute_temporal_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Features calendaires dérivées de order_purchase_timestamp uniquement — aucun risque de
    fuite, cette date est connue par définition au moment de la décision.
    delay_est_days reprend le calcul de fct_orders (délai promis au client, connu à l'achat).
    """
    df = con.execute("""
        select order_id, order_purchase_timestamp, order_estimated_delivery_date
        from staging.stg_orders
    """).fetchdf()

    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday  # 0=lundi
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["is_weekend"] = df["purchase_weekday"].isin([5, 6]).astype(int)
    df["delay_est_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    return df[
        ["order_id", "purchase_weekday", "purchase_month", "purchase_hour",
         "is_weekend", "delay_est_days"]
    ]


def load_order_item_measures(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Mesures commande déjà agrégées par dbt (intermediate.int_order_items) — connues au
    complet dès la création des order_items, donc point-in-time par construction.
    """
    return con.execute("""
        select order_id, nb_items, total_price, total_freight,
               nb_distinct_sellers, freight_ratio
        from intermediate.int_order_items
    """).fetchdf()


def load_payment_measures(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Mesures paiement déjà agrégées par dbt (intermediate.int_order_payments).

    HYPOTHÈSE MÉTIER NON PROUVÉE : on suppose que le mode de paiement et le nombre
    d'échéances sont fixés au checkout, donc disponibles avant order_approved_at.
    Le dataset ne fournit AUCUN timestamp sur order_payments permettant de le vérifier —
    c'est une hypothèse standard e-commerce (le paiement est saisi avant validation de la
    commande), pas un fait constaté dans les données. À citer explicitement en limite du
    README : si cette hypothèse est fausse (ex. relance de paiement après approbation),
    ce bloc constitue une fuite non détectée par les assertions de ce script.
    """
    return con.execute("""
        select order_id, total_payment, nb_payment_installments, dominant_payment_type
        from intermediate.int_order_payments
    """).fetchdf()


def compute_product_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Agrégats produit au grain commande : somme du poids, somme du volume, nb catégories
    distinctes. Le volume est calculé PAR ARTICLE (length × height × width) puis sommé —
    pas de max(dimension) séparé par axe, ce qui créerait un "produit fantôme" (la longueur
    d'un article combinée à la hauteur d'un autre n'a pas de sens physique). La somme des
    volumes par article est cohérente avec la somme du poids : les deux mesurent le volume
    logistique total de la commande, pas un pire cas isolé.
    """
    return con.execute("""
        with item_products as (
            select
                oi.order_id,
                p.product_weight_g,
                p.product_length_cm * p.product_height_cm * p.product_width_cm as volume_cm3,
                p.product_category_name
            from staging.stg_order_items oi
            join staging.stg_products p on oi.product_id = p.product_id
        )
        select
            order_id,
            sum(product_weight_g)                as product_weight_g_sum,
            sum(volume_cm3)                       as product_volume_cm3_sum,
            count(distinct product_category_name) as nb_distinct_categories
        from item_products
        group by order_id
    """).fetchdf()


def _haversine_km(lat1, lon1, lat2, lon2):
    """Distance orthodromique en km entre deux points (vectorisé numpy)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def compute_seller_distance_max(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Distance vendeur→client, agrégée au grain commande par le MAX (décision par défaut,
    à confirmer) — symétrique avec seller_late_rate_max : le maillon logistique le plus
    éloigné pilote plausiblement le risque, même raisonnement que pour le pire vendeur.

    geo_is_unknown=1 quand AUCUNE distance n'est calculable pour la commande (client sans
    géoloc connue, OU tous les vendeurs de la commande sans géoloc connue). NULL + flag,
    pas d'imputation par centroïde d'état (décision actée : biais silencieux pire que NULL).
    """
    pairs = con.execute("""
        select distinct
            oi.order_id,
            oi.seller_id,
            cg.centroid_lat as customer_lat,
            cg.centroid_lng as customer_lng,
            sg.centroid_lat as seller_lat,
            sg.centroid_lng as seller_lng
        from staging.stg_order_items oi
        join staging.stg_orders    o on oi.order_id = o.order_id
        join staging.stg_customers c on o.customer_id = c.customer_id
        join staging.stg_sellers   s on oi.seller_id = s.seller_id
        left join intermediate.int_geolocation cg on c.customer_zip_code_prefix = cg.zip_code_prefix
        left join intermediate.int_geolocation sg on s.seller_zip_code_prefix = sg.zip_code_prefix
    """).fetchdf()

    pairs["distance_km"] = _haversine_km(
        pairs["customer_lat"], pairs["customer_lng"],
        pairs["seller_lat"], pairs["seller_lng"],
    )

    result = (
        pairs.groupby("order_id")["distance_km"]
        .max()  # skipna par défaut : NaN seulement si TOUTES les paires sont NaN
        .reset_index()
        .rename(columns={"distance_km": "seller_distance_km_max"})
    )
    result["geo_is_unknown"] = result["seller_distance_km_max"].isna().astype(int)
    return result


def build_features_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Assemble tous les blocs de features au grain order_id. Vérifie après CHAQUE jointure
    que le nombre de lignes ne bouge pas (garde-fou anti fan-out : une jointure qui
    dupliquerait des order_id casserait le grain sans erreur SQL visible).
    """
    base = con.execute("select order_id from staging.stg_orders").fetchdf()
    n_expected = len(base)

    blocks = [
        compute_seller_late_rate(con),
        compute_temporal_features(con),
        load_order_item_measures(con),
        load_payment_measures(con),
        compute_product_features(con),
        compute_seller_distance_max(con),
    ]

    features = base
    for block in blocks:
        features = features.merge(block, on="order_id", how="left")
        if len(features) != n_expected:
            raise AssertionError(
                f"Fan-out détecté après jointure des colonnes {list(block.columns)} : "
                f"{len(features)} lignes au lieu de {n_expected} attendues."
            )

    leaked = FORBIDDEN_FEATURE_COLUMNS & set(features.columns)
    if leaked:
        raise AssertionError(f"Colonnes interdites présentes dans features_orders : {leaked}")

    return features


def write_features_table(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """Matérialise le DataFrame de features dans main.features_orders."""
    con.execute("CREATE SCHEMA IF NOT EXISTS main")
    con.register("features_df", df)
    con.execute("CREATE OR REPLACE TABLE main.features_orders AS SELECT * FROM features_df")
    con.unregister("features_df")


if __name__ == "__main__":
    with duckdb.connect(str(DB_PATH)) as con:
        features_df = build_features_table(con)
        print(f"features_orders : {features_df.shape[0]:,} lignes x {features_df.shape[1]} colonnes")
        write_features_table(con, features_df)
        print(f"Écrit dans main.features_orders -> {DB_PATH}")
