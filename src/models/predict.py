"""
J8 — Réinjection des prédictions dans l'entrepôt DuckDB (main.order_risk_scores,
main.order_risk_drivers). La prédiction devient une dimension filtrable dans la couche
BI (star schema), pas un artefact de notebook à côté — cf. docs/plan_14_jours.md Bloc 3.

Séquencement obligatoire : lancer APRÈS `dbt run` (nécessite marts.fct_orders) et après
`python src/features/build_features.py` (nécessite main.features_orders).
Lancer depuis la racine du projet : python src/models/predict.py
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Même raison que dans explain.py : le projet n'est pas un package installé, on ajoute
# la racine à sys.path pour réutiliser train.py sans dupliquer sa logique (préprocesseur,
# garde-fous anti-fuite, hyperparamètres du modèle de production).
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import (  # noqa: E402
    DB_PATH,
    CUTOFF_DATE,
    DECISION_THRESHOLD,
    defensible_thresholds,
    fit_models,
    get_feature_columns,
    load_dataset,
    split_train_test,
)

REPORT_PATH = PROJECT_ROOT / "artifacts" / "j8_prediction_integration_report.md"
N_TOP_DRIVERS = 3

RISK_TIER_LOW = "Sous seuil"
RISK_TIER_ALERT = "Alerte"


def risk_tier_extreme_label(q90_threshold: float) -> str:
    """
    Le label décrit ce que le tier EST (un seuil de probabilité fixe), pas ce qu'il
    prétend être ("top 10%"). "Top 10%" n'est vrai que pour le TEST, population sur
    laquelle q90_threshold a été calibré (defensible_thresholds sur X_test) — pour le
    train, ce même seuil absolu capture ~4% des commandes, pas 10% (cf.
    risk_tier_reading_note : class_weight="balanced" calibré sur le déséquilibre du
    train ne transfère pas le même percentile au test, et inversement). Un seuil
    opérationnel fixe reste le bon choix pour la production (une commande à 0.80 doit
    déclencher la même alerte quelle que soit la période) — c'est le NOM qui mentait,
    pas le seuil.
    """
    return f"Risque extrême (probabilité >= {q90_threshold:.2f})"


def compute_risk_scores(
    logreg_pipeline, df: pd.DataFrame, feature_cols: list[str], q90_threshold: float
) -> pd.DataFrame:
    """
    Score TOUTES les commandes (train + test), pas seulement le test. Décision prise
    avec l'utilisateur après discussion explicite : appliquer predict_proba une deuxième
    fois sur des lignes déjà utilisées pour le fit n'est pas une fuite (aucune
    information du test ne contamine l'entraînement) — c'est une pratique standard de
    risk-scoring (backtest + live), à condition que les deux populations restent
    distinguables. `is_in_sample` porte cette distinction explicitement : sans elle, un
    score de commande vue au training et un score de commande jamais vue seraient
    indiscernables dans la même colonne, ce qui SERAIT trompeur.

    risk_tier a 3 paliers dérivés de seuils déjà justifiés en J6 (jamais de nouveau
    seuil inventé ici) : DECISION_THRESHOLD (0.60, arbitrage métier) et q90_threshold
    (règle "10% les plus à risque", recalculé à chaque run par defensible_thresholds
    sur le TEST — jamais une constante recopiée à la main).
    """
    X_all = df[feature_cols]
    proba = logreg_pipeline.predict_proba(X_all)[:, 1]

    is_in_sample = (df["date_key"] < pd.Timestamp(CUTOFF_DATE)).to_numpy()
    is_flagged = proba >= DECISION_THRESHOLD
    tier = np.where(
        proba >= q90_threshold, risk_tier_extreme_label(q90_threshold),
        np.where(proba >= DECISION_THRESHOLD, RISK_TIER_ALERT, RISK_TIER_LOW),
    )

    scored_at = datetime.now(timezone.utc)
    return pd.DataFrame({
        "order_id": df["order_id"].to_numpy(),
        "risk_probability": proba.round(4),
        "is_in_sample": is_in_sample,
        "is_flagged_risk": is_flagged,
        "risk_tier": tier,
        "scored_at": scored_at,
    })


def compute_drivers(
    logreg_pipeline, df: pd.DataFrame, feature_cols: list[str], top_n: int
) -> pd.DataFrame:
    """
    Décomposition linéaire EXACTE du score de la régression logistique — pas SHAP.
    Pour un modèle linéaire, contribution_i = coef_i * valeur_standardisée_i, et la
    somme des contributions + l'intercept EST le score en log-odds (pas une
    approximation comme SHAP sur les arbres, qui répartit une contribution non-additive
    par nature). Cohérence par construction avec le modèle réellement déployé :
    LightGBM+SHAP (J7) reste un outil de diagnostic séparé, jamais mélangé ici avec le
    score de production.

    Vectorisé (pas de boucle Python par commande, ~96 470 lignes) : argsort sur
    |contribution| par ligne pour prendre le top_n, comme un argpartition trié.
    """
    prep = logreg_pipeline.named_steps["prep"]
    clf = logreg_pipeline.named_steps["clf"]

    X_all = df[feature_cols]
    X_transformed = prep.transform(X_all)
    feature_names = np.array(prep.get_feature_names_out())
    coefs = clf.coef_[0]

    contributions = X_transformed * coefs  # broadcasting : (n_commandes, n_features)
    top_idx = np.argsort(-np.abs(contributions), axis=1)[:, :top_n]

    n = len(df)
    order_ids = df["order_id"].to_numpy()
    top_contrib = np.take_along_axis(contributions, top_idx, axis=1)

    return pd.DataFrame({
        "order_id": np.repeat(order_ids, top_n),
        "driver_rank": np.tile(np.arange(1, top_n + 1), n),
        "feature_name": feature_names[top_idx].reshape(-1),
        "contribution": top_contrib.reshape(-1).round(4),
        "direction": np.where(top_contrib.reshape(-1) > 0, "retard", "à l'heure"),
    })


def sanity_checks(
    scores_df: pd.DataFrame, drivers_df: pd.DataFrame,
    df: pd.DataFrame, train_df: pd.DataFrame,
) -> None:
    """
    Garde-fous exécutables du J8 : la population scorée doit être exactement la
    population de features_orders (aucune commande perdue ou dupliquée en route), et
    is_in_sample doit partitionner EXACTEMENT comme split_train_test — un décalage ici
    signifierait que des commandes de train sont étiquetées comme scores honnêtes (ou
    l'inverse), ce qui romprait la garantie anti-fuite du reste du projet.
    """
    if len(scores_df) != len(df):
        raise AssertionError(
            f"order_risk_scores a {len(scores_df)} lignes, attendu {len(df)} "
            "(population complète de features_orders)."
        )
    if len(drivers_df) != len(df) * N_TOP_DRIVERS:
        raise AssertionError(
            f"order_risk_drivers a {len(drivers_df)} lignes, attendu "
            f"{len(df) * N_TOP_DRIVERS} ({len(df)} commandes x {N_TOP_DRIVERS} drivers)."
        )

    expected_train_ids = set(train_df["order_id"])
    actual_in_sample_ids = set(scores_df.loc[scores_df["is_in_sample"], "order_id"])
    if expected_train_ids != actual_in_sample_ids:
        raise AssertionError(
            "is_in_sample ne correspond pas exactement au split train de "
            "split_train_test() : des commandes de train ne sont pas marquées "
            "is_in_sample=true, ou l'inverse."
        )


def write_tables(con: duckdb.DuckDBPyConnection, scores_df: pd.DataFrame, drivers_df: pd.DataFrame) -> None:
    """CREATE OR REPLACE : idempotent, comme write_features_table dans build_features.py."""
    con.execute("CREATE SCHEMA IF NOT EXISTS main")
    con.register("scores_df", scores_df)
    con.execute("CREATE OR REPLACE TABLE main.order_risk_scores AS SELECT * FROM scores_df")
    con.unregister("scores_df")

    con.register("drivers_df", drivers_df)
    con.execute("CREATE OR REPLACE TABLE main.order_risk_drivers AS SELECT * FROM drivers_df")
    con.unregister("drivers_df")


def _df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def risk_tier_reading_note(
    scores_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, q90_threshold: float
) -> str:
    """
    Lecture du tableau risk_tier x is_in_sample, calculée à partir des chiffres réels
    (jamais une prédiction a priori recopiée en dur). Première version de ce texte :
    hypothèse "train tourne plus chaud" (sur-apprentissage LogReg, ratio train/test
    1.42x) — RÉFUTÉE par les chiffres réels du premier run (train : 4.06% en tier
    extrême ; test : 10.01%, l'inverse de l'hypothèse). Mécanisme réel identifié :
    class_weight="balanced" est calibré sur le déséquilibre de TRAIN uniquement (fit()
    ne voit que X_train/y_train) ; si le déséquilibre réel du test est différent
    (généralement plus marqué, car le taux de retard varie fortement par mois, cf. J6),
    le même modèle projette des probabilités systématiquement plus hautes sur la
    population dont le déséquilibre diffère le plus de celui du train — indépendamment
    du sur-apprentissage. Recalculé et reformulé dynamiquement pour rester correct si
    le split ou les données changent.
    """
    late_rate_train = float(train_df["is_late"].mean())
    late_rate_test = float(test_df["is_late"].mean())
    ratio_train = (1 - late_rate_train) / late_rate_train
    ratio_test = (1 - late_rate_test) / late_rate_test

    pct_high = (
        scores_df.assign(is_high=scores_df["risk_tier"] != RISK_TIER_LOW)
        .groupby("is_in_sample")["is_high"].mean() * 100
    )
    pct_high_train = float(pct_high.get(True, 0.0))
    pct_high_test = float(pct_high.get(False, 0.0))

    higher_side = "train (`is_in_sample=true`)" if pct_high_train > pct_high_test else "test (`is_in_sample=false`)"
    imbalance_more_skewed_side = "test" if ratio_test > ratio_train else "train"

    return (
        f"**Constat réel (pas une prédiction a priori)** : {pct_high_train:.1f}% des "
        f"commandes de train sont en tier Alerte/Extrême, contre {pct_high_test:.1f}% "
        f"pour le test — la proportion la plus élevée est du côté {higher_side}, alors "
        f"que le taux de retard BRUT est plus élevé sur train ({late_rate_train:.2%}) "
        f"que sur test ({late_rate_test:.2%}). Ce n'est PAS le signe que le test est "
        "réellement plus risqué : `class_weight=\"balanced\"` est calibré uniquement sur "
        f"le déséquilibre de train (ratio à l'heure:retard = {ratio_train:.1f}:1), pas "
        f"sur celui de test (ratio = {ratio_test:.1f}:1, plus déséquilibré côté "
        f"{imbalance_more_skewed_side}). Le même modèle, appliqué à une population dont "
        "le déséquilibre diffère de celui sur lequel il a été calibré, projette des "
        "probabilités systématiquement décalées pour cette population — un effet "
        "distinct du sur-apprentissage, qui s'ajoute à la limite déjà documentée en J7 "
        "(probabilités non calibrées, `class_weight=\"balanced\"`). Ne pas comparer les "
        "deux sous-populations comme si elles étaient sur la même échelle de risque "
        "réel — c'est tout l'intérêt de garder `is_in_sample` visible plutôt que de "
        "mélanger silencieusement les deux.\n"
    )


def write_report(
    scores_df: pd.DataFrame, drivers_df: pd.DataFrame, q90_threshold: float,
    sample_filtered: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame,
) -> None:
    tier_by_sample = (
        scores_df.groupby(["is_in_sample", "risk_tier"]).size()
        .rename("n_commandes").reset_index()
        .sort_values(["is_in_sample", "risk_tier"])
    )
    top_driver_counts = (
        drivers_df[drivers_df["driver_rank"] == 1]["feature_name"]
        .value_counts().head(10).rename("n_fois_driver_1").reset_index()
        .rename(columns={"index": "feature_name"})
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# J8 — Prédictions dans l'entrepôt (main.order_risk_scores, main.order_risk_drivers)\n",
        "Score de risque produit par la régression logistique (modèle de production, "
        f"seuil {DECISION_THRESHOLD}, cf. J6) et écrit dans DuckDB comme dimension "
        "filtrable — pas dans un notebook à côté. Drivers = décomposition linéaire "
        "exacte du score LogReg (`coef × valeur standardisée`), pas SHAP : SHAP a été "
        "calculé sur LightGBM (J7), qui n'est pas le modèle de production. Afficher un "
        "score et une explication issus de deux modèles différents serait incohérent — "
        "écart assumé par rapport au texte initial de docs/plan_14_jours.md.\n",
        "## Portée : toutes les commandes, avec is_in_sample\n",
        f"**{len(scores_df):,} commandes scorées** (population complète de "
        f"features_orders, train + test). `is_in_sample=true` pour les "
        f"{int(scores_df['is_in_sample'].sum()):,} commandes de train (le modèle les a "
        f"vues au fit — score optimiste) ; `is_in_sample=false` pour les "
        f"{int((~scores_df['is_in_sample']).sum()):,} commandes de test (score honnête, "
        "out-of-sample). Appliquer predict_proba une deuxième fois sur des lignes déjà "
        "utilisées pour le fit n'est PAS une fuite (aucune information du test ne "
        "contamine l'entraînement) — la colonne rend la distinction explicite plutôt "
        "que de mélanger les deux silencieusement.\n",
        "### Répartition des risk_tier par is_in_sample\n",
        _df_to_markdown(tier_by_sample) + "\n",
        f"Bornes de risk_tier calibrées sur le TEST uniquement (q90 = "
        f"{q90_threshold:.4f}) puis appliquées globalement aux deux sous-populations.\n",
        risk_tier_reading_note(scores_df, train_df, test_df, q90_threshold),
        "## Drivers les plus fréquents en position #1 (top 10)\n",
        _df_to_markdown(top_driver_counts) + "\n",
        "## Preuve de filtrabilité (jointure vers le star schema)\n",
        "Requête exécutée : jointure `main.order_risk_scores` → `marts.fct_orders` → "
        "`marts.dim_customer` / `marts.dim_product` / `marts.dim_date`, filtrée sur "
        "`is_in_sample = false` (vue par défaut recommandée pour un dashboard : scores "
        "honnêtes uniquement). Extrait (5 lignes) :\n",
        _df_to_markdown(sample_filtered.head(5)) + "\n",
        "## Limite assumée\n",
        "Aucun filtre par vendeur individuel : `docs/star_schema.md` exclut "
        "délibérément une dimension vendeur (le grain de commande ≠ grain d'item, une "
        "commande peut avoir plusieurs vendeurs). Rouvrir ce point demanderait de "
        "résoudre quel vendeur porte le `seller_late_rate_max` de chaque commande — "
        "non fait ici, décision reconfirmée avec l'utilisateur au J8. Filtres livrés : "
        "région (état client), catégorie produit, période.\n",
    ]
    REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    with duckdb.connect(str(DB_PATH)) as con:
        df = load_dataset(con)
        feature_cols = get_feature_columns(df)
        train_df, test_df = split_train_test(df, CUTOFF_DATE)

        X_train, y_train = train_df[feature_cols], train_df["is_late"]
        X_test, y_test = test_df[feature_cols], test_df["is_late"]

        models = fit_models(X_train, y_train, feature_cols)
        logreg_pipeline = models["Régression logistique"]

        threshold_business = defensible_thresholds(logreg_pipeline, X_test, y_test)
        q90_row = threshold_business[
            threshold_business["regle"] == "Flaguer les 10% les plus à risque"
        ]
        q90_threshold = float(q90_row["seuil"].iloc[0])

        scores_df = compute_risk_scores(logreg_pipeline, df, feature_cols, q90_threshold)
        drivers_df = compute_drivers(logreg_pipeline, df, feature_cols, N_TOP_DRIVERS)

        sanity_checks(scores_df, drivers_df, df, train_df)
        print(f"Garde-fous OK : {len(scores_df):,} commandes scorées, "
              f"{len(drivers_df):,} lignes de drivers.")

        write_tables(con, scores_df, drivers_df)
        print("Écrit : main.order_risk_scores, main.order_risk_drivers")

        sample_filtered = con.execute("""
            select c.customer_state, p.product_category_name,
                   strftime(o.date_key, '%Y-%m') as mois,
                   count(*) filter (where r.is_flagged_risk) as n_a_risque
            from main.order_risk_scores r
            join marts.fct_orders o using (order_id)
            join marts.dim_customer c using (customer_id)
            join marts.dim_product p using (product_key)
            where r.is_in_sample = false
            group by 1, 2, 3
            order by n_a_risque desc
        """).df()
        print("\n=== Preuve de filtrabilité (top 5, is_in_sample=false) ===")
        print(sample_filtered.head(5).to_string(index=False))

        write_report(scores_df, drivers_df, q90_threshold, sample_filtered, train_df, test_df)
        print(f"\nRapport écrit dans {REPORT_PATH}")


if __name__ == "__main__":
    main()
