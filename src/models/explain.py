"""
J7 — Explicabilité SHAP sur LightGBM (outil de diagnostic, PAS le modèle final).
Le modèle final retenu pour la prédiction en production est la régression logistique
(cf. artifacts/j6_modeling_report.md) : meilleur PR-AUC test, meilleure généralisation.
LightGBM est gardé uniquement ici pour lire les interactions non-linéaires entre
features via SHAP — une lecture que la régression logistique (linéaire, coefficients
globaux) ne permet pas nativement.
Lancer depuis la racine du projet : python src/models/explain.py
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Le projet n'est pas installé comme package (pas de pyproject.toml/__init__.py) : train.py
# et explain.py se lancent tous deux en script direct depuis la racine. On ajoute la racine
# à sys.path pour réutiliser les fonctions de train.py (mêmes garde-fous anti-fuite, mêmes
# hyperparamètres) sans les dupliquer — la duplication serait le vrai risque ici (un
# correctif anti-fuite appliqué dans un fichier et oublié dans l'autre).
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train import (  # noqa: E402
    DB_PATH,
    CUTOFF_DATE,
    FORBIDDEN_FEATURE_COLUMNS,
    fit_models,
    get_feature_columns,
    load_dataset,
    split_train_test,
)

REPORT_PATH = PROJECT_ROOT / "artifacts" / "j7_shap_report.md"

N_TOP_FEATURES_GLOBAL = 15
N_TOP_FEATURES_LOCAL = 8


def compute_shap_values(lgbm_pipeline, X_test: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Calcule les valeurs SHAP sur le TEST (jamais le train : on explique ce que le modèle
    fait sur des données qu'il n'a jamais vues, pas comment il a mémorisé le train).
    TreeExplainer est exact et rapide pour les modèles à arbres (contrairement à
    KernelExplainer, model-agnostique mais approximatif et coûteux) — LightGBM le
    supporte nativement.

    Le préprocesseur (imputation + scaling + one-hot) est appliqué AVANT SHAP : shap
    explique donc les features transformées (ex: "cat__onehot__dominant_payment_type_boleto"),
    pas les colonnes brutes. C'est le bon niveau de lecture ici : one-hot signifie que
    chaque modalité de paiement a sa propre contribution, ce qui est plus lisible qu'une
    contribution agrégée sur une colonne catégorielle.
    """
    preprocessor = lgbm_pipeline.named_steps["prep"]
    clf = lgbm_pipeline.named_steps["clf"]

    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_df)

    # LightGBM binaire : shap_values peut être une liste [classe0, classe1] selon la
    # version. On ne garde que la contribution vers la classe positive (retard).
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, X_test_df, explainer.expected_value


def global_importance(shap_values: np.ndarray, feature_names: list[str], top_n: int) -> pd.DataFrame:
    """
    Importance globale = moyenne de |SHAP| par feature sur tout le test. Contrairement à
    l'importance native de LightGBM (comptage de splits, biaisé vers les features à
    haute cardinalité), l'importance SHAP est dans l'unité du modèle (impact sur le
    log-odds de retard) et cohérente avec les explications locales ci-dessous — même
    échelle, même calcul, pas deux mesures d'importance qui se contredisent.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    df = df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)
    df["mean_abs_shap"] = df["mean_abs_shap"].round(4)
    return df


def local_explanation(
    idx: int,
    order_id: str,
    y_true: int,
    y_proba: float,
    shap_row: np.ndarray,
    feature_names: list[str],
    feature_values: pd.Series,
    base_value: float,
    top_n: int,
) -> str:
    """
    Explication locale d'UNE commande : les top_n features qui ont le plus poussé la
    prédiction, dans quel sens (positif = pousse vers "retard"), avec leur valeur réelle
    (après transformation) pour que le sens de la contribution soit interprétable.
    """
    order = np.argsort(-np.abs(shap_row))[:top_n]
    lines = [
        f"**Commande `{order_id}`** — retard réel = {y_true}, probabilité prédite = {y_proba:.4f} "
        f"(valeur de base SHAP = {1 / (1 + np.exp(-base_value)):.4f} — PAS le taux de retard "
        f"réel du test [5.48%] : `class_weight=\"balanced\"` recalibre les probabilités, donc "
        f"cette valeur de base n'est interprétable qu'en ranking relatif, pas comme une "
        f"fréquence)\n",
        "| feature | valeur (transformée) | contribution SHAP | sens |",
        "|---|---|---|---|",
    ]
    for i in order:
        contrib = shap_row[i]
        sens = "-> retard" if contrib > 0 else "-> à l'heure"
        lines.append(
            f"| {feature_names[i]} | {feature_values.iloc[i]:.3f} | {contrib:+.4f} | {sens} |"
        )
    return "\n".join(lines) + "\n"


def pick_examples(y_test: pd.Series, y_proba: np.ndarray, order_ids: pd.Series) -> dict:
    """
    Sélectionne 3 commandes représentatives plutôt qu'un échantillon aléatoire :
    - la commande en retard la mieux classée (le modèle réussit ici, on montre pourquoi) ;
    - la commande à l'heure la plus faussement flaguée (coût des fausses alertes : SHAP
      montre ce qui trompe le modèle) ;
    - la commande en retard la moins bien classée (angle mort du modèle : le retard que
      les features point-in-time ne peuvent pas voir venir).
    """
    # Index par défaut 0..n-1, dans le même ordre positionnel que test_df / X_test_transformed
    # / shap_values : row.name ci-dessous EST directement la position à utiliser dans ces
    # tableaux, pas besoin de re-chercher par order_id.
    df = pd.DataFrame({"y_true": y_test.values, "y_proba": y_proba, "order_id": order_ids.values})

    best_tp = df[df["y_true"] == 1].nlargest(1, "y_proba").iloc[0]
    worst_fp = df[df["y_true"] == 0].nlargest(1, "y_proba").iloc[0]
    worst_fn = df[df["y_true"] == 1].nsmallest(1, "y_proba").iloc[0]

    return {
        "Retard bien détecté (vrai positif le plus net)": best_tp,
        "Fausse alerte la plus nette (faux positif)": worst_fp,
        "Retard manqué (faux négatif le plus net)": worst_fn,
    }


def leakage_sanity_check(feature_names: list[str]) -> str:
    """
    Re-vérification anti-fuite spécifique au J7 : les colonnes interdites ne doivent
    apparaître ni dans les features brutes (déjà garanti par train.py) ni, transformées,
    dans les noms de features vus par SHAP. Un score LightGBM "trop beau" serait le
    premier signal à investiguer ici — ce n'est PAS le cas (PR-AUC test 0.087, sous la
    régression logistique à 0.113 et sous son propre score train à 0.281 : le modèle est
    plutôt en sur-apprentissage que suspect de fuite).
    """
    leaked = [f for f in feature_names if any(forb in f for forb in FORBIDDEN_FEATURE_COLUMNS)]
    if leaked:
        raise AssertionError(f"Fuite potentielle détectée dans les features transformées : {leaked}")
    return (
        "Vérifié : aucune colonne interdite (cible, timestamps post-achat, review_score) "
        "n'apparaît dans les features transformées vues par SHAP. Le PR-AUC LightGBM "
        "test (0.087) est inférieur à son propre PR-AUC train (0.281) et inférieur au "
        "PR-AUC test de la régression logistique (0.113) : signature d'un sur-apprentissage "
        "modéré, pas d'une fuite (une fuite gonflerait le score test lui-même, pas "
        "seulement le score train)."
    )


def metric_and_limits_section() -> str:
    """
    Section narrative (pas de calcul ici, synthèse de faits déjà établis en J6/J7) :
    justification de la métrique + limites honnêtes. Texte statique volontairement — ce
    n'est pas un nombre à recalculer, c'est un raisonnement à assumer par écrit, exigé
    par le brief (section limites) et par la grille d'évaluation.
    """
    return "\n".join([
        "## Métrique retenue et pourquoi\n",
        "**PR-AUC**, jamais l'accuracy. Sur un test à 5.48% de retards, un modèle qui "
        "prédit toujours \"à l'heure\" atteint 94.5% d'accuracy sans capturer un seul "
        "retard — l'accuracy ne distingue pas ça d'un modèle utile. Le ROC-AUC est écarté "
        "aussi : il intègre le taux de vrais négatifs (majoritaire et facile ici), ce qui "
        "le rend optimiste sur des classes très déséquilibrées. Le PR-AUC ne regarde que "
        "la classe positive (précision et recall du retard), la classe qui compte pour la "
        "décision métier. Il est systématiquement rapporté à côté de son plancher "
        "no-skill (le taux de retard de la période évaluée) car sans ce plancher un "
        "PR-AUC n'est pas interprétable en absolu.\n",
        "## Limites (section honnête)\n",
        "1. **Plafond de signal, pas de seuil.** PR-AUC test ≈ 0.11-0.13 selon le seuil : "
        "un signal réel (2x le plancher no-skill) mais modeste. Les causes dominantes de "
        "retard sont post-achat (aléas transporteur, dernier kilomètre) et non "
        "observables au moment de la commande par construction — c'est une contrainte du "
        "problème (anti-fuite), pas un manque d'effort de feature engineering.\n",
        "2. **Probabilités non calibrées.** `class_weight=\"balanced\"` (LogReg et "
        "LightGBM) recalibre les probabilités pour compenser le déséquilibre : la "
        "probabilité de sortie n'est PAS une fréquence réelle de retard (ex. le point de "
        "base SHAP moyen de LightGBM est ≈40%, très au-dessus du 5.48% réel du test). Les "
        "probabilités ne sont fiables qu'en ranking relatif (\"plus risqué que\"), jamais "
        "en lecture absolue (\"X% de chances de retard\"). Implication directe pour le "
        "seuil retenu (0.60, cf. J6) : ce n'est pas \"60% de chances de retard\", c'est un "
        "point de coupure choisi sur la courbe précision/recall.\n",
        "3. **LightGBM surapprend plus que la régression logistique** (ratio PR-AUC "
        "train/test 3.2x contre 1.4x, même après réduction de `num_leaves` et ajout de "
        "`reg_lambda`). C'est la raison directe pour laquelle LightGBM n'est pas le "
        "modèle de production — il est gardé seulement pour SHAP, où le sur-apprentissage "
        "biaise l'échelle des contributions mais pas leur ordre de grandeur relatif.\n",
        "4. **Un seul split temporel, pas de validation glissante.** Le taux de retard "
        "mensuel varie de 1.36% à 21.36% sur la période disponible (cf. J6) : un cutoff "
        "unique capture une fenêtre de test possiblement non représentative des mois "
        "hors échantillon. Une validation walk-forward (plusieurs cutoffs successifs) "
        "donnerait une estimation plus robuste — non faite ici par contrainte de temps.\n",
        "5. **Échantillons faibles aux seuils extrêmes.** L'estimation de précision à des "
        "seuils élevés (ex. 0.90 : 80 commandes flaguées) est bruitée par un petit "
        "dénominateur — un seul faux positif de plus ou de moins change la précision de "
        "plusieurs points. Ne pas sur-interpréter les seuils au-delà de ~0.85.\n",
    ])


def _df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        df = load_dataset(con)
        feature_cols = get_feature_columns(df)
        train_df, test_df = split_train_test(df, CUTOFF_DATE)

        X_train, y_train = train_df[feature_cols], train_df["is_late"]
        X_test, y_test = test_df[feature_cols], test_df["is_late"]

        models = fit_models(X_train, y_train, feature_cols)
        lgbm_pipeline = models["LightGBM"]

        shap_values, X_test_transformed, base_value = compute_shap_values(lgbm_pipeline, X_test)
        feature_names = list(X_test_transformed.columns)

        sanity_note = leakage_sanity_check(feature_names)
        print(sanity_note)

        global_df = global_importance(shap_values, feature_names, N_TOP_FEATURES_GLOBAL)
        print(f"\n=== Top {N_TOP_FEATURES_GLOBAL} features (importance SHAP globale) ===")
        print(global_df.to_string(index=False))

        y_proba_lgbm = lgbm_pipeline.predict_proba(X_test)[:, 1]
        examples = pick_examples(y_test, y_proba_lgbm, test_df["order_id"])

        local_sections = []
        for title, row in examples.items():
            pos = row.name
            explanation = local_explanation(
                idx=pos,
                order_id=row["order_id"],
                y_true=int(row["y_true"]),
                y_proba=float(row["y_proba"]),
                shap_row=shap_values[pos],
                feature_names=feature_names,
                feature_values=X_test_transformed.iloc[pos],
                base_value=base_value,
                top_n=N_TOP_FEATURES_LOCAL,
            )
            local_sections.append(f"### {title}\n\n{explanation}")

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        parts = [
            "# J7 — Explicabilité SHAP (LightGBM, outil de diagnostic)\n",
            "Le modèle final de production est la régression logistique (cf. J6). LightGBM "
            "est utilisé ICI uniquement pour lire les interactions non-linéaires entre "
            "features via SHAP — une lecture qu'un modèle linéaire ne permet pas.\n",
            "## Vérification anti-fuite (avant toute lecture des résultats)\n",
            sanity_note + "\n",
            f"## Importance globale (top {N_TOP_FEATURES_GLOBAL} features, moyenne |SHAP| sur le test)\n",
            _df_to_markdown(global_df) + "\n",
            "## Explications locales (3 commandes représentatives)\n",
        ]
        parts += local_sections
        parts.append(metric_and_limits_section())

        REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")
        print(f"\nRapport SHAP écrit dans {REPORT_PATH}")


if __name__ == "__main__":
    main()
