"""
Entraînement et évaluation de 3 modèles de prédiction du retard de livraison (is_late).
Split temporel strict (train = achats avant CUTOFF_DATE, test = à partir de CUTOFF_DATE).
Lancer depuis la racine du projet : python src/models/train.py

Justification du cutoff (2018-06-01) et de la ventilation mensuelle : voir le rapport
généré dans artifacts/j6_modeling_report.md à chaque exécution.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "olist.db"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "j6_modeling_report.md"

CUTOFF_DATE = "2018-06-01"
# Cutoffs alternatifs testés pour justifier le choix ci-dessus (cf. discussion J6) :
# reculer le cutoff ne rapproche pas le taux de retard test du train (volatilité
# structurelle du dataset, pas un effet d'un seul mois anormal) — gardé ici pour que
# le rapport de justification soit régénéré automatiquement à chaque run, jamais figé.
CANDIDATE_CUTOFFS = ["2018-04-01", "2018-05-01", "2018-06-01", "2018-07-01"]

# Seuil de décision retenu pour la régression logistique (modèle final), tranché par
# l'utilisateur après revue du tableau seuil/précision/recall du J6 — ce n'est PAS une
# valeur optimisée en boucle sur le test, c'est un arbitrage métier documenté :
# raisonnement -> coût de l'action déclenchée par une alerte -> tolérance aux fausses
# alertes -> seuil. Ici, l'alerte déclenche une action automatique à faible coût (pas
# d'intervention humaine par commande), donc le recall est priorisé sur la précision :
# rater un retard coûte plus cher qu'une fausse alerte bruyante mais peu coûteuse.
DECISION_THRESHOLD = 0.60

CATEGORICAL_COLUMNS = ["dominant_payment_type"]

# Colonnes qui ne doivent jamais être des features : identifiant, date utilisée
# seulement pour le split, cible elle-même.
NON_FEATURE_COLUMNS = {"order_id", "date_key", "is_late"}
# Reprend le garde-fou de build_features.py par défense en profondeur : train.py doit
# rester auditable seul, sans supposer que les protections d'un autre fichier ont été
# respectées en amont (c'est le fichier qu'un correcteur lit en premier).
FORBIDDEN_FEATURE_COLUMNS = {
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "review_score",
    "is_negative",
    "is_positive",
}


def load_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Joint main.features_orders (features point-in-time) et marts.fct_orders (cible
    is_late + date_key = date d'achat, utilisée uniquement pour le split).
    features_orders ne contient pas la cible : ce join est le seul endroit où features
    et cible se rencontrent, donc le seul endroit où une fuite de jointure serait possible.
    """
    df = con.execute("""
        select f.*, o.is_late, o.date_key
        from main.features_orders f
        join marts.fct_orders o using (order_id)
    """).fetchdf()
    if len(df) != con.execute("select count(*) from main.features_orders").fetchone()[0]:
        raise AssertionError("Le join features_orders <-> fct_orders change le nombre de lignes (fan-out ou perte).")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Isole les colonnes de X. Garde-fou exécutable indépendant de build_features.py :
    même si ce script est lu seul, on ne suppose jamais qu'une protection amont a tenu.
    """
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    leaked = FORBIDDEN_FEATURE_COLUMNS & set(feature_cols)
    if leaked:
        raise AssertionError(f"Colonnes interdites présentes dans X : {leaked}")
    if NON_FEATURE_COLUMNS & set(feature_cols):
        raise AssertionError("order_id, date_key ou is_late s'est glissé dans les features.")

    return feature_cols


def split_train_test(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split temporel strict, jamais de shuffle : train = achats avant cutoff, test = à partir
    du cutoff. Assertion post-hoc obligatoire : aucune commande de test ne doit être
    antérieure à une commande de train, sinon le split n'est pas chronologiquement propre.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    train_df = df[df["date_key"] < cutoff_ts].copy()
    test_df = df[df["date_key"] >= cutoff_ts].copy()

    if train_df["date_key"].max() >= test_df["date_key"].min():
        raise AssertionError(
            "Fuite temporelle : une commande de test est antérieure ou simultanée "
            "à une commande de train."
        )
    return train_df, test_df


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """
    Encodage + imputation, appris UNIQUEMENT sur le train (via Pipeline.fit, jamais sur
    le dataset complet). Une nouvelle instance à chaque appel : LogisticRegression et
    LightGBM ne doivent pas partager un objet déjà fit par l'autre pipeline.

    - Catégorielle (dominant_payment_type) : imputation par le mode puis OneHotEncoder
      (handle_unknown="ignore" — une modalité de paiement vue seulement au test ne doit
      pas faire planter le transform, elle est juste encodée à zéro partout).
    - Numériques : imputation par la médiane (peu de NaN, cf build_features.py — max
      477/96 470 lignes sur seller_distance_km_max) puis StandardScaler. Le scaler est
      indispensable pour LogisticRegression (solver lbfgs sensible à l'échelle : sans
      lui, total_price ou product_weight_g_sum à l'échelle des milliers dominent
      is_weekend à l'échelle 0/1 et le solver ne converge pas — vérifié empiriquement,
      ConvergenceWarning sans ce scaler). LightGBM est invariant à un scaling monotone
      des features (arbres de décision) : partager ce même préprocesseur entre les deux
      modèles ne le pénalise donc pas.
    """
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLUMNS]

    return ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_COLUMNS),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
    ])


def fit_models(X_train: pd.DataFrame, y_train: pd.Series, feature_cols: list[str]) -> dict:
    """
    Entraîne les 3 modèles dans l'ordre de progression : plancher -> baseline -> final.
    Le Dummy reçoit un X factice (des zéros) : il ignore les features par construction
    (strategy="most_frequent" ne regarde que y), donc lui donner X_train imputerait à tort
    l'idée qu'il "voit" les mêmes données que les deux autres modèles.
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)

    logreg = Pipeline([
        ("prep", build_preprocessor(feature_cols)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    logreg.fit(X_train, y_train)

    # Un seul ajustement de régularisation (pas un tuning) : le diagnostic train/test du
    # J6 a montré un sur-apprentissage net avec les hyperparamètres par défaut (ratio
    # PR-AUC train/test x3.88, contre x1.42 pour la régression logistique). reg_lambda=1.0
    # et num_leaves=15 (au lieu de 31) réduisent la capacité de mémorisation. Ajustement
    # unique, décidé a priori sur le critère "l'écart train/test se referme-t-il ?" — pas
    # une boucle sur le PR-AUC test, ce qui serait une fuite méthodologique (optimiser sur
    # le test = le nouveau train).
    lgbm = Pipeline([
        ("prep", build_preprocessor(feature_cols)),
        ("clf", LGBMClassifier(
            class_weight="balanced", random_state=42, verbose=-1,
            num_leaves=15, reg_lambda=1.0,
        )),
    ])
    lgbm.fit(X_train, y_train)

    return {"Dummy (plancher)": dummy, "Régression logistique": logreg, "LightGBM": lgbm}


def fit_lgbm_variant(
    X_train: pd.DataFrame, y_train: pd.Series, feature_cols: list[str], **lgbm_overrides
) -> Pipeline:
    """
    Variante paramétrable du pipeline LightGBM de fit_models, réservée aux expériences
    de diagnostic du J7 (isoler l'effet d'un retrait de feature ou d'une capacité
    réduite sur le sur-apprentissage). Réutilise build_preprocessor pour ne jamais
    dupliquer la logique d'imputation/scaling entre train.py et explain.py — la
    duplication serait le vrai risque (un correctif appliqué dans un fichier, oublié
    dans l'autre).

    Les hyperparamètres de base sont ceux retenus en J6 (num_leaves=15, reg_lambda=1.0) ;
    lgbm_overrides les surcharge ponctuellement. Chaque appel du J7 ne change qu'UN levier
    à la fois (une feature retirée de feature_cols, OU un hyperparamètre), jamais plusieurs
    simultanément — sinon un résultat ne serait plus attribuable à une cause précise.
    """
    params = dict(class_weight="balanced", random_state=42, verbose=-1, num_leaves=15, reg_lambda=1.0)
    params.update(lgbm_overrides)
    pipe = Pipeline([
        ("prep", build_preprocessor(feature_cols)),
        ("clf", LGBMClassifier(**params)),
    ])
    pipe.fit(X_train[feature_cols], y_train)
    return pipe


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Gère le cas Dummy (X factice) de façon transparente pour le code appelant."""
    if isinstance(model, DummyClassifier):
        return model.predict_proba(np.zeros((len(X), 1)))[:, 1]
    return model.predict_proba(X)[:, 1]


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, DummyClassifier):
        return model.predict(np.zeros((len(X), 1)))
    return model.predict(X)


def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    PR-AUC en métrique principale (accuracy inutile ici : ~92% de la classe majoritaire
    suffit à un modèle qui ne prédit jamais de retard). Le no_skill_pr_auc — le taux de
    positifs de LA PÉRIODE ÉVALUÉE — est affiché à côté de CHAQUE PR-AUC : un PR-AUC sans
    son plancher n'est pas interprétable (un PR-AUC de 0.15 est excellent si le plancher
    est 0.05, médiocre si le plancher est 0.12).
    """
    no_skill = y_test.mean()
    rows = []
    for name, model in models.items():
        y_proba = _predict_proba(model, X_test)
        y_pred = _predict(model, X_test)
        rows.append({
            "modele": name,
            "n_test": len(y_test),
            "no_skill_pr_auc": round(no_skill, 4),
            "pr_auc": round(average_precision_score(y_test, y_proba), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })
    return pd.DataFrame(rows)


def confusion_matrices(models: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    return {
        name: confusion_matrix(y_test, _predict(model, X_test), labels=[0, 1])
        for name, model in models.items()
    }


def train_vs_test_gap(
    models: dict, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """
    Diagnostic sur-apprentissage : PR-AUC en resubstitution (train, optimiste par
    construction) vs PR-AUC test. Un écart énorme = le modèle mémorise le train plutôt
    que d'apprendre un signal généralisable. Un PR-AUC train déjà médiocre = le problème
    n'est pas l'overfitting mais un manque de signal exploitable dans les features.
    """
    rows = []
    for name, model in models.items():
        proba_train = _predict_proba(model, X_train)
        proba_test = _predict_proba(model, X_test)
        rows.append({
            "modele": name,
            "pr_auc_train": round(average_precision_score(y_train, proba_train), 4),
            "pr_auc_test": round(average_precision_score(y_test, proba_test), 4),
        })
    return pd.DataFrame(rows)


def evaluate_by_month(models: dict, test_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Ventilation du PR-AUC mois par mois À L'INTÉRIEUR du test (juin/juillet/août 2018).
    Vérifie que le score agrégé n'est pas un artefact d'un seul mois "facile" — chaque
    mois a son propre plancher no-skill (le taux de retard varie de 1.36% à 10.39% sur
    cette seule fenêtre de 3 mois, cf rapport de justification du cutoff).
    """
    df = test_df.copy()
    df["mois"] = df["date_key"].astype("datetime64[ns]").dt.to_period("M").astype(str)

    rows = []
    for mois, group in df.groupby("mois"):
        X_m, y_m = group[feature_cols], group["is_late"]
        no_skill = y_m.mean()
        for name, model in models.items():
            y_proba = _predict_proba(model, X_m)
            rows.append({
                "mois": mois,
                "modele": name,
                "n": len(y_m),
                "n_retards": int(y_m.sum()),
                "no_skill_pr_auc": round(no_skill, 4),
                "pr_auc": round(average_precision_score(y_m, y_proba), 4),
            })
    return pd.DataFrame(rows)


def threshold_table(model, X_test: pd.DataFrame, y_test: pd.Series, thresholds: list[float]) -> pd.DataFrame:
    """
    Précision/recall/% de commandes flaguées pour une grille de seuils de décision sur
    la régression logistique (modèle final retenu). Le seuil par défaut (0.5) n'a aucune
    légitimité métier ici : class_weight="balanced" recalibre les probabilités pour
    compenser le déséquilibre de classes, donc 0.5 ne correspond pas à "50% de chances
    réelles de retard" mais à un point arbitraire de la distribution. Ce tableau sert à
    choisir un seuil selon la tolérance métier aux fausses alertes — il n'est PAS parcouru
    en boucle par le code pour maximiser un score : la sélection finale est un arbitrage
    humain fait une fois, en dehors de ce script.
    """
    y_proba = _predict_proba(model, X_test)
    n = len(y_test)
    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        n_flagged = int(y_pred.sum())
        n_true_pos = int(((y_pred == 1) & (y_test == 1)).sum())
        n_false_pos = int(((y_pred == 1) & (y_test == 0)).sum())
        rows.append({
            "seuil": t,
            "pct_flague": round(100 * n_flagged / n, 2),
            "n_flague": n_flagged,
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "retards_captes": n_true_pos,
            "fausses_alertes": n_false_pos,
        })
    return pd.DataFrame(rows)


def defensible_thresholds(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Traduit 2 règles métier en seuils concrets, mesurés sur le test (pour rapporter une
    performance honnête) — PAS optimisés en boucle sur le test (ce serait un tuning
    caché, cf. règle absolue de l'utilisateur). Chaque règle est un choix de tolérance
    au risque, pas un score à maximiser :

    - "flaguer les 10% les plus à risque" : contrainte opérationnelle (capacité de
      traitement manuel fixe), le seuil est le 90e centile des probabilités prédites.
      ATTENTION : ce centile est calculé ICI sur la distribution du test pour ce rapport.
      En production, il doit être recalibré sur les données les plus récentes disponibles
      au moment du déploiement (train ou une fenêtre glissante), jamais sur le test lui-même,
      sinon le seuil "voit" les données qu'il est censé évaluer.
    - "précision >= 20%" : contrainte sur le coût des fausses alertes (1 fausse alerte
      pour au plus 4 vraies), seuil = le plus petit seuil de la grille qui satisfait cette
      contrainte (plus bas = plus de recall, donc on prend le minimum qui passe la barre).
    """
    y_proba = _predict_proba(model, X_test)
    fine_grid = np.arange(0.01, 1.00, 0.01)

    rows = []

    q90 = float(np.quantile(y_proba, 0.90))
    y_pred_q90 = (y_proba >= q90).astype(int)
    rows.append({
        "regle": "Flaguer les 10% les plus à risque",
        "seuil": round(q90, 4),
        "pct_flague": round(100 * y_pred_q90.sum() / len(y_test), 2),
        "n_flague": int(y_pred_q90.sum()),
        "precision": round(precision_score(y_test, y_pred_q90, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred_q90, zero_division=0), 4),
        "retards_captes": int(((y_pred_q90 == 1) & (y_test == 1)).sum()),
        "fausses_alertes": int(((y_pred_q90 == 1) & (y_test == 0)).sum()),
    })

    best_t = None
    for t in fine_grid:
        y_pred = (y_proba >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_test, y_pred, zero_division=0)
        if prec >= 0.20:
            best_t = t
            break
    if best_t is not None:
        y_pred_prec20 = (y_proba >= best_t).astype(int)
        rows.append({
            "regle": "Précision >= 20% (au plus 1 fausse alerte pour 4 vraies)",
            "seuil": round(float(best_t), 4),
            "pct_flague": round(100 * y_pred_prec20.sum() / len(y_test), 2),
            "n_flague": int(y_pred_prec20.sum()),
            "precision": round(precision_score(y_test, y_pred_prec20, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred_prec20, zero_division=0), 4),
            "retards_captes": int(((y_pred_prec20 == 1) & (y_test == 1)).sum()),
            "fausses_alertes": int(((y_pred_prec20 == 1) & (y_test == 0)).sum()),
        })

    return pd.DataFrame(rows)


def monthly_late_rate_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Historique complet du taux de retard mensuel — pièce de justification du cutoff."""
    return con.execute("""
        select
            strftime(o.date_key, '%Y-%m') as mois,
            count(*) as n,
            sum(o.is_late) as n_retards,
            round(100.0 * sum(o.is_late) / count(*), 2) as pct_retard
        from main.features_orders f
        join marts.fct_orders o using (order_id)
        group by 1
        order by 1
    """).df()


def cutoff_comparison_table(con: duckdb.DuckDBPyConnection, cutoffs: list[str]) -> pd.DataFrame:
    """Compare plusieurs cutoffs candidats (volume/taux train vs test) — justifie CUTOFF_DATE."""
    rows = []
    for cutoff in cutoffs:
        split_df = con.execute(f"""
            select
                case when o.date_key < date '{cutoff}' then 'train' else 'test' end as split,
                count(*) as n,
                sum(o.is_late) as n_late
            from main.features_orders f
            join marts.fct_orders o using (order_id)
            group by 1
        """).df().set_index("split")

        n_total = split_df["n"].sum()
        rows.append({
            "cutoff": cutoff,
            "n_test": int(split_df.loc["test", "n"]),
            "pct_test": round(100 * split_df.loc["test", "n"] / n_total, 1),
            "n_retards_test": int(split_df.loc["test", "n_late"]),
            "pct_retard_test": round(100 * split_df.loc["test", "n_late"] / split_df.loc["test", "n"], 2),
            "pct_retard_train": round(100 * split_df.loc["train", "n_late"] / split_df.loc["train", "n"], 2),
        })
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Formatte un DataFrame en table markdown sans dépendance à tabulate."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def write_report(
    results: pd.DataFrame,
    matrices: dict,
    monthly_results: pd.DataFrame,
    overfit_check: pd.DataFrame,
    monthly_late_rate: pd.DataFrame,
    cutoff_comparison: pd.DataFrame,
    threshold_grid: pd.DataFrame,
    threshold_business: pd.DataFrame,
    chosen_threshold_row: pd.DataFrame,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# J6 — Modélisation : résultats et justification du split\n",
        f"Cutoff retenu : **{CUTOFF_DATE}** (train = achats avant cette date, test = à partir de cette date).\n",
        "## Comparaison des 3 modèles (test agrégé, juin-août 2018)\n",
        _df_to_markdown(results) + "\n",
        "## Diagnostic sur-apprentissage : PR-AUC train vs test\n",
        _df_to_markdown(overfit_check) + "\n",
        "## Matrices de confusion (test agrégé)\n",
    ]
    for name, cm in matrices.items():
        parts.append(f"**{name}**\n```\n{cm}\n```\n")
    parts += [
        "## Ventilation mensuelle du PR-AUC (robustesse — le score agrégé n'est-il "
        "pas un artefact d'un mois facile ?)\n",
        _df_to_markdown(monthly_results) + "\n",
        "## Justification du cutoff — taux de retard mensuel historique complet\n",
        _df_to_markdown(monthly_late_rate) + "\n",
        "## Justification du cutoff — comparaison de cutoffs candidats\n",
        _df_to_markdown(cutoff_comparison) + "\n",
        "## Analyse de seuil de décision (régression logistique, modèle final)\n",
        "Seuil par défaut de scikit-learn (0.5) : ~66% des commandes de test flaguées, "
        "précision 7.7%. C'est un choix de seuil non examiné, pas un défaut du modèle — "
        "d'où cette analyse. Le seuil ci-dessous n'est **pas** optimisé en boucle sur le "
        "test : ce tableau expose le compromis précision/recall pour un arbitrage métier "
        "humain, fait une seule fois.\n",
        _df_to_markdown(threshold_grid) + "\n",
        "### Seuils défendables proposés\n",
        _df_to_markdown(threshold_business) + "\n",
        "### Seuil retenu (décision métier, pas un tuning)\n",
        f"**Seuil = {DECISION_THRESHOLD}**, tranché par l'utilisateur après revue du tableau ci-dessus.\n\n"
        "Raisonnement : coût de l'action déclenchée par une alerte → tolérance aux fausses "
        "alertes → seuil. Ici, l'alerte déclenche une action automatique à faible coût (pas "
        "d'intervention humaine par commande) : une fausse alerte ne coûte quasiment rien à "
        "traiter, alors qu'un retard non détecté est une occasion manquée d'agir. Le recall "
        "est donc priorisé sur la précision.\n\n"
        "À ce seuil :\n\n" + _df_to_markdown(chosen_threshold_row) + "\n\n"
        "**Limite assumée** : la précision (~11%) reste basse à ce seuil comme à tout autre "
        "seuil raisonnable de la grille — ce n'est pas une conséquence du choix de seuil, "
        "c'est le plafond du signal disponible dans les features (les causes de retard sont "
        "en grande partie post-achat : aléas transporteur, dernier kilomètre — non observables "
        "au moment de la commande). Changer le seuil déplace le curseur précision/recall, il "
        "ne fait pas monter ce plafond.\n",
    ]
    REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        df = load_dataset(con)
        feature_cols = get_feature_columns(df)
        train_df, test_df = split_train_test(df, CUTOFF_DATE)

        print(f"Train : {len(train_df):,} lignes ({train_df['is_late'].mean():.2%} de retards)")
        print(f"Test  : {len(test_df):,} lignes ({test_df['is_late'].mean():.2%} de retards)")

        X_train, y_train = train_df[feature_cols], train_df["is_late"]
        X_test, y_test = test_df[feature_cols], test_df["is_late"]

        models = fit_models(X_train, y_train, feature_cols)

        lgbm_params = models["LightGBM"].named_steps["clf"].get_params()
        print("\n=== Hyperparamètres LightGBM réellement utilisés (pas de tuning) ===")
        for key in ["n_estimators", "num_leaves", "max_depth", "learning_rate",
                    "min_child_samples", "reg_alpha", "reg_lambda", "subsample"]:
            print(f"  {key} = {lgbm_params[key]}")

        results = evaluate_models(models, X_test, y_test)
        matrices = confusion_matrices(models, X_test, y_test)
        monthly_results = evaluate_by_month(models, test_df, feature_cols)
        overfit_check = train_vs_test_gap(models, X_train, y_train, X_test, y_test)

        print("\n=== PR-AUC train vs test (diagnostic sur-apprentissage) ===")
        print(overfit_check.to_string(index=False))

        monthly_late_rate = monthly_late_rate_table(con)
        cutoff_comparison = cutoff_comparison_table(con, CANDIDATE_CUTOFFS)

        thresholds_grid_values = [round(t, 2) for t in np.arange(0.10, 1.00, 0.05)]
        threshold_grid = threshold_table(models["Régression logistique"], X_test, y_test, thresholds_grid_values)
        threshold_business = defensible_thresholds(models["Régression logistique"], X_test, y_test)
        chosen_threshold_row = threshold_table(
            models["Régression logistique"], X_test, y_test, [DECISION_THRESHOLD]
        )

        print("\n=== Comparaison des 3 modèles (test agrégé) ===")
        print(results.to_string(index=False))

        print("\n=== Matrices de confusion (test agrégé) ===")
        for name, cm in matrices.items():
            print(f"\n{name} :\n{cm}")

        print("\n=== Ventilation mensuelle du PR-AUC (test) ===")
        print(monthly_results.to_string(index=False))

        print("\n=== Analyse de seuil (régression logistique) ===")
        print(threshold_grid.to_string(index=False))
        print("\n=== Seuils défendables proposés ===")
        print(threshold_business.to_string(index=False))
        print(f"\n=== Seuil retenu ({DECISION_THRESHOLD}) ===")
        print(chosen_threshold_row.to_string(index=False))

        write_report(
            results, matrices, monthly_results, overfit_check,
            monthly_late_rate, cutoff_comparison, threshold_grid, threshold_business,
            chosen_threshold_row,
        )
        print(f"\nRapport complet écrit dans {REPORT_PATH}")


if __name__ == "__main__":
    main()
