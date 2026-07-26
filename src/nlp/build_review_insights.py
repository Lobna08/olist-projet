"""
Étiquetage des motifs d'insatisfaction (avis négatifs, note <= 2) et écriture de
main.review_insights (order_id, motif, sentiment, texte_nettoye).

Catégories FIGÉES le 2026-07-26 (décision utilisateur, ne plus rouvrir) :
retard_livraison, livraison_incomplete, produit_incorrect, produit_endommage
(4 motifs -- aspect_mismatch retiré le même jour : 51 avis, F1=0.33, trop rare et trop
bruité pour être un filtre BI fiable, absorbé dans "autre" plutôt que présenté comme un
5e motif bancal), autre (longue traîne : rupture de stock, annulation, nota fiscal,
aspect_mismatch...).

La colonne `motif` de la table finale distingue 3 cas, à ne jamais confondre :
- un des 4 motifs réels (règle certaine, ou classifieur confiant >= CONFIDENCE_THRESHOLD)
- "autre" : avis NÉGATIF avec texte, mais sans motif clair (ni règle, ni confiance
  suffisante du classifieur) -- une vraie observation, pas un défaut de couverture.
- "non_applicable" : avis pas négatif (neutre/positif) ou sans texte -- la notion même
  de "motif d'insatisfaction" n'a pas de sens ici. Toute analyse de répartition des
  motifs DOIT filtrer sur sentiment = 'negatif', sinon "non_applicable" (~80% des
  lignes) écrase visuellement les 4 vrais motifs.

Pipeline en 2 temps :
1. Règles regex (figées) -> étiquette faible sur le corpus négatif, priorité fixe
   quand plusieurs motifs matchent le même avis.
2. TF-IDF + régression logistique multinomiale, entraîné sur les avis étiquetés par
   les règles (hors "autre"), utilisé ensuite pour reclasser UNIQUEMENT les avis que
   les règles ont laissés en "autre" -- et seulement si sa probabilité prédite dépasse
   CONFIDENCE_THRESHOLD ; sinon l'avis reste "autre" (cf. docstring de
   reclassify_autre_with_classifier pour la justification de ce choix).

Lancer depuis la racine du projet : python src/nlp/build_review_insights.py
Séquencement : nécessite `dbt run` au préalable (marts.fct_orders, staging.stg_order_reviews).
"""
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "olist.db"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "review_insights_report.md"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Seuil de confiance (proba max prédite par le classifieur) sous lequel un avis
# "autre" par les règles RESTE "autre" au lieu d'être forcé dans un des 4 motifs.
# Relevé de 0.5 à 0.6 (décision utilisateur) : sur une classification à 4 classes,
# 0.5 n'est que 2x le hasard (1/4 = 0.25) -- un seuil trop permissif pour une
# reclassification qui n'a aucune vérité terrain à confronter (les avis "autre" n'ont
# jamais été relus manuellement). 0.6 reste cohérent avec l'approche conservatrice du
# reste du projet (seuils métier justifiés plutôt qu'optimisés en boucle) : un "autre"
# légèrement plus large mais où chaque reclassement restant est mieux défendable.
CONFIDENCE_THRESHOLD = 0.6

# Ordre de priorité quand un avis matche plusieurs motifs à la fois (ex: "atrasou e
# chegou quebrado" matche retard_livraison ET produit_endommage) : le premier motif
# de cette liste qui matche gagne. Figé en même temps que les règles.
PRIORITY = [
    "retard_livraison",
    "livraison_incomplete",
    "produit_incorrect",
    "produit_endommage",
]

# Règles regex figées (cf. src/nlp/_validate_rules.py pour l'historique de validation
# empirique : comptage indépendant vs après priorité, échantillons manuels de
# non-catégorisés). Dernier ajustement : élargissement de livraison_incomplete pour
# capturer "pedi X e recebi Y" / "só veio um" / "não recebi ... completo".
RULES = {
    "retard_livraison": [
        r"\batraso",
        r"\bnão.*recebi.*prazo",
        r"\bvenceu.*prazo",
        r"prazo.*venc",
        r"\bfora.*prazo",
        r"\bacima.*prazo",
        r"\bprazo.*não.*cum",
        r"\bdemora",
        r"\blento",
        r"\bperdeu",
        r"\bcorreios.*perderam",
    ],
    "livraison_incomplete": [
        r"\bfaltou",
        r"\bmissing",
        r"\breceber.*menos",
        r"\brecebemos.*menos",
        r"\brecebida?.*incompleta",
        r"\bfalta.*item",
        r"\bfaltam.*item",
        r"\bapenass?.*uma?",
        r"\bapenas.*dois",
        r"\bapenas.*duas",
        r"\bsó.*recebi",
        r"\bsomente.*recebi",
        r"\brecebido?.*só",
        r"\brecebido?.*apenas",
        r"\bchegaram.*\d+",
        r"\bfalta.*uma",
        r"\bfalta.*um",
        r"\bincompleto",
        r"\bitem.*falta",
        r"\bquantidade.*err",
        r"\bquantidade.*meno",
        r"\bchegou.*(apenas|só)",
        r"\brecebi.*(apenas|só|menos)",
        r"\bpedi\s+(um|uma|dois|duas|tr[êe]s|quatro|cinco|seis|sete|oito|nove|dez|\d+)\b",
        r"\bs[óo]\s+veio\b",
        r"\bn[ãa]o\s+receb\w*.*complet",
        r"\bpedi\b.*\breceb(i|emos)\b",
    ],
    "produit_incorrect": [
        r"\bdiferente do anúncio",
        r"\bdiferente do anuncio",
        r"\bproduto diferente",
        r"\bmodelo errado",
        r"\bcor errada",
        r"\bcor diferente",
        r"\bfragrancia diferente",
        r"\bperfume diferente",
        r"\brecebi outro",
        r"\bmandaram uma fragrancia diferente",
        r"\brecebi um produto que não tem nada haver",
        r"\bfoi enviado produto diferente",
        r"\bnão foi o que pedi",
        r"\bnao foi o que pedi",
        r"\brecebi.*(outro|outra|diferente)",
        r"\bproduto.*(errado|diferente)",
    ],
    "produit_endommage": [
        r"\bquebr",
        r"\bromp",
        r"\bfragi",
        r"\bdano",
        r"\bdanado",
        r"\bdefeit",
        r"\bmanutenção",
        r"\bdefeituoso",
        r"\bem.*ruim",
        r"\bproduto.*ruim",
        r"\bqualidade.*péss",
        r"\bpéssima",
        r"\bmal.*embal",
        r"\bestoura",
        r"\bfuro",
        r"\b(estou|tá).*solta",
        r"\bsuja",
        r"\bsujo",
    ],
}
# aspect_mismatch retiré le 2026-07-26 (51 avis, F1=0.33 -- cf. docstring module) :
# ces avis retombent dans "autre", ils ne sont plus assignés à un motif dédié.


def normalize_text(text) -> str:
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def clean_text(title, message) -> str:
    """Concatène titre + message normalisés. Même fonction utilisée pour le matching
    des règles ET comme feature du classifieur -> pas de décalage entre les deux."""
    return (normalize_text(title) + " " + normalize_text(message)).strip()


def label_by_rules(text: str) -> str:
    """Applique les règles figées, retourne le premier motif de PRIORITY qui matche,
    sinon 'autre'."""
    matched = set()
    for category, patterns in RULES.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matched.add(category)
                break
    for category in PRIORITY:
        if category in matched:
            return category
    return "autre"


def load_reviewed_orders(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Population = main.fct_orders (grain star schema, commandes livrées, 96 470 lignes
    -- identique à orders_delivered des scripts d'exploration) en LEFT JOIN vers l'avis
    le plus récent par commande (même dédoublonnage que dbt/models/intermediate/
    int_order_reviews.sql : review_creation_date desc, review_id asc en tie-break).
    LEFT JOIN (pas INNER) : les 117 commandes livrées sans avis restent dans la table
    finale avec motif/sentiment/texte à NULL, pour garder le même grain qu'une commande
    de fct_orders -- une jointure BI sur order_id ne doit pas silencieusement perdre
    des lignes.
    """
    query = """
        with reviews_deduped as (
            select
                order_id, review_id, review_score,
                review_comment_title, review_comment_message,
                row_number() over (
                    partition by order_id
                    order by review_creation_date desc, review_id asc
                ) as rn
            from staging.stg_order_reviews
        )
        select
            f.order_id,
            r.review_score,
            r.review_comment_title,
            r.review_comment_message
        from marts.fct_orders f
        left join reviews_deduped r on f.order_id = r.order_id and r.rn = 1
    """
    return con.execute(query).df()


def build_classifier() -> Pipeline:
    """
    TF-IDF (1-2 grammes, min_df=3 pour ignorer le bruit de fautes de frappe isolées)
    + régression logistique. solver="lbfgs" (défaut) bascule automatiquement en
    multinomial (softmax) dès que y a plus de 2 classes -- le paramètre explicite
    multi_class="multinomial" est déprécié depuis sklearn 1.5 et supprimé en 1.7, donc
    volontairement omis ici : le comportement demandé est déjà celui par défaut.
    class_weight="balanced" : livraison_incomplete (~1176 avis) pèse environ 2x plus
    que produit_incorrect (~553) sur les 4 motifs retenus -- déséquilibre plus modéré
    qu'avec aspect_mismatch dans la boucle (ratio qui montait à 1:24), mais la
    repondération reste un défaut peu coûteux à garder pour ne pas laisser le F1 des
    classes minoritaires se dégrader silencieusement.
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])


def evaluate(labeled: pd.DataFrame) -> tuple[str, np.ndarray, list[str]]:
    """
    Split ALÉATOIRE stratifié (pas temporel) : décision assumée, différente du reste
    du projet. Le split temporel anti-fuite protège contre une fuite de FUTUR dans une
    tâche de PRÉDICTION (le modèle de retard ne doit jamais voir, même indirectement,
    des informations postérieures à la commande). Ici la tâche est une classification
    RÉTROSPECTIVE sur du texte déjà écrit après livraison -- il n'y a pas de "futur" à
    protéger, la seule fuite possible serait un avis présent à la fois en train et en
    test, ce que train_test_split empêche par construction (chaque avis est unique et
    assigné à un seul côté). stratify=y : sans ça, un tirage aléatoire pourrait envoyer
    la quasi-totalité des ~50 avis aspect_mismatch d'un seul côté du split.
    """
    X = labeled["texte_nettoye"]
    y = labeled["motif_regle"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    pipeline = build_classifier()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    labels = sorted(y.unique())
    report = classification_report(y_test, y_pred, labels=labels, digits=3, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return report, cm, labels


def reclassify_autre_with_classifier(
    negative: pd.DataFrame, confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[pd.Series, int]:
    """
    Réentraîne un classifieur sur TOUT le labeled set (train+test de l'évaluation
    fusionnés -- le split n'a servi qu'à mesurer une performance honnête, une fois le
    F1 rapporté il n'y a plus de raison de laisser 20% des données de côté pour la
    version qui sert réellement en production) et l'applique uniquement aux avis que
    les règles ont classés "autre".

    Choix assumé : les motifs déjà assignés par les règles ne sont PAS écrasés par le
    classifieur. Les règles sont un jugement humain auditable (chaque motif se relit et
    se justifie mot par mot) ; le classifieur, entraîné sur ces mêmes règles, ne peut
    par construction pas être plus fiable qu'elles sur les avis qu'elles ont déjà
    tranchés -- il ne peut qu'étendre la couverture aux avis que le vocabulaire figé
    n'a pas anticipés.

    Seuil de confiance : predict_proba().max() < confidence_threshold -> l'avis reste
    "autre" (correction apportée après relecture utilisateur -- sans ce seuil, le
    classifieur ne pouvait structurellement jamais prédire "autre" puisqu'il n'a jamais
    vu cette classe à l'entraînement, et forçait donc 100% des avis "autre" dans l'un
    des 4 motifs, ce qui sur-affirme un motif clair là où il n'y en a pas). Piège à
    connaître : le classifieur hérite du biais des règles (mêmes mots-clés dominants),
    donc les avis reclassés avec confiance seront statistiquement proches des avis déjà
    catégorisés -- il ne découvre pas un motif caché, il généralise le vocabulaire des 4
    motifs existants aux formulations synonymes que les règles n'ont pas anticipées.
    """
    labeled = negative[negative["motif_regle"] != "autre"]
    pipeline = build_classifier()
    pipeline.fit(labeled["texte_nettoye"], labeled["motif_regle"])

    autre_mask = negative["motif_regle"] == "autre"
    final = pd.Series(index=negative.index, dtype=object)
    final.loc[~autre_mask] = negative.loc[~autre_mask, "motif_regle"]

    proba = pipeline.predict_proba(negative.loc[autre_mask, "texte_nettoye"])
    classes = pipeline.classes_
    best_idx = np.argmax(proba, axis=1)
    best_proba = proba[np.arange(len(proba)), best_idx]
    predicted_labels = classes[best_idx]
    confident = best_proba >= confidence_threshold
    final.loc[autre_mask] = np.where(confident, predicted_labels, "autre")

    return final, int(confident.sum())


def write_tables(con: duckdb.DuckDBPyConnection, review_insights: pd.DataFrame) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS main")
    con.register("review_insights_df", review_insights)
    con.execute("CREATE OR REPLACE TABLE main.review_insights AS SELECT * FROM review_insights_df")
    con.unregister("review_insights_df")


def write_report(
    counts: pd.Series, report: str, cm: np.ndarray, labels: list[str],
    n_reclassified: int, n_autre_pool: int, review_insights: pd.DataFrame,
    full_distribution: pd.DataFrame, negative_distribution: pd.DataFrame,
) -> None:
    cm_df = pd.DataFrame(cm, index=[f"vrai:{l}" for l in labels], columns=[f"prédit:{l}" for l in labels])
    parts = [
        "# Motifs d'insatisfaction : règles + classifieur (main.review_insights)\n",
        "4 catégories figées (plus \"autre\" pour la longue traîne) : retard_livraison, "
        "livraison_incomplete, produit_incorrect, produit_endommage. aspect_mismatch a "
        "été retiré le 2026-07-26 : 51 avis, F1=0.33 sur le classifieur, trop rare et "
        "trop bruité pour être un filtre BI défendable -- absorbé dans \"autre\" plutôt "
        "que présenté comme un 5e motif fragile. Étiquetage par règles regex, coverage "
        "étendue par un classifieur TF-IDF + régression logistique entraîné sur ces "
        "mêmes règles, appliqué uniquement au-dessus d'un seuil de confiance "
        f"({CONFIDENCE_THRESHOLD}) -- sous ce seuil l'avis reste \"autre\".\n",
        "## Comptage figé (corpus négatif, note <= 2, avec texte, 4 motifs)\n",
        "```\n" + counts.to_string() + "\n```\n",
        "## Évaluation du classifieur (split aléatoire stratifié, 20% test, 4 classes)\n",
        "```\n" + report + "\n```\n",
        "### Matrice de confusion (test)\n",
        "```\n" + cm_df.to_string() + "\n```\n",
        f"## Application : {n_reclassified:,} / {n_autre_pool:,} avis 'autre' reclassés "
        f"avec confiance >= {CONFIDENCE_THRESHOLD}\n",
        "Les motifs déjà assignés par les règles ne sont jamais écrasés par le "
        "classifieur (cf. docstring de reclassify_autre_with_classifier). Les avis "
        "'autre' dont la probabilité max prédite reste sous le seuil restent 'autre' "
        "dans la table finale -- ce n'est pas un residual de couverture manquante, "
        "c'est une observation honnête.\n",
        "## Les 3 valeurs de `motif` dans main.review_insights -- ne pas confondre\n",
        "- un motif réel (règle certaine, ou classifieur confiant) : concerne UNIQUEMENT "
        "les avis `sentiment = 'negatif'` avec texte.\n",
        "- `autre` : avis négatif avec texte, mais sans motif clair.\n",
        "- `non_applicable` : avis neutre/positif, ou avis négatif sans texte -- la "
        "notion de motif d'insatisfaction n'a pas de sens ici. Toute analyse de "
        "répartition des motifs doit filtrer sur `sentiment = 'negatif'`, sinon "
        "`non_applicable` (~80% des lignes) écrase visuellement les 4 vrais motifs.\n",
        "### Distribution complète de `motif` (96 470 lignes, les 3 cas)\n",
        "```\n" + full_distribution.to_string(index=False) + "\n```\n",
        "### Distribution de `motif` filtrée sur `sentiment = 'negatif'` uniquement\n",
        "```\n" + negative_distribution.to_string(index=False) + "\n```\n",
        "## Limites\n",
        "**1. Biais de couverture textuelle.** Les notes basses sont sur-représentées "
        "parmi les avis AVEC texte (un client mécontent commente plus souvent qu'un "
        "client satisfait qui se contente de noter). `main.review_insights` hérite "
        "donc de ce biais : le motif dominant d'une note basse reflète en partie qui a "
        "pris le temps d'écrire, pas uniquement la vraie distribution des problèmes de "
        "livraison. Aucune pondération de correction n'est appliquée ici -- ce serait "
        "une hypothèse supplémentaire non vérifiable sur ce dataset, donc mieux vaut "
        "documenter le biais que le masquer sous un chiffre corrigé arbitrairement.\n",
        "**2. aspect_mismatch absorbé dans 'autre'.** Retiré comme motif dédié le "
        "2026-07-26 : 51 avis (0.5% du corpus négatif), F1=0.33 en évaluation isolée -- "
        "trop rare et trop bruité pour être un filtre BI fiable. Conséquence directe : "
        "un vrai désaccord d'aspect produit (couleur, taille, matière différente de "
        "l'annonce) n'a plus de case dédiée et se retrouve soit dans 'autre', soit "
        "absorbé par 'produit_incorrect' si son vocabulaire recoupe cette règle "
        "(chevauchement déjà présent dans les regex : 'cor diferente' apparaît dans "
        "les deux catégories avant ce retrait) -- une perte de granularité assumée, pas "
        "un oubli.\n",
        "## Grain de la table finale\n",
        f"{len(review_insights):,} lignes, une par commande de marts.fct_orders "
        "(LEFT JOIN sur l'avis le plus récent) : `order_id`, `motif` (voir section "
        "ci-dessus), `sentiment` (négatif/neutre/positif dérivé du score, NULL si pas "
        "d'avis), `texte_nettoye` (NULL si pas d'avis ou texte vide).\n",
        "## Séparation stricte avec le pipeline prédictif (anti-fuite)\n",
        "`main.review_insights` est un module BI à part, jamais consommé par "
        "`src/models/train.py` ni `src/features/build_features.py`. review_score et "
        "review_comment_* sont systématiquement POSTÉRIEURS à la livraison (l'avis est "
        "laissé après réception de la commande) : `dbt/models/staging/"
        "stg_order_reviews.sql` et `int_order_reviews.sql` portent tous les deux un "
        "commentaire d'avertissement explicite en tête de fichier sur ce point, et "
        "`NON_FEATURE_COLUMNS` / `FORBIDDEN_FEATURE_COLUMNS` dans train.py excluent "
        "`review_score` par nom, en défense en profondeur. `motif` et `sentiment` "
        "seraient donc une fuite de cible instantanée s'ils étaient un jour ajoutés "
        "comme feature du modèle de retard (`is_late`) -- ce module reste un enrichissement "
        "de la couche BI (dimension filtrable dans le star schema), jamais une entrée du "
        "modèle prédictif.\n",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    with duckdb.connect(str(DB_PATH)) as con:
        df = load_reviewed_orders(con)

        df["texte_nettoye"] = [
            clean_text(t, m) for t, m in
            zip(df["review_comment_title"], df["review_comment_message"])
        ]
        df["sentiment"] = np.select(
            [df["review_score"] <= 2, df["review_score"] == 3],
            ["negatif", "neutre"],
            default="positif",
        )
        df.loc[df["review_score"].isna(), "sentiment"] = pd.NA

        has_text = df["texte_nettoye"].str.strip() != ""
        is_negative = df["sentiment"] == "negatif"
        negative = df.loc[is_negative & has_text].copy()

        negative["motif_regle"] = negative["texte_nettoye"].apply(label_by_rules)

        counts = negative["motif_regle"].value_counts()
        print("=" * 80)
        print(f"CORPUS NÉGATIF AVEC TEXTE : {len(negative):,} avis")
        print("=" * 80)
        print(counts.to_string())
        print(f"\n% autre : {100 * counts.get('autre', 0) / len(negative):.1f}%")

        labeled = negative[negative["motif_regle"] != "autre"]
        report, cm, labels = evaluate(labeled)
        print("\n" + "=" * 80)
        print("ÉVALUATION DU CLASSIFIEUR (test, 20%, split aléatoire stratifié, 4 classes)")
        print("=" * 80)
        print(report)

        n_autre_pool = int((negative["motif_regle"] == "autre").sum())
        negative["motif_final"], n_reclassified = reclassify_autre_with_classifier(negative)
        print(f"\nReclassés depuis 'autre' avec confiance >= {CONFIDENCE_THRESHOLD} : "
              f"{n_reclassified:,} / {n_autre_pool:,}")

        # motif : "non_applicable" par défaut (pas négatif, ou négatif sans texte) ;
        # remplacé par le motif réel ou "autre" uniquement pour les avis négatifs avec texte.
        df["motif"] = "non_applicable"
        df.loc[negative.index, "motif"] = negative["motif_final"]

        review_insights = df[["order_id", "motif", "sentiment", "texte_nettoye"]].copy()
        review_insights.loc[~has_text, "texte_nettoye"] = pd.NA

        full_distribution = (
            review_insights["motif"].value_counts(dropna=False)
            .rename_axis("motif").reset_index(name="n")
        )
        full_distribution["pct"] = (100 * full_distribution["n"] / len(review_insights)).round(1)

        negative_only = review_insights[review_insights["sentiment"] == "negatif"]
        negative_distribution = (
            negative_only["motif"].value_counts(dropna=False)
            .rename_axis("motif").reset_index(name="n")
        )
        negative_distribution["pct"] = (100 * negative_distribution["n"] / len(negative_only)).round(1)

        print("\n" + "=" * 80)
        print("DISTRIBUTION COMPLÈTE DE motif (96 470 lignes)")
        print("=" * 80)
        print(full_distribution.to_string(index=False))

        print("\n" + "=" * 80)
        print("DISTRIBUTION DE motif -- FILTRÉE sentiment = 'negatif'")
        print("=" * 80)
        print(negative_distribution.to_string(index=False))

        write_tables(con, review_insights)
        print(f"\nÉcrit : main.review_insights ({len(review_insights):,} lignes)")

        write_report(
            counts, report, cm, labels, n_reclassified, n_autre_pool,
            review_insights, full_distribution, negative_distribution,
        )
        print(f"Rapport écrit dans {REPORT_PATH}")


if __name__ == "__main__":
    main()
