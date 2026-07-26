# Documentation détaillée

Le [README principal](../README.md) raconte le projet en 5 minutes. Cette page indique
où trouver la preuve derrière chaque affirmation.

## Cadrage et conception

| Document | Contenu |
|---|---|
| [`cadrage.md`](cadrage.md) | Définition de la cible (`is_late`), moment de prédiction, frontière de fuite de données — le document de référence anti-fuite. |
| [`star_schema.md`](star_schema.md) | Schéma complet de l'entrepôt : le fait `fct_orders`, les quatre dimensions, la décision assumée de ne pas modéliser le vendeur comme dimension. |
| [`spec_staging_dbt.md`](spec_staging_dbt.md) | Spécification de la couche staging/intermediate dbt : nettoyage, agrégations, exclusions chiffrées, points de fuite identifiés avant l'écriture du code. |

## Résultats du modèle

| Document | Contenu |
|---|---|
| [`modeling_report.md`](modeling_report.md) | Comparaison des 3 modèles (baseline, régression logistique, LightGBM), justification du split temporel et du cutoff, analyse complète du seuil de décision. |
| [`explainability_report.md`](explainability_report.md) | Comparaison croisée des coefficients de la régression logistique et de l'importance SHAP de LightGBM ; diagnostic du sur-apprentissage de LightGBM avec test d'ablation. |
| [`prediction_integration_report.md`](prediction_integration_report.md) | Preuve que la prédiction est une dimension filtrable de l'entrepôt : structure des tables, requête de jointure à cinq tables, limites assumées. |

## Analyse texte des avis clients

| Document | Contenu |
|---|---|
| [`review_insights_report.md`](review_insights_report.md) | Motifs d'insatisfaction détectés dans les avis négatifs (règles figées + classifieur TF-IDF/régression logistique, F1 macro 0,91 sur 4 motifs), comptages figés, limites (biais de couverture, une catégorie marginale absorbée dans autre), séparation anti-fuite avec le pipeline prédictif. |

## Déploiement

| Document | Contenu |
|---|---|
| [`deployment.md`](deployment.md) | Les deux modes du dashboard (local en direct sur DuckDB / démo publique sur instantané Parquet), pourquoi, et comment rafraîchir la démo publique. |

## Gestion de projet

| Document | Contenu |
|---|---|
| [`plan_14_jours.md`](plan_14_jours.md) | Feuille de route du projet par blocs (fondations BI, cœur IA, intégration AI↔BI, livraison). |

## Captures d'écran

Le dossier [`images/`](images/) contient les trois captures du dashboard utilisées dans
le README (vue d'ensemble, analyse, prédiction).
