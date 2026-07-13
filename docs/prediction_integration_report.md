# Intégration des prédictions dans l'entrepôt (main.order_risk_scores, main.order_risk_drivers)

> Instantané versionné pour la soutenance, généré par `python src/models/predict.py`
> et copié depuis `artifacts/prediction_integration_report.md` (non versionné, régénéré à chaque
> run). Ce fichier n'est PAS auto-mis-à-jour : recopier manuellement la version fraîche ici après un changement du script.

Score de risque produit par la régression logistique (modèle de production, seuil 0.6, cf. rapport de modélisation) et écrit dans DuckDB comme dimension filtrable — pas dans un notebook à côté. Drivers = décomposition linéaire exacte du score LogReg (`coef × valeur standardisée`), pas SHAP : SHAP a été calculé sur LightGBM (rapport d'explicabilité), qui n'est pas le modèle de production. Afficher un score et une explication issus de deux modèles différents serait incohérent — écart assumé par rapport au texte initial de docs/plan_14_jours.md.

## Portée : toutes les commandes, avec is_in_sample

**96,470 commandes scorées** (population complète de features_orders, train + test). `is_in_sample=true` pour les 77,867 commandes de train (le modèle les a vues au fit — score optimiste) ; `is_in_sample=false` pour les 18,603 commandes de test (score honnête, out-of-sample). Appliquer predict_proba une deuxième fois sur des lignes déjà utilisées pour le fit n'est PAS une fuite (aucune information du test ne contamine l'entraînement) — la colonne rend la distinction explicite plutôt que de mélanger les deux silencieusement.

### Répartition des risk_tier par is_in_sample

| is_in_sample | risk_tier | n_commandes |
|---|---|---|
| False | Alerte | 4986 |
| False | Risque extrême (probabilité >= 0.71) | 1862 |
| False | Sous seuil | 11755 |
| True | Alerte | 6975 |
| True | Risque extrême (probabilité >= 0.71) | 3158 |
| True | Sous seuil | 67734 |

Bornes de risk_tier calibrées sur le TEST uniquement (q90 = 0.7082) puis appliquées globalement aux deux sous-populations.

**Constat réel (pas une prédiction a priori)** : 13.0% des commandes de train sont en tier Alerte/Extrême, contre 36.8% pour le test — la proportion la plus élevée est du côté test (`is_in_sample=false`), alors que le taux de retard BRUT est plus élevé sur train (8.74%) que sur test (5.48%). Ce n'est PAS le signe que le test est réellement plus risqué : `class_weight="balanced"` est calibré uniquement sur le déséquilibre de train (ratio à l'heure:retard = 10.4:1), pas sur celui de test (ratio = 17.3:1, plus déséquilibré côté test). Le même modèle, appliqué à une population dont le déséquilibre diffère de celui sur lequel il a été calibré, projette des probabilités systématiquement décalées pour cette population — un effet distinct du sur-apprentissage, qui s'ajoute à la limite déjà documentée dans le rapport d'explicabilité (probabilités non calibrées, `class_weight="balanced"`). Ne pas comparer les deux sous-populations comme si elles étaient sur la même échelle de risque réel — c'est tout l'intérêt de garder `is_in_sample` visible plutôt que de mélanger silencieusement les deux.

## Drivers les plus fréquents en position #1 (top 10)

| feature_name | n_fois_driver_1 |
|---|---|
| num__delay_est_days | 49909 |
| num__seller_distance_km_max | 33129 |
| num__seller_late_rate_max | 6914 |
| num__nb_items | 1763 |
| num__total_freight | 1595 |
| num__nb_distinct_sellers | 1250 |
| num__purchase_month | 538 |
| num__nb_payment_installments | 368 |
| cat__dominant_payment_type_credit_card | 345 |
| num__geo_is_unknown | 325 |

## Preuve de filtrabilité (jointure vers le star schema)

Requête exécutée : jointure `main.order_risk_scores` → `marts.fct_orders` → `marts.dim_customer` / `marts.dim_product` / `marts.dim_date`, filtrée sur `is_in_sample = false` (vue par défaut recommandée pour un dashboard : scores honnêtes uniquement). Extrait (5 lignes) :

| customer_state | product_category_name | mois | n_a_risque |
|---|---|---|---|
| SP | health_beauty | 2018-08 | 232 |
| SP | bed_bath_table | 2018-08 | 220 |
| SP | housewares | 2018-08 | 180 |
| SP | sports_leisure | 2018-08 | 132 |
| SP | watches_gifts | 2018-08 | 122 |

## Limite assumée

Aucun filtre par vendeur individuel : `docs/star_schema.md` exclut délibérément une dimension vendeur (le grain de commande ≠ grain d'item, une commande peut avoir plusieurs vendeurs). Rouvrir ce point demanderait de résoudre quel vendeur porte le `seller_late_rate_max` de chaque commande — non fait ici, décision reconfirmée avec l'utilisateur à l'intégration des prédictions. Filtres livrés : région (état client), catégorie produit, période.
