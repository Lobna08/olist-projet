# J7 — Explicabilité SHAP (LightGBM, outil de diagnostic)

> Instantané versionné pour la soutenance, généré par `python src/models/explain.py`
> et copié depuis `artifacts/j7_shap_report.md` (non versionné, régénéré à chaque
> run). Ce fichier n'est PAS auto-mis-à-jour : si `explain.py` ou `train.py` changent,
> recopier manuellement la version fraîche ici.

Le modèle final de production est la régression logistique (cf. J6). LightGBM est utilisé ICI uniquement pour lire les interactions non-linéaires entre features via SHAP — une lecture qu'un modèle linéaire ne permet pas.

## Vérification anti-fuite (avant toute lecture des résultats)

Vérifié : aucune colonne interdite (cible, timestamps post-achat, review_score) n'apparaît dans les features transformées vues par SHAP. Le PR-AUC LightGBM test (0.087) est inférieur à son propre PR-AUC train (0.281) et inférieur au PR-AUC test de la régression logistique (0.113) : signature d'un sur-apprentissage modéré, pas d'une fuite (une fuite gonflerait le score test lui-même, pas seulement le score train).

## Bloc A — Coefficients LogReg et comparaison croisée avec SHAP

Le modèle de production (régression logistique) et l'outil de diagnostic (LightGBM + SHAP) sont deux familles de modèles indépendantes : l'un linéaire, l'autre à base d'arbres. S'ils s'accordent sur les features dominantes, c'est une validation croisée du signal — moins susceptible d'être un artefact propre à une seule famille de modèle.

### Top 15 coefficients (régression logistique, valeur absolue)

| feature | coef | abs_coef |
|---|---|---|
| num__delay_est_days | -0.5101 | 0.5101 |
| num__seller_distance_km_max | 0.4791 | 0.4791 |
| num__seller_late_rate_max | 0.1969 | 0.1969 |
| num__nb_distinct_sellers | -0.1946 | 0.1946 |
| num__total_freight | 0.1405 | 0.1405 |
| num__nb_items | -0.1205 | 0.1205 |
| cat__dominant_payment_type_voucher | -0.1158 | 0.1158 |
| cat__dominant_payment_type_boleto | 0.0855 | 0.0855 |
| cat__dominant_payment_type_credit_card | -0.0773 | 0.0773 |
| num__purchase_month | -0.0703 | 0.0703 |
| num__nb_payment_installments | 0.0585 | 0.0585 |
| num__purchase_hour | 0.0365 | 0.0365 |
| num__product_weight_g_sum | 0.0349 | 0.0349 |
| num__freight_ratio | 0.0318 | 0.0318 |
| num__geo_is_unknown | 0.0307 | 0.0307 |

### Comparaison des rangs d'importance (top 15 de chaque modèle, 23 features au total)

**Spearman rho = 0.697** (p = 0.0002) entre le rang |coefficient| LogReg et le rang mean|SHAP| LightGBM sur l'ensemble des features -> corrélation forte et significative.

Recouvrement du top 15 : **13/15** features communes.

- Communes : cat__dominant_payment_type_boleto, cat__dominant_payment_type_credit_card, num__delay_est_days, num__freight_ratio, num__nb_distinct_sellers, num__nb_items, num__nb_payment_installments, num__product_weight_g_sum, num__purchase_hour, num__purchase_month, num__seller_distance_km_max, num__seller_late_rate_max, num__total_freight

- Seulement dans le top LogReg : cat__dominant_payment_type_voucher, num__geo_is_unknown

- Seulement dans le top SHAP : num__product_volume_cm3_sum, num__total_price

Lecture générique de ce type de divergence (mécanisme, pas un constat figé sur les features listées ci-dessus, qui peuvent changer d'un run à l'autre) : une feature qui monte côté LogReg mais pas côté SHAP a typiquement un effet global stable capté par un coefficient linéaire unique ; une feature qui monte côté SHAP mais pas côté LogReg porte typiquement un signal non-linéaire ou conditionnel à d'autres features (interaction), que la régression logistique ne peut pas représenter par construction (modèle additif linéaire).

| feature | logreg_coef | rank_logreg | mean_abs_shap | rank_shap |
|---|---|---|---|---|
| num__delay_est_days | -0.5101 | 1 | 0.6076 | 1 |
| num__purchase_month | -0.0703 | 10 | 0.4775 | 2 |
| num__seller_distance_km_max | 0.4791 | 2 | 0.3152 | 3 |
| num__seller_late_rate_max | 0.1969 | 3 | 0.2022 | 4 |
| num__total_freight | 0.1405 | 5 | 0.1963 | 5 |
| num__nb_items | -0.1205 | 6 | 0.0765 | 6 |
| num__nb_distinct_sellers | -0.1946 | 4 | 0.041 | 7 |
| num__purchase_hour | 0.0365 | 12 | 0.0392 | 8 |
| num__product_volume_cm3_sum | -0.0176 | 19 | 0.0314 | 9 |
| num__nb_payment_installments | 0.0585 | 11 | 0.0299 | 10 |
| num__total_price | -0.0018 | 23 | 0.0288 | 11 |
| num__freight_ratio | 0.0318 | 14 | 0.0275 | 12 |
| cat__dominant_payment_type_boleto | 0.0855 | 8 | 0.026 | 13 |
| cat__dominant_payment_type_credit_card | -0.0773 | 9 | 0.0254 | 14 |
| num__product_weight_g_sum | 0.0349 | 13 | 0.0228 | 15 |

### Point signalé : coefficient contre-intuitif de nb_distinct_sellers

Coefficient négatif (-0.19) : plus une commande a de vendeurs distincts, plus sa probabilité prédite de retard est FAIBLE. Contre-intuitif a priori — on s'attendrait à ce que coordonner plusieurs vendeurs (livraisons séparées à synchroniser) augmente le risque, pas le réduise.

**Le pattern existe déjà dans les données brutes**, avant tout modèle (taux de retard réel par nombre de vendeurs distincts, train) :

| nb_distinct_sellers | n commandes | taux de retard réel |
|---|---|---|
| 1 | 76908 | 8.83% |
| 2 | 915 | 1.86% |
| 3 | 41 | 0.00% |
| 4 | 2 | 0.00% |
| 5 | 1 | 0.00% |

**Limite de fiabilité statistique** : 1.23% seulement des commandes du train ont plus d'un vendeur (959/77,867). Le taux à 0% pour 3+ vendeurs repose sur une quarantaine de commandes — trop peu pour conclure à un effet causal stable, mais le palier à 2 vendeurs (915 commandes, taux divisé par ~4.7 par rapport à 1 vendeur) est basé sur un échantillon assez grand pour ne pas être du seul bruit.

**Hypothèse non prouvée (corrélationnel, pas causal)** : la corrélation entre nb_distinct_sellers et seller_late_rate_max est faible (0.057) — l'explication 'les commandes multi-vendeurs utilisent par hasard des vendeurs historiquement plus fiables' est donc peu soutenue par les données. Piste plus plausible mais non vérifiée ici : les commandes multi-vendeurs sont rares (marketplace) et pourraient être structurellement différentes (vendeurs plus expérimentés à gérer des envois fractionnés, ou catégories de produits différentes) — non testé faute de temps, à citer comme limite si interrogé.

## Importance globale SHAP (top 15 features, moyenne |SHAP| sur le test, LightGBM seul)

| feature | mean_abs_shap |
|---|---|
| num__delay_est_days | 0.6076 |
| num__purchase_month | 0.4775 |
| num__seller_distance_km_max | 0.3152 |
| num__seller_late_rate_max | 0.2022 |
| num__total_freight | 0.1963 |
| num__nb_items | 0.0765 |
| num__nb_distinct_sellers | 0.041 |
| num__purchase_hour | 0.0392 |
| num__product_volume_cm3_sum | 0.0314 |
| num__nb_payment_installments | 0.0299 |
| num__total_price | 0.0288 |
| num__freight_ratio | 0.0275 |
| cat__dominant_payment_type_boleto | 0.026 |
| cat__dominant_payment_type_credit_card | 0.0254 |
| num__product_weight_g_sum | 0.0228 |

## Explications locales (3 commandes représentatives)

### Retard bien détecté (vrai positif le plus net)

**Commande `a236ae70310b60403c3adac81f96d5e6`** — retard réel = 1, probabilité prédite = 0.8742 (valeur de base SHAP = 0.3993 — PAS le taux de retard réel du test [5.48%] : `class_weight="balanced"` recalibre les probabilités, donc cette valeur de base n'est interprétable qu'en ranking relatif, pas comme une fréquence)

| feature | valeur (transformée) | contribution SHAP | sens |
|---|---|---|---|
| num__seller_distance_km_max | 2.746 | +1.0437 | -> retard |
| num__delay_est_days | -0.499 | +0.5395 | -> retard |
| num__seller_late_rate_max | 3.424 | +0.5178 | -> retard |
| num__purchase_month | 0.341 | -0.3161 | -> à l'heure |
| num__total_freight | 2.009 | +0.2858 | -> retard |
| num__purchase_hour | 1.538 | +0.0491 | -> retard |
| num__nb_items | -0.264 | +0.0446 | -> retard |
| num__purchase_weekday | -1.405 | +0.0422 | -> retard |

### Fausse alerte la plus nette (faux positif)

**Commande `a1e313e06320d5c725c2ba02d0cf5be2`** — retard réel = 0, probabilité prédite = 0.8839 (valeur de base SHAP = 0.3993 — PAS le taux de retard réel du test [5.48%] : `class_weight="balanced"` recalibre les probabilités, donc cette valeur de base n'est interprétable qu'en ranking relatif, pas comme une fréquence)

| feature | valeur (transformée) | contribution SHAP | sens |
|---|---|---|---|
| num__seller_distance_km_max | 2.534 | +1.0160 | -> retard |
| num__delay_est_days | -0.622 | +0.5603 | -> retard |
| num__seller_late_rate_max | 0.910 | +0.3736 | -> retard |
| num__total_price | 10.251 | +0.3186 | -> retard |
| num__purchase_month | 0.624 | -0.1944 | -> à l'heure |
| num__total_freight | 2.068 | +0.1541 | -> retard |
| num__purchase_hour | 0.977 | +0.0487 | -> retard |
| num__nb_payment_installments | 2.575 | +0.0484 | -> retard |

### Retard manqué (faux négatif le plus net)

**Commande `7e708aed151d6a8601ce8f2eaa712bf4`** — retard réel = 1, probabilité prédite = 0.1299 (valeur de base SHAP = 0.3993 — PAS le taux de retard réel du test [5.48%] : `class_weight="balanced"` recalibre les probabilités, donc cette valeur de base n'est interprétable qu'en ranking relatif, pas comme une fréquence)

| feature | valeur (transformée) | contribution SHAP | sens |
|---|---|---|---|
| num__delay_est_days | 1.953 | -1.0218 | -> à l'heure |
| num__purchase_month | 0.058 | -0.4893 | -> à l'heure |
| num__total_price | -0.534 | -0.1253 | -> à l'heure |
| num__seller_late_rate_max | -0.140 | -0.0556 | -> à l'heure |
| num__freight_ratio | 1.232 | +0.0534 | -> retard |
| num__nb_items | -0.264 | +0.0414 | -> retard |
| num__total_freight | -0.200 | +0.0306 | -> retard |
| num__seller_distance_km_max | 0.420 | +0.0268 | -> retard |

## Bloc B — Diagnostic du sur-apprentissage LightGBM : hypothèse seller_late_rate_max

Constat de départ (J6) : LightGBM sur-apprend nettement plus que la régression logistique (ratio PR-AUC train/test **3.23** contre **1.42**). Hypothèse testée : seller_late_rate_max — un taux de retard vendeur lissé mais quasi unique par ligne (70,456/77,867 valeurs distinctes sur le train, soit 90.5%) — permettrait à l'arbre de mémoriser des historiques vendeur individuels plutôt que d'apprendre un signal généralisable.

### Table d'ablation

| variante | pr_auc_train | pr_auc_test | ratio_train_test |
|---|---|---|---|
| Référence — Régression logistique (J6) | 0.1599 | 0.1128 | 1.42 |
| Baseline — LightGBM (J6, toutes features) | 0.2809 | 0.0871 | 3.23 |
| Hypothèse — sans seller_late_rate_max | 0.2633 | 0.0964 | 2.73 |
| Contrôle — sans delay_est_days (top SHAP) | 0.2475 | 0.053 | 4.67 |
| Contrôle — sans purchase_month (top SHAP) | 0.2323 | 0.0736 | 3.16 |
| Contrôle — sans seller_distance_km_max (top SHAP) | 0.2678 | 0.0786 | 3.41 |
| Capacité réduite — num_leaves=7 (toutes features) | 0.2607 | 0.0916 | 2.85 |
| Capacité réduite — n_estimators=30 (toutes features) | 0.2553 | 0.0858 | 2.98 |
| Combo — sans seller_late_rate_max + num_leaves=7 | 0.2385 | 0.0972 | 2.45 |

### Test de spécificité (groupe de contrôle)

Retirer seller_late_rate_max : ratio 2.73 (test 0.0964), améliore le PR-AUC test vs baseline (0.0871).

- sans delay_est_days : ratio 4.67 (test 0.053), dégrade le PR-AUC test vs baseline (0.0871)
- sans purchase_month : ratio 3.16 (test 0.0736), dégrade le PR-AUC test vs baseline (0.0871)
- sans seller_distance_km_max : ratio 3.41 (test 0.0786), dégrade le PR-AUC test vs baseline (0.0871)

**Verdict de spécificité : Spécifique à cette feature.** Retirer n'importe quelle feature forte ne réduit pas mécaniquement le sur-apprentissage — retirer delay_est_days (le signal SHAP le plus fort) DÉGRADE le PR-AUC test et AUGMENTE le ratio (le modèle, privé de son signal principal, sur-apprend davantage sur ce qui lui reste). seller_late_rate_max est la seule feature du top SHAP dont le retrait améliore simultanément la généralisation ET réduit le sur-apprentissage.

### Verdict

**Hypothèse CONFIRMÉE comme contributeur réel et spécifique, mais PARTIEL.** Retirer seller_late_rate_max referme 28% de l'écart de ratio entre LightGBM baseline (3.23) et la régression logistique (1.42) — un effet mesurable, pas un artefact de bruit (confirmé par le groupe de contrôle ci-dessus). Le mécanisme plausible : une feature de type 'target encoding' (taux calculé à partir de l'historique de la cible elle-même) à cardinalité quasi unique donne à un modèle à base d'arbres la possibilité de découper l'espace en tranches qui isolent presque un vendeur individuel — un degré de liberté que la régression logistique, avec un coefficient global unique sur cette variable, ne possède pas.

**Mais 72% de l'écart reste inexpliqué par cette seule feature.** Le ratio après retrait (2.73) reste bien au-dessus de celui de la régression logistique (1.42). Deux réglages de capacité, testés indépendamment sur le modèle complet (aucune feature retirée), réduisent aussi le ratio sans toucher aux features : num_leaves=7 seul -> 2.85, n_estimators=30 seul -> 2.98. En combinant retrait + capacité réduite, le ratio descend à **2.45** (PR-AUC test 0.0972) — le plus bas obtenu ici, sans jamais rejoindre celui de la régression logistique.

**Explications classées par plausibilité :**

1. **seller_late_rate_max (target encoding à cardinalité quasi unique) — CONFIRMÉ**, contributeur mesurable et spécifique (~28% de l'écart, validé par groupe de contrôle).
2. **Capacité intrinsèque de l'ensemble d'arbres (num_leaves, n_estimators) — PLAUSIBLE**, effet mesuré indépendamment et cumulatif avec (1), mais ne ferme pas l'écart restant à lui seul non plus.
3. **Écart résiduel après (1)+(2) combinés** (2.45 contre 1.42 pour la régression logistique) : cohérent avec la conclusion déjà actée en J6 — LightGBM, même régularisé et même privé de sa feature la plus problématique, généralise structurellement moins bien que la régression logistique sur ce problème à faible signal. Ce test confirme ce choix de modèle de production plutôt qu'il ne le remet en cause.

## Métrique retenue et pourquoi

**PR-AUC**, jamais l'accuracy. Sur un test à 5.48% de retards, un modèle qui prédit toujours "à l'heure" atteint 94.5% d'accuracy sans capturer un seul retard — l'accuracy ne distingue pas ça d'un modèle utile. Le ROC-AUC est écarté aussi : il intègre le taux de vrais négatifs (majoritaire et facile ici), ce qui le rend optimiste sur des classes très déséquilibrées. Le PR-AUC ne regarde que la classe positive (précision et recall du retard), la classe qui compte pour la décision métier. Il est systématiquement rapporté à côté de son plancher no-skill (le taux de retard de la période évaluée) car sans ce plancher un PR-AUC n'est pas interprétable en absolu.

## Limites (section honnête)

1. **Plafond de signal, pas de seuil.** PR-AUC test ≈ 0.11-0.13 selon le seuil : un signal réel (2x le plancher no-skill) mais modeste. Les causes dominantes de retard sont post-achat (aléas transporteur, dernier kilomètre) et non observables au moment de la commande par construction — c'est une contrainte du problème (anti-fuite), pas un manque d'effort de feature engineering.

2. **Probabilités non calibrées.** `class_weight="balanced"` (LogReg et LightGBM) recalibre les probabilités pour compenser le déséquilibre : la probabilité de sortie n'est PAS une fréquence réelle de retard (ex. le point de base SHAP moyen de LightGBM est ≈40%, très au-dessus du 5.48% réel du test). Les probabilités ne sont fiables qu'en ranking relatif ("plus risqué que"), jamais en lecture absolue ("X% de chances de retard"). Implication directe pour le seuil retenu (0.60, cf. J6) : ce n'est pas "60% de chances de retard", c'est un point de coupure choisi sur la courbe précision/recall.

3. **LightGBM surapprend plus que la régression logistique** (ratio PR-AUC train/test 3.2x contre 1.4x, même après réduction de `num_leaves` et ajout de `reg_lambda`). C'est la raison directe pour laquelle LightGBM n'est pas le modèle de production — il est gardé seulement pour SHAP, où le sur-apprentissage biaise l'échelle des contributions mais pas leur ordre de grandeur relatif.

4. **Un seul split temporel, pas de validation glissante.** Le taux de retard mensuel varie de 1.36% à 21.36% sur la période disponible (cf. J6) : un cutoff unique capture une fenêtre de test possiblement non représentative des mois hors échantillon. Une validation walk-forward (plusieurs cutoffs successifs) donnerait une estimation plus robuste — non faite ici par contrainte de temps.

5. **Échantillons faibles aux seuils extrêmes.** L'estimation de précision à des seuils élevés (ex. 0.90 : 80 commandes flaguées) est bruitée par un petit dénominateur — un seul faux positif de plus ou de moins change la précision de plusieurs points. Ne pas sur-interpréter les seuils au-delà de ~0.85.
