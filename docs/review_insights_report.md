# Motifs d'insatisfaction : règles + classifieur (main.review_insights)

> Instantané versionné pour la soutenance, généré par `python src/nlp/build_review_insights.py`
> et copié depuis `artifacts/review_insights_report.md` (non versionné, régénéré à chaque
> run). Ce fichier n'est PAS auto-mis-à-jour : recopier manuellement la version fraîche ici après un changement du script.

4 catégories figées (plus "autre" pour la longue traîne) : retard_livraison, livraison_incomplete, produit_incorrect, produit_endommage. aspect_mismatch a été retiré le 2026-07-26 : 51 avis, F1=0.33 sur le classifieur, trop rare et trop bruité pour être un filtre BI défendable -- absorbé dans "autre" plutôt que présenté comme un 5e motif fragile. Étiquetage par règles regex, coverage étendue par un classifieur TF-IDF + régression logistique entraîné sur ces mêmes règles, appliqué uniquement au-dessus d'un seuil de confiance (0.6) -- sous ce seuil l'avis reste "autre".

## Comptage figé (corpus négatif, note <= 2, avec texte, 4 motifs)

```
motif_regle
autre                   6319
livraison_incomplete    1176
produit_endommage        698
retard_livraison         576
produit_incorrect        553
```

## Évaluation du classifieur (split aléatoire stratifié, 20% test, 4 classes)

```
                      precision    recall  f1-score   support

livraison_incomplete      0.932     0.928     0.930       235
   produit_endommage      0.928     0.914     0.921       140
   produit_incorrect      0.890     0.874     0.882       111
    retard_livraison      0.883     0.922     0.902       115

            accuracy                          0.913       601
           macro avg      0.908     0.909     0.909       601
        weighted avg      0.914     0.913     0.913       601

```

### Matrice de confusion (test)

```
                           prédit:livraison_incomplete  prédit:produit_endommage  prédit:produit_incorrect  prédit:retard_livraison
vrai:livraison_incomplete                          218                         4                         8                        5
vrai:produit_endommage                               7                       128                         1                        4
vrai:produit_incorrect                               7                         2                        97                        5
vrai:retard_livraison                                2                         4                         3                      106
```

## Application : 1,168 / 6,319 avis 'autre' reclassés avec confiance >= 0.6

Les motifs déjà assignés par les règles ne sont jamais écrasés par le classifieur (cf. docstring de reclassify_autre_with_classifier). Les avis 'autre' dont la probabilité max prédite reste sous le seuil restent 'autre' dans la table finale -- ce n'est pas un residual de couverture manquante, c'est une observation honnête.

## Les 3 valeurs de `motif` dans main.review_insights -- ne pas confondre

- un motif réel (règle certaine, ou classifieur confiant) : concerne UNIQUEMENT les avis `sentiment = 'negatif'` avec texte.

- `autre` : avis négatif avec texte, mais sans motif clair.

- `non_applicable` : avis neutre/positif, ou avis négatif sans texte -- la notion de motif d'insatisfaction n'a pas de sens ici. Toute analyse de répartition des motifs doit filtrer sur `sentiment = 'negatif'`, sinon `non_applicable` (~80% des lignes) écrase visuellement les 4 vrais motifs.

### Distribution complète de `motif` (96 470 lignes, les 3 cas)

```
               motif     n  pct
      non_applicable 87148 90.3
               autre  5151  5.3
livraison_incomplete  1596  1.7
    retard_livraison  1065  1.1
   produit_endommage   832  0.9
   produit_incorrect   678  0.7
```

### Distribution de `motif` filtrée sur `sentiment = 'negatif'` uniquement

```
               motif    n  pct
               autre 5151 42.0
      non_applicable 2946 24.0
livraison_incomplete 1596 13.0
    retard_livraison 1065  8.7
   produit_endommage  832  6.8
   produit_incorrect  678  5.5
```

## Limites

**1. Biais de couverture textuelle.** Les notes basses sont sur-représentées parmi les avis AVEC texte (un client mécontent commente plus souvent qu'un client satisfait qui se contente de noter). `main.review_insights` hérite donc de ce biais : le motif dominant d'une note basse reflète en partie qui a pris le temps d'écrire, pas uniquement la vraie distribution des problèmes de livraison. Aucune pondération de correction n'est appliquée ici -- ce serait une hypothèse supplémentaire non vérifiable sur ce dataset, donc mieux vaut documenter le biais que le masquer sous un chiffre corrigé arbitrairement.

**2. aspect_mismatch absorbé dans 'autre'.** Retiré comme motif dédié le 2026-07-26 : 51 avis (0.5% du corpus négatif), F1=0.33 en évaluation isolée -- trop rare et trop bruité pour être un filtre BI fiable. Conséquence directe : un vrai désaccord d'aspect produit (couleur, taille, matière différente de l'annonce) n'a plus de case dédiée et se retrouve soit dans 'autre', soit absorbé par 'produit_incorrect' si son vocabulaire recoupe cette règle (chevauchement déjà présent dans les regex : 'cor diferente' apparaît dans les deux catégories avant ce retrait) -- une perte de granularité assumée, pas un oubli.

## Grain de la table finale

96,470 lignes, une par commande de marts.fct_orders (LEFT JOIN sur l'avis le plus récent) : `order_id`, `motif` (voir section ci-dessus), `sentiment` (négatif/neutre/positif dérivé du score, NULL si pas d'avis), `texte_nettoye` (NULL si pas d'avis ou texte vide).

## Séparation stricte avec le pipeline prédictif (anti-fuite)

`main.review_insights` est un module BI à part, jamais consommé par `src/models/train.py` ni `src/features/build_features.py`. review_score et review_comment_* sont systématiquement POSTÉRIEURS à la livraison (l'avis est laissé après réception de la commande) : `dbt/models/staging/stg_order_reviews.sql` et `int_order_reviews.sql` portent tous les deux un commentaire d'avertissement explicite en tête de fichier sur ce point, et `NON_FEATURE_COLUMNS` / `FORBIDDEN_FEATURE_COLUMNS` dans train.py excluent `review_score` par nom, en défense en profondeur. `motif` et `sentiment` seraient donc une fuite de cible instantanée s'ils étaient un jour ajoutés comme feature du modèle de retard (`is_late`) -- ce module reste un enrichissement de la couche BI (dimension filtrable dans le star schema), jamais une entrée du modèle prédictif.
