# Plan de développement — Olist Delivery-Risk Command Center

## Bloc 1 — Fondations BI

- **Setup + cadrage.** Repo git (structure `data/ src/ models/ notebooks/ docs/`, `.gitignore`, `requirements.txt`), instructions projet + skill en place. Définir la cible par écrit (livré en retard = 1), figer le moment de prédiction = order time. Démarrer l'EDA. *Livrable : repo propre + page de cadrage.*
- **EDA + schéma.** Profilage, grain, clés, valeurs manquantes. Dessiner le star schema (fait = commandes ; dims = client, vendeur, produit, date, géo). Charger les CSV bruts dans un schéma `raw` DuckDB. *Livrable : star schema validé + raw dans DuckDB.*
- **dbt : sources + staging.** Init dbt Core (`dbt-duckdb`), sources, un staging par table, traduction des catégories en seed. *Livrable : couche staging fonctionnelle.*
- **dbt : marts + tests → POINT DE CONTRÔLE.** Fait + dimensions, tests `unique`/`not_null`/`relationships`. Décision : pipeline OK → on garde dbt ; en retard → bascule SQL DuckDB, dbt en "future work". *Livrable : star schema testé + capture du DAG `dbt docs`.*

## Bloc 2 — Cœur IA

- **Feature mart + baseline.** Features **point-in-time uniquement** (fret, distance vendeur→client, poids/dimensions, retard historique vendeur, région, paiement, largeur fenêtre promise, saisonnalité). Baseline (classe majoritaire / régression logistique). *Livrable : features anti-fuite + baseline chiffrée.* **[Partie porteuse — je dois savoir l'expliquer]**
- **Modèle principal.** LightGBM dans un `Pipeline` scikit-learn, gestion du déséquilibre, **split temporel**. *Livrable : modèle > baseline.* **[Partie porteuse]**
- **Évaluation + SHAP.** Métrique justifiée (precision/recall/AUC, pas accuracy), section limites. SHAP global + local. ⚠️ Score trop beau = fuite : vérifier cible, timing, split. *Livrable : éval défendable + SHAP.* **[Partie porteuse]**

## Bloc 3 — Intégration AI↔BI

- **Réinjection des prédictions.** Score de risque (régression logistique, modèle de production) + top drivers → colonnes/dimensions dans DuckDB, filtrables. Drivers = décomposition linéaire du score LogReg (`coef × valeur standardisée`), pas SHAP : SHAP a été calculé sur LightGBM (étape d'explicabilité), qui n'est pas le modèle de production — afficher un score et une explication issus de deux modèles différents serait incohérent. *Livrable : prédictions dans l'entrepôt.* **[Partie porteuse — cœur de la complémentarité]**
- **Dashboard Streamlit.** Risque par région/SLA, vue d'ensemble (pas de filtre par vendeur individuel : `docs/star_schema.md` exclut délibérément une dimension vendeur, décision reconfirmée à l'intégration des prédictions). *Livrable : app locale.*
- **Drill-down + démo.** Segment risqué → pourquoi (SHAP). Déploiement Streamlit Community Cloud (lien démo CV). *Livrable : dashboard intégré + URL publique.*

## Bloc 4 — Livraison 

- **README + repro.** README narratif (problème→données→schéma→modèle→résultat→lancer). Notebooks → modules `.py`. *Livrable : README qui raconte l'histoire.*
- **Présentation + feuille de route.** Deck de validation (idée, pertinence, faisabilité, approche) + roadmap. *Livrable : deck + roadmap.*
- *Livrable : projet reproductible de bout en bout.*
