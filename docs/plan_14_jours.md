# Plan 14 jours — Olist Delivery-Risk Command Center

> Mode : l'agent (Claude Code) code, j'explique/valide/défends. Deadline stricte.
> Les 3 derniers jours (J11–J14) sont sanctuarisés pour la finition — ne pas les grignoter.

## Bloc 1 — Fondations BI (J1–J4)

- **J1 — Setup + cadrage.** Repo git (structure `data/ src/ models/ notebooks/ docs/`, `.gitignore`, `requirements.txt`), CLAUDE.md + skill en place. Définir la cible par écrit (livré en retard = 1), figer le moment de prédiction = order time. Démarrer l'EDA. *Livrable : repo propre + page de cadrage.*
- **J2 — EDA + schéma.** Profilage, grain, clés, valeurs manquantes. Dessiner le star schema (fait = commandes ; dims = client, vendeur, produit, date, géo). Charger les CSV bruts dans un schéma `raw` DuckDB. *Livrable : star schema validé + raw dans DuckDB.*
- **J3 — dbt : sources + staging.** Init dbt Core (`dbt-duckdb`), sources, un staging par table, traduction des catégories en seed. *Livrable : couche staging fonctionnelle.*
- **J4 — dbt : marts + tests → ⚠️ POINT DE CONTRÔLE.** Fait + dimensions, tests `unique`/`not_null`/`relationships`. Décision : pipeline OK → on garde dbt ; en retard → bascule SQL DuckDB, dbt en "future work". *Livrable : star schema testé + capture du DAG `dbt docs`.*

## Bloc 2 — Cœur IA (J5–J7)

- **J5 — Feature mart + baseline.** Features **point-in-time uniquement** (fret, distance vendeur→client, poids/dimensions, retard historique vendeur, région, paiement, largeur fenêtre promise, saisonnalité). Baseline (classe majoritaire / régression logistique). *Livrable : features anti-fuite + baseline chiffrée.* **[Partie porteuse — je dois savoir l'expliquer]**
- **J6 — Modèle principal.** LightGBM dans un `Pipeline` scikit-learn, gestion du déséquilibre, **split temporel**. *Livrable : modèle > baseline.* **[Partie porteuse]**
- **J7 — Évaluation + SHAP.** Métrique justifiée (precision/recall/AUC, pas accuracy), section limites. SHAP global + local. ⚠️ Score trop beau = fuite : vérifier cible, timing, split. *Livrable : éval défendable + SHAP.* **[Partie porteuse]**

## Bloc 3 — Intégration AI↔BI (J8–J10)

- **J8 — Réinjection des prédictions.** Score de risque (régression logistique, modèle de production) + top drivers → colonnes/dimensions dans DuckDB, filtrables. Drivers = décomposition linéaire du score LogReg (`coef × valeur standardisée`), pas SHAP : SHAP a été calculé sur LightGBM (J7), qui n'est pas le modèle de production — afficher un score et une explication issus de deux modèles différents serait incohérent. *Livrable : prédictions dans l'entrepôt.* **[Partie porteuse — cœur de la complémentarité]**
- **J9 — Dashboard Streamlit.** Risque par région/SLA, vue d'ensemble (pas de filtre par vendeur individuel : `docs/star_schema.md` exclut délibérément une dimension vendeur, décision reconfirmée au J8). *Livrable : app locale.*
- **J10 — Drill-down + démo.** Segment risqué → pourquoi (SHAP). Déploiement Streamlit Community Cloud (lien démo CV). *Livrable : dashboard intégré + URL publique.*

## Bloc 4 — Livraison (J11–J14, sanctuarisé)

- **J11 — README + repro.** README narratif (problème→données→schéma→modèle→résultat→lancer). Notebooks → modules `.py`. *Livrable : README qui raconte l'histoire.*
- **J12 — Hygiène + gestion projet.** Historique git propre, board GitHub Projects, références citées. Buffer : NLP stretch seulement si le cœur est excellent. *Livrable : repo propre + board + biblio.*
- **J13 — Présentation + feuille de route.** Deck de validation (idée, pertinence, faisabilité, approche) + roadmap. *Livrable : deck + roadmap.*
- **J14 — Répétition de repro à blanc.** Cloner dans un dossier vierge, installer, une commande, vérifier. Finition. *Livrable : projet reproductible de bout en bout.*

## Règles d'arbitrage
- NLP = bonus (J12), jamais une étape. Pas avant que le cœur soit excellent.
- J11–14 intouchables. Si un jour glisse, le scope du cœur se réduit (un modèle au lieu de trois), jamais la repro ni le README.
- À chaque partie porteuse : si je ne sais pas l'expliquer sans l'agent, on ne passe pas à la suite.
