# Déploiement du dashboard

## Deux modes, un seul code

`app/app.py` détecte automatiquement son mode au démarrage (`is_local_mode()`,
teste si `data/duckdb/olist.db` existe) — aucune configuration manuelle, aucune
requête SQL ne change entre les deux modes.

| | **Local** | **Déployé (Streamlit Community Cloud)** |
|---|---|---|
| Source | `data/duckdb/olist.db` en direct | `data/dashboard_export/*.parquet` (instantané figé) |
| Fraîcheur | Live — relancer `predict.py` met à jour l'app immédiatement | Figée à la date du dernier export, **pas de mise à jour automatique** |
| Pourquoi | Le fichier `.db` (93 Mo) est gitignored, non publié | Streamlit Cloud clone le dépôt GitHub — pas de fichier `.db` disponible |
| Signalé à l'écran | Rien (comportement normal) | Bandeau "Démonstration publique — données figées au [date]" |

Le mode déployé est un **complément** pour obtenir un lien public (démo CV) — il ne
remplace pas le mode local, qui reste la preuve de l'intégration réelle à l'entrepôt
DuckDB (star schema + prédictions comme dimension filtrable, cf. rapport
d'intégration des prédictions).

## Pourquoi Parquet plutôt que d'autres options

- **Pas d'agrégats pré-calculés** : les filtres croisés (état × catégorie × période)
  ont besoin du grain commande pour rester combinables ; pré-agréger figerait une
  granularité et casserait cette flexibilité — inutile de toute façon, le grain
  commande tient déjà en ~14 Mo.
- **Pas toute la base** : `olist.db` complet fait 93 Mo (raw + staging + tables
  intermédiaires inutiles au dashboard). L'export ne reprend que les colonnes que
  `app/app.py` lit réellement (vérifié en lisant le code, 7 tables) — voir le détail
  dans `src/models/export_dashboard_data.py`.
- **Vues DuckDB sur Parquet plutôt qu'un second jeu de requêtes** : en mode déployé,
  `_connect()` (dans `app/app.py`) crée des `VIEW` schema-qualifiées
  (`marts.fct_orders`, `main.order_risk_scores`, ...) qui pointent sur les fichiers
  Parquet — sous les mêmes noms que le warehouse réel. Résultat : le SQL de
  `app/app.py` est identique dans les deux modes, testé et vérifié (mêmes résultats
  agrégés que contre `olist.db`).

## Cycle de rafraîchissement de la démo publique

Le rafraîchissement **n'est pas automatique**. Relancer `predict.py` en local ne met
à jour que `olist.db` (local) — la démo publique reste figée tant que l'export n'est
pas régénéré et poussé sur GitHub.

```
dbt run                                    # star schema à jour
python src/features/build_features.py      # features point-in-time
python src/models/predict.py               # scores + drivers dans le warehouse
python src/models/export_dashboard_data.py # instantané Parquet pour la démo
git add data/dashboard_export/
git commit -m "..."
git push                                   # Streamlit Cloud redéploie automatiquement
```

`data/dashboard_export/MANIFEST.txt` (généré par le script d'export) donne la date
et la taille de chaque fichier — seule façon de vérifier, en regardant le dépôt,
quand la démo a été figée pour la dernière fois.

## Déployer pour la première fois

Étapes côté Streamlit Community Cloud (interface web, à faire manuellement) :

1. Aller sur [share.streamlit.io](https://share.streamlit.io) et se connecter avec
   le compte GitHub qui héberge `Lobna08/olist-projet`.
2. Cliquer **"New app"**.
3. Renseigner :
   - **Repository** : `Lobna08/olist-projet`
   - **Branch** : `main`
   - **Main file path** : `app/app.py`
4. Streamlit Cloud détecte automatiquement `app/requirements.txt` (prioritaire sur
   le `requirements.txt` racine, car situé dans le même dossier que le fichier
   d'entrée) et installe uniquement les 4 paquets nécessaires au dashboard.
5. Cliquer **"Deploy"**. Premier déploiement : quelques minutes.
6. Une fois en ligne, vérifier que le bandeau "Démonstration publique — données
   figées au [date]" s'affiche — s'il est absent, l'app a probablement trouvé un
   `olist.db` quelque part (à investiguer, ne devrait pas arriver sur un clone
   propre).
7. L'URL générée (`https://<nom-app>.streamlit.app`) est le lien à mettre sur le CV.

## Vérification avant de déployer

Simuler le mode déployé en local (sans toucher au vrai `olist.db`) :

```
# Renommer temporairement (PAS supprimer) le fichier local pour forcer le mode Parquet
mv data/duckdb/olist.db data/duckdb/olist.db.bak
streamlit run app/app.py
# Vérifier : bandeau "Démonstration publique" visible, les 3 onglets fonctionnent,
# les filtres répondent, aucune erreur dans le terminal.
mv data/duckdb/olist.db.bak data/duckdb/olist.db   # restaurer avant de continuer
```
