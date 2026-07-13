# dbt Pipeline — Olist Delivery-Risk

## Commandes

Depuis la **racine du projet** (`olist-projet/`) :

```bash
# 1. Charger les CSV dans DuckDB (schéma raw)
python src/ingestion/load_raw.py

# 2. Charger la seed statique (traduction catégories)
dbt --project-dir dbt --profiles-dir dbt seed

# 3. Construire tous les modèles (staging + intermediate)
dbt --project-dir dbt --profiles-dir dbt run

# 4. Vérifier les tests d'intégrité (27 tests grain + FK)
dbt --project-dir dbt --profiles-dir dbt test

# Optionnel — reconstruire uniquement une couche
dbt --project-dir dbt --profiles-dir dbt run --select staging
dbt --project-dir dbt --profiles-dir dbt run --select intermediate
```

Base DuckDB produite : `data/duckdb/olist.db`

---

## Architecture des trois couches

```
raw (DuckDB)          stg_* (views)             int_* (tables)
──────────────        ─────────────────         ──────────────────────
9 CSV chargés    →    typage + nettoyage    →    agrégation maille
tels quels            grain = source             commande + jointures
```

**Règle d'or :** toute table multi-lignes par commande (`order_items`, `payments`, `reviews`, `geolocation`) est agrégée dans un `int_` avant d'entrer dans le fait. Joindre sans agréger = explosion de grain = métriques fausses.

| Couche | Modèles | Matérialisation |
|--------|---------|-----------------|
| Staging | `stg_orders`, `stg_order_items`, `stg_order_payments`, `stg_order_reviews`, `stg_customers`, `stg_sellers`, `stg_products`, `stg_geolocation` | view |
| Intermediate | `int_order_items`, `int_order_payments`, `int_order_reviews`, `int_geolocation`, `int_orders_enriched` | table |

---

## Décisions clés

**1 — Incohérences temporelles (`stg_orders`)**
EDA (2026-06-30) : 0 incohérence trouvée sur les commandes delivered. Le contrôle est posé en colonne `has_temporal_inconsistency` pour reproductibilité, rien n'est exclu.

**2 — Aberrations prix/fret (`stg_order_items`)**
Exclus : `price <= 0` ou `freight_value < 0`. Conservé : `freight_value = 0` (livraison gratuite légitime). Volume exclu : 0 ligne sur 112 650 — dataset propre.

**3 — `nb_payment_installments` (`int_order_payments`)**
`MAX(payment_installments)` = durée de financement la plus longue sur la commande. `SUM` serait absurde (additionner des durées de plans différents).

**4 — `dominant_payment_type` (`int_order_payments`)**
Type représentant le montant le plus élevé (par valeur, pas par count). Tie-breaker alphabétique pour la reproductibilité.

**5 — Déduplication reviews (`int_order_reviews`)**
Garder la review la plus récente (`review_creation_date DESC`). Tie-breaker : `review_id ASC` (déterministe si même date).

**6 — Bornes géolocalisation (`stg_geolocation`)**
Filtrage : latitude ∈ [-34, +6], longitude ∈ [-74, -28]. Points hors bornes exclus : 31 sur 1 000 163 (0.003 %). Centroïde calculé dans `int_geolocation` (moyenne lat/lng par préfixe).

**7 — Test FK `stg_order_items → stg_orders`**
Ce test serait un faux positif structurel : `stg_orders` filtre sur `delivered`, mais `stg_order_items` couvre tous les statuts. Les 2 461 "orphelins" sont les items des commandes non-delivered — comportement attendu, pas une erreur de données. Remplacé par le test inverse : `stg_orders.order_id → int_order_items.order_id` ("toute commande delivered a au moins un item").

---

## Frontières fuite

Les colonnes suivantes sont présentes dans `stg_orders` et `int_orders_enriched` mais **interdites comme features du modèle ML** :

- `order_delivered_customer_date` — inconnue à l'achat, sert uniquement à construire `is_late`
- `order_delivered_carrier_date` — idem
- `review_score`, `is_negative`, `is_positive` — postérieures à la livraison, KPIs BI uniquement

Colonnes autorisées en feature (connues à `order_purchase_timestamp`) : `order_estimated_delivery_date`, `order_approved_at`, et toutes les mesures `int_order_items` / `int_order_payments`.

---

## Exclusions documentées

| Modèle | Règle | Volume exclu | % |
|--------|-------|-------------|---|
| `stg_orders` | `order_status != 'delivered'` | 2 963 | 3.0 % |
| `stg_orders` | `order_delivered_customer_date IS NULL` | 8 | 0.01 % |
| `stg_order_items` | `price <= 0` ou `freight_value < 0` | 0 | 0.00 % |
| `stg_geolocation` | hors bornes Brésil | 31 | 0.003 % |

**Population finale de modélisation : 96 470 commandes (97.0 % du dataset initial).**
