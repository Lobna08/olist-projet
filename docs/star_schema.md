# Star Schema — Olist Delivery-Risk & Satisfaction Command Center

## Diagramme

```
                    ┌─────────────────┐
                    │  dim_date       │
                    │─────────────────│
                    │ date_key (PK)   │
                    │ jour            │
                    │ mois            │
                    │ annee           │
                    │ jour_semaine    │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
┌─────────┴───────┐ ┌────────┴────────┐ ┌──────┴──────────┐
│  dim_customer   │ │   fct_orders    │ │  dim_product    │
│─────────────────│ │─────────────────│ │─────────────────│
│ customer_id(PK) │ │ order_id (PK)   │ │ product_key(PK) │
│ customer_state  │ │                 │ │ category_name   │
│ customer_city   │ │ — Mesures —     │ │ weight_g        │
└─────────────────┘ │ is_late         │ │ length_cm       │
                    │ total_price     │ │ height_cm       │
                    │ total_freight   │ │ width_cm        │
                    │ nb_items        │ └─────────────────┘
                    │ nb_dist_sellers │
                    │ freight_ratio   │ ┌─────────────────┐
                    │ delay_est_days  │ │  dim_geography  │
                    │                 │ │─────────────────│
                    │ — FK —          │ │ geo_key (PK)    │
                    │ customer_id ────┼─│ zip_prefix      │
                    │ product_key ────┼─│ centroid_lat    │
                    │ date_key    ────┼─│ centroid_lng    │
                    │ geo_key     ────┼─│ state           │
                    └─────────────────┘ └─────────────────┘
```

---

## Fait : `fct_orders`

**Grain** : une ligne = une commande (`order_id`). Population : commandes `delivered` avec dates non nulles (voir `docs/cadrage.md`).

### Mesures

| Colonne | Calcul | Rôle |
|---|---|---|
| `is_late` | `delivered_date > estimated_date` → 1, sinon 0 | Cible du modèle ML et KPI principal du dashboard |
| `total_price` | `SUM(price)` sur `order_items` | Valeur de la commande |
| `total_freight` | `SUM(freight_value)` sur `order_items` | Coût logistique |
| `nb_items` | `COUNT(order_item_id)` sur `order_items` | Volume de la commande |
| `nb_distinct_sellers` | `COUNT(DISTINCT seller_id)` sur `order_items` | Complexité logistique (commande multi-vendeurs) |
| `freight_ratio` | `total_freight / total_price` | Part du fret dans la valeur — signal de distance/poids |
| `delay_estimated_days` | `(estimated_date - purchase_timestamp).days` | Délai promis au client, connu à l'achat |

### Clés étrangères

| FK | Dimension cible |
|---|---|
| `customer_id` | `dim_customer` |
| `product_key` | `dim_product` (produit principal = premier item par `order_item_id`) |
| `date_key` | `dim_date` (basé sur `order_purchase_timestamp`) |
| `geo_key` | `dim_geography` (basé sur `customer_zip_code_prefix`) |

---

## Dimensions

### `dim_customer`
**Colonnes** : `customer_id` (PK), `customer_state`, `customer_city`  
**Sert à** : filtrer et agréger le taux de retard par région client dans le dashboard (heatmap état / ville).

### `dim_product`
**Colonnes** : `product_key` (PK), `product_category_name`, `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm`  
**Sert à** : analyser si certaines catégories (produits lourds, volumineux) sont sur-représentées dans les retards.

### `dim_date`
**Colonnes** : `date_key` (PK), `jour`, `mois`, `annee`, `jour_semaine`  
**Sert à** : détecter des patterns temporels — pic de retard en fin de mois, saisonnalité, effet week-end.

### `dim_geography`
**Colonnes** : `geo_key` (PK), `zip_prefix`, `centroid_lat`, `centroid_lng`, `state`  
**Sert à** : calculer la distance approximative vendeur→client et alimenter une carte des zones à risque dans le dashboard.

---

## Décision assumée : le vendeur n'est pas une dimension

Une commande peut avoir plusieurs vendeurs (`order_items` grain = item, pas commande). Choisir un vendeur arbitraire comme FK violerait le grain du fait ou imposerait une dénormalisation injustifiable.

**Choix retenu** : le vendeur est capté de deux façons dans le schéma —
1. `nb_distinct_sellers` en mesure (capte la multiplicité).
2. En feature engineering : taux de retard historique du vendeur calculé en point-in-time et joint à la table de features — sans créer de dimension dédiée.

Cette décision est une limite assumée à mentionner dans le README.
