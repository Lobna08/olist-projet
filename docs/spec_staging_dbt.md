# Spec staging dbt — Préparation & nettoyage Olist

> À donner à Claude Code au J3. Décrit la couche raw → staging → intermediate (agrégation à la maille commande).
> Le star schema (marts) vient APRÈS, dans un document séparé.
>
> Légende : ⚠️ = point fuite de données (discipline stricte) · 🔑 = décision à trancher et documenter (ne pas déléguer en aveugle).
>
> Convention de nommage : `stg_<source>` (nettoyage 1:1) · `int_<sujet>` (agrégation/jointure) · seeds pour le statique.

## Architecture des couches

```
raw (9 CSV chargés tels quels)
  └─ stg_*   : 1 modèle par source, nettoyage + typage, MÊME grain que la source
       └─ int_* : agrégation à la maille COMMANDE + jointures intermédiaires
            └─ marts (star schema) — document séparé, étape ultérieure
```

Règle d'or : **toute table multi-lignes par commande doit être agrégée à la maille commande dans un `int_` AVANT d'entrer dans le fait.** C'est le point n°1 du projet.

---

## 1. `stg_orders` (source : olist_orders) — grain : 1 ligne / commande

- Caster en timestamp : `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_estimated_delivery_date`.
- 🔑 **Filtrage statut** : pour le périmètre modèle, ne garder que `order_status = 'delivered'`. Documenter le volume retiré et le % que ça représente.
- 🔑 Exclure les commandes `delivered` dont `order_delivered_customer_date` est NULL (cible incalculable).
- 🔑 Exclure ou corriger les incohérences temporelles : `delivered_customer_date < purchase_timestamp`, délais négatifs. Décider exclusion vs correction, chiffrer.
- ⚠️ **Frontière fuite** : les colonnes `order_delivered_*` servent UNIQUEMENT à fabriquer la cible plus tard. Les marquer mentalement comme « interdites en features ». `order_estimated_delivery_date` est connue à l'achat → autorisée en feature.
- Tests dbt : `unique` + `not_null` sur `order_id`.

## 2. `stg_order_items` → `int_order_items` (source : olist_order_items)

- `stg_order_items` : grain = 1 ligne / **article** (ne pas agréger ici). Caster `price`, `freight_value` en numérique.
- ⚠️ Aberrations : `price <= 0`, `freight_value < 0` → inspecter, 🔑 décider seuil/exclusion.
- `int_order_items` : **agréger à la maille commande** :
  - `nb_items` = count
  - `total_price` = sum(price)
  - `total_freight` = sum(freight_value)
  - `nb_distinct_sellers` = count distinct seller_id
  - `freight_ratio` = total_freight / nullif(total_price,0)  *(feature + KPI)*
- Test : `unique` sur `order_id` dans `int_order_items`.

## 3. `stg_order_payments` → `int_order_payments` (source : olist_order_payments)

- Grain source = plusieurs lignes / commande (échéances/moyens multiples).
- `int_order_payments` : agréger à la commande :
  - `total_payment` = sum(payment_value)
  - `nb_payment_installments` = max ou sum (🔑 décider la sémantique)
  - `dominant_payment_type` = type majoritaire (par valeur ou par count, 🔑 à fixer)
- Test : `unique` sur `order_id`.

## 4. `stg_order_reviews` → `int_order_reviews` (source : olist_order_reviews)

- ⚠️⚠️ **Fuite majeure** : la note et le texte de review arrivent APRÈS la livraison. INTERDITS comme features du modèle de retard. Ils servent aux KPIs satisfaction (BI) uniquement, jamais en entrée du modèle.
- Déduplication : 🔑 plusieurs reviews possibles par commande + doublons. Choisir une règle déterministe (ex. garder la review au `review_creation_date` la plus récente) et l'appliquer.
- `int_order_reviews` : grain = 1 ligne / commande : `review_score` (une valeur), flags `is_negative` (≤2), `is_positive` (≥4).
- Test : `unique` sur `order_id`.

## 5. `stg_customers` (source : olist_customers) — grain : 1 ligne / customer_id

- Nettoyer `customer_zip_code_prefix`, `customer_state`, `customer_city` (casse, espaces).
- Garder `customer_unique_id` (vrai identifiant client, ≠ `customer_id` qui est par commande). 🔑 Bien distinguer les deux — confusion classique.
- Test : `unique` sur `customer_id`.

## 6. `stg_sellers` (source : olist_sellers) — grain : 1 ligne / seller_id

- Nettoyer `seller_zip_code_prefix`, `seller_state`, `seller_city`.
- Test : `unique` sur `seller_id`.

## 7. `stg_products` (source : olist_products + seed traduction)

- Caster dimensions : `product_weight_g`, `product_length_cm`, `product_height_cm`, `product_width_cm` (features potentielles).
- Joindre la **seed** `product_category_name_translation` pour le libellé anglais.
- 🔑 Catégories sans traduction ou NULL → fallback à décider (« unknown » recommandé, documenté).
- Valeurs manquantes dimensions : 🔑 stratégie par colonne (imputation médiane ? flag « inconnu » ?), pas globale.
- Test : `unique` sur `product_id`.

## 8. `stg_geolocation` → `int_geolocation` (source : olist_geolocation)

- Source = plusieurs lat/long par `zip_code_prefix` + coordonnées aberrantes.
- Nettoyage : filtrer les points **hors bornes du Brésil** (lat/long aberrantes). 🔑 Fixer les bornes.
- `int_geolocation` : **agréger en un centroïde unique par `zip_code_prefix`** (moyenne lat, moyenne long).
- Sert ensuite à calculer `distance_seller_customer` (feature + KPI) — étape feature engineering, J5, partie porteuse.
- Test : `unique` sur `zip_code_prefix` dans `int_geolocation`.

## 9. Seed : `product_category_name_translation`

- Charger en **seed dbt** (petite table statique de référence). Ne PAS la traiter comme une source volumineuse.

---

## Modèle de synthèse `int_orders_enriched` (préparation du fait)

Joindre, **à la maille commande**, `stg_orders` + `int_order_items` + `int_order_payments` + `int_order_reviews`.
- ⚠️ Vérifier qu'après jointure le nombre de lignes = nombre de commandes (pas d'explosion de grain). Test : `unique` sur `order_id`.
- Ne PAS encore calculer la cible ni les features ici : ce modèle prépare les briques propres. Cible + features = étapes J5–J7 (parties porteuses), avec la discipline fuite.

---

## Checklist de fin d'étape (à exiger de Claude Code)

- [ ] Chaque `int_*` a un grain commande prouvé par un test `unique` sur `order_id`.
- [ ] Toutes les colonnes post-livraison (dates de livraison réelle, review) sont identifiées et isolées des futures features.
- [ ] Chaque exclusion est **chiffrée** (combien de lignes / quel %) et documentée.
- [ ] Tests `relationships` posés sur les clés étrangères (order→customer, item→order, etc.).
- [ ] `dbt docs generate` produit un DAG lisible (capture pour README/slides).

## Pièges rappelés

1. **Le grain (n°1)** : joindre sans agréger d'abord = CA et taux de retard faux. C'est ce qu'un correcteur teste en premier.
2. **Fuite** : review et dates de livraison réelle sont des poisons en features. Frontière nette dès le staging.
3. **Documenter les exclusions** : un nettoyage non chiffré se lit « junior ».
