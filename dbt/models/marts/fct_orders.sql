-- Grain : 1 ligne / order_id. Population : commandes delivered avec date de livraison
-- non nulle (filtrage fait dans stg_orders → 96 470 lignes attendues).
--
-- Règle anti-fuite :
--   order_delivered_customer_date et order_estimated_delivery_date sont consommées
--   ici dans des calculs dérivés et n'apparaissent PAS comme colonnes du fait.
--   is_late est la SEULE sortie autorisée de order_delivered_customer_date.
with orders as (
    select * from {{ ref('int_orders_enriched') }}
),

-- product_key = produit du premier item (order_item_id = 1).
-- Vérifié empiriquement : 0 commande dans la population n'a pas d'item order_item_id = 1.
first_item as (
    select
        order_id,
        product_id as product_key
    from {{ ref('stg_order_items') }}
    where order_item_id = 1
),

-- geo_key = zip_code_prefix du client, requis pour la FK vers dim_geography.
-- 265 commandes (0.27%) ont un code postal absent de la table de géolocalisation.
-- On vérifie l'existence via int_geolocation et on route vers le sentinel 'UNKNOWN'.
customers as (
    select
        c.customer_id,
        coalesce(g.zip_code_prefix, 'UNKNOWN') as geo_key
    from {{ ref('stg_customers') }} c
    left join {{ ref('int_geolocation') }} g
        on c.customer_zip_code_prefix = g.zip_code_prefix
)

select
    -- ── Clé du fait ────────────────────────────────────────────────────────────
    o.order_id,

    -- ── Clés étrangères ────────────────────────────────────────────────────────
    o.customer_id,
    fi.product_key,
    o.order_purchase_timestamp::date          as date_key,
    c.geo_key,

    -- ── Cible (partie porteuse) ────────────────────────────────────────────────
    -- order_delivered_customer_date ne sort PAS du CASE WHEN : colonne brute absente.
    case
        when o.order_delivered_customer_date > o.order_estimated_delivery_date then 1
        else 0
    end                                       as is_late,

    -- ── Mesures issues de int_order_items ──────────────────────────────────────
    o.nb_items,
    o.total_price,
    o.total_freight,
    o.nb_distinct_sellers,
    o.freight_ratio,

    -- ── Feature temporelle pré-achat (pas de fuite) ────────────────────────────
    -- order_estimated_delivery_date est connue du client à l'achat.
    datediff('day',
        o.order_purchase_timestamp,
        o.order_estimated_delivery_date
    )                                         as delay_est_days

from orders o
left join first_item fi on o.order_id = fi.order_id
left join customers  c  on o.customer_id = c.customer_id
