-- Table de synthèse à la maille commande : stg_orders + 3 tables int_*
-- ⚠️ Toutes les jointures sont many-to-one : le grain commande est préservé
-- Preuve : test unique sur order_id dans schema.yml obligatoire
with orders as (
    select * from {{ ref('stg_orders') }}
),

items as (
    select * from {{ ref('int_order_items') }}
),

payments as (
    select * from {{ ref('int_order_payments') }}
),

reviews as (
    select * from {{ ref('int_order_reviews') }}
)

select
    -- identifiants
    o.order_id,
    o.customer_id,

    -- dates autorisées en feature (connues à l'achat)
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_estimated_delivery_date,

    -- ⚠️ INTERDIT EN FEATURE — nécessaires uniquement pour is_late dans la couche marts
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,

    o.has_temporal_inconsistency,

    -- mesures items (int_order_items)
    i.nb_items,
    i.total_price,
    i.total_freight,
    i.nb_distinct_sellers,
    i.freight_ratio,

    -- mesures paiements (int_order_payments)
    p.total_payment,
    p.nb_payment_installments,
    p.dominant_payment_type,

    -- satisfaction BI — ⚠️ INTERDIT EN FEATURE
    r.review_score,
    r.is_negative,
    r.is_positive

from orders o
left join items    i on o.order_id = i.order_id
left join payments p on o.order_id = p.order_id
left join reviews  r on o.order_id = r.order_id
