with source as (
    select * from {{ source('raw', 'orders') }}
),

-- Exclusions chiffrées (EDA 2026-06-30) :
--   non-delivered : 2 963 lignes (3.0 %) → hors périmètre modèle
--   delivered sans order_delivered_customer_date : 8 lignes (0.01 %) → cible incalculable
filtered as (
    select *
    from source
    where order_status = 'delivered'
      and order_delivered_customer_date is not null
),

typed as (
    select
        order_id,
        customer_id,

        -- ✅ Colonnes autorisées en feature (connues à l'instant de l'achat)
        cast(order_purchase_timestamp      as timestamp) as order_purchase_timestamp,
        cast(order_approved_at             as timestamp) as order_approved_at,
        cast(order_estimated_delivery_date as timestamp) as order_estimated_delivery_date,

        -- ⚠️ INTERDIT EN FEATURE — postérieures à l'achat
        -- Conservées uniquement pour construire is_late dans la couche marts
        cast(order_delivered_carrier_date  as timestamp) as order_delivered_carrier_date,
        cast(order_delivered_customer_date as timestamp) as order_delivered_customer_date,

        -- 🔑 Décision J3-1 : 0 incohérence trouvée en EDA → flag documenté, rien exclu
        (
            cast(order_delivered_customer_date as timestamp)
                < cast(order_purchase_timestamp as timestamp)
            or
            cast(order_estimated_delivery_date as timestamp)
                < cast(order_purchase_timestamp as timestamp)
        ) as has_temporal_inconsistency

    from filtered
)

select * from typed
