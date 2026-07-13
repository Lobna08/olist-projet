-- Grain : 1 ligne / article (ne pas agréger ici — agrégation dans int_order_items)
with source as (
    select * from {{ source('raw', 'order_items') }}
),

typed as (
    select
        order_id,
        order_item_id,
        product_id,
        seller_id,
        cast(shipping_limit_date as timestamp) as shipping_limit_date,
        cast(price         as double) as price,
        cast(freight_value as double) as freight_value
    from source
),

-- Décision : exclure price <= 0 OU freight_value < 0 (erreurs saisie)
-- freight_value = 0 conservé (livraison gratuite légitime)
-- Exclusions chiffrées : 0 ligne sur 112 650 (0.000 %) — dataset propre
cleaned as (
    select *
    from typed
    where price > 0
      and freight_value >= 0
)

select * from cleaned
