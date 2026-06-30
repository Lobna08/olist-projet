-- Agrégation à la maille commande — grain prouvé par test unique sur order_id (schema.yml)
with stg as (
    select * from {{ ref('stg_order_items') }}
)

select
    order_id,
    count(*)                                       as nb_items,
    sum(price)                                     as total_price,
    sum(freight_value)                             as total_freight,
    count(distinct seller_id)                      as nb_distinct_sellers,
    sum(freight_value) / nullif(sum(price), 0)     as freight_ratio
from stg
group by order_id
