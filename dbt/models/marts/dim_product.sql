-- 1 ligne / product_id. Renommé product_key pour aligner sur la FK de fct_orders.
-- stg_products est déjà à ce grain (source Olist = 1 ligne/produit).
with stg as (
    select * from {{ ref('stg_products') }}
)

select
    product_id              as product_key,
    product_category_name,
    product_weight_g,
    product_length_cm,
    product_height_cm,
    product_width_cm
from stg
