-- 1 ligne / customer_id.
-- Dans Olist, customer_id est un identifiant PAR COMMANDE (un même acheteur a
-- plusieurs customer_id). La PK de cette dimension est donc customer_id, pas
-- customer_unique_id — la FK dans fct_orders vient de orders.customer_id.
with stg as (
    select * from {{ ref('stg_customers') }}
)

select
    customer_id,
    customer_state,
    customer_city,
    -- conservé pour la jointure géo dans fct_orders
    customer_zip_code_prefix
from stg
