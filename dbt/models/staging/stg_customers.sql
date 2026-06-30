with source as (
    select * from {{ source('raw', 'customers') }}
)

select
    -- customer_id : identifiant PAR COMMANDE — un même client a plusieurs customer_id
    -- customer_unique_id : vrai identifiant client (à utiliser pour l'analyse client)
    -- Confusion classique Olist : ne pas les intervertir
    customer_id,
    customer_unique_id,
    lpad(cast(customer_zip_code_prefix as varchar), 5, '0') as customer_zip_code_prefix,
    upper(trim(customer_state))                             as customer_state,
    lower(trim(customer_city))                              as customer_city
from source
