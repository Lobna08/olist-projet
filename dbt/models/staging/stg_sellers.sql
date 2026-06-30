with source as (
    select * from {{ source('raw', 'sellers') }}
)

select
    seller_id,
    lpad(cast(seller_zip_code_prefix as varchar), 5, '0') as seller_zip_code_prefix,
    upper(trim(seller_state))                             as seller_state,
    lower(trim(seller_city))                              as seller_city
from source
