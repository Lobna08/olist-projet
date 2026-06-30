-- Grain : 1 ligne / paiement (plusieurs par commande — agrégation dans int_order_payments)
with source as (
    select * from {{ source('raw', 'order_payments') }}
)

select
    order_id,
    payment_sequential,
    payment_type,
    cast(payment_installments as integer) as payment_installments,
    cast(payment_value        as double)  as payment_value
from source
where cast(payment_value as double) >= 0
