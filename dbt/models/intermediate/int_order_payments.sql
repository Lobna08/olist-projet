with stg as (
    select * from {{ ref('stg_order_payments') }}
),

aggregated as (
    select
        order_id,
        sum(payment_value)        as total_payment,
        -- 🔑 Décision J3-3 : MAX = durée de financement la plus longue sur la commande
        max(payment_installments) as nb_payment_installments
    from stg
    group by order_id
),

-- 🔑 Décision J3-4 : type dominant = celui représentant le montant le plus élevé
-- Tie-breaker alphabétique sur payment_type pour la reproductibilité
dominant as (
    select
        order_id,
        payment_type,
        row_number() over (
            partition by order_id
            order by sum(payment_value) desc, payment_type asc
        ) as rn
    from stg
    group by order_id, payment_type
)

select
    a.order_id,
    a.total_payment,
    a.nb_payment_installments,
    d.payment_type as dominant_payment_type
from aggregated a
join dominant d on a.order_id = d.order_id and d.rn = 1
