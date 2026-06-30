-- ⚠️ Toutes les colonnes sont post-livraison → KPIs BI uniquement, jamais en feature ML
with stg as (
    select * from {{ ref('stg_order_reviews') }}
),

-- 🔑 Décision J3-5 : garder la review la plus récente par commande
-- Tie-breaker : review_id ASC (déterministe) si review_creation_date identique
deduped as (
    select
        *,
        row_number() over (
            partition by order_id
            order by review_creation_date desc, review_id asc
        ) as rn
    from stg
)

select
    order_id,
    review_id,
    review_score,
    review_creation_date,
    (review_score <= 2) as is_negative,
    (review_score >= 4) as is_positive
from deduped
where rn = 1
