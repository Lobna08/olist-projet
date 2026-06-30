-- ⚠️⚠️ FUITE MAJEURE — toutes les colonnes de cette table sont postérieures à la livraison
-- review_score, review_comment_* : INTERDITS EN FEATURE du modèle de retard
-- Autorisés uniquement pour les KPIs satisfaction dans la couche BI
with source as (
    select * from {{ source('raw', 'order_reviews') }}
)

select
    review_id,
    order_id,
    cast(review_score as integer)         as review_score,
    review_comment_title,
    review_comment_message,
    cast(review_creation_date    as timestamp) as review_creation_date,    -- ⚠️ post-livraison
    cast(review_answer_timestamp as timestamp) as review_answer_timestamp   -- ⚠️ post-livraison
from source
