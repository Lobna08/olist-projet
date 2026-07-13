with source as (
    select * from {{ source('raw', 'products') }}
),

-- Décision : fallback 'unknown' pour catégories sans traduction ou NULL
-- Dimensions numériques manquantes (2 lignes) : conservées NULL → imputation médiane au feature engineering
translated as (
    select
        p.product_id,
        coalesce(t.product_category_name_english, 'unknown') as product_category_name,
        cast(p.product_weight_g          as double)  as product_weight_g,
        cast(p.product_length_cm         as double)  as product_length_cm,
        cast(p.product_height_cm         as double)  as product_height_cm,
        cast(p.product_width_cm          as double)  as product_width_cm,
        cast(p.product_name_lenght        as integer) as product_name_length,       -- typo source conservé, alias corrigé
        cast(p.product_description_lenght as integer) as product_description_length, -- idem
        cast(p.product_photos_qty        as integer) as product_photos_qty
    from source p
    left join {{ ref('product_category_name_translation') }} t
        on p.product_category_name = t.product_category_name
)

select * from translated
