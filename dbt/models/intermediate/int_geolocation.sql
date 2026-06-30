-- Centroïde unique par préfixe postal, après filtrage bornes Brésil (stg_geolocation)
-- 31 points aberrants exclus en amont (0.003 %) — chiffré dans stg_geolocation.sql
with stg as (
    select * from {{ ref('stg_geolocation') }}
)

select
    zip_code_prefix,
    avg(lat)   as centroid_lat,
    avg(lng)   as centroid_lng,
    -- max(state) : cohérent par préfixe dans Olist (préfixes mono-état)
    max(state) as state
from stg
group by zip_code_prefix
