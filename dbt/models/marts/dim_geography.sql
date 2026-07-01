-- 1 ligne / zip_code_prefix (déduplication déjà faite dans int_geolocation par AVG + MAX).
-- geo_key = zip_code_prefix (clé naturelle, pas de substitut nécessaire à cette échelle).
-- Sentinel 'UNKNOWN' : 265 commandes (0.27%) ont des codes postaux absents du dataset
-- de géolocalisation Olist — couverture incomplète de la source, pas une erreur de pipeline.
with geo as (
    select * from {{ ref('int_geolocation') }}
)

select
    zip_code_prefix as geo_key,
    zip_code_prefix,
    centroid_lat,
    centroid_lng,
    state
from geo

union all

select
    'UNKNOWN'      as geo_key,
    'UNKNOWN'      as zip_code_prefix,
    null::double   as centroid_lat,
    null::double   as centroid_lng,
    null::varchar  as state
