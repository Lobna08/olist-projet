-- 1 ligne / zip_code_prefix (déduplication déjà faite dans int_geolocation par AVG + MAX).
-- geo_key = zip_code_prefix (clé naturelle, pas de substitut nécessaire à cette échelle).
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
