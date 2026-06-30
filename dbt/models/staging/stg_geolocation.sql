-- 🔑 Décision J3-6 : bornes physiques Brésil lat[-34, +6] / lng[-74, -28]
-- Exclusions chiffrées : 31 points sur 1 000 163 (0.003 %) hors bornes
with source as (
    select * from {{ source('raw', 'geolocation') }}
),

within_bounds as (
    select
        lpad(cast(geolocation_zip_code_prefix as varchar), 5, '0') as zip_code_prefix,
        cast(geolocation_lat as double) as lat,
        cast(geolocation_lng as double) as lng,
        lower(trim(geolocation_city))   as city,
        upper(trim(geolocation_state))  as state
    from source
    where cast(geolocation_lat as double) between -34 and 6
      and cast(geolocation_lng as double) between -74 and -28
)

select * from within_bounds
