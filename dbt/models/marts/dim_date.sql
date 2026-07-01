-- Calendrier complet 2016-2019 : 1 ligne / jour, généré par generate_series.
-- Couvre le dataset Olist (2016-10 → 2018-10) avec marge d'un an de chaque côté.
-- Ne pas remplacer par SELECT DISTINCT sur les dates présentes :
-- les jours sans commande disparaîtraient et briseraient les séries temporelles BI.
with calendar as (
    select generate_series::date as date_key
    from generate_series(date '2016-01-01', date '2019-12-31', interval '1 day')
)

select
    date_key,
    extract(day       from date_key)::integer  as jour,
    extract(month     from date_key)::integer  as mois,
    extract(year      from date_key)::integer  as annee,
    -- DOW : 0 = dimanche, 6 = samedi (convention DuckDB / ISO SQL)
    extract(dow       from date_key)::integer  as jour_semaine,
    extract(quarter   from date_key)::integer  as trimestre,
    extract(dow       from date_key) in (0, 6) as is_weekend
from calendar
