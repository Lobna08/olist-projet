-- Par défaut dbt préfixe le custom schema : main_staging, main_intermediate.
-- Ce macro donne des noms propres : staging, intermediate, main.
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
