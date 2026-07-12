"""
J9 — Dashboard Streamlit : lit main.order_risk_scores + main.order_risk_drivers
joints au star schema (marts.*) EN DIRECT depuis DuckDB, aucune donnée figée.
Lancer depuis la racine du projet : streamlit run app/app.py

Portée : vue d'ensemble + filtres région/catégorie/période + drivers agrégés.
Le drill-down (cliquer une commande -> voir SES drivers individuels) est hors scope,
réservé au J10 (cf. docs/plan_14_jours.md).
"""
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "olist.db"

# Palette de statut (skill dataviz, palette de référence) : réservée à la sémantique
# d'ÉTAT d'une commande (good -> warning -> critical), jamais réutilisée comme couleur
# de série arbitraire. Toujours appariée à un libellé visible (axe/légende Streamlit),
# jamais la couleur seule.
COLOR_GOOD = "#0ca30c"      # Sous seuil
COLOR_WARNING = "#fab219"   # Alerte
COLOR_CRITICAL = "#d03b3b"  # Risque extrême

# is_in_sample : couleur primaire pour le test (honnête, vue par défaut), couleur
# secondaire/muted pour le train (in-sample, optimiste) — renforce visuellement que
# le train est la population à interpréter avec prudence.
COLOR_TEST = "#2a78d6"
COLOR_TRAIN = "#898781"

st.set_page_config(page_title="Olist — Delivery Risk Command Center", layout="wide")


def run_query(sql: str, params: list | None = None) -> pd.DataFrame:
    """
    Connexion DuckDB COURTE : ouverte, exécutée, fermée à chaque appel — jamais une
    connexion tenue pour la durée de vie de l'app.

    Correction faite en cours de session après un test qui a échoué : une connexion
    read_only laissée ouverte (via st.cache_resource, l'approche initialement prévue
    dans le plan) empêche quand même un AUTRE processus d'ouvrir le fichier en
    écriture — DuckDB exige qu'AUCUNE connexion, même read_only, ne soit ouverte pour
    qu'un écrivain puisse attacher le fichier. Reproduit concrètement :
    `python src/models/predict.py` a échoué avec une IOException tant que l'app
    tournait avec une connexion cache_resource ouverte. Fermer après chaque requête
    réduit la fenêtre de verrou à la durée d'une requête (millisecondes ici, tables
    à ~96 470 / ~289 410 lignes) au lieu de toute la session Streamlit — pas une
    garantie absolue contre une collision pile au même instant, mais rend la
    coexistence dashboard + pipeline praticable au lieu d'un blocage permanent.
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        if params:
            return con.execute(sql, params).df()
        return con.execute(sql).df()


def get_filter_options():
    """Options de filtre peuplées par requête live — jamais une liste codée en dur."""
    states = run_query(
        "select distinct customer_state from marts.dim_customer order by 1"
    )["customer_state"].tolist()
    categories = run_query(
        "select distinct product_category_name from marts.dim_product "
        "where product_category_name is not null order by 1"
    )["product_category_name"].tolist()
    months_df = run_query("""
        select distinct d.annee, d.mois
        from marts.fct_orders o
        join marts.dim_date d on d.date_key = o.date_key
        order by 1, 2
    """)
    month_labels = [f"{int(r.annee)}-{int(r.mois):02d}" for r in months_df.itertuples()]
    return states, categories, month_labels


def build_base_cte(
    is_in_sample_filter: bool | None,
    states: list[str],
    categories: list[str],
    ym_min: int,
    ym_max: int,
) -> tuple[str, list]:
    """
    CTE réutilisée par toutes les sections — une seule jointure
    order_risk_scores x fct_orders x dims, pas une par section. Quand le train est
    inclus (is_in_sample_filter=None), is_in_sample REND une colonne du résultat au
    lieu d'être filtrée : chaque section l'utilise comme facette de couleur, pour ne
    jamais fusionner silencieusement scores honnêtes et scores in-sample (cf. J8).
    """
    where = ["(d.annee * 100 + d.mois) between ? and ?"]
    params: list = [ym_min, ym_max]
    if is_in_sample_filter is not None:
        where.append("r.is_in_sample = ?")
        params.append(is_in_sample_filter)
    if states:
        where.append("c.customer_state = ANY(?)")
        params.append(states)
    if categories:
        where.append("p.product_category_name = ANY(?)")
        params.append(categories)
    where_sql = " and ".join(where)
    cte = f"""
        with base as (
            select
                r.order_id, r.risk_probability, r.is_in_sample, r.is_flagged_risk,
                r.risk_tier, o.is_late, c.customer_state, p.product_category_name,
                d.annee, d.mois
            from main.order_risk_scores r
            join marts.fct_orders o using (order_id)
            join marts.dim_customer c using (customer_id)
            join marts.dim_product p using (product_key)
            join marts.dim_date d on d.date_key = o.date_key
            where {where_sql}
        )
    """
    return cte, params


def query_kpis(cte, params) -> pd.DataFrame:
    sql = cte + """
        select
            is_in_sample,
            count(*) as n_commandes,
            avg(is_late) as taux_retard_reel,
            sum(case when risk_tier = 'Alerte' then 1 else 0 end) as n_alerte,
            sum(case when risk_tier like 'Risque extr%' then 1 else 0 end) as n_extreme,
            avg(case when risk_tier like 'Risque extr%' then is_late end) as precision_extreme
        from base
        group by is_in_sample
        order by is_in_sample
    """
    return run_query(sql, params)


def query_tier_breakdown(cte, params) -> pd.DataFrame:
    sql = cte + "select is_in_sample, risk_tier, count(*) as n from base group by 1, 2"
    return run_query(sql, params)


def query_region(cte, params) -> pd.DataFrame:
    sql = cte + """
        select is_in_sample, customer_state,
               count(*) as n_commandes, avg(is_late) as taux_retard_reel
        from base group by 1, 2
    """
    return run_query(sql, params)


def query_category(cte, params) -> pd.DataFrame:
    sql = cte + """
        select is_in_sample, product_category_name,
               count(*) as n_commandes, avg(is_late) as taux_retard_reel
        from base group by 1, 2
    """
    return run_query(sql, params)


def query_period(cte, params) -> pd.DataFrame:
    sql = cte + """
        select is_in_sample, annee, mois,
               count(*) as n_commandes, avg(is_late) as taux_retard_reel
        from base group by 1, 2, 3 order by 2, 3
    """
    return run_query(sql, params)


def query_drivers(cte, params) -> pd.DataFrame:
    sql = cte + """
        select b.is_in_sample, dr.feature_name, count(*) as n
        from base b
        join main.order_risk_drivers dr on dr.order_id = b.order_id
        where dr.driver_rank = 1
        group by 1, 2
        order by n desc
    """
    return run_query(sql, params)


def render_kpis(kpi_df: pd.DataFrame) -> None:
    st.subheader("Vue d'ensemble")
    if len(kpi_df) == 1:
        row = kpi_df.iloc[0]
        cols = st.columns(5)
        cols[0].metric("Commandes scorées", f"{int(row.n_commandes):,}")
        cols[1].metric("Taux de retard réel", f"{row.taux_retard_reel:.1%}")
        cols[2].metric("En Alerte", f"{int(row.n_alerte):,}")
        cols[3].metric("Risque extrême", f"{int(row.n_extreme):,}")
        precision = row.precision_extreme
        cols[4].metric(
            "Précision (risque extrême)",
            f"{precision:.1%}" if pd.notna(precision) else "n/a",
        )
    else:
        # Train inclus : jamais fusionné avec le test dans une seule ligne de KPIs
        # (cf. J8 — les deux populations ne sont pas comparables en valeur absolue).
        display = kpi_df.copy()
        display["population"] = display["is_in_sample"].map(
            {False: "Test (honnête)", True: "Train (in-sample, optimiste)"}
        )
        display = display[
            ["population", "n_commandes", "taux_retard_reel", "n_alerte", "n_extreme", "precision_extreme"]
        ]
        st.dataframe(display, hide_index=True, use_container_width=True)


def render_tier_breakdown(df: pd.DataFrame, include_train: bool) -> None:
    st.subheader("Répartition des risk_tier")
    if df.empty:
        st.info("Aucune commande pour ces filtres.")
        return
    tier_order = sorted(df["risk_tier"].unique(), key=lambda t: (t != "Sous seuil", t != "Alerte"))
    if include_train:
        pivot = df.pivot_table(
            index="risk_tier", columns="is_in_sample", values="n", fill_value=0
        ).reindex(tier_order)
        pivot.columns = ["Test (honnête)" if not c else "Train (in-sample)" for c in pivot.columns]
        st.bar_chart(pivot, color=[COLOR_TEST, COLOR_TRAIN], stack=False)
    else:
        # Un seul y ("n") -> le param color de bar_chart attend une couleur PAR SÉRIE,
        # pas par catégorie de l'axe x. Pour une couleur par barre, il faut une colonne
        # de couleurs littérales (hex) référencée par nom, pas une liste positionnelle.
        status_colors = {"Sous seuil": COLOR_GOOD, "Alerte": COLOR_WARNING}
        plot_df = df.copy()
        plot_df["risk_tier"] = pd.Categorical(plot_df["risk_tier"], categories=tier_order, ordered=True)
        plot_df = plot_df.sort_values("risk_tier")
        plot_df["color"] = plot_df["risk_tier"].map(lambda t: status_colors.get(t, COLOR_CRITICAL))
        st.bar_chart(plot_df, x="risk_tier", y="n", color="color")


def render_rate_vs_volume(df: pd.DataFrame, dim_col: str, label: str, include_train: bool) -> None:
    st.subheader(f"Risque par {label} (taux vs volume)")
    if df.empty:
        st.info("Aucune commande pour ces filtres.")
        return
    plot_df = df.rename(columns={dim_col: label, "n_commandes": "Volume", "taux_retard_reel": "Taux de retard réel"})
    if include_train:
        plot_df["Population"] = plot_df["is_in_sample"].map({False: "Test (honnête)", True: "Train (in-sample)"})
        st.scatter_chart(
            plot_df, x="Volume", y="Taux de retard réel", color="Population",
            size=80,
        )
    else:
        st.scatter_chart(
            plot_df, x="Volume", y="Taux de retard réel", color="#2a78d6", size=80,
        )
    table = plot_df.sort_values("Taux de retard réel", ascending=False)
    cols = [label, "Volume", "Taux de retard réel"] + (["Population"] if include_train else [])
    st.dataframe(
        table[cols].style.format({"Taux de retard réel": "{:.1%}"}),
        hide_index=True, use_container_width=True,
    )


def render_period(df: pd.DataFrame, include_train: bool) -> None:
    st.subheader("Risque par période")
    if df.empty:
        st.info("Aucune commande pour ces filtres.")
        return
    df = df.copy()
    df["periode"] = df["annee"].astype(int).astype(str) + "-" + df["mois"].astype(int).astype(str).str.zfill(2)
    if include_train:
        df["Population"] = df["is_in_sample"].map({False: "Test (honnête)", True: "Train (in-sample)"})
        rate_pivot = df.pivot_table(index="periode", columns="Population", values="taux_retard_reel")
        vol_pivot = df.pivot_table(index="periode", columns="Population", values="n_commandes", fill_value=0)
        st.caption("Taux de retard réel par mois")
        st.line_chart(rate_pivot, color=[COLOR_TEST, COLOR_TRAIN])
        st.caption("Volume de commandes par mois (graphique séparé — jamais un double axe)")
        st.bar_chart(vol_pivot, color=[COLOR_TEST, COLOR_TRAIN], stack=False)
    else:
        rate = df.set_index("periode")["taux_retard_reel"]
        vol = df.set_index("periode")["n_commandes"]
        st.caption("Taux de retard réel par mois")
        st.line_chart(rate, color=COLOR_TEST)
        st.caption("Volume de commandes par mois (graphique séparé — jamais un double axe)")
        st.bar_chart(vol, color=COLOR_TEST)


def render_drivers(df: pd.DataFrame, include_train: bool) -> None:
    st.subheader("Drivers dominants (feature la plus poussante, agrégé)")
    st.caption(
        "Décomposition linéaire exacte du score de la régression logistique "
        "(coef × valeur standardisée), pas SHAP — cf. J8. Vue agrégée sur les "
        "commandes actuellement filtrées ; le détail par commande est réservé au J10."
    )
    if df.empty:
        st.info("Aucune commande pour ces filtres.")
        return
    if include_train:
        df = df.copy()
        df["Population"] = df["is_in_sample"].map({False: "Test (honnête)", True: "Train (in-sample)"})
        pivot = df.pivot_table(index="feature_name", columns="Population", values="n", fill_value=0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        st.bar_chart(pivot, color=[COLOR_TEST, COLOR_TRAIN], stack=False)
    else:
        pivot = df.set_index("feature_name")["n"].sort_values(ascending=False)
        st.bar_chart(pivot, color=COLOR_TEST)


def main() -> None:
    states_all, categories_all, month_labels = get_filter_options()

    st.title("Olist — Delivery Risk Command Center")
    st.caption(
        "Score de risque = régression logistique (modèle de production, J6). "
        "Drivers = décomposition linéaire du même modèle (J8), pas SHAP (diagnostic "
        "LightGBM séparé, J7). Toutes les requêtes ci-dessous tournent en direct sur "
        "DuckDB — aucune donnée figée."
    )

    with st.sidebar:
        st.header("Filtres")
        include_train = st.toggle(
            "Inclure les commandes de train (in-sample, scores optimistes)",
            value=False,
        )
        states_sel = st.multiselect("État client", states_all, default=[])
        categories_sel = st.multiselect("Catégorie produit", categories_all, default=[])
        ym_range = st.select_slider(
            "Période", options=month_labels, value=(month_labels[0], month_labels[-1]),
        )

    if include_train:
        st.warning(
            "Le train est inclus : ces commandes ont été vues par le modèle pendant "
            "l'entraînement, leurs scores sont optimistes. `class_weight=\"balanced\"` "
            "est calibré sur le déséquilibre du train, pas transférable tel quel au "
            "test (cf. J8) — chaque graphique distingue les deux populations par "
            "couleur, jamais fusionnées dans un même total."
        )

    is_in_sample_filter = None if include_train else False
    ym_min = int(ym_range[0].replace("-", ""))
    ym_max = int(ym_range[1].replace("-", ""))
    cte, params = build_base_cte(is_in_sample_filter, states_sel, categories_sel, ym_min, ym_max)

    render_kpis(query_kpis(cte, params))
    render_tier_breakdown(query_tier_breakdown(cte, params), include_train)
    render_rate_vs_volume(query_region(cte, params), "customer_state", "état client", include_train)
    render_rate_vs_volume(query_category(cte, params), "product_category_name", "catégorie produit", include_train)
    render_period(query_period(cte, params), include_train)
    render_drivers(query_drivers(cte, params), include_train)


if __name__ == "__main__":
    main()
