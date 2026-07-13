"""
J9 — Dashboard Streamlit pour un public métier : lit le star schema DuckDB EN DIRECT
(main.order_risk_scores/order_risk_drivers + marts.* + intermediate.int_orders_enriched
pour la satisfaction), aucune donnée figée. Lancer depuis la racine du projet :
streamlit run app/app.py

Principe directeur : un décideur métier doit tout comprendre sans connaissance
technique. Le vocabulaire d'ingénierie (risk_tier, is_in_sample, noms de features
préfixés) est traduit ici, jamais affiché tel quel — sauf dans l'expander
Méthodologie, qui documente la rigueur pour qui veut creuser.

Le drill-down interactif (cliquer une commande -> explorer SES drivers) reste hors
scope, réservé au jalon suivant.
"""
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "duckdb" / "olist.db"

# Un seul bleu séquentiel pour les onglets BI (région, catégorie, évolution
# mensuelle) : ce sont des classements de MAGNITUDE, pas des états, donc pas la
# palette de statut (règle de la skill dataviz : sequential = une teinte, status =
# réservé à un état good/warning/critical).
COLOR_SEQUENTIAL = "#2a78d6"

# Palette de statut — réservée à l'onglet Prédiction (niveau de risque = état d'une
# commande), jamais réutilisée ailleurs comme couleur de série arbitraire.
COLOR_LOW = "#0ca30c"
COLOR_MODERATE = "#fab219"
COLOR_HIGH = "#d03b3b"

MIN_VOLUME = 30  # cf. plan : en dessous, le taux affiché est un artefact d'échantillon

TIER_ORDER = ["Faible", "Modéré", "Élevé"]
TIER_COLORS = {"Faible": COLOR_LOW, "Modéré": COLOR_MODERATE, "Élevé": COLOR_HIGH}

DIRECTION_LABELS = {"retard": "augmente le risque", "à l'heure": "réduit le risque"}

FEATURE_LABELS = {
    "num__delay_est_days": "Délai de livraison promis",
    "num__seller_distance_km_max": "Distance vendeur-client",
    "num__seller_late_rate_max": "Fiabilité du vendeur",
    "num__total_freight": "Frais de port",
    "num__nb_items": "Nombre d'articles",
    "num__nb_distinct_sellers": "Nombre de vendeurs différents",
    "num__purchase_month": "Mois de l'achat",
    "num__purchase_hour": "Heure de l'achat",
    "num__purchase_weekday": "Jour de la semaine de l'achat",
    "num__is_weekend": "Achat le week-end",
    "num__product_volume_cm3_sum": "Volume du colis",
    "num__product_weight_g_sum": "Poids du colis",
    "num__nb_payment_installments": "Nombre d'échéances de paiement",
    "num__total_price": "Montant de la commande",
    "num__total_payment": "Montant payé",
    "num__freight_ratio": "Part du transport dans le prix",
    "num__geo_is_unknown": "Localisation du client inconnue",
    "num__nb_distinct_categories": "Nombre de catégories de produits",
    "cat__dominant_payment_type_boleto": "Paiement par boleto",
    "cat__dominant_payment_type_credit_card": "Paiement par carte de crédit",
    "cat__dominant_payment_type_voucher": "Paiement par bon d'achat",
    "cat__dominant_payment_type_debit_card": "Paiement par carte de débit",
}

BR_STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas", "BA": "Bahia",
    "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo", "GO": "Goiás",
    "MA": "Maranhão", "MT": "Mato Grosso", "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais", "PA": "Pará", "PB": "Paraíba", "PR": "Paraná",
    "PE": "Pernambuco", "PI": "Piauí", "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte", "RS": "Rio Grande do Sul", "RO": "Rondônia",
    "RR": "Roraima", "SC": "Santa Catarina", "SP": "São Paulo", "SE": "Sergipe",
    "TO": "Tocantins",
}

# Les 69 catégories du dataset (liste vérifiée par requête, pas devinée), traduites
# en français plutôt que juste nettoyées (underscore -> espace) : l'anglais brut
# ("fashion_underwear_beach") reste illisible pour un décideur même une fois
# capitalisé. Quelques clés portent les coquilles d'origine du fichier de traduction
# Olist ("fashio_female_clothing", "costruction_tools_*", "home_confort") — la clé
# doit matcher exactement la valeur en base, la traduction corrige le sens, pas la clé.
CATEGORY_LABELS = {
    "agro_industry_and_commerce": "Agro-industrie et commerce",
    "air_conditioning": "Climatisation",
    "art": "Art",
    "arts_and_craftmanship": "Artisanat d'art",
    "audio": "Audio",
    "auto": "Automobile",
    "baby": "Bébé",
    "bed_bath_table": "Linge de maison",
    "books_general_interest": "Livres généraux",
    "books_imported": "Livres importés",
    "books_technical": "Livres techniques",
    "cds_dvds_musicals": "CD, DVD et musique",
    "christmas_supplies": "Articles de Noël",
    "cine_photo": "Cinéma et photo",
    "computers": "Ordinateurs",
    "computers_accessories": "Accessoires informatiques",
    "consoles_games": "Consoles et jeux vidéo",
    "construction_tools_construction": "Outillage de construction",
    "construction_tools_lights": "Éclairage de chantier",
    "construction_tools_safety": "Équipement de sécurité chantier",
    "cool_stuff": "Objets tendance",
    "costruction_tools_garden": "Outillage de jardin",
    "costruction_tools_tools": "Outils de bricolage",
    "diapers_and_hygiene": "Couches et hygiène",
    "drinks": "Boissons",
    "dvds_blu_ray": "DVD et Blu-ray",
    "electronics": "Électronique",
    "fashio_female_clothing": "Mode féminine",
    "fashion_bags_accessories": "Sacs et accessoires de mode",
    "fashion_childrens_clothes": "Mode enfant",
    "fashion_male_clothing": "Mode masculine",
    "fashion_shoes": "Chaussures",
    "fashion_sport": "Mode sport",
    "fashion_underwear_beach": "Sous-vêtements et maillots de bain",
    "fixed_telephony": "Téléphonie fixe",
    "flowers": "Fleurs",
    "food": "Alimentation",
    "food_drink": "Alimentation et boissons",
    "furniture_bedroom": "Mobilier chambre",
    "furniture_decor": "Meubles et décoration",
    "furniture_living_room": "Mobilier salon",
    "furniture_mattress_and_upholstery": "Matelas et literie",
    "garden_tools": "Outils de jardin",
    "health_beauty": "Santé et beauté",
    "home_appliances": "Électroménager",
    "home_appliances_2": "Électroménager (autre)",
    "home_comfort_2": "Confort de la maison (autre)",
    "home_confort": "Confort de la maison",
    "home_construction": "Rénovation et construction",
    "housewares": "Articles ménagers",
    "industry_commerce_and_business": "Industrie, commerce et entreprises",
    "kitchen_dining_laundry_garden_furniture": "Cuisine, salle à manger et jardin",
    "la_cuisine": "Cuisine",
    "luggage_accessories": "Bagagerie",
    "market_place": "Marketplace",
    "music": "Musique",
    "musical_instruments": "Instruments de musique",
    "office_furniture": "Mobilier de bureau",
    "party_supplies": "Articles de fête",
    "perfumery": "Parfumerie",
    "pet_shop": "Animalerie",
    "security_and_services": "Sécurité et services",
    "signaling_and_security": "Signalisation et sécurité",
    "small_appliances": "Petit électroménager",
    "small_appliances_home_oven_and_coffee": "Petit électroménager cuisine",
    "sports_leisure": "Sport et loisirs",
    "stationery": "Papeterie",
    "tablets_printing_image": "Tablettes et impression",
    "telephony": "Téléphonie",
    "toys": "Jouets",
    "unknown": "Catégorie inconnue",
    "watches_gifts": "Montres et cadeaux",
}

st.set_page_config(page_title="Olist — Pilotage risque livraison", layout="wide")


def translate_tier(raw: str) -> str:
    if raw == "Sous seuil":
        return "Faible"
    if raw == "Alerte":
        return "Modéré"
    if raw.startswith("Risque extrême"):
        return "Élevé"
    return raw  # filet de sécurité si une nouvelle valeur apparaît un jour


def translate_feature(raw: str) -> str:
    return FEATURE_LABELS.get(raw, raw)


def translate_state(code: str) -> str:
    return BR_STATE_NAMES.get(code, code)


def translate_category(raw: str) -> str:
    return CATEGORY_LABELS.get(raw, raw.replace("_", " ").capitalize())


def run_query(sql: str, params: list | None = None) -> pd.DataFrame:
    """
    Connexion DuckDB COURTE (ouverte/exécutée/fermée à chaque appel), jamais tenue
    pour la durée de vie de l'app — DuckDB exige qu'aucune connexion, même read_only,
    ne reste ouverte pour qu'un autre processus (predict.py, dbt run) puisse écrire.
    Vérifié empiriquement lors du premier jet de ce dashboard : une connexion tenue
    ouverte via st.cache_resource bloquait predict.py avec une IOException.
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        if params:
            return con.execute(sql, params).df()
        return con.execute(sql).df()


def get_filter_options():
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


def _filter_clause(states, categories, ym_min, ym_max):
    where = ["(d.annee * 100 + d.mois) between ? and ?"]
    params: list = [ym_min, ym_max]
    if states:
        where.append("c.customer_state = ANY(?)")
        params.append(states)
    if categories:
        where.append("p.product_category_name = ANY(?)")
        params.append(categories)
    return " and ".join(where), params


def build_bi_cte(states, categories, ym_min, ym_max) -> tuple[str, list]:
    """
    Base des onglets BI (Vue d'ensemble, Analyse) : population COMPLÈTE (train +
    test, ~96 470 commandes) — is_late, review_score et total_price sont des faits
    historiques connus pour TOUTE commande livrée, is_in_sample ne les concerne pas
    (cette distinction n'existe que pour les prédictions du modèle). Conséquence
    voulue : l'évolution mensuelle du taux de retard montre 2 ans d'historique, pas
    seulement la fenêtre de test de 3 mois.
    """
    where_sql, params = _filter_clause(states, categories, ym_min, ym_max)
    cte = f"""
        with base as (
            select
                o.order_id, o.is_late, o.total_price,
                c.customer_state, p.product_category_name, d.annee, d.mois,
                e.review_score, e.order_purchase_timestamp,
                e.order_estimated_delivery_date, e.order_delivered_customer_date
            from marts.fct_orders o
            join marts.dim_customer c using (customer_id)
            join marts.dim_product p using (product_key)
            join marts.dim_date d on d.date_key = o.date_key
            left join intermediate.int_orders_enriched e on e.order_id = o.order_id
            where {where_sql}
        )
    """
    return cte, params


def build_pred_cte(states, categories, ym_min, ym_max) -> tuple[str, list]:
    """
    Base de l'onglet Prédiction : uniquement les scores honnêtes (is_in_sample=false,
    la population test — le modèle n'a jamais vu ces commandes). Filtré ici une fois
    pour toutes, jamais exposé comme choix à l'écran principal (cf. Méthodologie).
    """
    where_sql, params = _filter_clause(states, categories, ym_min, ym_max)
    cte = f"""
        with base as (
            select
                r.order_id, r.risk_probability, r.risk_tier, o.total_price,
                c.customer_state, p.product_category_name, d.annee, d.mois
            from main.order_risk_scores r
            join marts.fct_orders o using (order_id)
            join marts.dim_customer c using (customer_id)
            join marts.dim_product p using (product_key)
            join marts.dim_date d on d.date_key = o.date_key
            where r.is_in_sample = false and {where_sql}
        )
    """
    return cte, params


# ---------------------------------------------------------------------------
# Onglet 1 — Vue d'ensemble
# ---------------------------------------------------------------------------

def render_overview(cte, params) -> None:
    kpi = run_query(cte + """
        select
            count(*) as n_commandes,
            sum(total_price) as ca_total,
            avg(total_price) as panier_moyen,
            avg(is_late) as taux_retard,
            avg(datediff('day', order_purchase_timestamp, order_delivered_customer_date))
                as delai_reel_jours,
            avg(datediff('day', order_purchase_timestamp, order_estimated_delivery_date))
                as delai_promis_jours,
            avg(review_score) as note_moyenne
        from base
    """, params)
    if kpi.empty or kpi.iloc[0].n_commandes == 0:
        st.info("Aucune commande pour ces filtres.")
        return
    row = kpi.iloc[0]

    row1 = st.columns(4)
    row1[0].metric("Commandes", f"{int(row.n_commandes):,}")
    row1[1].metric("Chiffre d'affaires total", f"{row.ca_total:,.0f} R$")
    row1[2].metric("Panier moyen", f"{row.panier_moyen:,.0f} R$")
    row1[3].metric("Taux de livraison en retard", f"{row.taux_retard:.1%}")

    row2 = st.columns(3)
    row2[0].metric("Délai de livraison réel (moyen)", f"{row.delai_reel_jours:.1f} jours")
    row2[1].metric("Délai promis au client (moyen)", f"{row.delai_promis_jours:.1f} jours")
    note = row.note_moyenne
    row2[2].metric("Note de satisfaction moyenne", f"{note:.2f} / 5" if pd.notna(note) else "n/a")

    st.subheader("Évolution mensuelle du taux de retard")
    monthly = run_query(
        cte + "select annee, mois, count(*) as n, avg(is_late) as taux_retard from base group by 1, 2 order by 1, 2",
        params,
    )
    if not monthly.empty:
        monthly["periode"] = (
            monthly["annee"].astype(int).astype(str) + "-" + monthly["mois"].astype(int).astype(str).str.zfill(2)
        )
        # Même seuil de volume que région/catégorie : les tout premiers mois du
        # dataset n'ont qu'une poignée de commandes (2016-09 : 1 commande, 100% de
        # retard) — un artefact qui écrase l'échelle de tout le graphique et masque
        # la vraie volatilité (1.36% à 21.36% sur les mois avec un volume réel).
        included = monthly[monthly["n"] >= MIN_VOLUME]
        excluded = monthly[monthly["n"] < MIN_VOLUME]
        st.line_chart(included.set_index("periode")["taux_retard"], color=COLOR_SEQUENTIAL)
        if not excluded.empty:
            mois_exclus = ", ".join(excluded["periode"])
            st.caption(
                f"{len(excluded)} mois exclus du graphique (< {MIN_VOLUME} commandes, "
                f"pas assez de volume pour un taux fiable) : {mois_exclus}."
            )


# ---------------------------------------------------------------------------
# Onglet 2 — Analyse
# ---------------------------------------------------------------------------

def _split_by_volume(df: pd.DataFrame, group_col: str):
    included = df[df["n"] >= MIN_VOLUME].copy()
    excluded = df[df["n"] < MIN_VOLUME]
    return included, excluded


def _horizontal_bar(
    df: pd.DataFrame, category_col: str, value_col: str,
    value_format: str, color: str, tooltip: list[str],
) -> None:
    """
    Barres horizontales avec un ORDRE EXPLICITE.

    Le premier jet utilisait un pandas Categorical ordonné en espérant que la
    promotion du champ vers le type Vega-Lite 'ordinal' ferait respecter l'ordre du
    DataFrame — ÇA NE MARCHAIT PAS EN RENDU RÉEL (rapporté par l'utilisateur : les
    états s'affichaient par ordre alphabétique). Le vrai mécanisme Vega-Lite : sans
    `sort` explicite, le tri par défaut d'un axe est ascendant sur la VALEUR du
    champ — peu importe le type nominal/ordinal, ça retombe sur l'alphabétique pour
    du texte. `alt.Y(..., sort=<liste explicite>)` est le mécanisme documenté et sans
    ambiguïté d'Altair pour imposer un ordre — utilisé ici à la place de st.bar_chart,
    qui ne permet pas de contrôler `sort` directement.

    Le volume (n) est en tooltip, jamais dans le libellé de l'axe — évite les
    libellés tronqués ("Mato Grosso do S…") signalés sur la version précédente.
    """
    if df.empty:
        return
    order = df[category_col].tolist()
    chart = (
        alt.Chart(df)
        .mark_bar(color=color)
        .encode(
            x=alt.X(f"{value_col}:Q", title=None, axis=alt.Axis(format=value_format)),
            y=alt.Y(f"{category_col}:N", sort=order, title=None),
            tooltip=tooltip,
        )
        .properties(height=26 * len(df) + 10)
    )
    st.altair_chart(chart, use_container_width=True)


TOP_N_REGIONS = 10


def render_region_analysis(cte, params) -> None:
    st.subheader(f"Taux de retard par région — top {TOP_N_REGIONS} les plus à risque")
    df = run_query(
        cte + "select customer_state, count(*) as n, avg(is_late) as taux_retard from base group by 1",
        params,
    )
    if df.empty:
        st.info("Aucune commande pour ces filtres.")
        return
    included, excluded = _split_by_volume(df, "customer_state")
    if included.empty:
        st.info(f"Aucune région n'atteint le minimum de {MIN_VOLUME} commandes pour ces filtres.")
        return
    included["État"] = included["customer_state"].map(translate_state)
    included["Commandes"] = included["n"]
    included["Taux de retard"] = included["taux_retard"]
    top = included.sort_values("Taux de retard", ascending=False).head(TOP_N_REGIONS)

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        _horizontal_bar(
            top, category_col="État", value_col="Taux de retard", value_format=".0%",
            color=COLOR_SEQUENTIAL, tooltip=["État", "Commandes", alt.Tooltip("Taux de retard", format=".1%")],
        )
    with col_table:
        st.dataframe(
            top[["État", "Commandes", "Taux de retard"]]
                .style.format({"Commandes": "{:,}", "Taux de retard": "{:.1%}"}),
            hide_index=True, use_container_width=True,
        )
    n_not_shown = len(included) - len(top)
    footnote = []
    if n_not_shown > 0:
        footnote.append(f"{n_not_shown} autre(s) région(s) au-dessus du seuil, non affichée(s) (hors top {TOP_N_REGIONS}).")
    if not excluded.empty:
        noms = ", ".join(translate_state(s) for s in excluded["customer_state"])
        footnote.append(f"{len(excluded)} état(s) exclus (< {MIN_VOLUME} commandes) : {noms}.")
    if footnote:
        st.caption(" ".join(footnote))


def render_category_analysis(cte, params) -> None:
    st.subheader("Taux de retard par catégorie de produit")
    df = run_query(
        cte + "select product_category_name, count(*) as n, avg(is_late) as taux_retard from base group by 1",
        params,
    )
    if df.empty:
        st.info("Aucune commande pour ces filtres.")
        return
    included, excluded = _split_by_volume(df, "product_category_name")
    if included.empty:
        st.info(f"Aucune catégorie n'atteint le minimum de {MIN_VOLUME} commandes pour ces filtres.")
        return
    included["Catégorie"] = included["product_category_name"].map(translate_category)
    included["Commandes"] = included["n"]
    included["Taux de retard"] = included["taux_retard"]
    included = included.sort_values("Taux de retard", ascending=False)

    tooltip = ["Catégorie", "Commandes", alt.Tooltip("Taux de retard", format=".1%")]
    col_worst, col_best = st.columns(2)
    with col_worst:
        st.caption("10 catégories les plus à risque")
        worst = included.head(10)
        _horizontal_bar(worst, "Catégorie", "Taux de retard", ".1%", COLOR_SEQUENTIAL, tooltip)
    with col_best:
        st.caption("10 catégories les plus fiables")
        best = included.tail(10).sort_values("Taux de retard", ascending=False)
        # .1% (pas .0%) : les catégories les plus fiables sont souvent sous 5%, où un
        # arrondi à l'entier écrase toute la variation ("0%, 0%, 1%, 1%" illisible).
        _horizontal_bar(best, "Catégorie", "Taux de retard", ".1%", COLOR_SEQUENTIAL, tooltip)

    if not excluded.empty:
        st.caption(f"{len(excluded)} catégorie(s) exclue(s) (< {MIN_VOLUME} commandes), non affichées.")


def render_satisfaction_link(cte, params) -> None:
    st.subheader("Impact du retard sur la satisfaction client")
    df = run_query(
        cte + "select is_late, avg(review_score) as note, count(review_score) as n_avis from base group by 1",
        params,
    )
    if df.empty or df["n_avis"].sum() == 0:
        st.info("Pas assez d'avis clients pour ces filtres.")
        return
    note_a_temps = df.loc[df["is_late"] == 0, "note"]
    note_retard = df.loc[df["is_late"] == 1, "note"]
    if note_a_temps.empty or note_retard.empty:
        st.info("Pas assez de données pour comparer (commandes en retard et à l'heure toutes les deux nécessaires).")
        return
    a_temps, retard = float(note_a_temps.iloc[0]), float(note_retard.iloc[0])
    delta = a_temps - retard
    st.markdown(
        f"**Une commande en retard obtient en moyenne {retard:.2f}/5, contre "
        f"{a_temps:.2f}/5 pour une commande livrée à l'heure — soit {delta:.2f} "
        f"point(s) de satisfaction en moins.**"
    )


# ---------------------------------------------------------------------------
# Onglet 3 — Prédiction
# ---------------------------------------------------------------------------

def render_prediction(cte, params) -> None:
    kpi = run_query(cte + """
        select
            sum(total_price) filter (where risk_tier != 'Sous seuil') as ca_a_risque,
            count(*) filter (where risk_tier = 'Alerte') as n_modere,
            count(*) filter (where risk_tier like 'Risque extr%') as n_eleve
        from base
    """, params)
    if kpi.empty:
        st.info("Aucune commande pour ces filtres.")
        return
    row = kpi.iloc[0]
    ca_risque = row.ca_a_risque or 0

    cols = st.columns(3)
    cols[0].metric("Chiffre d'affaires exposé au risque", f"{ca_risque:,.0f} R$")
    cols[1].metric("Commandes niveau Modéré", f"{int(row.n_modere or 0):,}")
    cols[2].metric("Commandes niveau Élevé", f"{int(row.n_eleve or 0):,}")

    st.subheader("Facteurs de risque les plus fréquents")
    drivers = run_query(cte + """
        select dr.feature_name, count(*) as n
        from base b
        join main.order_risk_drivers dr on dr.order_id = b.order_id
        where dr.driver_rank = 1
        group by 1 order by n desc
    """, params)
    if not drivers.empty:
        drivers["Facteur"] = drivers["feature_name"].map(translate_feature)
        drivers["Commandes"] = drivers["n"]
        drivers = drivers.sort_values("Commandes", ascending=False)
        _horizontal_bar(
            drivers, "Facteur", "Commandes", ",.0f", COLOR_SEQUENTIAL,
            ["Facteur", "Commandes"],
        )
    else:
        st.info("Aucun facteur de risque pour ces filtres.")

    st.subheader("Commandes prioritaires")
    st.caption("Les 50 commandes au niveau de risque le plus élevé pour les filtres actuels.")
    priority = run_query(cte + """
        select b.customer_state, b.product_category_name, b.total_price,
               b.risk_tier, b.risk_probability, dr.feature_name as top_driver
        from base b
        left join main.order_risk_drivers dr
            on dr.order_id = b.order_id and dr.driver_rank = 1
        where b.risk_tier != 'Sous seuil'
        order by b.risk_probability desc
        limit 50
    """, params)
    if priority.empty:
        st.info("Aucune commande à risque pour ces filtres.")
        return
    priority["État"] = priority["customer_state"].map(translate_state)
    priority["Catégorie"] = priority["product_category_name"].map(translate_category)
    priority["Niveau de risque"] = priority["risk_tier"].map(translate_tier)
    priority["Facteur principal"] = priority["top_driver"].map(translate_feature)
    display = priority.rename(columns={
        "total_price": "Montant (R$)",
        "risk_probability": "Score de risque",
    })[["État", "Catégorie", "Montant (R$)", "Niveau de risque", "Score de risque", "Facteur principal"]]
    st.dataframe(
        display.style.format({"Montant (R$)": "{:,.0f}", "Score de risque": "{:.0%}"}),
        hide_index=True, use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Méthodologie
# ---------------------------------------------------------------------------

def render_methodology() -> None:
    with st.expander("Méthodologie (pour aller plus loin)", expanded=False):
        st.markdown(
            f"""
- **Onglets "Vue d'ensemble" et "Analyse"** : calculés sur l'historique complet des
  commandes livrées (2 ans). Les régions et catégories avec moins de
  {MIN_VOLUME} commandes sont exclues des classements — sur un petit nombre de
  commandes, un taux de retard de 0% ou 100% ne reflète que la taille de
  l'échantillon, pas un vrai signal.
- **Onglet "Prédiction"** : le modèle est évalué sur une période qu'il n'a jamais
  vue pendant son apprentissage (une pratique standard pour mesurer une performance
  honnête). Les commandes utilisées pour entraîner le modèle ne sont pas affichées
  ici — leurs scores seraient artificiellement optimistes.
- Le niveau de risque "Élevé" correspond aux commandes dans le décile de risque le
  plus haut (calibré sur la période d'évaluation) ; "Modéré" correspond au seuil de
  décision retenu pour déclencher une alerte.
- Détail technique complet (choix du modèle, métriques, limites) :
  `docs/j6_modeling_report.md`, `docs/j7_explainability_report.md`,
  `docs/j8_prediction_integration_report.md`.
            """
        )


# ---------------------------------------------------------------------------

def main() -> None:
    states_all, categories_all, month_labels = get_filter_options()

    st.title("Olist — Pilotage du risque de livraison")
    st.caption(
        "Vue d'ensemble et analyse de l'activité de livraison, avec anticipation des "
        "commandes à risque."
    )

    with st.sidebar:
        st.header("Filtres")
        states_sel = st.multiselect(
            "État client", states_all, default=[], format_func=translate_state,
        )
        categories_sel = st.multiselect("Catégorie produit", categories_all, default=[])
        ym_range = st.select_slider(
            "Période", options=month_labels, value=(month_labels[0], month_labels[-1]),
        )

    ym_min = int(ym_range[0].replace("-", ""))
    ym_max = int(ym_range[1].replace("-", ""))

    tab_overview, tab_analysis, tab_prediction = st.tabs(
        ["Vue d'ensemble", "Analyse", "Prédiction"]
    )

    with tab_overview:
        bi_cte, bi_params = build_bi_cte(states_sel, categories_sel, ym_min, ym_max)
        render_overview(bi_cte, bi_params)

    with tab_analysis:
        bi_cte, bi_params = build_bi_cte(states_sel, categories_sel, ym_min, ym_max)
        render_region_analysis(bi_cte, bi_params)
        render_category_analysis(bi_cte, bi_params)
        render_satisfaction_link(bi_cte, bi_params)

    with tab_prediction:
        pred_cte, pred_params = build_pred_cte(states_sel, categories_sel, ym_min, ym_max)
        render_prediction(pred_cte, pred_params)

    render_methodology()


if __name__ == "__main__":
    main()
