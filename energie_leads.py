"""
Energie Leads Dashboard
Streamlit app om snel potentiële klanten te bereiken voor thuisbatterijen,
zonnepanelen en laadpalen.
"""

import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta

# ---------------------------------------------------------------------------
# Constanten / data
# ---------------------------------------------------------------------------

PRODUCTEN = ["Thuisbatterij", "Zonnepanelen", "Laadpaal"]

KANALEN = {
    "Google Ads (zoekadvertenties)": {
        "snelheid": 9,
        "kosten": 7,
        "conversieratio": 8,
        "bereik": 9,
        "toelichting": (
            "Potentiële klanten zoeken actief naar 'zonnepanelen installateur' of "
            "'thuisbatterij plaatsen'. Met goed ingestelde zoekadvertenties heb je "
            "binnen uren aanvragen. Kosten per klik zijn hoog, maar de intentie van "
            "de bezoeker is sterk."
        ),
        "tips": [
            "Target zoektermen als 'zonnepanelen installeren [stad]' en 'laadpaal thuis'",
            "Gebruik responsive search ads met lokale extensies",
            "Stel callextensions in zodat mensen direct bellen",
            "Begin met een dagbudget van €30-50 om data te verzamelen",
        ],
    },
    "Facebook / Instagram Ads": {
        "snelheid": 8,
        "kosten": 6,
        "conversieratio": 6,
        "bereik": 10,
        "toelichting": (
            "Grote visuele impact via foto's en video's van installaties. Met "
            "lookalike audiences op bestaande klanten kun je snel nieuwe leads "
            "genereren. Lead-ads sturen mensen niet weg van het platform."
        ),
        "tips": [
            "Gebruik 'Lead Ads' zodat mensen hun gegevens zonder website kunnen achterlaten",
            "Maak video van een echte installatie (vóór/na) voor hoog engagement",
            "Target eigenhuisbezitters van 35-65 jaar in jouw servicegebied",
            "Retarget websitebezoekers met een speciale aanbieding",
        ],
    },
    "Deur-aan-deur canvassing": {
        "snelheid": 8,
        "kosten": 4,
        "conversieratio": 7,
        "bereik": 5,
        "toelichting": (
            "Persoonlijk contact levert de hoogste conversie op. Focus op wijken "
            "waar al zonnepanelen zichtbaar zijn – buren zijn dan al warm gemaakt "
            "voor het idee. Snel te starten, geen technische kennis vereist."
        ),
        "tips": [
            "Begin in straten waar je al installaties hebt gedaan (social proof)",
            "Gebruik een tablet om direct een quickscan/offerte te tonen",
            "Noteer adressen van huizen met al zonnepanelen als leadlijst voor laadpalen/batterijen",
            "Vraag altijd om een referral-naam bij elk gesprek",
        ],
    },
    "Lokale SEO & Google Bedrijfsprofiel": {
        "snelheid": 4,
        "kosten": 9,
        "conversieratio": 8,
        "bereik": 7,
        "toelichting": (
            "Op de lange termijn de meest rendabele bron. Zorg dat je goed scoort op "
            "'[stad] zonnepanelen installateur'. Google Bedrijfsprofiel levert "
            "direct telefonische aanvragen op – gratis."
        ),
        "tips": [
            "Vraag elke klant om een Google-recensie direct na installatie",
            "Upload foto's van elke installatie met locatietag",
            "Zorg voor NAP-consistentie (naam, adres, telefoon) op alle directories",
            "Schrijf lokale blogposts: 'Kosten zonnepanelen in [stad] 2025'",
        ],
    },
    "Referral / mond-tot-mondreclame": {
        "snelheid": 5,
        "kosten": 10,
        "conversieratio": 10,
        "bereik": 6,
        "toelichting": (
            "De goedkoopste leads zijn van tevreden klanten. Bied een aantrekkelijke "
            "bonus (€50-200 korting, cadeaukaart) voor elke doorverwijzing die leidt "
            "tot een installatie. Converteer hoog omdat er vertrouwen is."
        ),
        "tips": [
            "Stuur 2 weken na installatie een e-mail met referral-aanbod",
            "Maak een WhatsApp-status update makkelijk te delen voor klanten",
            "Geef klant een unieke referralcode voor tracking",
            "Overweeg een 'buddy-deal': korting voor zowel referrer als nieuwe klant",
        ],
    },
    "Energieleverancier-partnerschap": {
        "snelheid": 3,
        "kosten": 9,
        "conversieratio": 7,
        "bereik": 10,
        "toelichting": (
            "Samenwerken met energieleveranciers (Eneco, Vattenfall, Essent) of "
            "subsidieprogramma's (ISDE, Salderingsregeling) levert warme leads op "
            "uit hun klantenbestand. Vergt meer tijd om op te zetten maar schaalbaar."
        ),
        "tips": [
            "Neem contact op met lokale energiecoöperaties",
            "Registreer bij ISDE-subsidie als erkend installateur",
            "Partner met woningcorporaties voor collectieve inkoop",
            "Werk samen met VvE-beheerders voor appartementencomplexen",
        ],
    },
    "Flyers & Direct Mail": {
        "snelheid": 6,
        "kosten": 6,
        "conversieratio": 4,
        "bereik": 7,
        "toelichting": (
            "Fysiek materiaal in de brievenbus heeft minder concurrentie dan digitaal. "
            "Gerichte verspreiding in postcodegebieden met hoge woningbezit en "
            "weinig zonnepanelen werkt het best."
        ),
        "tips": [
            "Target postcodes met veel koopwoningen via PostNL Reclamepost",
            "Vermeld altijd een QR-code naar een landingspagina met leadformulier",
            "Gebruik een aanbod met tijdsdruk ('Gratis quickscan t/m [datum]')",
            "A/B test twee versies van de flyer in verschillende wijken",
        ],
    },
    "Online marktplaatsen (Werkspot, Homeadvice)": {
        "snelheid": 7,
        "kosten": 6,
        "conversieratio": 7,
        "bereik": 7,
        "toelichting": (
            "Platforms zoals Werkspot, Homeadvice en Solar-quotes brengen al "
            "geïnteresseerde huiseigenaren direct in contact met installateurs. "
            "Reageer snel op aanvragen – binnen 5 minuten verhoogt de kans op "
            "opdracht sterk."
        ),
        "tips": [
            "Stel e-mail- en sms-meldingen in voor nieuwe aanvragen",
            "Reageer altijd binnen 5 minuten met een persoonlijk bericht",
            "Bouw je profiel uit met foto's, certificaten en recensies",
            "Vraag klanten die via het platform binnenkwamen om een review",
        ],
    },
}

LEAD_STATUSSEN = ["Nieuw", "Gecontacteerd", "Offerte verstuurd", "Gewonnen", "Verloren"]

STATUS_KLEUREN = {
    "Nieuw": "#3498db",
    "Gecontacteerd": "#f39c12",
    "Offerte verstuurd": "#9b59b6",
    "Gewonnen": "#2ecc71",
    "Verloren": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

LEADS_KEY = "energie_leads"


def _init_leads():
    if LEADS_KEY not in st.session_state:
        st.session_state[LEADS_KEY] = pd.DataFrame(
            columns=[
                "Datum",
                "Naam",
                "Telefoon",
                "E-mail",
                "Postcode",
                "Product",
                "Kanaal",
                "Status",
                "Notities",
            ]
        )


def _add_lead(naam, telefoon, email, postcode, product, kanaal, notities=""):
    _init_leads()
    new_row = {
        "Datum": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Naam": naam,
        "Telefoon": telefoon,
        "E-mail": email,
        "Postcode": postcode,
        "Product": product,
        "Kanaal": kanaal,
        "Status": "Nieuw",
        "Notities": notities,
    }
    st.session_state[LEADS_KEY] = pd.concat(
        [st.session_state[LEADS_KEY], pd.DataFrame([new_row])],
        ignore_index=True,
    )


def _get_leads() -> pd.DataFrame:
    _init_leads()
    return st.session_state[LEADS_KEY]


# ---------------------------------------------------------------------------
# Pagina's
# ---------------------------------------------------------------------------


def pagina_kanalen_overzicht():
    st.header("📡 Marketingkanalen – Snelste weg naar nieuwe klanten")
    st.markdown(
        """
Hieronder staan de meest effectieve kanalen om snel potentiële klanten te bereiken
voor **thuisbatterijen**, **zonnepanelen** en **laadpalen**. De kanalen zijn gesorteerd
op snelheid van eerste leads.
"""
    )

    # Radar chart
    kanaal_namen = list(KANALEN.keys())
    df_radar = pd.DataFrame(
        {
            "Kanaal": kanaal_namen,
            "Snelheid": [KANALEN[k]["snelheid"] for k in kanaal_namen],
            "Lage kosten": [KANALEN[k]["kosten"] for k in kanaal_namen],
            "Conversieratio": [KANALEN[k]["conversieratio"] for k in kanaal_namen],
            "Bereik": [KANALEN[k]["bereik"] for k in kanaal_namen],
        }
    )
    df_radar["Totaalscore"] = (
        df_radar["Snelheid"] * 0.4
        + df_radar["Lage kosten"] * 0.2
        + df_radar["Conversieratio"] * 0.25
        + df_radar["Bereik"] * 0.15
    )
    df_radar = df_radar.sort_values("Snelheid", ascending=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig_bar = go.Figure(
            go.Bar(
                x=df_radar["Snelheid"],
                y=df_radar["Kanaal"],
                orientation="h",
                marker_color=[
                    f"rgba(52,152,219,{0.5 + v * 0.05})" for v in df_radar["Snelheid"]
                ],
                text=df_radar["Snelheid"],
                textposition="inside",
            )
        )
        fig_bar.update_layout(
            title="Snelheid van eerste leads (score 1-10)",
            xaxis=dict(range=[0, 10]),
            height=380,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("#### 🏆 Top 3 snelste kanalen")
        for i, row in df_radar.head(3).iterrows():
            medal = ["🥇", "🥈", "🥉"][list(df_radar.head(3).index).index(i)]
            st.markdown(f"{medal} **{row['Kanaal']}**  \nSnelheid: {row['Snelheid']}/10")
            st.markdown("---")

    st.subheader("Vergelijking per dimensie")
    dim_cols = st.columns(4)
    dim_labels = ["Snelheid", "Lage kosten", "Conversieratio", "Bereik"]
    for col, dim in zip(dim_cols, dim_labels):
        best = df_radar.sort_values(dim, ascending=False).iloc[0]
        col.metric(f"Beste: {dim}", best["Kanaal"][:25], f"{best[dim]}/10")

    st.subheader("📋 Kanaaldetails & praktische tips")
    for kanaal_naam, info in sorted(
        KANALEN.items(), key=lambda x: x[1]["snelheid"], reverse=True
    ):
        with st.expander(
            f"**{kanaal_naam}** — Snelheid {info['snelheid']}/10 · Conversie {info['conversieratio']}/10",
            expanded=False,
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Snelheid", f"{info['snelheid']}/10")
            c2.metric("Lage kosten", f"{info['kosten']}/10")
            c3.metric("Conversieratio", f"{info['conversieratio']}/10")
            c4.metric("Bereik", f"{info['bereik']}/10")
            st.markdown(f"**Toelichting:** {info['toelichting']}")
            st.markdown("**Praktische tips:**")
            for tip in info["tips"]:
                st.markdown(f"- {tip}")


def pagina_roi_calculator():
    st.header("💶 ROI-calculator per marketingkanaal")
    st.markdown(
        "Bereken de verwachte return on investment van elk kanaal op basis van uw eigen cijfers."
    )

    col1, col2 = st.columns(2)
    with col1:
        kanaal_keuze = st.selectbox("Kies kanaal", list(KANALEN.keys()))
        maandbudget = st.number_input(
            "Maandbudget (€)", min_value=0, max_value=50000, value=500, step=50
        )
    with col2:
        gem_orderwaarde = st.number_input(
            "Gemiddelde orderwaarde (€)", min_value=0, max_value=50000, value=8000, step=500
        )
        gem_marge = st.slider("Brutomarge (%)", 5, 60, 25)

    info = KANALEN[kanaal_keuze]
    conv_ratio_pct = info["conversieratio"] / 10 * 15  # max ~15% conversie
    kosten_per_lead_factor = (11 - info["kosten"]) * 10  # hogere kosten = hogere CPL
    geschatte_cpl = max(20, kosten_per_lead_factor)

    leads_per_maand = max(1, int(maandbudget / geschatte_cpl))
    gewonnen_opdrachten = max(0, int(leads_per_maand * conv_ratio_pct / 100))
    omzet = gewonnen_opdrachten * gem_orderwaarde
    winst = omzet * gem_marge / 100
    roi = ((winst - maandbudget) / max(1, maandbudget)) * 100

    st.markdown("---")
    st.subheader("Prognose")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Leads/maand", f"~{leads_per_maand}")
    m2.metric("Opdrachten/maand", f"~{gewonnen_opdrachten}")
    m3.metric("Geschatte omzet", f"€{omzet:,.0f}")
    m4.metric("Geschatte winst", f"€{winst:,.0f}")
    m5.metric("ROI", f"{roi:.0f}%", delta=f"{'positief' if roi > 0 else 'negatief'}")

    st.info(
        "ℹ️ Dit zijn schattingen op basis van branchegemiddelden. "
        "Werkelijke resultaten kunnen afwijken afhankelijk van targeting, kwaliteit van "
        "advertenties en marktomstandigheden."
    )

    # Vergelijkingstabel alle kanalen
    st.subheader("Vergelijking alle kanalen bij huidig budget")
    rows = []
    for k, v in KANALEN.items():
        cpl = max(20, (11 - v["kosten"]) * 10)
        lpm = max(1, int(maandbudget / cpl))
        conv = v["conversieratio"] / 10 * 15
        opdrachten = max(0, int(lpm * conv / 100))
        omzet_k = opdrachten * gem_orderwaarde
        winst_k = omzet_k * gem_marge / 100
        roi_k = ((winst_k - maandbudget) / max(1, maandbudget)) * 100
        rows.append(
            {
                "Kanaal": k,
                "Leads/mnd": lpm,
                "Opdrachten/mnd": opdrachten,
                "Omzet (€)": f"{omzet_k:,.0f}",
                "Winst (€)": f"{winst_k:,.0f}",
                "ROI (%)": f"{roi_k:.0f}%",
            }
        )
    df_comparison = pd.DataFrame(rows).sort_values("Leads/mnd", ascending=False).reset_index(drop=True)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)


def pagina_leads_beheer():
    st.header("📋 Lead Management")
    st.markdown("Registreer en beheer al uw leads voor thuisbatterijen, zonnepanelen en laadpalen.")

    tabs = st.tabs(["➕ Nieuwe lead", "📊 Lead overzicht", "✏️ Status bijwerken"])

    # --- Tab: Nieuwe lead ---
    with tabs[0]:
        st.subheader("Nieuwe lead invoeren")
        with st.form("nieuw_lead_formulier", clear_on_submit=True):
            c1, c2 = st.columns(2)
            naam = c1.text_input("Naam *", placeholder="Jan de Vries")
            telefoon = c2.text_input("Telefoon *", placeholder="+31 6 12345678")
            email = c1.text_input("E-mailadres", placeholder="jan@example.com")
            postcode = c2.text_input("Postcode", placeholder="1234 AB")
            product = c1.selectbox("Geïnteresseerd in", PRODUCTEN)
            kanaal = c2.selectbox("Bron / herkomst lead", list(KANALEN.keys()))
            notities = st.text_area("Notities", placeholder="Bijv. heeft al 10 zonnepanelen, wil opslagbatterij")
            ingediend = st.form_submit_button("Lead opslaan", type="primary")

            if ingediend:
                if not naam.strip() or not telefoon.strip():
                    st.error("Naam en telefoon zijn verplicht.")
                else:
                    _add_lead(naam, telefoon, email, postcode, product, kanaal, notities)
                    st.success(f"✅ Lead '{naam}' opgeslagen!")

    # --- Tab: Lead overzicht ---
    with tabs[1]:
        df = _get_leads()
        if df.empty:
            st.info("Nog geen leads geregistreerd. Voeg je eerste lead toe via het tabblad 'Nieuwe lead'.")
        else:
            # KPI row
            totaal = len(df)
            gewonnen = len(df[df["Status"] == "Gewonnen"])
            conversie = gewonnen / totaal * 100 if totaal else 0
            per_product = df.groupby("Product").size()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Totaal leads", totaal)
            k2.metric("Gewonnen", gewonnen)
            k3.metric("Conversieratio", f"{conversie:.1f}%")
            k4.metric("Meest gevraagd", per_product.idxmax() if not per_product.empty else "—")

            # Status donut
            status_counts = df["Status"].value_counts()
            fig_donut = go.Figure(
                go.Pie(
                    labels=status_counts.index,
                    values=status_counts.values,
                    hole=0.5,
                    marker=dict(
                        colors=[STATUS_KLEUREN.get(s, "#95a5a6") for s in status_counts.index]
                    ),
                )
            )
            fig_donut.update_layout(title="Leads per status", height=300, margin=dict(t=40, b=10))
            c_d, c_k = st.columns([1, 1])
            c_d.plotly_chart(fig_donut, use_container_width=True)

            # Kanaal verdeling
            kanaal_counts = df["Kanaal"].value_counts().head(5)
            fig_kanaal = go.Figure(
                go.Bar(
                    x=kanaal_counts.values,
                    y=kanaal_counts.index,
                    orientation="h",
                    marker_color="#3498db",
                )
            )
            fig_kanaal.update_layout(
                title="Top 5 herkomstkanalen",
                height=300,
                margin=dict(t=40, b=10),
            )
            c_k.plotly_chart(fig_kanaal, use_container_width=True)

            st.subheader("Alle leads")
            st.dataframe(df, use_container_width=True, hide_index=True)

    # --- Tab: Status bijwerken ---
    with tabs[2]:
        df = _get_leads()
        if df.empty:
            st.info("Nog geen leads beschikbaar.")
        else:
            lead_opties = [
                f"{row['Datum']} — {row['Naam']} ({row['Product']})"
                for _, row in df.iterrows()
            ]
            geselecteerde_lead = st.selectbox("Selecteer lead", lead_opties)
            idx = lead_opties.index(geselecteerde_lead)
            huidige_status = df.iloc[idx]["Status"]
            nieuwe_status = st.selectbox(
                "Nieuwe status",
                LEAD_STATUSSEN,
                index=LEAD_STATUSSEN.index(huidige_status),
            )
            extra_notitie = st.text_area("Notitie toevoegen", placeholder="Bijv. offerte verstuurd op 15 mei")

            if st.button("Status bijwerken", type="primary"):
                st.session_state[LEADS_KEY].at[idx, "Status"] = nieuwe_status
                if extra_notitie:
                    huidig = st.session_state[LEADS_KEY].at[idx, "Notities"]
                    st.session_state[LEADS_KEY].at[idx, "Notities"] = (
                        f"{huidig}\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {extra_notitie}".strip()
                    )
                st.success(f"✅ Status bijgewerkt naar **{nieuwe_status}**")
                st.rerun()


def pagina_actieplan():
    st.header("🚀 Actieplan: Snelste start naar meer klanten")
    st.markdown(
        """
Wil je **zo snel mogelijk** leads genereren voor thuisbatterijen, zonnepanelen en laadpalen?
Volg dan dit bewezen stappenplan.
"""
    )

    stappen = [
        {
            "titel": "Week 1 – Directe online zichtbaarheid",
            "kleur": "#2ecc71",
            "acties": [
                "📌 **Google Bedrijfsprofiel** optimaliseren: voeg producten, foto's en openingstijden toe",
                "📌 Maak een **Google Ads campagne** aan met zoektermen als 'zonnepanelen installeren [jouw stad]'",
                "📌 Registreer op **Werkspot** en **Homeadvice** – vul profiel volledig in",
                "📌 Stuur een **WhatsApp/SMS-bericht** naar alle bestaande klanten met een referral-aanbieding",
            ],
        },
        {
            "titel": "Week 2 – Offline bereik vergroten",
            "kleur": "#3498db",
            "acties": [
                "📌 **Deur-aan-deur** in wijken waar al installaties zijn gedaan",
                "📌 Verspreid **flyers** in straten met veel koopwoningen (PostNL Reclamepost)",
                "📌 Neem contact op met **lokale energiecoöperaties** voor samenwerking",
                "📌 Bezoek een **lokale beurs of braderie** met een standplaats of flyers",
            ],
        },
        {
            "titel": "Week 3-4 – Social media & content",
            "kleur": "#9b59b6",
            "acties": [
                "📌 Start een **Facebook/Instagram Lead Ad** campagne gericht op huiseigenaren 35-65 jaar",
                "📌 Post wekelijks een **voor/na foto** van een recente installatie",
                "📌 Maak een korte **video** (60s) van een installatie voor Instagram Reels",
                "📌 Vraag elke tevreden klant om een **Google recensie** (stuur een directe link)",
            ],
        },
        {
            "titel": "Doorlopend – Systemen en opvolging",
            "kleur": "#f39c12",
            "acties": [
                "📌 Gebruik dit **lead dashboard** om alle contacten bij te houden",
                "📌 Bel elke nieuwe lead **binnen 2 uur** terug (opvolgsnelheid = conversie)",
                "📌 Stuur **3 dagen na eerste contact** een herinnerings-sms als er geen reactie is",
                "📌 Vraag bij elke opdracht om **2 referrals** (namen van buren of vrienden)",
                "📌 Meet maandelijks welk kanaal de meeste opdrachten oplevert via dit dashboard",
            ],
        },
    ]

    for stap in stappen:
        st.markdown(
            f"""
<div style="border-left: 4px solid {stap['kleur']}; padding: 10px 20px; margin-bottom: 16px;
     background-color: #f8f9fa; border-radius: 4px;">
<h4 style="color: {stap['kleur']}; margin-top: 0;">{stap['titel']}</h4>
{''.join(f"<p style='margin: 4px 0;'>{a}</p>" for a in stap['acties'])}
</div>
""",
            unsafe_allow_html=True,
        )

    st.subheader("📞 Snelste kanalen – samenvatting")
    st.markdown(
        """
| Kanaal | Eerste lead verwacht | Kosten om te starten |
|--------|---------------------|----------------------|
| Google Ads | Dezelfde dag | €30-50/dag budget |
| Werkspot / Homeadvice | Zelfde dag (profiel) | €0 profiel, betaal per lead |
| Deur-aan-deur | Zelfde dag | Alleen tijd |
| Facebook Lead Ads | 1-2 dagen | €10-20/dag budget |
| Flyers | 3-5 dagen na verspreiding | €100-300 voor druk + bezorging |
| Referral programma | 1-2 weken | €50-200 per geslaagde doorverwijzing |
| Lokale SEO | 3-6 maanden | Tijd / content |
"""
    )

    st.info(
        "💡 **Tip:** Combineer altijd minimaal 2 kanalen – een met direct effect (Google Ads / deur-aan-deur) "
        "en een met langetermijn effect (SEO / referrals). Zo bouw je een stabiele leadstroom op."
    )


# ---------------------------------------------------------------------------
# Hoofdfunctie
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Energie Leads Dashboard",
        page_icon="⚡",
        layout="wide",
    )

    st.title("⚡ Energie Leads Dashboard")
    st.markdown(
        "**Snelste manieren om potentiële klanten te bereiken voor "
        "thuisbatterijen, zonnepanelen en laadpalen.**"
    )

    pagina = st.sidebar.radio(
        "Navigatie",
        [
            "📡 Marketingkanalen",
            "💶 ROI-calculator",
            "📋 Lead Management",
            "🚀 Actieplan",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
**Over dit dashboard**

Gebruik dit dashboard om:
- De beste kanalen te vinden
- Leads bij te houden
- Je ROI te berekenen
- Een concreet actieplan te volgen
"""
    )

    if pagina == "📡 Marketingkanalen":
        pagina_kanalen_overzicht()
    elif pagina == "💶 ROI-calculator":
        pagina_roi_calculator()
    elif pagina == "📋 Lead Management":
        pagina_leads_beheer()
    elif pagina == "🚀 Actieplan":
        pagina_actieplan()


if __name__ == "__main__":
    main()
