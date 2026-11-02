"""Konstantprodukt-Markt und LVR - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 6 der Reihe „Marktmechanismen“: Ein Pool tauscht ein riskantes Gut gegen Zahlungsmittel nach der Regel x · y = k. Die Demo rechnet Preisauswirkung und Gebühr, zeigt am Beispiel des Konstantprodukts,
dass ein solcher Pool ein Kostenfunktions-Prognosemarkt ist (Brücke zu Stück 5), und misst den Verlust gegenüber dem Rebalancieren (LVR), den die Liquiditätsgeber tragen.
Modell, keine Marktdaten, keine Anlageberatung.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import kpm_algorithm as A
import kpm_constants as C
from kpm_evaluation import (
    BRIDGE_HEADER,
    DRIFT_HEADER,
    FEE_HEADER,
    HAND_HEADER,
    STEPS_HEADER,
    Settings,
    analyse,
    bridge_rows,
    bridge_table,
    de,
    drift_rows,
    fee_rows,
    hand_rows,
    impact_header,
    impact_rows,
    impact_table,
    pct,
    round_trip_header,
    round_trip_rows,
    round_trip_table,
    split_header,
    split_rows,
    split_table,
    steps_rows,
    verdict_bridge,
    verdict_fee,
    verdict_il,
    verdict_lvr,
    verdict_trade,
)
from kpm_layout import dunkles_design, kopf_css
from kpm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from kpm_visualization import build_bridge, build_drift, build_fees, build_impact, build_path

st.set_page_config(page_title="Konstantprodukt-Markt und LVR – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Rechne die Pfade (Schrittzahlen, Gebühren, Drift) …")
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _hand():
    return hand_rows()


@st.cache_data(show_spinner=False)
def _bridge():
    return bridge_table()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kpm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔄 Konstantprodukt-Markt und LVR")
st.info(C.DISCLAIMER)
st.markdown(
    """
Ein **Pool** hält ein riskantes Gut $x$ und Zahlungsmittel $y$ und tauscht jederzeit nach der Regel $x \\cdot y = k$: Der **Preis** ist $y/x$, und wer kauft, schiebt ihn hoch (**Preisauswirkung**).
Der Pool braucht dafür kein Orderbuch – aber er kennt den Preis **außerhalb** nicht. Wenn der sich bewegt, holen **Arbitrageure** die Differenz ab; das, was der Pool dabei gegenüber dem **Rebalancieren**
verliert, heißt **LVR** (Loss-versus-Rebalancing) und ist der Preis des Mechanismus für die, die ihn bereitstellen. Die Demo rechnet das nach, zeigt, dass ein Konstantprodukt-Pool am Beispiel ein
**Kostenfunktions-Prognosemarkt** ist (Brücke zur LMSR aus Stück 5), und misst, was **Gebühren** daran ändern.
Sechstes Stück der Reihe **Marktmechanismen**; alle Pfade sind synthetisch.

Weiter unten: der Abschnitt **🎯 Ergebnis**, drei **🔬 Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Pool", expanded=True):
    st.markdown(
        """
1. **Handelsregel.** Der Pool hält $(x, y)$ mit $x \\cdot y = k$. Wer $\\Delta x$ kauft, zahlt $\\Delta y = y\\,\\Delta x/(x - \\Delta x)$; danach gilt wieder $x' y' = k$. Mit **Gebühr** $f$ gehen nur $1 - f$ des Eingangs in die Invariante ein, die Gebühr bleibt im Pool.
2. **Preisauswirkung.** Kauft man den Anteil $s$ der Reserve, ist der mittlere Preis das $1/(1-s)$-fache des Anfangspreises und der Preis danach das $1/(1-s)^2$-fache.
3. **LVR.** Springt der Referenzpreis, bringen Arbitrageure den Pool auf den neuen Preis und behalten den Unterschied; der Pool-Wert wächst langsamer als der eines Bestands, der laufend zum Marktpreis nachgezogen wird. Die Rate ist $\\sigma^2/8$ des Poolwerts.
4. **Warum das gilt:** Der Poolwert bei Preis $P$ ist $2\\sqrt{kP}$, eine konkave Funktion von $P$; ihre Krümmung mal die Varianz des Preises ist genau, was verloren geht.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Der Pool (Ergebnis)**")
    s = st.slider("Gekaufter Anteil der Reserve", *bounds("s_slider"), key="s_slider", step=0.01, help="Welchen Anteil der Reserve x ein Händler kauft (der Pool startet mit 1 000 und 1 000, Preis 1).")
    fee = st.select_slider("Gebühr", options=C.FEE_CHOICES, key="fee_select", format_func=lambda v: pct(v, 2), help="Anteil jeder Einzahlung, der als Gebühr im Pool bleibt.")
    st.markdown("**Prognosemarkt (Experiment 1)**")
    p = st.slider("Zielpreis von A", *bounds("p_slider"), key="p_slider", step=0.01, help="Der Händler kauft Anteile A, bis der Preis von A diesen Wert erreicht (Start 0,5).")
    st.markdown("**Referenzpreis (Experimente 2 und 3)**")
    sigma = st.slider("Volatilität σ (je Jahr)", *bounds("sigma_slider"), key="sigma_slider", step=0.1, help="Geometrische Brownsche Bewegung des Referenzpreises.")
    mu = st.slider("Drift μ (je Jahr)", *bounds("mu_slider"), key="mu_slider", step=0.1, help="Erwartete Wachstumsrate des Referenzpreises.")
    steps = st.select_slider("Arbitrage-Schritte im Jahr", options=C.STEP_CHOICES, key="steps_select", help="Wie oft Arbitrageure den Pool auf den Referenzpreis bringen (12 = monatlich, 365 = täglich).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Bestimmt den einzelnen Pfad, der gezeichnet wird.")
    st.button("🎲 Neuer Pfad", width="stretch", on_click=randomize_seed)

sync_query_params({"s_slider": float(s), "fee_select": float(fee), "p_slider": float(p), "sigma_slider": float(sigma), "mu_slider": float(mu), "steps_select": int(steps), "seed_input": int(seed)})

settings = Settings(float(s), float(fee), float(p), float(sigma), float(mu), int(steps), int(seed))
an = _analysis(settings)
t = an.trade

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Der Händler zahlt", de(t.dy), delta=f"für {de(t.dx)} Einheiten", delta_color="off")
m2.metric("Mittlerer Preis", de(t.avg_price, 4), delta=f"{de(t.avg_ratio, 4)}-faches des Anfangspreises", delta_color="off")
m3.metric("Preis danach", de(t.price_after, 4), delta=f"{de(t.after_ratio, 4)}-faches des Anfangspreises", delta_color="off")
m4.metric("Gebühr im Pool", de(t.fee_paid), delta=f"k: {de(t.k_before, 0)} → {de(t.k_after, 0)}", delta_color="off")
st.plotly_chart(build_impact(settings.fee), width="stretch", key="main_chart")
st.caption("Preisauswirkung eines Kaufs: Wer den Anteil s der Reserve kauft, zahlt im Mittel das 1/(1−s)-fache und lässt den Preis auf das 1/(1−s)²-fache steigen (Pool 1 000 / 1 000; gestrichelt: der mittlere Preis mit der gewählten Gebühr).")
st.markdown("\n".join(impact_header(settings.fee) + impact_rows(impact_table(settings.fee))))
with st.expander("Mit Gebühr ist der Pfad nicht mehr gleichgültig – und Hin-und-zurück kostet"):
    st.markdown("**Dasselbe Ziel (10 % der Reserve kaufen) in n Stücken:**")
    st.markdown("\n".join(split_header(settings.fee) + split_rows(split_table(settings.fee))))
    st.markdown("**Kaufen und sofort alles zurückverkaufen:**")
    st.markdown("\n".join(round_trip_header(settings.fee) + round_trip_rows(round_trip_table(settings.fee))))
    st.caption("Ohne Gebühr bleibt das Produkt k erhalten: Aufteilen kostet nichts extra und der Rückweg ist gratis. Mit Gebühr wächst k bei jedem Handel, und jeder Teilhandel zahlt Gebühr auf sich selbst.")
with st.expander("Vier Handbeispiele zum Nachrechnen"):
    st.markdown("\n".join(HAND_HEADER + _hand()))
    st.caption("A: $\\Delta y = 1000 \\cdot 100/900$. B: Reserven $(a, b) = (100\\sqrt{0{,}2/0{,}8},\\ 100\\sqrt{0{,}8/0{,}2}) = (50;\\ 200)$, 100 Sätze geprägt, $100 + 100 - 50 = 150$ Anteile A. "
               "C: Poolwert $2\\sqrt{10^6 \\cdot 4} = 4000$ gegen Halten $4 \\cdot 1000 + 1000 = 5000$. D: $0{,}05^2/8 = 3{,}125 \\cdot 10^{-4}$.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wer zahlt für den Preis, den der Pool nicht kennt – und wie viel?")
kind_v, text_v = verdict_trade(t.avg_ratio, t.after_ratio, t.dy, settings.s, settings.fee, t.fee_paid)
getattr(st, kind_v)(text_v)

st.markdown("---")

# --- Experiment 1: Prognosemarkt -----------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Ein Konstantprodukt-Pool als Prognosemarkt")
st.markdown(
    "Zwei Ergebnis-Anteile $A$ und $B$ (zahlen 1 bei Eintritt); ein vollständiger Satz $A + B$ lässt sich für 1 prägen. Ein Pool mit Reserven $(a, b)$, $ab = k$, hat den Preis $p_A = b/(a+b)$. "
    "Wer $q_A$ Anteile $A$ will, zahlt $m$ Sätze, und das Produkt muss $k$ bleiben: $(r + m - q_A)(r + m) = r^2$. Das ist eine **Kostenfunktion** $C(q)$ – am Beispiel die Brücke zur LMSR aus Stück 5."
)
st.plotly_chart(build_bridge(), width="stretch", key="bridge_chart")
st.caption(f"Anteile A bis zum Preis (Start 0,5; Reserve r = {de(C.BRIDGE_R, 0)}): Am Start reagiert der Konstantprodukt-Markt wie die LMSR mit b = r (Preisänderung 1/(4r) je Anteil), an den Rändern ist er viel tiefer.")
st.markdown("\n".join(BRIDGE_HEADER + bridge_rows(_bridge())))
kind_b, text_b = verdict_bridge(an.bridge)
getattr(st, kind_b)(text_b)
st.caption("Der Konstantprodukt-Markt hat die Verlustgrenze r (nie erreicht), die LMSR mit gleicher Anfangsreaktion die Grenze r · ln 2. Gerechnet wird nur dieses Beispiel; der allgemeine Satz von Frongillo, Papireddygari und Waggoner (jeder CFMM mit konkavem Potential ist ein Kostenfunktionsmarkt) wird nicht nachgerechnet.")

st.markdown("---")

# --- Experiment 2: LVR -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie viel verliert der Pool gegenüber dem Rebalancieren?")
st.markdown(
    f"Der Referenzpreis folgt einer geometrischen Brownschen Bewegung (σ = {de(settings.sigma, 1)}, μ = {de(settings.mu, 1)}); nach jedem der {settings.steps} Schritte im Jahr bringen Arbitrageure den Pool auf den Referenzpreis. "
    "**LVR** ist die Summe ihrer Gewinne: was der Pool gegenüber einem Bestand verliert, der laufend zum Marktpreis nachgezogen wird."
)
st.plotly_chart(build_path(an.path), width="stretch", key="path_chart")
end = an.path
st.caption(f"Ein einzelner Pfad (Seed {settings.seed}): Der Abstand zwischen Rebalancieren und Pool ist der aufgelaufene LVR ({de(end['lvr'][-1], 0)} am Ende bei einem Anfangswert von {de(C.POOL_V0, 0)}); Halten ist der Bestand vom Start. "
           f"Die gezeichnete Linie rechnet mit der gewählten Gebühr ({pct(settings.fee, 2)}).")
row = next(r for r in an.steps_table if r["steps"] == settings.steps) if settings.steps in C.STEP_TABLE else an.steps_table[-1]
kind_l, text_l = verdict_lvr(row["exact"], row["sim"], row["se"], row["steps"], settings.sigma)
getattr(st, kind_l)(text_l)
st.markdown("\n".join(STEPS_HEADER + steps_rows(an.steps_table)))
st.caption(f"{C.LVR_PATHS} Pfade je Zeile (Seed {C.LVR_SEED}), ohne Gebühr. Die exakte Formel gilt für jede Zahl von Schritten: E[LVR] = V₀ Σ gⁱ (a − g) mit g = E[√(P'/P)] und a = (E[P'/P] + 1)/2; "
           "bei Drift 0 ist das 1 − exp(−σ²T/8), unabhängig von der Zahl der Schritte.")
st.markdown("**Impermanent Loss gegen LVR.** Der Impermanent Loss vergleicht den Pool mit dem **Halten** des Anfangsbestands, der LVR mit dem **Rebalancieren**. Beide sind nur ohne Drift gleich groß.")
st.plotly_chart(build_drift(an.drift_table), width="stretch", key="drift_chart")
st.markdown("\n".join(DRIFT_HEADER + drift_rows(an.drift_table)))
kind_i, text_i = verdict_il(an.drift_table)
getattr(st, kind_i)(text_i)
st.caption(f"Drift μ = Vielfaches von σ² (−½, 0, ½, 1), {C.DRIFT_STEPS} Schritte, {C.DRIFT_PATHS} Pfade (Seed {C.DRIFT_SEED}), ohne Gebühr; „exakt“ ist die Formel für diskrete Schritte beziehungsweise E[V_T] − E[H_T]. Die Standardfehler gelten für die Simulation.")

st.markdown("---")

# --- Experiment 3: Gebühren ----------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Was ändern Gebühren?")
st.markdown(
    "Mit Gebühr handeln Arbitrageure nur, wenn die Abweichung größer ist als das Band $[ (1-f)\\,p_{\\text{Pool}},\\ p_{\\text{Pool}}/(1-f) ]$. Gemessen werden dieselben Pfade für jede Gebühr: "
    "der **Arbitrage-Gewinn** (was der Pool gegenüber dem Rebalancieren verliert, einschließlich der Gebühren, die im Pool bleiben), die **Gebühreneinnahmen** und das **Ausgleichsvolumen**."
)
st.plotly_chart(build_fees(an.fee_table), width="stretch", key="fee_chart")
st.markdown("\n".join(FEE_HEADER + fee_rows(an.fee_table)))
kind_f, text_f = verdict_fee(an.fee_table, settings.fee)
getattr(st, kind_f)(text_f)
st.caption(f"{C.LVR_PATHS} Pfade (Seed {C.LVR_SEED}), {settings.steps} Schritte im Jahr, σ = {de(settings.sigma, 1)}, μ = {de(settings.mu, 1)}. Ausgleichsvolumen = (Arbitrage-Gewinn / V₀) / Gebühr: das Handelsvolumen von Nicht-Informierten "
           "(als Vielfaches des Anfangswerts im Jahr), dessen Gebühren den Arbitrage-Verlust gerade ausgleichen würden; eine Rechengröße, keine Simulation von Noise-Tradern.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Geometrische Brownsche Bewegung** | Sprünge, schwankende Volatilität und echte Kurse sind nicht gemessen; die Formeln gelten für dieses Modell. | – |
| **Arbitrage zu festen Zeitpunkten, ohne Konkurrenz um die Reihenfolge** | Echte Plattformen haben Blockzeiten, Reihenfolgekonkurrenz und Gas; die Schrittzahl ist nur ein Stellvertreter. | – |
| **Zwei Güter, volle Preisspanne** | Konzentrierte Liquidität (Uniswap v3) und mehr als zwei Güter werden nicht gerechnet. | – |
| **Reine Arbitrage als Fluss** | Nicht-Informierte (Noise-Trader) werden nicht simuliert; das Ausgleichsvolumen ist eine Rechengröße. | – |
| **Brücke nur am Beispiel** | Der allgemeine Satz (jeder CFMM mit konkavem Potential ist ein Kostenfunktionsmarkt) wird nicht nachgerechnet. | – |
| **Modell, keine Marktdaten, keine Anlageberatung** | Die Zahlen sagen nichts über echte Pools oder Erträge; sie sind keine Empfehlung. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Pool.** Reserven $(x, y)$, $xy = k$, Preis $P = y/x$. Kauf von $\Delta x$: $\Delta y = y\,\Delta x/(x - \Delta x)$. Anteil $s = \Delta x/x$: mittlerer Preis $P/(1-s)$, Preis danach $P/(1-s)^2$.
Mit Gebühr $f$, $\gamma = 1 - f$: $\Delta y = \frac{y}{\gamma}\frac{\Delta x}{x - \Delta x}$, die Gebühr bleibt in $y$, $k$ wächst.

**Brücke.** Zwei Ergebnis-Anteile, ein Satz kostet 1. Reserven $(a, b)$ mit $ab = k = r^2$, $p_A = b/(a+b)$. Der Kauf $q = (q_A, q_B)$ prägt $m$ Sätze: $(r + m - q_A)(r + m - q_B) = r^2$,
also $m = C(q) = \tfrac12\bigl(-S + \sqrt{S^2 - 4c}\bigr)$ mit $S = 2r - q_A - q_B$, $c = (r - q_A)(r - q_B) - r^2$. Die Preise sind $\nabla C$ und summieren sich zu 1. Bis zum Preis $p$: $a = r\sqrt{(1-p)/p}$, $b = r\sqrt{p/(1-p)}$,
$m = b - r$, $q_A = r + m - a$; der Verlust bei Eintritt von $A$ ist $r\,(1 - \sqrt{(1-p)/p}) < r$. Die LMSR hat $b\ln(2p) < b\ln 2$.

**LVR.** Poolwert bei Preis $P$: $V(P) = 2\sqrt{kP}$, Reserve $x^*(P) = \sqrt{k/P}$. Der Arbitrage-Gewinn eines Schritts ist $x_i P_{i+1} + y_i - V(P_{i+1}) \ge 0$; die Summe ist der LVR.
Wegen $x_i P_i = V_i/2$ und $E[V_{i+1}] = g\,E[V_i]$ mit $g = E[\sqrt{P_{i+1}/P_i}] = e^{(\mu/2 - \sigma^2/8)\,dt}$ gilt $E[\mathrm{LVR}] = V_0 \sum_{i<n} g^i\,(a - g)$, $a = (e^{\mu\,dt} + 1)/2$.
Bei $\mu = 0$ ist das $V_0\,(1 - e^{-\sigma^2 T/8})$; die Rate ist $\sigma^2/8$ des Poolwerts.

**Impermanent Loss.** Pool gegen Halten bei Preisverhältnis $r$: $2\sqrt{r}/(1+r) - 1$. Erwartet: $E[V_T - H_T] = V_0\,e^{(\mu/2 - \sigma^2/8)T} - \tfrac{V_0}{2}(e^{\mu T} + 1)$.

**Gebühren.** Kauf von $x$ bis der Grenzpreis inklusive Gebühr $P$ ist: $(y + \Delta y)(y + \gamma\Delta y) = \gamma P\,x y$; Verkauf: $P\gamma\,\Delta x^2 + P x(1+\gamma)\,\Delta x + P x^2 - \gamma x y = 0$.

**Literatur** (nur Titel, Abstract und Sekundärquellen geprüft, keine Volltexte). Milionis, Moallemi, Roughgarden, Zhang: *Automated market making and loss-versus-rebalancing.* arXiv 2208.06046.
Frongillo, Papireddygari, Waggoner: *An axiomatic characterization of CFMMs and equivalence to prediction markets.* ITCS 2024, LIPIcs 287, 51:1–51:21 (arXiv 2302.00196).
Die Formel für diskrete Schritte und die Brücke über vollständige Sätze sind eigene Rechnung, durch Simulation und Aufzählung geprüft.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
