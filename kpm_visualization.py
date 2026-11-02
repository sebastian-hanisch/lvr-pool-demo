"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import kpm_algorithm as A
import kpm_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=60), legend=dict(orientation="h", y=-0.4), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_impact(f: float) -> go.Figure:
    """Mittlerer Preis und Preis danach im Verhältnis zum Anfangspreis über den Anteil s der gekauften Reserve (ohne Gebühr; mit Gebühr der mittlere Preis gestrichelt)."""
    xs = [i / 100 for i in range(1, 91)]
    t0 = [A.buy(C.POOL_X, C.POOL_Y, s, 0.0) for s in xs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[100 * s for s in xs], y=[t.avg_ratio for t in t0], mode="lines", name="mittlerer Preis (ohne Gebühr)", line=dict(color=C.COLOR_PRICE, width=2)))
    fig.add_trace(go.Scatter(x=[100 * s for s in xs], y=[t.after_ratio for t in t0], mode="lines", name="Preis danach", line=dict(color=C.COLOR_LOSS, width=2)))
    if f > 0:
        fig.add_trace(go.Scatter(x=[100 * s for s in xs], y=[A.buy(C.POOL_X, C.POOL_Y, s, f).avg_ratio for s in xs], mode="lines", name=f"mittlerer Preis (Gebühr {100 * f:.2f} %)".replace(".", ","),
                                 line=dict(color=C.COLOR_BOUND, width=2, dash="dash")))
    fig.update_xaxes(title="gekaufter Anteil der Reserve x in %")
    fig.update_yaxes(title="Verhältnis zum Anfangspreis", range=[0.9, 12])
    return _base(fig, 340)


def build_bridge(r: float = C.BRIDGE_R) -> go.Figure:
    """Anteile A bis zum Preis p: Konstantprodukt-Markt gegen LMSR mit gleicher Anfangsreaktion (b = r)."""
    ps = [0.5 + i * 0.0049 for i in range(0, 100)]
    ps = [p for p in ps if p < 0.995]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ps, y=[A.cp_buy_to_price(p, r)[0] for p in ps], mode="lines", name="Konstantprodukt", line=dict(color=C.COLOR_CP, width=2)))
    fig.add_trace(go.Scatter(x=ps, y=[A.lmsr_shares_to_price(p, r) for p in ps], mode="lines", name="LMSR (b = r)", line=dict(color=C.COLOR_LMSR, width=2, dash="dash")))
    fig.update_xaxes(title="Preis von A", range=[0.5, 1.0])
    fig.update_yaxes(title="Anteile A bis zu diesem Preis")
    return _base(fig, 340)


def build_path(path: dict) -> go.Figure:
    """Ein Pfad: Poolwert, Halten und Rebalancieren (alle in Einheiten des Zahlungsmittels); der Abstand Rebalancieren minus Pool ist der aufgelaufene LVR."""
    n = len(path["P"])
    x = [i / (n - 1) for i in range(n)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=path["rebalance"], mode="lines", name="Rebalancieren", line=dict(color=C.COLOR_GAIN, width=2)))
    fig.add_trace(go.Scatter(x=x, y=path["hold"], mode="lines", name="Halten", line=dict(color=C.COLOR_BEFORE, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=path["pool"], mode="lines", name="Pool", line=dict(color=C.COLOR_PRICE, width=2)))
    fig.update_xaxes(title="Zeit (Jahre)")
    fig.update_yaxes(title="Wert")
    return _base(fig, 340)


def build_drift(rows: list) -> go.Figure:
    """Impermanent Loss und LVR (jeweils im Verhältnis zum Anfangswert) über die Drift."""
    x = [r["mu"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[-r["il_exact"] for r in rows], mode="lines+markers", name="Impermanent Loss (Betrag)", line=dict(color=C.COLOR_LOSS, width=2)))
    fig.add_trace(go.Scatter(x=x, y=[r["lvr_exact"] for r in rows], mode="lines+markers", name="LVR", line=dict(color=C.COLOR_PRICE, width=2)))
    fig.update_xaxes(title="Drift μ des Referenzpreises")
    fig.update_yaxes(title="Anteil am Anfangswert", tickformat=".0%", rangemode="tozero")
    return _base(fig, 320)


def build_fees(rows: list) -> go.Figure:
    """Arbitrage-Gewinn und Gebühreneinnahmen über die Gebühr (im Verhältnis zum Anfangswert)."""
    x = [100 * r["fee"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{v:.2f} %".replace(".", ",") for v in x], y=[r["profit"] for r in rows], name="Arbitrage-Gewinn (Verlust des Pools)", marker_color=C.COLOR_LOSS))
    fig.add_trace(go.Bar(x=[f"{v:.2f} %".replace(".", ",") for v in x], y=[r["fees"] for r in rows], name="Gebühreneinnahmen", marker_color=C.COLOR_GAIN))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Gebühr")
    fig.update_yaxes(title="Anteil am Anfangswert", tickformat=".0%", rangemode="tozero")
    return _base(fig, 320)
