"""Kennzahlen, Verdikt-Logik, Tabellen. Verdikte sind reine Funktionen (kind, text) auf Zahlen; die App und tools/readme_rows.py rechnen über dieselben Funktionen."""

import math
from dataclasses import dataclass

import kpm_algorithm as A
import kpm_constants as C

V0 = C.POOL_V0


def de(v: float, d: int = 2) -> str:
    """Zahl mit Dezimalkomma; eine Zahl, die auf null rundet, steht ohne Vorzeichen (nie „-0,00“, das wäre Rundungsrauschen)."""
    text = f"{v:.{d}f}"
    if float(text) == 0.0:
        text = text.lstrip("-")
    return text.replace(".", ",")


def pct(p: float, d: int = 1) -> str:
    return f"{de(100 * p, d)} %"


@dataclass(frozen=True)
class Settings:
    s: float = C.DEFAULT_S
    fee: float = C.DEFAULT_FEE
    p: float = C.DEFAULT_P
    sigma: float = C.DEFAULT_SIGMA
    mu: float = C.DEFAULT_MU
    steps: int = C.DEFAULT_STEPS
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    trade: A.Trade
    bridge: dict                # Zeile des Vergleichs Konstantprodukt / LMSR beim gewählten Preis
    exact_share: float          # E[LVR] / V0 für die gewählten Schritte, exakt (ohne Gebühr)
    steps_table: list           # LVR nach Schrittzahl: exakt gegen Simulation (ohne Gebühr)
    drift_table: list           # Impermanent Loss gegen LVR über die Drift
    fee_table: list             # Gebühren: Arbitrage-Gewinn, Gebühreneinnahmen, Handelsanteil
    path: dict                  # ein einzelner Pfad (Seed)


# --- Pool ------------------------------------------------------------------------------------------------------------------------------------

def impact_table(f: float) -> list:
    out = []
    for s in C.IMPACT_S:
        t0, tf = A.buy(C.POOL_X, C.POOL_Y, s, 0.0), A.buy(C.POOL_X, C.POOL_Y, s, f)
        out.append({"s": s, "avg_ratio": t0.avg_ratio, "after_ratio": t0.after_ratio, "pay": t0.dy, "pay_fee": tf.dy})
    return out


def split_table(f: float) -> list:
    return [{"n": n, "paid0": A.split_cost(C.POOL_X, C.POOL_Y, 0.1, 0.0, n)[0], "paid": A.split_cost(C.POOL_X, C.POOL_Y, 0.1, f, n)[0]} for n in C.SPLITS]


def round_trip_table(f: float) -> list:
    return [{"s": s, "loss0": A.round_trip(C.POOL_X, C.POOL_Y, s, 0.0)[3], "loss": A.round_trip(C.POOL_X, C.POOL_Y, s, f)[3]} for s in C.RT_S]


def bridge_row(p: float, r: float = C.BRIDGE_R) -> dict:
    qa, m = A.cp_buy_to_price(p, r)
    return {"p": p, "cp_shares": qa, "cp_cost": m, "cp_loss": A.cp_loss(p, r), "lm_shares": A.lmsr_shares_to_price(p, r), "lm_cost": A.lmsr_cost_to_price(p, r), "lm_loss": A.lmsr_loss(p, r), "r": r}


def bridge_table() -> list:
    return [bridge_row(p) for p in C.BRIDGE_PS]


# --- LVR -------------------------------------------------------------------------------------------------------------------------------------

def steps_table(sigma: float, mu: float) -> list:
    """E[LVR] / V0 nach Zahl der Schritte: exakt (Formel für diskrete Schritte) gegen Simulation ohne Gebühr."""
    out = []
    for n in C.STEP_TABLE:
        run = A.simulate_pool(0.0, sigma, mu, C.HORIZON, n, C.LVR_PATHS, C.LVR_SEED)
        st = A.mean_se(run.profit / V0)
        out.append({"steps": n, "exact": A.exact_lvr_gbm(V0, sigma, mu, C.HORIZON, n) / V0, "sim": st.mean, "se": st.se})
    return out


def drift_table(sigma: float) -> list:
    """Impermanent Loss (Wert gegen Halten) gegen LVR über die Drift mu = m sigma^2: exakt und simuliert (DRIFT_STEPS Schritte)."""
    out = []
    for m in C.DRIFT_MULTIPLES:
        mu = m * sigma ** 2
        run = A.simulate_pool(0.0, sigma, mu, C.HORIZON, C.DRIFT_STEPS, C.DRIFT_PATHS, C.DRIFT_SEED)
        lv, il = A.mean_se(run.profit / V0), A.mean_se((run.V_T - run.H_T) / V0)
        out.append({"mu": mu, "lvr_exact": A.exact_lvr_gbm(V0, sigma, mu, C.HORIZON, C.DRIFT_STEPS) / V0, "il_exact": A.exact_il_value(V0, sigma, mu, C.HORIZON) / V0,
                    "lvr_sim": lv.mean, "lvr_se": lv.se, "il_sim": il.mean, "il_se": il.se})
    return out


def fee_table(sigma: float, mu: float, steps: int) -> list:
    """Dieselben Pfade für jede Gebühr: Arbitrage-Gewinn (was der Pool gegenüber dem Rebalancieren verliert, einschließlich der Gebühren im Pool), Gebühren, Handelsanteil, Ausgleichsvolumen."""
    rows, base = [], None
    for f in C.FEE_CHOICES:
        run = A.simulate_pool(f, sigma, mu, C.HORIZON, steps, C.LVR_PATHS, C.LVR_SEED)
        share = float(run.profit.mean()) / V0
        base = share if base is None else base
        rows.append({"fee": f, "profit": share, "profit_se": A.mean_se(run.profit / V0).se, "rel": share / base if base > 0 else float("nan"), "fees": float(run.fees.mean()) / V0,
                     "trade_share": float(run.trades.mean()) / steps, "min_profit": float(run.profit.min()) / V0, "break_even": A.break_even_volume(share, f)})
    return rows


def analyse(settings: Settings) -> Analysis:
    path = A.simulate_pool(settings.fee, settings.sigma, settings.mu, C.HORIZON, settings.steps, 1, settings.seed, record=True).path
    return Analysis(settings, A.buy(C.POOL_X, C.POOL_Y, settings.s, settings.fee), bridge_row(settings.p), A.exact_lvr_gbm(V0, settings.sigma, settings.mu, C.HORIZON, settings.steps) / V0,
                    steps_table(settings.sigma, settings.mu), drift_table(settings.sigma), fee_table(settings.sigma, settings.mu, settings.steps), path)


# --- Verdikte (reine Funktionen) -----------------------------------------------------------------------------------------------------------

def verdict_trade(avg_ratio: float, after_ratio: float, pay: float, s: float, f: float, fee_paid: float) -> tuple[str, str]:
    """Ergebnis des Handels: mittlerer Preis und Preis danach im Verhältnis zum Anfangspreis."""
    text = (f"Wer {pct(s, 0)} der Reserve kauft, zahlt {de(pay)} und im Mittel das {de(avg_ratio, 4)}-fache des Anfangspreises; der Preis steigt auf das {de(after_ratio, 4)}-fache. "
            + (f"Davon sind {de(fee_paid)} Gebühr ({pct(f, 2)} der Zahlung), die im Pool bleibt." if f > 0 else "Ohne Gebühr bleibt das Produkt k gleich und der Pfad ist gleichgültig."))
    return ("warning" if after_ratio >= 4 else "info"), text


def verdict_bridge(row: dict) -> tuple[str, str]:
    """Konstantprodukt gegen LMSR (b = r) beim Zielpreis: wer braucht mehr Anteile, und wie hoch ist der Verlust des Marktmachers im Verhältnis zur Reserve?"""
    cp, lm, r = row["cp_shares"], row["lm_shares"], row["r"]
    if abs(cp - lm) < 1e-9 * max(cp, lm, 1.0):
        deeper = "gleich viele Anteile wie"
    else:
        deeper = "mehr Anteile als" if cp > lm else "weniger Anteile als"
    text = (f"Bis zum Preis {de(row['p'], 2)} braucht der Konstantprodukt-Markt {de(cp)} Anteile ({de(row['cp_cost'])} Kosten), also {deeper} die LMSR mit gleicher Anfangsreaktion ({de(lm)} Anteile, {de(row['lm_cost'])} Kosten). "
            f"Der Verlust des Marktmachers bei Eintritt des Ausgangs ist {de(row['cp_loss'] / r, 3)} r (Grenze r), bei der LMSR {de(row['lm_loss'] / r, 3)} r (Grenze {de(math.log(2), 4)} r).")
    return "info", text


def verdict_lvr(exact: float, sim: float, se: float, steps: int, sigma: float) -> tuple[str, str]:
    """Erwarteter LVR im Verhältnis zum Anfangswert: exakte Formel gegen Simulation."""
    z = abs(sim - exact) / se if se > 0 else 0.0
    text = (f"Über ein Jahr mit {steps} Schritten verliert der Pool gegenüber dem Rebalancieren im Erwartungswert {pct(exact, 2)} seines Anfangswerts (exakte Formel); die Simulation misst {pct(sim, 2)} ± {pct(se, 2)}, "
            f"{de(z, 1)} Standardfehler daneben. Die Rate ist σ²/8 = {pct(A.lvr_rate(sigma), 2)} des Poolwerts je Jahr (Volatilität {de(sigma, 2)}).")
    return ("success" if z <= 3 else "warning"), text


def verdict_il(rows: list) -> tuple[str, str]:
    """Impermanent Loss gegen LVR über die Drift: ohne Drift gleich im Betrag; wie weit gehen sie bei der größten Drift auseinander?"""
    zero = min(rows, key=lambda r: abs(r["mu"]))
    top = max(rows, key=lambda r: abs(r["il_exact"] + r["lvr_exact"]))
    gap = abs(top["il_exact"]) - top["lvr_exact"]
    if gap > 1e-9:
        return "info", (f"Ohne Drift sind Impermanent Loss ({pct(zero['il_exact'], 2)}) und LVR ({pct(zero['lvr_exact'], 2)}) im Betrag gleich. Mit Drift μ = {de(top['mu'], 2)} liegt der Impermanent Loss bei {pct(top['il_exact'], 2)}, "
                        f"der LVR bei {pct(top['lvr_exact'], 2)}: Der LVR ist die laufende Last der Liquiditätsgeber, der Impermanent Loss ein Vergleich mit dem Halten.")
    return "info", f"Bei diesen Einstellungen unterscheiden sich Impermanent Loss ({pct(top['il_exact'], 2)}) und LVR ({pct(top['lvr_exact'], 2)}) kaum."


def verdict_fee(rows: list, fee: float) -> tuple[str, str]:
    """Gebühren: wie viel vom gebührenfreien Arbitrage-Gewinn bleibt, und bleibt der Pool in jedem Pfad hinter dem Rebalancieren zurück?"""
    row = next((r for r in rows if abs(r["fee"] - fee) < 1e-12), rows[-1])
    if row["fee"] == 0:
        return "info", f"Ohne Gebühr verliert der Pool {pct(row['profit'], 2)} seines Anfangswerts an die Arbitrage; es gibt keine Einnahmen, die dagegen stehen."
    behind = "In jedem simulierten Pfad bleibt der Pool hinter dem Rebalancieren zurück" if row["min_profit"] > 0 else "Nicht in jedem simulierten Pfad bleibt der Pool hinter dem Rebalancieren zurück"
    text = (f"Mit Gebühr {pct(row['fee'], 2)} sinkt der Arbitrage-Gewinn auf {pct(row['rel'], 1)} des gebührenfreien ({pct(row['profit'], 2)} des Anfangswerts), die Gebühren betragen {pct(row['fees'], 2)}; "
            f"es handeln {pct(row['trade_share'], 1)} der Schritte. {behind} (kleinster Pfad {pct(row['min_profit'], 2)}). Ausgleichen würde ihn erst ein Handel von Nicht-Informierten im Wert von {de(row['break_even'], 1)}-mal dem Anfangswert im Jahr.")
    return ("warning" if row["min_profit"] > 0 else "info"), text


# --- Tabellenzeilen (App und README: dieselbe Rundung) ---------------------------------------------------------------------------------------

HAND_HEADER = ["| Beispiel | Aufgabe | Ergebnis |", "|---|---|---|"]


def hand_rows() -> list:
    t, tf = A.buy(C.POOL_X, C.POOL_Y, 0.1, 0.0), A.buy(C.POOL_X, C.POOL_Y, 0.1, 0.003)
    rows = [f"| A: Pool 1000 / 1000, 100 Einheiten kaufen, ohne Gebühr | Zahlung {de(t.dy)}, mittlerer Preis {de(t.avg_price, 4)}, Preis danach {de(t.price_after, 4)} | mit Gebühr 0,3 %: Zahlung {de(tf.dy)} |"]
    b = bridge_row(0.8)
    st = A.cp_state((b["cp_shares"], 0.0), C.BRIDGE_R)
    rows.append(f"| B: Konstantprodukt als Prognosemarkt, Reserve 100, Preis auf 0,8 | {de(b['cp_shares'])} Anteile A für {de(b['cp_cost'])}; Reserven ({de(st['a'])}; {de(st['b'])}) | Verlust bei A {de(b['cp_loss'])}; LMSR (b = 100): {de(b['lm_shares'])} Anteile für {de(b['lm_cost'])} |")
    xn, yn, prof, _ = A.arbitrage(C.POOL_X, C.POOL_Y, 4.0, 0.0)
    pool, hold = float(4.0 * xn + yn), 4.0 * C.POOL_X + C.POOL_Y
    rows.append(f"| C: ein Schritt von P = 1 auf P = 4, Pool 1000 / 1000 | Reserven ({de(float(xn), 0)}; {de(float(yn), 0)}), Poolwert {de(pool, 0)}, Halten {de(hold, 0)} | LVR {de(float(prof), 0)}, Impermanent Loss {pct(A.impermanent_loss_ratio(4.0), 1)} |")
    rows.append(f"| D: Tagesvolatilität 5 % | LVR-Rate σ²/8 | {de(A.lvr_rate(0.05) * 1e4, 3)} Basispunkte je Tag |")
    return rows


def impact_header(f: float) -> list:
    return [f"| Anteil s der Reserve | mittlerer Preis / Anfangspreis | Preis danach / Anfangspreis | Zahlung ohne Gebühr | Zahlung mit Gebühr {pct(f, 2)} |", "|---|---|---|---|---|"]


def impact_rows(rows: list) -> list:
    return [f"| {pct(r['s'], 1)} | {de(r['avg_ratio'], 6)} | {de(r['after_ratio'], 6)} | {de(r['pay'], 4)} | {de(r['pay_fee'], 4)} |" for r in rows]


def split_header(f: float) -> list:
    return [f"| Stücke | Zahlung ohne Gebühr | Zahlung mit Gebühr {pct(f, 2)} |", "|---|---|---|"]


def split_rows(rows: list) -> list:
    return [f"| {r['n']} | {de(r['paid0'], 5)} | {de(r['paid'], 5)} |" for r in rows]


def round_trip_header(f: float) -> list:
    return [f"| Anteil s der Reserve | Verlust beim Hin und zurück ohne Gebühr | mit Gebühr {pct(f, 2)} |", "|---|---|---|"]


def round_trip_rows(rows: list) -> list:
    return [f"| {pct(r['s'], 0)} | {pct(r['loss0'], 4)} | {pct(r['loss'], 4)} |" for r in rows]


BRIDGE_HEADER = ["| Preis p | Konstantprodukt: Anteile / Kosten | LMSR (b = r): Anteile / Kosten | Verlust Konstantprodukt (in r) | Verlust LMSR (in r) |", "|---|---|---|---|---|"]


def bridge_rows(rows: list) -> list:
    return [f"| {de(r['p'], 2)} | {de(r['cp_shares'], 3)} / {de(r['cp_cost'], 3)} | {de(r['lm_shares'], 3)} / {de(r['lm_cost'], 3)} | {de(r['cp_loss'] / r['r'], 6)} | {de(r['lm_loss'] / r['r'], 6)} |" for r in rows]


STEPS_HEADER = ["| Schritte im Jahr | E[LVR] / V₀ (exakt) | E[LVR] / V₀ (Simulation) |", "|---|---|---|"]


def steps_rows(rows: list) -> list:
    return [f"| {r['steps']} | {pct(r['exact'], 3)} | {pct(r['sim'], 3)} ± {pct(r['se'], 3)} |" for r in rows]


DRIFT_HEADER = ["| Drift μ | E[LVR] / V₀ (exakt) | E[LVR] / V₀ (Simulation) | Impermanent Loss / V₀ (exakt) | Impermanent Loss / V₀ (Simulation) |", "|---|---|---|---|---|"]


def drift_rows(rows: list) -> list:
    return [f"| {de(r['mu'], 2)} | {pct(r['lvr_exact'], 2)} | {pct(r['lvr_sim'], 2)} ± {pct(r['lvr_se'], 2)} | {pct(r['il_exact'], 2)} | {pct(r['il_sim'], 2)} ± {pct(r['il_se'], 2)} |" for r in rows]


FEE_HEADER = ["| Gebühr | Arbitrage-Gewinn / V₀ | Anteil des gebührenfreien | Gebühren / V₀ | Schritte mit Handel | Ausgleichsvolumen / V₀ |", "|---|---|---|---|---|---|"]


def fee_rows(rows: list) -> list:
    return [f"| {pct(r['fee'], 2)} | {pct(r['profit'], 2)} | {pct(r['rel'], 1)} | {pct(r['fees'], 2)} | {pct(r['trade_share'], 1)} | {'–' if r['break_even'] is None else de(r['break_even'], 1)} |" for r in rows]
