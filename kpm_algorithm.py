"""Kern der Demo (nur numpy/Standardbibliothek): der Konstantprodukt-Pool, seine Brücke zur Kostenfunktion und der LVR.

Pool mit Reserven (x, y): x riskantes Gut, y Zahlungsmittel; Preis von x in y: P = y / x; Invariante x y = k. Gebühr f (Uniswap-v2-Art): Nur gamma = 1 - f des Eingangs geht in die Invariante ein,
die Gebühr bleibt im Pool und lässt k wachsen. Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b)."""

import math
from dataclasses import dataclass

import numpy as np

import kpm_constants as C
import kpm_scenario as S


# --- Der Pool ---------------------------------------------------------------------------------------------------------------------------------

def check_pool(x: float, y: float, f: float = 0.0):
    if x <= 0 or y <= 0:
        raise ValueError("die Reserven müssen positiv sein")
    if not 0.0 <= f < 1.0:
        raise ValueError("die Gebühr muss in [0, 1) liegen")


def swap_in_y(x: float, y: float, dy: float, f: float = 0.0):
    """Ein Händler zahlt dy (brutto) in y ein. Rückgabe: (ausgezahltes x, x neu, y neu); die Gebühr bleibt in y."""
    check_pool(x, y, f)
    g = 1.0 - f
    dx = x - x * y / (y + g * dy)
    return dx, x - dx, y + dy


def swap_in_x(x: float, y: float, dx: float, f: float = 0.0):
    """Ein Händler zahlt dx (brutto) in x ein. Rückgabe: (ausgezahltes y, x neu, y neu)."""
    check_pool(x, y, f)
    g = 1.0 - f
    dy = y - x * y / (x + g * dx)
    return dy, x + dx, y - dy


def buy_payment(x: float, y: float, dx: float, f: float = 0.0) -> float:
    """Zahlung in y für den Kauf von dx aus dem Pool: y / (1 - f) * dx / (x - dx)."""
    check_pool(x, y, f)
    if not 0 < dx < x:
        raise ValueError("gekauft werden kann nur ein Teil der Reserve")
    return y / (1.0 - f) * dx / (x - dx)


@dataclass(frozen=True)
class Trade:
    dx: float                 # gekaufte Menge
    dy: float                 # Zahlung
    avg_price: float          # dy / dx
    price_before: float
    price_after: float
    k_before: float
    k_after: float
    fee_paid: float           # f * dy
    avg_ratio: float          # mittlerer Preis / Anfangspreis
    after_ratio: float        # Preis danach / Anfangspreis


def buy(x: float, y: float, s: float, f: float = 0.0) -> Trade:
    """Ein Händler kauft den Anteil s der Reserve x."""
    dx = s * x
    dy = buy_payment(x, y, dx, f)
    got, xn, yn = swap_in_y(x, y, dy, f)
    p0 = y / x
    return Trade(got, dy, dy / got, p0, yn / xn, x * y, xn * yn, f * dy, dy / got / p0, yn / xn / p0)


def round_trip(x: float, y: float, s: float, f: float = 0.0):
    """Kaufen und sofort alles zurückverkaufen. Rückgabe: (gezahlt, zurückbekommen, Verlust, Verlust als Anteil der Zahlung)."""
    t = buy(x, y, s, f)
    _, xn, yn = swap_in_y(x, y, t.dy, f)
    back, _, _ = swap_in_x(xn, yn, t.dx, f)
    return t.dy, back, t.dy - back, (t.dy - back) / t.dy


def split_cost(x: float, y: float, s: float, f: float, n: int):
    """Dasselbe Ziel (den Anteil s der Anfangsreserve kaufen) in n gleichen Stücken. Rückgabe: (gezahlt, erhalten)."""
    want = s * x / n
    paid = got = 0.0
    xc, yc = x, y
    for _ in range(n):
        dy = buy_payment(xc, yc, want, f)
        dxg, xc, yc = swap_in_y(xc, yc, dy, f)
        paid += dy
        got += dxg
    return paid, got


# --- Brücke zur Kostenfunktion: Konstantprodukt zwischen zwei Ergebnis-Anteilen --------------------------------------------------------------

def cp_cost(q, r: float) -> float:
    """Kosten C(q) des Konstantprodukt-Prognosemarkts (Reserven r und r, k = r^2, vollständige Sätze A + B für 1 prägbar): kleinste Zahl m von Sätzen mit (r + m - qA)(r + m - qB) = r^2."""
    qa, qb = float(q[0]), float(q[1])
    s = 2 * r - qa - qb
    c = (r - qa) * (r - qb) - r * r
    return 0.5 * (-s + math.sqrt(s * s - 4 * c))


def cp_state(q, r: float) -> dict:
    """Reserven, Preise und Kasse nach dem Kauf q."""
    m = cp_cost(q, r)
    a, b = r + m - q[0], r + m - q[1]
    return {"m": m, "a": a, "b": b, "p_a": b / (a + b), "p_b": a / (a + b)}


def cp_buy_to_price(p: float, r: float):
    """Kauf von A bis zum Preis p_A = p (Start 1/2): Rückgabe (Anteile A, Kosten). Reserven (a, b) = (r sqrt((1-p)/p), r sqrt(p/(1-p)))."""
    if not 0.5 <= p < 1:
        raise ValueError("der Zielpreis muss in [1/2, 1) liegen")
    a = r * math.sqrt((1 - p) / p)
    b = r * math.sqrt(p / (1 - p))
    m = b - r
    return r + m - a, m


def cp_loss(p: float, r: float) -> float:
    """Verlust des Marktmachers bei Eintritt von A nach dem Kauf bis Preis p: Auszahlung minus Kasse = r (1 - sqrt((1-p)/p)); Grenze r."""
    return r * (1.0 - math.sqrt((1 - p) / p))


def lmsr_shares_to_price(p: float, b: float) -> float:
    """LMSR (Stück 5, binär): Anteile A bis zum Preis p: b ln(p / (1 - p))."""
    return b * math.log(p / (1 - p))


def lmsr_cost_to_price(p: float, b: float) -> float:
    """Kosten dafür: -b ln(2 (1 - p))."""
    return -b * math.log(2 * (1 - p))


def lmsr_loss(p: float, b: float) -> float:
    """Verlust bei Eintritt von A: b ln(2 p); Grenze b ln 2."""
    return b * math.log(2 * p)


# --- LVR -------------------------------------------------------------------------------------------------------------------------------------

def pool_value(k: float, P):
    """Wert des Pools bei Preis P (ohne Gebühr): 2 sqrt(k P)."""
    return 2.0 * np.sqrt(k * np.asarray(P, dtype=float))


def reserves_at_price(k: float, P):
    return np.sqrt(k / np.asarray(P, dtype=float)), np.sqrt(k * np.asarray(P, dtype=float))


def impermanent_loss_ratio(r: float) -> float:
    """Wertunterschied Pool gegen Halten bei Preisverhältnis r = P_T / P_0, ohne Gebühr: 2 sqrt(r) / (1 + r) - 1."""
    return 2.0 * math.sqrt(r) / (1.0 + r) - 1.0


def lvr_rate(sigma: float) -> float:
    """LVR je Zeit im Verhältnis zum Poolwert: sigma^2 / 8."""
    return sigma ** 2 / 8.0


def arbitrage(x, y, P, f: float = 0.0):
    """Arbitrageure bringen die Pools (Felder) auf den Referenzpreis P; mit Gebühr nur außerhalb des Bandes [g p_Pool, p_Pool / g] (g = 1 - f), bis der Grenzpreis inklusive Gebühr gleich P ist.
    Kauf von x (Pool zu billig): (y + dy)(y + g dy) = g P x y; Verkauf von x (Pool zu teuer): P g dx^2 + P x (1 + g) dx + P x^2 - g x y = 0.
    Rückgabe: (x neu, y neu, Gewinn des Arbitrageurs in y zum Preis P, Gebühren in y zum Preis P)."""
    x, y, P = (np.asarray(v, dtype=float) for v in np.broadcast_arrays(x, y, P))
    g = 1.0 - f
    p = y / x
    buy_mask = p / g < P
    sell_mask = p * g > P
    xn, yn = x.copy(), y.copy()
    prof, fee = np.zeros_like(x), np.zeros_like(x)
    if buy_mask.any():
        xb, yb, Pb = x[buy_mask], y[buy_mask], P[buy_mask]
        B = yb * (1 + g)
        cc = yb * yb - g * Pb * xb * yb
        dy = (-B + np.sqrt(B * B - 4 * g * cc)) / (2 * g)
        dx = xb - xb * yb / (yb + g * dy)
        xn[buy_mask], yn[buy_mask] = xb - dx, yb + dy
        prof[buy_mask], fee[buy_mask] = Pb * dx - dy, f * dy
    if sell_mask.any():
        xs, ys, Ps = x[sell_mask], y[sell_mask], P[sell_mask]
        a = Ps * g
        B = Ps * xs * (1 + g)
        cc = Ps * xs * xs - g * xs * ys
        dx = (-B + np.sqrt(B * B - 4 * a * cc)) / (2 * a)
        dy = ys - xs * ys / (xs + g * dx)
        xn[sell_mask], yn[sell_mask] = xs + dx, ys - dy
        prof[sell_mask], fee[sell_mask] = dy - Ps * dx, f * dx * Ps
    return xn, yn, prof, fee


def discrete_g(sigma: float, mu: float, dt: float) -> float:
    """E[sqrt(P_{i+1} / P_i)] bei geometrischer Brownscher Bewegung: exp((mu / 2 - sigma^2 / 8) dt)."""
    return math.exp((mu / 2.0 - sigma ** 2 / 8.0) * dt)


def exact_lvr(V0: float, g: float, a: float, steps: int) -> float:
    """Erwarteter LVR über `steps` Schritte, geschlossene Form: V0 sum_i g^i (a - g) mit a = (E[P_{i+1} / P_i] + 1) / 2 und g = E[sqrt(P_{i+1} / P_i)]
    (aus E[V_i] = V0 g^i und x_i P_i = V_i / 2, ohne Gebühr; gilt für jede Verteilung des Faktors, deren beide Erwartungswerte bekannt sind)."""
    return V0 * sum(g ** i * (a - g) for i in range(steps))


def exact_lvr_gbm(V0: float, sigma: float, mu: float, T: float, steps: int) -> float:
    dt = T / steps
    return exact_lvr(V0, discrete_g(sigma, mu, dt), (math.exp(mu * dt) + 1.0) / 2.0, steps)


def exact_pool_value(V0: float, sigma: float, mu: float, T: float) -> float:
    """E[V_T] = V0 exp((mu / 2 - sigma^2 / 8) T) (ohne Gebühr)."""
    return V0 * math.exp((mu / 2.0 - sigma ** 2 / 8.0) * T)


def exact_hold_value(V0: float, mu: float, T: float) -> float:
    """E[H_T] des Bestands, der beim Start gehalten wird: V0 / 2 (exp(mu T) + 1)."""
    return V0 / 2.0 * (math.exp(mu * T) + 1.0)


def exact_il_value(V0: float, sigma: float, mu: float, T: float) -> float:
    """Erwarteter Wertunterschied Pool gegen Halten E[V_T - H_T] (nichtpositiv)."""
    return exact_pool_value(V0, sigma, mu, T) - exact_hold_value(V0, mu, T)


@dataclass(frozen=True)
class PoolRun:
    profit: np.ndarray        # Summe der Arbitrage-Gewinne je Pfad (in y)
    fees: np.ndarray          # Summe der Gebühren je Pfad (in y, zum Referenzpreis bewertet)
    trades: np.ndarray        # Zahl der Schritte mit Handel je Pfad
    V_T: np.ndarray           # Poolwert am Ende (zum Referenzpreis, einschließlich der Gebühren)
    H_T: np.ndarray           # Wert des beim Start gehaltenen Bestands am Ende
    P_T: np.ndarray
    steps: int
    path: dict | None         # nur bei record=True: Zeitreihen des ersten Pfads


def simulate_pool(f: float, sigma: float, mu: float, T: float, steps: int, paths: int, seed: int, record: bool = False) -> PoolRun:
    """`paths` Referenzpreispfade (Start 1), Arbitrage nach jedem Schritt. Vektorisiert über die Pfade; derselbe Zufallszug (seed) für alle Gebühren."""
    S.check_params(sigma, T, steps, paths)
    rng = S.rng_for(seed)
    dt = T / steps
    x = np.full(paths, C.POOL_X)
    y = np.full(paths, C.POOL_Y)
    P = np.ones(paths)
    profit, fees, trades = np.zeros(paths), np.zeros(paths), np.zeros(paths)
    rec = {"P": [1.0], "pool": [C.POOL_V0], "hold": [C.POOL_V0], "rebalance": [C.POOL_V0], "lvr": [0.0]} if record else None
    rebal = C.POOL_V0
    for _ in range(steps):
        P_new = P * S.step_factor(sigma, mu, dt, rng.standard_normal(paths))
        if record:
            rebal += float(x[0]) * (float(P_new[0]) - float(P[0]))
        x_old = x
        x, y, prof, fee = arbitrage(x, y, P_new, f)
        profit += prof
        fees += fee
        trades += prof > 0
        P = P_new
        if record:
            rec["P"].append(float(P[0]))
            rec["pool"].append(float(P[0] * x[0] + y[0]))
            rec["hold"].append(float(C.POOL_X * P[0] + C.POOL_Y))
            rec["rebalance"].append(rebal)
            rec["lvr"].append(float(profit[0]))
        del x_old
    return PoolRun(profit, fees, trades, P * x + y, C.POOL_X * P + C.POOL_Y, P, steps, rec)


@dataclass(frozen=True)
class Stat:
    mean: float
    se: float


def mean_se(a) -> Stat:
    a = np.asarray(a, dtype=float)
    return Stat(float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0)


def break_even_volume(profit_share: float, f: float) -> float | None:
    """Handelsvolumen von Nicht-Informierten (als Vielfaches des Anfangswerts des Pools), dessen Gebühren den Arbitrage-Verlust ausgleichen würden: (Gewinn / V0) / f; ohne Gebühr gibt es keins."""
    return profit_share / f if f > 0 else None
