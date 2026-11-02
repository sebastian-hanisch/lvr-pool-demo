"""Orakel (ORACLE-PLAYBOOK): unabhängige Rechenwege für den Pool, die Brücke und den LVR.

Die Demo rechnet den Pool in geschlossenen Formeln (quadratische Gleichungen, Wurzeln); die Orakel unten rechnen dagegen mit
(1) Bisektion über die Invariante und die Tauschfunktionen (Handelsmenge, Arbitrage-Menge mit und ohne Gebühr),
(2) einer expliziten Kette „Sätze prägen, dann tauschen“ mit Bisektion statt quadratischer Formel, Preisen als finiten Differenzen und der Aufzählung beider Ausgänge,
(3) einer Aufzählung aller Pfade eines Binomialbaums (explizite Arbitrage Schritt für Schritt) gegen die Formel für den erwarteten LVR, einer numerischen Integration (scipy.integrate.quad) für einen Schritt
    bei lognormalem Faktor, und einer Simulation mit vier Standardfehlern,
(4) denselben Zufallszügen Schritt für Schritt über die Tauschfunktionen gegen die vektorisierte Simulation (mit und ohne Gebühr),
(5) der Umrechnung des Impermanent Loss über die Reserven."""

import itertools
import math

import numpy as np
import pytest
from scipy import integrate

import kpm_algorithm as A
import kpm_constants as C
import kpm_scenario as S


def bisect_arbitrage(x, y, P, f):
    """Arbitrage mit Gebühr über die Tauschfunktionen und Bisektion (statt der quadratischen Formeln). Rückgabe: x', y', Gewinn in y zum Preis P, Gebühr in y zum Preis P."""
    g = 1.0 - f
    if y / x / g < P:                                   # x ist im Pool zu billig: kaufen, bis der Grenzpreis inklusive Gebühr P ist
        lo, hi = 0.0, 1.0
        while True:
            _, xn, yn = A.swap_in_y(x, y, hi, f)
            if yn / xn >= g * P:
                break
            hi *= 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            _, xn, yn = A.swap_in_y(x, y, mid, f)
            lo, hi = (mid, hi) if yn / xn < g * P else (lo, mid)
        dy = 0.5 * (lo + hi)
        dx, xn, yn = A.swap_in_y(x, y, dy, f)
        return xn, yn, P * dx - dy, f * dy
    if y / x * g > P:                                   # x ist im Pool zu teuer: verkaufen, bis der Grenzpreis inklusive Gebühr P ist
        lo, hi = 0.0, x
        while True:
            _, xn, yn = A.swap_in_x(x, y, hi, f)
            if yn / xn <= P / g:
                break
            hi *= 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            _, xn, yn = A.swap_in_x(x, y, mid, f)
            lo, hi = (mid, hi) if yn / xn > P / g else (lo, mid)
        dx = 0.5 * (lo + hi)
        dy, xn, yn = A.swap_in_x(x, y, dx, f)
        return xn, yn, dy - P * dx, f * dx * P
    return x, y, 0.0, 0.0


# --- (1) Handelsmenge und Arbitrage-Menge ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("f", [0.0, 0.0005, 0.003, 0.01])
def test_the_payment_for_a_wanted_amount_against_bisection_over_the_swap(f):
    rng = np.random.default_rng(1)
    for _ in range(60):
        x, y = rng.uniform(100, 3000), rng.uniform(100, 3000)
        want = rng.uniform(0.001, 0.9) * x
        lo, hi = 0.0, 1.0
        while A.swap_in_y(x, y, hi, f)[0] < want:
            hi *= 2
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if A.swap_in_y(x, y, mid, f)[0] < want else (lo, mid)
        assert A.buy_payment(x, y, want, f) == pytest.approx(0.5 * (lo + hi), rel=1e-11)
        dxg, xn, yn = A.swap_in_y(x, y, A.buy_payment(x, y, want, f), f)
        assert dxg == pytest.approx(want, rel=1e-11)
        assert xn * (yn - f * A.buy_payment(x, y, want, f)) == pytest.approx(x * y, rel=1e-11)    # die wirksame Invariante bleibt (die Gebühr liegt daneben im Pool)


@pytest.mark.parametrize("f", [0.0, 0.0005, 0.003, 0.01])
def test_the_closed_form_arbitrage_against_bisection(f):
    rng = np.random.default_rng(2)
    for _ in range(80):
        x, y = rng.uniform(300, 2000), rng.uniform(300, 2000)
        P = (y / x) * math.exp(rng.normal(0, 0.1))
        xn, yn, prof, fee = (float(v) for v in A.arbitrage(x, y, P, f))
        xb, yb, pb, fb = bisect_arbitrage(x, y, P, f)
        assert (xn, yn) == (pytest.approx(xb, rel=1e-9), pytest.approx(yb, rel=1e-9))
        assert prof == pytest.approx(pb, abs=1e-8) and fee == pytest.approx(fb, abs=1e-8)
        assert prof >= -1e-9                                                         # der Arbitrageur handelt nur mit Gewinn
        if (yn, xn) != (y, x):                                                       # nach dem Handel ist der Grenzpreis inklusive Gebühr gleich P
            g = 1 - f
            marginal = yn / xn / g if y / x / g < P else yn / xn * g
            assert marginal == pytest.approx(P, rel=1e-9)


def test_inside_the_fee_band_nobody_trades():
    x, y, f = 1000.0, 1000.0, 0.01
    for P in (0.995, 1.0, 1.005):
        xn, yn, prof, fee = A.arbitrage(x, y, P, f)
        assert (float(xn), float(yn), float(prof), float(fee)) == (x, y, 0.0, 0.0)
    assert float(A.arbitrage(x, y, 1.02, f)[2]) > 0 and float(A.arbitrage(x, y, 0.98, f)[2]) > 0


# --- (2) Brücke: Sätze prägen, dann tauschen -------------------------------------------------------------------------------------------------

def explicit_cost(q, r):
    """Kleinste Zahl m von Sätzen mit (r + m - qA)(r + m - qB) = r^2 (beide Reserven positiv), per Bisektion statt der quadratischen Formel."""
    lo = max(q[0], q[1], 0.0) - r + 1e-12                                              # ab hier sind beide Reserven positiv
    step = 1.0
    while (r + lo + step - q[0]) * (r + lo + step - q[1]) < r * r:
        step *= 2
    hi = lo + step
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if (r + mid - q[0]) * (r + mid - q[1]) < r * r else (lo, mid)
    return 0.5 * (lo + hi)


def test_the_cost_function_against_the_explicit_minting_and_swapping():
    rng = np.random.default_rng(3)
    for r in (10.0, 100.0, 1000.0):
        for _ in range(100):
            q = rng.uniform(0, 0.8 * r, 2) * rng.choice([0.0, 1.0], 2)
            assert A.cp_cost(q, r) == pytest.approx(explicit_cost(q, r), rel=1e-9, abs=1e-9)
            st = A.cp_state(q, r)
            assert st["a"] * st["b"] == pytest.approx(r * r, rel=1e-9) and st["p_a"] + st["p_b"] == pytest.approx(1.0, abs=1e-12) and st["a"] > 0 and st["b"] > 0
            h = 1e-5 * r
            ga = (A.cp_cost(q + np.array([h, 0.0]), r) - A.cp_cost(q - np.array([h, 0.0]), r)) / (2 * h) if q[0] > h else None
            if ga is not None:
                assert st["p_a"] == pytest.approx(ga, abs=1e-6)                      # der Preis ist die Ableitung der Kosten


def test_the_cash_of_a_trade_sequence_depends_only_on_the_end_state_and_both_outcomes_are_bounded():
    rng = np.random.default_rng(4)
    r = 100.0
    for _ in range(300):
        q, paid = np.zeros(2), 0.0
        for _ in range(int(rng.integers(1, 8))):
            d = rng.uniform(0, 0.3 * r, 2) * rng.choice([0.0, 1.0], 2)
            paid += A.cp_cost(q + d, r) - A.cp_cost(q, r)
            q = q + d
        assert paid == pytest.approx(A.cp_cost(q, r), abs=1e-9 * r)
        for outcome in (0, 1):                                                       # Aufzählung beider Ausgänge: Verlust des Marktmachers = Auszahlung minus Kasse
            assert q[outcome] - paid < r                                             # Grenze r


@pytest.mark.parametrize("p", [0.55, 0.6, 0.8, 0.9, 0.99])
def test_the_bridge_numbers_against_the_explicit_chain(p):
    r = 100.0
    qa, m = A.cp_buy_to_price(p, r)
    assert explicit_cost((qa, 0.0), r) == pytest.approx(m, rel=1e-9)
    assert A.cp_state((qa, 0.0), r)["p_a"] == pytest.approx(p, abs=1e-12)
    assert qa - m == pytest.approx(A.cp_loss(p, r), rel=1e-9) and A.cp_loss(p, r) < r
    # LMSR zum Vergleich: Kosten über die logarithmische Kostenfunktion C(q) = b ln(e^{qA/b} + e^{qB/b})
    b = r
    q_lm = A.lmsr_shares_to_price(p, b)
    c_lm = b * math.log(math.exp(q_lm / b) + 1.0) - b * math.log(2.0)
    assert A.lmsr_cost_to_price(p, b) == pytest.approx(c_lm, rel=1e-9) and q_lm - c_lm == pytest.approx(A.lmsr_loss(p, b), rel=1e-9) and A.lmsr_loss(p, b) < b * math.log(2)


def test_the_initial_reaction_is_the_same_as_the_lmsr_with_b_equal_to_the_reserve():
    for r in (10.0, 100.0, 1000.0):
        h = 1e-4 * r
        cp = (A.cp_state((h, 0.0), r)["p_a"] - 0.5) / h
        lm = (1 / (1 + math.exp(-h / r)) - 0.5) / h
        assert cp == pytest.approx(1 / (4 * r), rel=1e-3) and lm == pytest.approx(1 / (4 * r), rel=1e-3)


# --- (3) LVR: Binomialbaum, numerische Integration, Simulation ---------------------------------------------------------------------------------

def tree_expected_lvr(n, sigma, mu, T):
    """Erwarteter LVR über n Schritte eines Binomialbaums (Faktoren u, d = 1/u, Wahrscheinlichkeit p so, dass E[Faktor] = exp(mu dt)): Aufzählung aller 2^n Pfade mit expliziter Arbitrage."""
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(mu * dt) - d) / (u - d)
    assert 0 < p < 1
    total = 0.0
    for path in itertools.product((True, False), repeat=n):
        x, y, P, prob, profit = C.POOL_X, C.POOL_Y, 1.0, 1.0, 0.0
        for up in path:
            P *= u if up else d
            prob *= p if up else 1 - p
            x, y, prof, _ = bisect_arbitrage(x, y, P, 0.0)
            profit += prof
        total += prob * profit
    return total, p, u, d


@pytest.mark.parametrize("n", [1, 2, 4, 6])
@pytest.mark.parametrize("mu", [0.0, 0.3, -0.3])
def test_the_exact_expected_lvr_against_the_enumeration_of_a_binomial_tree(n, mu):
    sigma, T = 0.5, 1.0
    total, p, u, d = tree_expected_lvr(n, sigma, mu, T)
    dt = T / n
    g = p * math.sqrt(u) + (1 - p) * math.sqrt(d)                                      # E[sqrt(Faktor)] des Baums
    a = (math.exp(mu * dt) + 1.0) / 2.0                                                # (E[Faktor] + 1) / 2
    assert A.exact_lvr(C.POOL_V0, g, a, n) == pytest.approx(total, rel=1e-9)


@pytest.mark.parametrize("sigma,mu,dt", [(0.8, 0.0, 1.0), (0.8, 0.5, 1.0), (0.3, -0.4, 0.25), (1.5, 1.0, 0.5)])
def test_one_step_of_the_exact_formula_against_numerical_integration(sigma, mu, dt):
    k = C.POOL_K

    def integrand(z):
        P = S.step_factor(sigma, mu, dt, z)
        profit = C.POOL_X * P + C.POOL_Y - 2 * math.sqrt(k * P)
        return profit * math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

    value, err = integrate.quad(integrand, -12, 12, epsabs=1e-12, epsrel=1e-12, limit=200)
    assert A.exact_lvr_gbm(C.POOL_V0, sigma, mu, dt, 1) == pytest.approx(value, rel=1e-8)


@pytest.mark.parametrize("mu", [0.0, 0.5])
def test_the_exact_expectations_against_the_simulation_with_four_standard_errors(mu):
    sigma, steps, paths = 0.8, 40, 40000
    run = A.simulate_pool(0.0, sigma, mu, 1.0, steps, paths, 5)
    lv, il = A.mean_se(run.profit), A.mean_se(run.V_T - run.H_T)
    assert abs(lv.mean - A.exact_lvr_gbm(C.POOL_V0, sigma, mu, 1.0, steps)) < 4 * lv.se
    assert abs(il.mean - A.exact_il_value(C.POOL_V0, sigma, mu, 1.0)) < 4 * il.se
    assert abs(float(run.V_T.mean()) - A.exact_pool_value(C.POOL_V0, sigma, mu, 1.0)) < 4 * float(run.V_T.std(ddof=1) / math.sqrt(paths))
    assert (run.profit >= -1e-9).all()                                                 # pfadweise nichtnegativ


# --- (4) Simulation gegen Schritt für Schritt über die Tauschfunktionen ----------------------------------------------------------------------

@pytest.mark.parametrize("f", [0.0, 0.003, 0.01])
def test_the_vectorised_simulation_equals_the_step_by_step_run_through_the_swap_functions(f):
    sigma, mu, steps, paths, seed = 0.8, 0.2, 12, 6, 7
    run = A.simulate_pool(f, sigma, mu, 1.0, steps, paths, seed)
    rng = S.rng_for(seed)
    x, y, P = np.full(paths, C.POOL_X), np.full(paths, C.POOL_Y), np.ones(paths)
    profit, fees = np.zeros(paths), np.zeros(paths)
    for _ in range(steps):
        P = P * S.step_factor(sigma, mu, 1.0 / steps, rng.standard_normal(paths))
        for j in range(paths):
            x[j], y[j], pr, fe = bisect_arbitrage(float(x[j]), float(y[j]), float(P[j]), f)
            profit[j] += pr
            fees[j] += fe
    assert run.profit == pytest.approx(profit, abs=1e-7) and run.fees == pytest.approx(fees, abs=1e-7)
    assert run.V_T == pytest.approx(P * x + y, rel=1e-10) and run.P_T == pytest.approx(P, rel=1e-12)
    # Wert-Identität: Rebalancieren minus Pool = Summe der Arbitrage-Gewinne (auch mit Gebühr)
    one = A.simulate_pool(f, sigma, mu, 1.0, steps, 1, seed, record=True).path
    assert one["rebalance"][-1] - one["pool"][-1] == pytest.approx(one["lvr"][-1], abs=1e-7)


# --- (5) Impermanent Loss über die Reserven --------------------------------------------------------------------------------------------------

def test_the_impermanent_loss_formula_against_the_reserves():
    for r in (0.25, 0.5, 0.9, 1.0, 1.3, 2.0, 4.0, 10.0):
        x, y = A.reserves_at_price(C.POOL_K, r)                                       # Reserven bei Preis r (Start Preis 1, also x0 = y0 = 1000)
        pool = r * float(x) + float(y)
        hold = r * C.POOL_X + C.POOL_Y
        assert pool / hold - 1 == pytest.approx(A.impermanent_loss_ratio(r), abs=1e-12)
        assert float(A.pool_value(C.POOL_K, r)) == pytest.approx(pool, rel=1e-12)
