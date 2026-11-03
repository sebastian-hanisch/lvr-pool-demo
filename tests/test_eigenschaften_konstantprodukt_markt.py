"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für JEDE Eingabe gelten müssen, auf Zufallszuständen mit festen Seeds.
Literaturtests (Milionis u. a.; Frongillo u. a.; nur über Abstracts und Trefferlisten) stehen am Ende."""

import math

import numpy as np
import pytest

import kpm_algorithm as A
import kpm_constants as C

SEED = 20261102


def test_the_product_is_kept_without_fee_and_grows_with_fee():
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        x, y = rng.uniform(50, 3000, 2)
        s = rng.uniform(0.001, 0.9)
        assert A.buy(x, y, s, 0.0).k_after == pytest.approx(x * y, rel=1e-12)
        f = rng.uniform(0.0001, 0.05)
        assert A.buy(x, y, s, f).k_after > x * y


def test_a_round_trip_is_free_without_fee_and_never_free_with_fee():
    rng = np.random.default_rng(SEED + 1)
    for _ in range(200):
        x, y = rng.uniform(50, 3000, 2)
        s = rng.uniform(0.001, 0.9)
        assert A.round_trip(x, y, s, 0.0)[2] == pytest.approx(0.0, abs=1e-9 * y)
        paid, back, loss, share = A.round_trip(x, y, s, rng.uniform(0.0005, 0.02))
        assert loss > 0 and back < paid and 0 < share < 1


def test_the_price_rises_with_every_purchase_and_the_average_price_lies_between():
    x, y = 1000.0, 1000.0
    last = 0.0
    for s in np.linspace(0.01, 0.9, 30):
        t = A.buy(x, y, float(s), 0.003)
        assert t.price_after > last and t.price_before < t.avg_price < t.price_after
        last = t.price_after


def test_scaling_and_relabelling_of_the_pool():
    rng = np.random.default_rng(SEED + 2)
    for _ in range(100):
        x, y = rng.uniform(50, 3000, 2)
        c = rng.uniform(0.1, 10)
        s, f = rng.uniform(0.01, 0.8), rng.choice([0.0, 0.003])
        t, tc = A.buy(x, y, s, f), A.buy(c * x, c * y, s, f)
        assert tc.dy == pytest.approx(c * t.dy, rel=1e-12) and tc.price_after == pytest.approx(t.price_after, rel=1e-12)       # die Größe des Pools ändert nur die Menge
        dy = 0.2 * y
        got, xn, yn = A.swap_in_y(x, y, dy, f)
        got2, yn2, xn2 = A.swap_in_x(y, x, dy, f)                                                                         # die Güter vertauschen: dasselbe Spiegelbild
        assert (got, xn, yn) == (pytest.approx(got2), pytest.approx(xn2), pytest.approx(yn2))


def test_the_arbitrage_profit_is_nonnegative_and_the_pool_ends_at_the_reference_price():
    rng = np.random.default_rng(SEED + 3)
    for f in (0.0, 0.003):
        x, y = np.full(400, 1000.0), np.full(400, 1000.0)
        P = np.exp(rng.normal(0, 0.3, 400))
        xn, yn, prof, fee = A.arbitrage(x, y, P, f)
        assert (prof >= -1e-9).all() and (fee >= 0).all() and (xn > 0).all() and (yn > 0).all()
        if f == 0.0:
            assert yn / xn == pytest.approx(P, rel=1e-9)
            assert xn * yn == pytest.approx(x * y, rel=1e-9)


def test_the_expected_lvr_scales_with_the_value_and_with_sigma_squared_for_small_sigma():
    assert A.exact_lvr_gbm(2 * C.POOL_V0, 0.8, 0.3, 1.0, 50) == pytest.approx(2 * A.exact_lvr_gbm(C.POOL_V0, 0.8, 0.3, 1.0, 50), rel=1e-12)
    small = [A.exact_lvr_gbm(C.POOL_V0, s, 0.0, 1.0, 100) / C.POOL_V0 for s in (0.01, 0.02, 0.04)]
    assert small[1] / small[0] == pytest.approx(4.0, rel=1e-3) and small[2] / small[1] == pytest.approx(4.0, rel=1e-3)
    assert small[0] == pytest.approx(0.01 ** 2 / 8, rel=1e-3)
    assert A.exact_lvr_gbm(C.POOL_V0, 0.0, 0.0, 1.0, 10) == pytest.approx(0.0, abs=1e-12)


def test_the_lvr_rate_per_value_does_not_depend_on_the_drift_for_fine_steps():
    """Je Schritt im Verhältnis zum erwarteten Poolwert: (a - g) / g -> sigma^2 dt / 8, gleich für jede Drift."""
    dt, sigma = 1e-4, 0.8
    for mu in (-1.0, 0.0, 0.5, 1.0):
        g = A.discrete_g(sigma, mu, dt)
        a = (math.exp(mu * dt) + 1) / 2
        assert (a - g) / g / dt == pytest.approx(sigma ** 2 / 8, rel=2e-3)


def test_the_expected_il_equals_minus_the_expected_lvr_without_drift_and_at_minus_sigma_squared_over_four_but_not_at_the_other_tested_drifts():
    for sigma in (0.3, 0.8, 1.2):
        for steps in (1, 10, 365):
            assert A.exact_il_value(C.POOL_V0, sigma, 0.0, 1.0) == pytest.approx(-A.exact_lvr_gbm(C.POOL_V0, sigma, 0.0, 1.0, steps), rel=1e-9)
            mu_cross = -sigma ** 2 / 4
            assert A.exact_il_value(C.POOL_V0, sigma, mu_cross, 1.0) == pytest.approx(-A.exact_lvr_gbm(C.POOL_V0, sigma, mu_cross, 1.0, steps), rel=1e-9)
        for mu in (-0.5, 0.5, 1.0):
            il, lvr = A.exact_il_value(C.POOL_V0, sigma, mu, 1.0), A.exact_lvr_gbm(C.POOL_V0, sigma, mu, 1.0, 365)
            assert il <= 0 and abs(il + lvr) > 1e-4 * C.POOL_V0


def test_more_fee_means_less_arbitrage_profit_on_the_same_paths_and_every_path_stays_behind_rebalancing():
    runs = [A.simulate_pool(f, 0.8, 0.0, 1.0, 100, 300, 4) for f in C.FEE_CHOICES]
    means = [float(r.profit.mean()) for r in runs]
    assert means == sorted(means, reverse=True)
    assert all((r.profit > 0).all() for r in runs)
    assert [float(r.fees.mean()) for r in runs] == sorted(float(r.fees.mean()) for r in runs) and float(runs[0].fees.sum()) == 0.0
    trades = [float(r.trades.mean()) for r in runs]
    assert trades == sorted(trades, reverse=True)


def test_the_bridge_properties():
    rng = np.random.default_rng(SEED + 4)
    r = 100.0
    ps = np.linspace(0.5, 0.995, 100)
    cost = [A.cp_buy_to_price(float(p), r)[1] for p in ps]
    assert cost == sorted(cost) and all(A.cp_loss(float(p), r) < r for p in ps)
    assert all(A.cp_buy_to_price(float(p), r)[0] > A.lmsr_shares_to_price(float(p), r) for p in ps[1:])               # an jedem Preis über 1/2 tiefer als die LMSR (b = r)
    assert all(A.cp_loss(float(p), r) > A.lmsr_loss(float(p), r) for p in ps[1:])
    for _ in range(100):                                                                                            # konvex: C(mitte) <= Mittel der Kosten
        q1, q2 = rng.uniform(0, 80, 2), rng.uniform(0, 80, 2)
        assert A.cp_cost((q1 + q2) / 2, r) <= 0.5 * (A.cp_cost(q1, r) + A.cp_cost(q2, r)) + 1e-9
    assert A.cp_cost(np.zeros(2), r) == pytest.approx(0.0, abs=1e-12) and A.cp_state(np.zeros(2), r)["p_a"] == pytest.approx(0.5)


def test_the_break_even_volume_is_profit_over_fee():
    assert A.break_even_volume(0.06, 0.003) == pytest.approx(20.0) and A.break_even_volume(0.06, 0.0) is None


# --- Literaturtests (nur Abstracts und Trefferlisten) ---------------------------------------------------------------------------------------------

def test_literature_lvr_rate_sigma_squared_over_eight_for_constant_product():
    """Milionis u. a. (Treffer-Auszug): Für das Konstantprodukt ist die LVR-Rate ein Achtel der Varianz mal Poolwert; 5 % Tagesvolatilität sind etwa 3,125 Basispunkte je Tag."""
    assert A.lvr_rate(0.05) * 1e4 == pytest.approx(3.125)
    dt = 1e-6
    assert A.exact_lvr_gbm(C.POOL_V0, 0.8, 0.0, dt, 1) / C.POOL_V0 / dt == pytest.approx(0.8 ** 2 / 8, rel=1e-5)


def test_literature_the_lvr_is_the_gap_to_the_rebalancing_strategy_and_not_the_impermanent_loss():
    """Milionis u. a.: LVR ist die Lücke zur Rebalancier-Strategie; Impermanent Loss ist Pool gegen Halten und ein anderes Maß (hier: mit Drift verschieden)."""
    mu, sigma = 0.64, 0.8
    il, lvr = A.exact_il_value(C.POOL_V0, sigma, mu, 1.0) / C.POOL_V0, A.exact_lvr_gbm(C.POOL_V0, sigma, mu, 1.0, 365) / C.POOL_V0
    assert il < -0.17 and 0.09 < lvr < 0.092


def test_literature_a_constant_product_pool_is_a_cost_function_market_on_the_example():
    """Frongillo, Papireddygari, Waggoner (Abstract): CFMMs und Kostenfunktions-Prognosemärkte sind gleichwertig; am Beispiel Konstantprodukt: Preise = Ableitung der Kosten, Summe 1, pfadunabhängig."""
    r = 50.0
    for q in ((10.0, 3.0), (25.0, 0.0), (0.0, 12.0)):
        st = A.cp_state(q, r)
        assert st["p_a"] + st["p_b"] == pytest.approx(1.0) and st["m"] == pytest.approx(A.cp_cost(q, r))
        h = 1e-5
        for i, key in ((0, "p_a"), (1, "p_b")):
            dq = np.zeros(2)
            dq[i] = h
            lo = np.maximum(np.array(q) - dq, 0.0)
            hi = np.array(q) + dq
            if np.array(q)[i] > h:
                assert (A.cp_cost(hi, r) - A.cp_cost(lo, r)) / (hi[i] - lo[i]) == pytest.approx(st[key], abs=1e-6)
