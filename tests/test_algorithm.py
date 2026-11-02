"""Handbeispiele A bis D von Hand gerechnet; jede Einheit des Kerns an einer Mini-Instanz."""

import math

import numpy as np
import pytest

import kpm_algorithm as A
import kpm_constants as C


def test_hand_a_the_pool_trade(hand_a):
    t = A.buy(hand_a["x"], hand_a["y"], 0.1)
    assert t.dy == pytest.approx(1000 * 100 / 900) and t.dx == pytest.approx(100.0)
    assert t.avg_price == pytest.approx(1000 / 900) and t.price_after == pytest.approx(1000 / 900 ** 1 * (1 / 0.9) ** 1) and t.avg_ratio == pytest.approx(1 / 0.9) and t.after_ratio == pytest.approx(1 / 0.81)
    assert (f"{t.dy:.2f}", f"{t.avg_price:.4f}", f"{t.price_after:.4f}") == ("111.11", "1.1111", "1.2346")
    assert t.k_after == pytest.approx(1e6) and t.fee_paid == 0.0
    tf = A.buy(hand_a["x"], hand_a["y"], 0.1, 0.003)
    assert f"{tf.dy:.2f}" == "111.45" and tf.dy == pytest.approx(1000 / 0.997 * 100 / 900) and tf.k_after > 1e6 and tf.fee_paid == pytest.approx(0.003 * tf.dy)


def test_swaps_by_hand():
    dx, xn, yn = A.swap_in_y(1000.0, 1000.0, 1000 * 100 / 900)
    assert (dx, xn, yn) == (pytest.approx(100.0), pytest.approx(900.0), pytest.approx(1111.1111111))
    dy, xn, yn = A.swap_in_x(1000.0, 1000.0, 1000.0)                               # 1000 x einzahlen: y halbiert sich
    assert (dy, xn, yn) == (pytest.approx(500.0), 2000.0, pytest.approx(500.0))
    dy, _, _ = A.swap_in_x(1000.0, 1000.0, 1000.0, 0.01)                             # mit Gebühr: nur 990 gehen in die Invariante
    assert dy == pytest.approx(1000 - 1e6 / 1990)


def test_hand_b_the_prediction_market(hand_b):
    qa, m = A.cp_buy_to_price(hand_b["p"], hand_b["r"])
    assert (qa, m) == (pytest.approx(150.0), pytest.approx(100.0))
    st = A.cp_state((150.0, 0.0), 100.0)
    assert (st["a"], st["b"], st["p_a"], st["p_b"]) == (pytest.approx(50.0), pytest.approx(200.0), pytest.approx(0.8), pytest.approx(0.2))
    assert A.cp_cost((150.0, 0.0), 100.0) == pytest.approx(100.0)
    assert A.cp_loss(0.8, 100.0) == pytest.approx(50.0) and qa - m == pytest.approx(50.0)
    assert A.lmsr_shares_to_price(0.8, 100.0) == pytest.approx(100 * math.log(4)) and A.lmsr_cost_to_price(0.8, 100.0) == pytest.approx(100 * math.log(2.5))
    assert (f"{A.lmsr_shares_to_price(0.8, 100.0):.2f}", f"{A.lmsr_cost_to_price(0.8, 100.0):.2f}") == ("138.63", "91.63")
    assert A.lmsr_loss(0.8, 100.0) == pytest.approx(100 * math.log(1.6))


def test_hand_c_one_step_of_lvr(hand_c):
    xn, yn, prof, fee = A.arbitrage(hand_c["x"], hand_c["y"], hand_c["P"])
    assert (float(xn), float(yn), float(prof), float(fee)) == (pytest.approx(500.0), pytest.approx(2000.0), pytest.approx(1000.0), 0.0)
    pool, hold = float(4 * xn + yn), 4 * 1000 + 1000
    assert (pool, hold) == (pytest.approx(4000.0), 5000) and float(A.pool_value(1e6, 4.0)) == pytest.approx(4000.0)
    assert hold - pool == pytest.approx(float(prof))                                    # bei einem Schritt: LVR = Wertunterschied zum Halten
    assert A.impermanent_loss_ratio(4.0) == pytest.approx(-0.2) and (hold - pool) / hold == pytest.approx(0.2)


def test_hand_d_the_lvr_rate(hand_d):
    assert A.lvr_rate(hand_d["sigma"]) == pytest.approx(3.125e-4) and f"{A.lvr_rate(hand_d['sigma']) * 1e4:.3f}" == "3.125"


def test_the_exact_lvr_by_hand():
    V0, sigma, T = 2000.0, 0.8, 1.0
    assert A.exact_lvr_gbm(V0, sigma, 0.0, T, 1) == pytest.approx(V0 * (1 - math.exp(-sigma ** 2 * T / 8)))
    # zwei Schritte, mu = 0: g = exp(-sigma^2 dt / 8), a = 1: V0 ((1 - g) + g (1 - g))
    g = math.exp(-sigma ** 2 * 0.5 / 8)
    assert A.exact_lvr_gbm(V0, sigma, 0.0, T, 2) == pytest.approx(V0 * ((1 - g) + g * (1 - g)))
    assert A.exact_lvr_gbm(V0, 0.0, 0.0, T, 5) == pytest.approx(0.0, abs=1e-12)
    assert A.exact_il_value(V0, sigma, 0.0, T) == pytest.approx(-V0 * (1 - math.exp(-sigma ** 2 * T / 8)))


def test_the_pool_constants():
    assert (C.POOL_X, C.POOL_Y, C.POOL_K, C.POOL_V0) == (1000.0, 1000.0, 1e6, 2000.0)


def test_a_two_trade_sequence_without_fee_is_one_trade():
    x, y = 1000.0, 1000.0
    paid = 0.0
    for want in (50.0, 50.0):
        dy = A.buy_payment(x, y, want)
        _, x, y = A.swap_in_y(x, y, dy)
        paid += dy
    assert paid == pytest.approx(A.buy_payment(1000.0, 1000.0, 100.0)) and x * y == pytest.approx(1e6)
    assert A.split_cost(1000.0, 1000.0, 0.1, 0.0, 2)[0] == pytest.approx(paid)


def test_the_simulation_basics():
    r = A.simulate_pool(0.0, 0.8, 0.0, 1.0, 10, 50, 1)
    assert r.profit.shape == (50,) and (r.profit > 0).all() and r.steps == 10 and r.path is None
    r1 = A.simulate_pool(0.0, 0.8, 0.0, 1.0, 10, 1, 1, record=True)
    assert len(r1.path["P"]) == 11 and r1.path["rebalance"][-1] - r1.path["pool"][-1] == pytest.approx(float(r1.profit[0]))
    assert np.array_equal(A.simulate_pool(0.003, 0.8, 0.0, 1.0, 10, 50, 1).profit, A.simulate_pool(0.003, 0.8, 0.0, 1.0, 10, 50, 1).profit)
