"""Randfälle: kleinste und größte Regler, ungültige Eingaben, Preise im Band, Rundung auf null, Rechenzeit am Worst Case."""

import math
import time

import numpy as np
import pytest

import kpm_algorithm as A
import kpm_constants as C
import kpm_evaluation as ev


@pytest.mark.parametrize("s", [C.S_MIN, 0.5, C.S_MAX])
@pytest.mark.parametrize("f", C.FEE_CHOICES)
def test_extreme_trade_sizes_and_fees_are_finite_and_consistent(s, f):
    t = A.buy(C.POOL_X, C.POOL_Y, s, f)
    assert all(math.isfinite(v) for v in (t.dy, t.avg_price, t.price_after, t.k_after, t.fee_paid))
    assert t.after_ratio == pytest.approx(1 / (1 - s) ** 2, rel=1e-9 if f == 0 else 5e-2) and t.dx == pytest.approx(s * C.POOL_X)
    assert t.avg_ratio >= 1 / (1 - s) - 1e-12


@pytest.mark.parametrize("p", [C.P_MIN, 0.8, C.P_MAX])
def test_extreme_bridge_prices(p):
    row = ev.bridge_row(p)
    assert all(math.isfinite(v) for v in row.values()) and row["cp_loss"] < row["r"] and row["lm_loss"] < row["r"] * math.log(2)
    assert row["cp_shares"] > row["lm_shares"] > 0


@pytest.mark.parametrize("steps", [C.STEP_CHOICES[0], C.STEP_CHOICES[-1]])
@pytest.mark.parametrize("sigma,mu", [(C.SIGMA_MIN, C.MU_MIN), (C.SIGMA_MAX, C.MU_MAX), (C.SIGMA_MAX, C.MU_MIN)])
def test_extreme_volatility_drift_and_steps_stay_finite_and_nonnegative(steps, sigma, mu):
    run = A.simulate_pool(0.003, sigma, mu, 1.0, steps, 200, 1)
    assert np.isfinite(run.profit).all() and np.isfinite(run.V_T).all() and (run.profit >= -1e-6 * C.POOL_V0).all()
    assert A.exact_lvr_gbm(C.POOL_V0, sigma, mu, 1.0, steps) > 0 and math.isfinite(A.exact_il_value(C.POOL_V0, sigma, mu, 1.0))


def test_a_single_step_and_no_volatility():
    run = A.simulate_pool(0.0, 0.8, 0.0, 1.0, 1, 50, 1)
    assert (run.profit > 0).all()
    flat = A.simulate_pool(0.0, 0.0, 0.0, 1.0, 10, 5, 1)
    assert np.allclose(flat.profit, 0.0, atol=1e-9) and np.allclose(flat.P_T, 1.0) and flat.trades.sum() == 0


def test_the_price_inside_the_band_means_no_trade_and_zero_profit():
    run = A.simulate_pool(0.01, 0.001, 0.0, 1.0, 20, 20, 1)                                 # winzige Schritte, Band 1 %: kein Handel
    assert run.trades.sum() == 0 and run.profit.sum() == 0.0 and run.fees.sum() == 0.0


def test_invalid_inputs_are_rejected():
    for bad in ((0.0, 1000.0), (1000.0, -1.0)):
        with pytest.raises(ValueError):
            A.check_pool(*bad)
    for f in (-0.1, 1.0, 2.0):
        with pytest.raises(ValueError):
            A.check_pool(1000.0, 1000.0, f)
    with pytest.raises(ValueError):
        A.buy_payment(1000.0, 1000.0, 1000.0)
    with pytest.raises(ValueError):
        A.buy_payment(1000.0, 1000.0, 0.0)
    with pytest.raises(ValueError):
        A.cp_buy_to_price(0.4, 100.0)
    with pytest.raises(ValueError):
        A.cp_buy_to_price(1.0, 100.0)
    for kw in ({"sigma": -0.1}, {"steps": 0}, {"paths": 0}, {"T": 0.0}):
        args = {"f": 0.0, "sigma": 0.8, "mu": 0.0, "T": 1.0, "steps": 5, "paths": 5, "seed": 1, **kw}
        with pytest.raises(ValueError):
            A.simulate_pool(**args)


def test_numbers_that_round_to_zero_are_shown_without_a_sign():
    assert ev.de(-1e-15, 4) == "0,0000" and ev.de(-0.0, 2) == "0,00" and ev.de(-0.004, 2) == "0,00" and ev.de(-0.006, 2) == "-0,01" and ev.pct(-1e-9, 2) == "0,00 %"
    assert ev.de(3.14159, 2) == "3,14" and ev.pct(0.0746, 1) == "7,5 %"


def test_the_slider_extremes_produce_finite_analyses():
    for kw in ({"s": C.S_MIN, "fee": 0.0, "p": C.P_MIN, "sigma": C.SIGMA_MIN, "mu": C.MU_MIN, "steps": 12},
               {"s": C.S_MAX, "fee": 0.01, "p": C.P_MAX, "sigma": C.SIGMA_MAX, "mu": C.MU_MAX, "steps": 365}):
        a = ev.analyse(ev.Settings(**kw))
        assert all(math.isfinite(r["sim"]) and math.isfinite(r["exact"]) for r in a.steps_table)
        assert all(math.isfinite(r["lvr_sim"]) and math.isfinite(r["il_sim"]) for r in a.drift_table)
        assert all(math.isfinite(r["profit"]) and math.isfinite(r["fees"]) for r in a.fee_table) and len(a.path["P"]) == kw["steps"] + 1


def test_the_worst_case_runs_in_a_few_seconds():
    t0 = time.time()
    ev.analyse(ev.Settings(s=C.S_MAX, fee=0.01, p=C.P_MAX, sigma=C.SIGMA_MAX, mu=C.MU_MAX, steps=2000))
    ev.hand_rows(), ev.bridge_table(), ev.impact_table(0.01), ev.split_table(0.01), ev.round_trip_table(0.01)
    assert time.time() - t0 < 60
