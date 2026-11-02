"""Jede Zahl der README steht hier und wird über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4, §11a).

Die Tabellen der README werden Zeile für Zeile gegen `tools/readme_rows.py` geprüft (dieselbe Rundung wie die App); die einzelnen Sätze der README
haben je einen Test unter der Bedingung, die in der README daneben steht."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import kpm_algorithm as A  # noqa: E402
import kpm_constants as C  # noqa: E402
import kpm_evaluation as ev  # noqa: E402
import readme_rows as R  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")
V0 = C.POOL_V0


def test_the_tables_in_the_readme_are_the_computed_rows():
    for name, rows in (("Tabelle 1", R.hand_rows()), ("Tabelle 2", R.impact_rows()), ("Tabelle 2b", R.split_rows()), ("Tabelle 2c", R.round_trip_rows()), ("Tabelle 3", R.bridge_rows()),
                       ("Tabelle 4", R.steps_rows()), ("Tabelle 5", R.drift_rows()), ("Tabelle 6", R.fee_rows())):
        for row in rows:
            assert row in README, (name, row)
    for header in (ev.HAND_HEADER, ev.impact_header(R.FEE), ev.split_header(R.FEE), ev.round_trip_header(R.FEE), ev.BRIDGE_HEADER, ev.STEPS_HEADER, ev.DRIFT_HEADER, ev.FEE_HEADER):
        for line in header:
            assert line in README, line


def test_the_default_settings_behind_the_tables():
    d = ev.Settings()
    assert (d.fee, d.sigma, d.mu, d.steps) == (0.003, 0.8, 0.0, 365) and (C.LVR_PATHS, C.LVR_SEED, C.DRIFT_PATHS, C.DRIFT_STEPS, C.DRIFT_SEED, C.HORIZON) == (5000, 3, 5000, 100, 2, 1.0)
    assert (C.POOL_X, C.POOL_Y, C.POOL_K, V0, C.BRIDGE_R) == (1000.0, 1000.0, 1e6, 2000.0, 100.0)
    for part in ("5 000 Pfade je Zeile, Seed 3", "100 Schritte, 5 000 Pfade, Seed 2", "365 Schritte, 5 000 Pfade, Seed 3", "Start: Pool 1000 / 1000, Preis 1, Wert $V_0 = 2000$; Horizont ein Jahr"):
        assert part in README, part


def test_the_hand_examples():
    t = A.buy(1000.0, 1000.0, 0.1)
    assert (f"{t.dy:.2f}", f"{t.avg_price:.4f}", f"{t.price_after:.4f}", f"{A.buy(1000.0, 1000.0, 0.1, 0.003).dy:.2f}") == ("111.11", "1.1111", "1.2346", "111.45")
    qa, m = A.cp_buy_to_price(0.8, 100.0)
    assert (qa, m, A.cp_loss(0.8, 100.0)) == (pytest.approx(150.0), pytest.approx(100.0), pytest.approx(50.0))
    xn, yn, prof, _ = A.arbitrage(1000.0, 1000.0, 4.0)
    assert (float(xn), float(yn), float(prof)) == (pytest.approx(500.0), pytest.approx(2000.0), pytest.approx(1000.0)) and f"{A.impermanent_loss_ratio(4.0):.3f}" == "-0.200"
    assert f"{A.lvr_rate(0.05) * 1e4:.3f}" == "3.125"
    for part in ("$\\Delta y = 1000 \\cdot 100/900$", "$(100\\sqrt{0{,}2/0{,}8};\\ 100\\sqrt{0{,}8/0{,}2}) = (50;\\ 200)$", "$100 + 100 - 50 = 150$ Anteile A", "$2\\sqrt{10^6 \\cdot 4} = 4000$ gegen Halten $4 \\cdot 1000 + 1000 = 5000$",
                 "$0{,}05^2/8 = 3{,}125 \\cdot 10^{-4}$"):
        assert part in README, part


def test_the_price_impact_sentences():
    for s in (0.001, 0.01, 0.1, 0.5):
        t = A.buy(1000.0, 1000.0, s)
        assert t.avg_ratio == pytest.approx(1 / (1 - s), rel=1e-12) and t.after_ratio == pytest.approx(1 / (1 - s) ** 2, rel=1e-12)
    half = A.buy(1000.0, 1000.0, 0.5)
    assert (half.avg_ratio, half.after_ratio, half.k_after) == (pytest.approx(2.0), pytest.approx(4.0), pytest.approx(1e6))
    s = ev.split_table(0.003)
    assert [round(r["paid"], 5) for r in s] == [111.44545, 111.45427, 111.46157, 111.46324] and {round(r["paid0"], 5) for r in s} == {111.11111}
    assert all(r["loss0"] == pytest.approx(0.0, abs=1e-12) for r in ev.round_trip_table(0.003)) and [f"{100 * r['loss']:.4f}" for r in ev.round_trip_table(0.003)] == ["0.5931", "0.5394", "0.3000"]
    for part in ("Ohne Gebühr bleibt $k$ gleich $1\\,000\\,000$", "zahlt im Mittel das Doppelte und lässt den Preis auf das Vierfache steigen", "**Mit Gebühr ist der Pfad nicht mehr gleichgültig:**"):
        assert part in README, part


def test_the_bridge_sentences():
    rows = {r["p"]: r for r in ev.bridge_table()}
    assert C.BRIDGE_PS == (0.6, 0.8, 0.9, 0.99) and f"{rows[0.99]['cp_shares']:.3f}" == "984.937" and f"{rows[0.99]['lm_shares']:.3f}" == "459.512"
    assert all(r["cp_shares"] > r["lm_shares"] and r["cp_loss"] < r["r"] and r["lm_loss"] < r["r"] * math.log(2) for r in rows.values()) and f"{math.log(2):.4f}" == "0.6931"
    r = 100.0
    assert (A.cp_state((1e-4, 0.0), r)["p_a"] - 0.5) / 1e-4 == pytest.approx(1 / (4 * r), rel=1e-3)
    for part in ("**an den Rändern viel tiefer** (bis zum Preis 0,99 braucht man 984,937 Anteile statt 459,512)", "**Verlustgrenze $r$**, die er nie erreicht", "$r \\ln 2 = 0{,}6931\\,r$", "$1/(4r)$ je Anteil"):
        assert part in README, part


def test_the_lvr_sentences():
    a = R.default_analysis()
    exact = V0 * (1 - math.exp(-0.8 ** 2 / 8))
    assert all(r["exact"] == pytest.approx(exact / V0, rel=1e-9) for r in a.steps_table) and f"{100 * exact / V0:.3f}" == "7.688"
    zs = [abs(r["sim"] - r["exact"]) / r["se"] for r in a.steps_table]
    assert max(zs) < 1.5 and f"{max(zs):.1f}" == "1.4"
    for steps in C.STEP_TABLE[:2]:
        run = A.simulate_pool(0.0, 0.8, 0.0, 1.0, steps, 500, 1)
        assert (run.profit > 0).all()
    for part in ("**unabhängig von der Zahl der Schritte**", "die Simulation liegt höchstens 1,4 Standardfehler daneben", "**Kein simulierter Pfad hat negativen LVR.**", "$V_0(1 - e^{-\\sigma^2 T/8}) = 7{,}688\\ \\%$"):
        assert part in README, part


def test_the_drift_sentences():
    a = R.default_analysis()
    rows = {round(r["mu"], 2): r for r in a.drift_table}
    assert set(rows) == {-0.32, 0.0, 0.32, 0.64}
    assert abs(rows[0.0]["il_exact"] + rows[0.0]["lvr_exact"]) < 1e-12 and f"{100 * rows[0.0]['lvr_exact']:.2f}" == "7.69"
    assert f"{100 * rows[0.64]['il_exact']:.2f}" == "-17.70" and f"{100 * rows[0.64]['lvr_exact']:.2f}" == "9.11"
    assert all(abs(rows[m]["il_exact"] + rows[m]["lvr_exact"]) > 1e-3 for m in (-0.32, 0.32, 0.64))
    zs = [abs(r["lvr_sim"] - r["lvr_exact"]) / r["lvr_se"] for r in a.drift_table] + [abs(r["il_sim"] - r["il_exact"]) / r["il_se"] for r in a.drift_table]
    assert len(zs) == 8 and max(zs) <= 1.0
    for part in ("**Nur ohne Drift sind beide im Betrag gleich** (7,69 %)", "(−17,70 % bei $\\mu = 0{,}64$)", "(9,11 %)", "Alle acht Simulationswerte der LVR- und Impermanent-Loss-Spalten liegen höchstens einen Standardfehler vom exakten Wert."):
        assert part in README, part


def test_the_fee_sentences():
    a = R.default_analysis()
    rows = {r["fee"]: r for r in a.fee_table}
    mins = [r["min_profit"] for r in a.fee_table]
    assert all(m > 0 for m in mins) and (f"{100 * min(mins):.1f}", f"{100 * max(mins):.1f}") == ("2.6", "3.6")                  # nur Kleinstes und Größtes: die Zwischenwerte sind nicht im README
    assert f"{100 * rows[0.01]['rel']:.1f}" == "72.3" and f"{rows[0.003]['break_even']:.1f}" == "23.1" and rows[0.0]["break_even"] is None
    assert [r["profit"] for r in a.fee_table] == sorted((r["profit"] for r in a.fee_table), reverse=True)
    fine = ev.fee_table(0.8, 0.0, 2000)
    assert 0.45 < next(r for r in fine if r["fee"] == 0.01)["rel"] < 0.6 and next(r for r in fine if r["fee"] == 0.01)["rel"] < rows[0.01]["rel"]
    for part in ("**Mit reinem Arbitrage-Fluss bleibt der Pool in jedem simulierten Pfad hinter dem Rebalancieren zurück**", "(der kleinste Pfad verliert zwischen 2,6 % und 3,6 % von $V_0$)", "bei 1 % auf 72,3 % des gebührenfreien Werts",
                 "bei 0,3 % Gebühr etwa das 23-Fache des Anfangswerts im Jahr", "bei 2 000 Schritten im Jahr und 1 % Gebühr auf etwa die Hälfte des gebührenfreien Werts"):
        assert part in README, part


def test_the_disclaimer_is_in_the_readme_and_the_app():
    assert C.DISCLAIMER.replace("**", "").startswith("Modell, keine Marktdaten, keine Anlageberatung.")
    assert "> " + C.DISCLAIMER in README
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "Modell, keine Marktdaten, keine Anlageberatung" in app and "C.DISCLAIMER" in app


def test_no_negative_zero_in_the_readme_tables():
    assert "-0,0000" not in README and "-0,00 " not in README


def test_the_readme_sentences_about_the_verification():
    for part in ("**Bisektion über die Tauschfunktion**", "**explizite Kette „Sätze prägen, dann tauschen“**", "**numerische Integration**", "**dieselben Zufallszüge Schritt für Schritt**", "Aufzählung aller $2^n$ Pfade eines Binomialbaums"):
        assert part in README or part.replace("**", "") in README, part
    assert np.isfinite(A.exact_lvr_gbm(V0, 0.8, 0.0, 1.0, 365))
