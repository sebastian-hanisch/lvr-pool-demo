"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset zeigt, was sein Hilfetext sagt."""

import pytest

import kpm_algorithm as A
import kpm_constants as C
import kpm_evaluation as ev
import kpm_presets as P

DEFAULT = "Standardfall (Voreinstellung)"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        assert C.S_MIN <= p["s"] <= C.S_MAX and p["fee"] in C.FEE_CHOICES and C.P_MIN <= p["p"] <= C.P_MAX, name
        assert C.SIGMA_MIN <= p["sigma"] <= C.SIGMA_MAX and C.MU_MIN <= p["mu"] <= C.MU_MAX and p["steps"] in C.STEP_CHOICES and 0 <= p["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_every_preset_differs_from_the_default_in_at_least_two_settings_and_from_every_other():
    default = C.PRESETS[DEFAULT]
    for name, p in C.PRESETS.items():
        if name != DEFAULT:
            assert sum(p[k] != default[k] for k in default) >= 2, name
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_the_presets_show_what_their_help_says():
    a = A.buy(C.POOL_X, C.POOL_Y, C.PRESETS["Handbeispiel A: 10 % der Reserve kaufen"]["s"], C.PRESETS["Handbeispiel A: 10 % der Reserve kaufen"]["fee"])
    assert (f"{a.dy:.2f}", f"{a.avg_price:.4f}", f"{a.price_after:.4f}") == ("111.11", "1.1111", "1.2346")
    b = ev.bridge_row(C.PRESETS["Handbeispiel B: Prognosemarkt auf 80 %"]["p"])
    assert (f"{b['cp_shares']:.2f}", f"{b['cp_cost']:.2f}", f"{b['lm_shares']:.2f}", f"{b['lm_cost']:.2f}") == ("150.00", "100.00", "138.63", "91.63")
    half = C.PRESETS["Halbe Reserve: der Preis vervierfacht sich"]
    t = A.buy(C.POOL_X, C.POOL_Y, half["s"], half["fee"])
    assert (f"{t.avg_ratio:.4f}", f"{t.after_ratio:.4f}") == ("2.0000", "4.0000")
    assert C.PRESETS["Hohe Volatilität: der Pool verliert mehr"]["sigma"] > C.DEFAULT_SIGMA and C.PRESETS["Steigender Kurs: Impermanent Loss wächst"]["mu"] > C.DEFAULT_MU
    fine = C.PRESETS["Hohe Gebühr, feine Schritte"]
    assert fine["fee"] == 0.01 and fine["steps"] == 2000


@pytest.mark.parametrize("name", [n for n in C.PRESETS if n != DEFAULT])
def test_every_preset_runs_and_its_message_holds(name):
    p = C.PRESETS[name]
    a = ev.analyse(ev.Settings(**p))
    assert all(r["sim"] > 0 for r in a.steps_table) and all(r["profit"] > 0 for r in a.fee_table)
    if name.startswith("Steigender Kurs"):
        top = max(a.drift_table, key=lambda r: r["mu"])
        assert abs(top["il_exact"]) > 1.5 * top["lvr_exact"]
    if name.startswith("Hohe Volatilität"):
        low = ev.analyse(ev.Settings())
        assert a.exact_share > low.exact_share
    if name.startswith("Hohe Gebühr"):
        row = next(r for r in a.fee_table if r["fee"] == 0.01)
        assert 0.45 < row["rel"] < 0.6 and row["min_profit"] > 0
