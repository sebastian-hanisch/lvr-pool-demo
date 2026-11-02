"""Verdikte (reine Funktionen in kpm_evaluation.py) an handgesetzten Werten: Schwellen, Zweige und jede Zahl im Text."""

import kpm_evaluation as ev


def test_verdict_trade():
    kind, text = ev.verdict_trade(1.1111, 1.2346, 111.11, 0.1, 0.0, 0.0)
    assert kind == "info" and "10 % der Reserve" in text and "zahlt 111,11" in text and "1,1111-fache" in text and "1,2346-fache" in text and "Ohne Gebühr" in text and "Gebühr (" not in text
    kind, text = ev.verdict_trade(1.1114, 1.2351, 111.45, 0.1, 0.003, 0.33)
    assert kind == "info" and "0,33 Gebühr (0,30 % der Zahlung)" in text and "Ohne Gebühr" not in text
    assert ev.verdict_trade(1.9, 3.99, 10.0, 0.5, 0.0, 0.0)[0] == "info" and ev.verdict_trade(2.0, 4.0, 10.0, 0.5, 0.0, 0.0)[0] == "warning"          # genau auf der Schwelle: Warnung


def _bridge(cp, lm, p=0.9):
    return {"p": p, "cp_shares": cp, "cp_cost": 200.0, "cp_loss": 66.6667, "lm_shares": lm, "lm_cost": 160.94, "lm_loss": 58.7787, "r": 100.0}


def test_verdict_bridge():
    kind, text = ev.verdict_bridge(_bridge(266.67, 219.72))
    assert kind == "info" and "Preis 0,90" in text and "266,67 Anteile (200,00 Kosten)" in text and "mehr Anteile als" in text and "219,72 Anteile, 160,94 Kosten" in text and "0,667 r (Grenze r)" in text and "0,588 r (Grenze 0,6931 r)" in text
    assert "weniger Anteile als" in ev.verdict_bridge(_bridge(200.0, 219.72))[1]
    assert "gleich viele Anteile wie" in ev.verdict_bridge(_bridge(219.72, 219.72))[1]


def test_verdict_lvr():
    kind, text = ev.verdict_lvr(0.0769, 0.0770, 0.0003, 365, 0.8)
    assert kind == "success" and "365 Schritten" in text and "7,69 %" in text and "7,70 % ± 0,03 %" in text and "0,3 Standardfehler" in text and "σ²/8 = 8,00 % des Poolwerts je Jahr" in text
    kind, text = ev.verdict_lvr(0.0769, 0.0800, 0.0003, 52, 0.8)
    assert kind == "warning" and "10,3 Standardfehler" in text
    assert ev.verdict_lvr(0.0769, 0.0778, 0.0003, 52, 0.8)[0] == "success"                                     # genau 3 Standardfehler zählen noch als gut
    assert ev.verdict_lvr(0.05, 0.05, 0.0, 12, 0.5)[0] == "success"                                            # ohne Streuung kein Teilen durch null


def _il(mu, il, lvr):
    return {"mu": mu, "il_exact": il, "lvr_exact": lvr}


def test_verdict_il():
    rows = [_il(-0.32, -0.0764, 0.0712), _il(0.0, -0.0769, 0.0769), _il(0.32, -0.1053, 0.0835), _il(0.64, -0.1770, 0.0911)]
    kind, text = ev.verdict_il(rows)
    assert kind == "info" and "Ohne Drift sind Impermanent Loss (-7,69 %) und LVR (7,69 %) im Betrag gleich" in text and "Mit Drift μ = 0,64" in text and "-17,70 %" in text and "9,11 %" in text
    same = [_il(0.0, -0.0769, 0.0769), _il(0.1, -0.0769, 0.0769)]
    kind, text = ev.verdict_il(same)
    assert kind == "info" and "unterscheiden sich" in text and "kaum" in text


def _fee(f, rel, profit, fees, trade, min_profit, be):
    return {"fee": f, "rel": rel, "profit": profit, "fees": fees, "trade_share": trade, "min_profit": min_profit, "break_even": be}


def test_verdict_fee():
    rows = [_fee(0.0, 1.0, 0.0769, 0.0, 1.0, 0.03, None), _fee(0.003, 0.9, 0.0692, 0.008, 0.943, 0.0325, 23.1), _fee(0.01, 0.723, 0.0556, 0.022, 0.814, 0.022, 5.6)]
    kind, text = ev.verdict_fee(rows, 0.003)
    assert kind == "warning" and "Gebühr 0,30 %" in text and "90,0 % des gebührenfreien (6,92 % des Anfangswerts)" in text and "0,80 %" in text and "94,3 %" in text
    assert "In jedem simulierten Pfad bleibt der Pool hinter dem Rebalancieren zurück (kleinster Pfad 3,25 %)" in text and "23,1-mal" in text
    kind, text = ev.verdict_fee(rows, 0.0)
    assert kind == "info" and "Ohne Gebühr verliert der Pool 7,69 %" in text and "keine Einnahmen" in text
    bad = [rows[0], _fee(0.003, 0.9, 0.0692, 0.008, 0.943, -0.001, 23.1)]
    kind, text = ev.verdict_fee(bad, 0.003)
    assert kind == "info" and "Nicht in jedem simulierten Pfad" in text and "In jedem simulierten Pfad" not in text
    assert "Gebühr 1,00 %" in ev.verdict_fee(rows, 0.5)[1]                                                    # unbekannte Gebühr: letzte Zeile
