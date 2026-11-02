"""Zeilen der README-Tabellen, gerechnet über dieselben Funktionen wie in der App (kpm_evaluation) und in der Rundung der App.

`tests/test_claims.py` prüft jede dieser Zeilen gegen die README; `python tools/readme_rows.py` druckt sie zum Einfügen.
Alle Zahlen sind deterministisch (feste Seeds, nur exp, log, sqrt und Summen: keine lineare Algebra, kein Optimierer)."""

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kpm_constants as C  # noqa: E402
import kpm_evaluation as ev  # noqa: E402

FEE = C.DEFAULT_FEE


@lru_cache(maxsize=None)
def default_analysis():
    return ev.analyse(ev.Settings())


def hand_rows() -> list[str]:
    return ev.hand_rows()


def impact_rows() -> list[str]:
    return ev.impact_rows(ev.impact_table(FEE))


def split_rows() -> list[str]:
    return ev.split_rows(ev.split_table(FEE))


def round_trip_rows() -> list[str]:
    return ev.round_trip_rows(ev.round_trip_table(FEE))


def bridge_rows() -> list[str]:
    return ev.bridge_rows(ev.bridge_table())


def steps_rows() -> list[str]:
    return ev.steps_rows(default_analysis().steps_table)


def drift_rows() -> list[str]:
    return ev.drift_rows(default_analysis().drift_table)


def fee_rows() -> list[str]:
    return ev.fee_rows(default_analysis().fee_table)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for title, header, rows in (("Handbeispiele", ev.HAND_HEADER, hand_rows()), ("Preisauswirkung", ev.impact_header(FEE), impact_rows()), ("Aufteilen", ev.split_header(FEE), split_rows()),
                                ("Hin und zurück", ev.round_trip_header(FEE), round_trip_rows()), ("Brücke", ev.BRIDGE_HEADER, bridge_rows()), ("LVR nach Schritten", ev.STEPS_HEADER, steps_rows()),
                                ("Drift", ev.DRIFT_HEADER, drift_rows()), ("Gebühren", ev.FEE_HEADER, fee_rows())):
        print(f"\n{title}"); print("\n".join(header + rows))
    a = default_analysis()
    print("\nStandardfall:", {k: round(v, 4) for k, v in a.trade.__dict__.items()})
    print("exakt 365 Schritte:", a.exact_share, "Pfad Ende:", {k: round(v[-1], 2) for k, v in a.path.items()})
    print("Min Pfadgewinn:", [round(r["min_profit"], 4) for r in a.fee_table])
