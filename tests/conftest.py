"""Gemeinsame Fixtures: die Handbeispiele, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand_a():
    """Pool 1000 / 1000 (k = 10^6, Preis 1): 100 Einheiten x kaufen kostet 1000 * 100 / 900 = 111,11; mittlerer Preis 1,1111, Preis danach 1,2346 (= 1/0,9^2); mit Gebühr 0,3 %: 111,45."""
    return {"x": 1000.0, "y": 1000.0, "dx": 100.0}


@pytest.fixture
def hand_b():
    """Konstantprodukt als Prognosemarkt, Reserve r = 100, Start p_A = 0,5, Preis auf 0,8: Reserven (a, b) = (100 sqrt(0,25), 100 sqrt(4)) = (50; 200), m = 200 - 100 = 100 Sätze geprägt,
    q_A = 100 + 100 - 50 = 150 Anteile A für 100; Verlust bei Eintritt von A: 150 - 100 = 50. LMSR (b = 100): 100 ln 4 = 138,63 Anteile für 100 ln 2,5 = 91,63."""
    return {"r": 100.0, "p": 0.8}


@pytest.fixture
def hand_c():
    """Ein Schritt von P = 1 auf P = 4, Pool 1000 / 1000: der Arbitrageur bringt den Pool auf (500; 2000); Poolwert 2 sqrt(10^6 * 4) = 4000, Halten 4 * 1000 + 1000 = 5000, LVR 1000,
    Impermanent Loss 2 * 2 / 5 - 1 = -20 %."""
    return {"x": 1000.0, "y": 1000.0, "P": 4.0}


@pytest.fixture
def hand_d():
    """Tagesvolatilität 5 %: LVR-Rate sigma^2 / 8 = 0,0025 / 8 = 3,125e-4, also 3,125 Basispunkte je Tag."""
    return {"sigma": 0.05}
