"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# Der Pool: feste Anfangsreserven (Preis 1, k = 10^6, Wert 2 000); die Regler betreffen Handel, Gebühr und Referenzpreis.
POOL_X, POOL_Y = 1000.0, 1000.0
POOL_K = POOL_X * POOL_Y
POOL_V0 = 2.0 * POOL_K ** 0.5
BRIDGE_R = 100.0                            # Anfangsreserve des Konstantprodukt-Prognosemarkts (jeweils r Anteile A und B)

# Regler. Komplexität am Maximum: Eine Simulation sind höchstens LVR_PATHS Pfade mal 2 000 Schritte (vektorisiert über die Pfade, je Schritt einige Rechenoperationen auf Feldern);
# die App rechnet je Analyse acht Läufe mit Pfadzahl LVR_PATHS (Gebühren, Schrittzahlen) und vier kurze für die Drift; Worst-Case-Test in tests/test_edge_cases.py.
S_MIN, S_MAX, DEFAULT_S = 0.01, 0.90, 0.05
FEE_CHOICES = (0.0, 0.0005, 0.003, 0.01)
DEFAULT_FEE = 0.003
P_MIN, P_MAX, DEFAULT_P = 0.55, 0.99, 0.90
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 0.1, 1.5, 0.8
MU_MIN, MU_MAX, DEFAULT_MU = -1.0, 1.0, 0.0
STEP_CHOICES = (12, 52, 365, 2000)
DEFAULT_STEPS = 365
SEED_MAX, DEFAULT_SEED = 999999, 8
HORIZON = 1.0                               # Jahre

# Zählungen und Simulationen über feste Zufallszüge (DEMO-PLAYBOOK §4: nie ein einzelner Seed)
LVR_PATHS, LVR_SEED = 5000, 3               # Pfade je Lauf (Gebühren, Schrittzahlen)
DRIFT_PATHS, DRIFT_STEPS, DRIFT_SEED = 5000, 100, 2
DRIFT_MULTIPLES = (-0.5, 0.0, 0.5, 1.0)     # Drift als Vielfaches von sigma^2
IMPACT_S = (0.001, 0.01, 0.1, 0.5)          # Handelsgrößen der Preisauswirkung
SPLITS = (1, 2, 10, 100)                    # Aufteilen eines Handels in so viele Stücke
BRIDGE_PS = (0.6, 0.8, 0.9, 0.99)           # Preise des Vergleichs Konstantprodukt gegen LMSR
RT_S = (0.01, 0.1, 0.5)                     # Handelsgrößen des Hin-und-zurück-Versuchs
STEP_TABLE = (12, 52, 365, 2000)            # Schrittzahlen der LVR-Tabelle
GAIN_TOL = 1e-9
IL_BAND = 1e-9                              # Toleranz für „Preis im Band, kein Handel“

COLOR_PRICE = "#2F6B65"
COLOR_BEFORE = "#9aa5a4"
COLOR_LOSS = "#c0392b"
COLOR_GAIN = "#4c78a8"
COLOR_BOUND = "#e08a2c"
COLOR_CP = "#2F6B65"
COLOR_LMSR = "#4c78a8"

DISCLAIMER = "**Modell, keine Marktdaten, keine Anlageberatung.** Alle Pfade und Zahlen sind synthetisch; die Demo erklärt einen Mechanismus und gibt keine Empfehlung für Käufe, Verkäufe oder Anlagen."

PRESETS = {
    "Standardfall (Voreinstellung)": {"s": DEFAULT_S, "fee": DEFAULT_FEE, "p": DEFAULT_P, "sigma": DEFAULT_SIGMA, "mu": DEFAULT_MU, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Handbeispiel A: 10 % der Reserve kaufen": {"s": 0.10, "fee": 0.0, "p": DEFAULT_P, "sigma": DEFAULT_SIGMA, "mu": DEFAULT_MU, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Handbeispiel B: Prognosemarkt auf 80 %": {"s": DEFAULT_S, "fee": 0.0, "p": 0.80, "sigma": DEFAULT_SIGMA, "mu": DEFAULT_MU, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Halbe Reserve: der Preis vervierfacht sich": {"s": 0.50, "fee": 0.0, "p": DEFAULT_P, "sigma": DEFAULT_SIGMA, "mu": DEFAULT_MU, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Hohe Volatilität: der Pool verliert mehr": {"s": DEFAULT_S, "fee": 0.0, "p": DEFAULT_P, "sigma": 1.2, "mu": DEFAULT_MU, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Steigender Kurs: Impermanent Loss wächst": {"s": DEFAULT_S, "fee": 0.0, "p": DEFAULT_P, "sigma": DEFAULT_SIGMA, "mu": 0.6, "steps": DEFAULT_STEPS, "seed": DEFAULT_SEED},
    "Hohe Gebühr, feine Schritte": {"s": DEFAULT_S, "fee": 0.01, "p": DEFAULT_P, "sigma": DEFAULT_SIGMA, "mu": DEFAULT_MU, "steps": 2000, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Ein Händler kauft 5 % der Reserve, Gebühr 0,3 %; der Referenzpreis folgt einer geometrischen Brownschen Bewegung mit Volatilität 0,8, täglich nachgezogen.",
    "Handbeispiel A: 10 % der Reserve kaufen": "Pool 1 000 / 1 000: 100 Einheiten kosten 111,11, der mittlere Preis ist 1,1111, der Preis danach 1,2346; ohne Gebühr ist der Pfad gleichgültig.",
    "Handbeispiel B: Prognosemarkt auf 80 %": "Konstantprodukt mit Reserve 100 als Prognosemarkt: 150 Anteile für 100 treiben den Preis auf 0,8, die LMSR braucht 138,63 Anteile für 91,63.",
    "Halbe Reserve: der Preis vervierfacht sich": "Wer die Hälfte der Reserve kauft, zahlt im Mittel das Doppelte und lässt den Preis auf das Vierfache steigen.",
    "Hohe Volatilität: der Pool verliert mehr": "Volatilität 1,2 statt 0,8: Die LVR-Rate wächst mit dem Quadrat der Volatilität (σ²/8 des Poolwerts).",
    "Steigender Kurs: Impermanent Loss wächst": "Drift 0,6: Der Impermanent Loss (Pool gegen Halten) wächst, der LVR deutlich weniger; ohne Drift sind beide gleich groß.",
    "Hohe Gebühr, feine Schritte": "Gebühr 1 % bei 2 000 Schritten im Jahr: Die Gebühren mindern den Arbitrage-Verlust auf etwa die Hälfte, bleiben aber hinter ihm zurück.",
}
