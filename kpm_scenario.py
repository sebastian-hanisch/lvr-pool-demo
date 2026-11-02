"""Szenario: synthetische Referenzpreise (nur numpy). Zufall nur über den übergebenen Generator, nie über `hash()`.

Vehikel: Ein Pool auf einer Marktplattform tauscht ein riskantes Gut gegen Zahlungsmittel; der Referenzpreis außerhalb des Pools folgt einer geometrischen Brownschen Bewegung
dP = P (mu dt + sigma dW). Keine Kurse, keine Börsen, keine Token."""

import math

import numpy as np


def step_factor(sigma: float, mu: float, dt: float, z):
    """Faktor P_{i+1} / P_i bei Normalverteilung z: exp((mu - sigma^2 / 2) dt + sigma sqrt(dt) z); sein Erwartungswert ist exp(mu dt)."""
    return np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z)


def rng_for(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def check_params(sigma: float, T: float, steps: int, paths: int):
    if sigma < 0:
        raise ValueError("die Volatilität darf nicht negativ sein")
    if T <= 0 or steps < 1 or paths < 1:
        raise ValueError("Horizont, Schritte und Pfade müssen positiv sein")
