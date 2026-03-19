"""
lab_sim.py – Surrogate model for a Suzuki-Miyaura coupling reaction.

The simulator uses a mixture of three Gaussians over the normalised
(temperature, catalyst_loading, solvent_polarity) space, plus 2 % Gaussian
noise, to approximate a realistic but non-trivial yield surface.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Parameter bounds
# ---------------------------------------------------------------------------
TEMP_MIN, TEMP_MAX = 30.0, 150.0          # °C
CAT_MIN, CAT_MAX   = 0.1, 5.0            # mol%
POL_MIN, POL_MAX   = 0.1, 1.0            # dimensionless

# ---------------------------------------------------------------------------
# Hidden objective – mixture of 3 Gaussians
# ---------------------------------------------------------------------------
# Each Gaussian: (centre_T, centre_C, centre_P, sigma, weight)
_GAUSSIANS = [
    # True optimum – moderate temp, moderate catalyst, mid-polarity
    {"mu": np.array([80.0,  2.5, 0.55]), "sigma": 0.20, "w": 85.0},
    # Local max – high-temp pathway
    {"mu": np.array([120.0, 1.0, 0.80]), "sigma": 0.25, "w": 60.0},
    # Trap – low-temp / high-catalyst
    {"mu": np.array([50.0,  4.0, 0.30]), "sigma": 0.15, "w": 45.0},
]

_NOISE_FRAC = 0.02          # 2 % of the max weight as noise σ
_MAX_WEIGHT = max(g["w"] for g in _GAUSSIANS)
_RNG = np.random.default_rng()


def _normalise(temp: float, catalyst_pct: float, polarity: float) -> np.ndarray:
    """Min-max normalise raw inputs to [0, 1]."""
    return np.array([
        (temp - TEMP_MIN)         / (TEMP_MAX - TEMP_MIN),
        (catalyst_pct - CAT_MIN)  / (CAT_MAX  - CAT_MIN),
        (polarity - POL_MIN)      / (POL_MAX  - POL_MIN),
    ])


def _normalise_centre(mu_raw: np.ndarray) -> np.ndarray:
    """Normalise a Gaussian centre from raw parameter space."""
    return np.array([
        (mu_raw[0] - TEMP_MIN) / (TEMP_MAX - TEMP_MIN),
        (mu_raw[1] - CAT_MIN)  / (CAT_MAX  - CAT_MIN),
        (mu_raw[2] - POL_MIN)  / (POL_MAX  - POL_MIN),
    ])


def simulate_reaction(
    temp: float,
    catalyst_pct: float,
    polarity: float,
) -> float:
    """
    Simulate a Suzuki-Miyaura coupling and return the % yield.

    Parameters
    ----------
    temp : float
        Reaction temperature in °C  (30 – 150).
    catalyst_pct : float
        Catalyst loading in mol%    (0.1 – 5.0).
    polarity : float
        Solvent polarity index      (0.1 – 1.0).

    Returns
    -------
    float
        Simulated isolated yield clipped to [0, 100].

    Raises
    ------
    ValueError
        If any parameter is outside the allowed range.
    """
    # --- validation ---
    if not (TEMP_MIN <= temp <= TEMP_MAX):
        raise ValueError(
            f"Temperature {temp} °C is outside [{TEMP_MIN}, {TEMP_MAX}]."
        )
    if not (CAT_MIN <= catalyst_pct <= CAT_MAX):
        raise ValueError(
            f"Catalyst loading {catalyst_pct} mol% is outside "
            f"[{CAT_MIN}, {CAT_MAX}]."
        )
    if not (POL_MIN <= polarity <= POL_MAX):
        raise ValueError(
            f"Polarity {polarity} is outside [{POL_MIN}, {POL_MAX}]."
        )

    x = _normalise(temp, catalyst_pct, polarity)

    # --- evaluate mixture of Gaussians ---
    y = 0.0
    for g in _GAUSSIANS:
        mu_n = _normalise_centre(g["mu"])
        dist_sq = np.sum((x - mu_n) ** 2)
        y += g["w"] * np.exp(-dist_sq / (2.0 * g["sigma"] ** 2))

    # --- add 2 % Gaussian noise ---
    noise = _RNG.normal(0.0, _NOISE_FRAC * _MAX_WEIGHT)
    y += noise

    return float(np.clip(y, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Evaluate near the true optimum
    for _ in range(5):
        y = simulate_reaction(80.0, 2.5, 0.55)
        print(f"Yield at (80, 2.5, 0.55): {y:.2f}%")

    # Boundary check
    try:
        simulate_reaction(200, 2.5, 0.5)
    except ValueError as e:
        print(f"Caught expected error: {e}")
