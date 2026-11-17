"""Vectorised bioenergetics (Structure of Arrays).

Every function here operates on NumPy arrays where index ``i`` is the same
individual in every array. No Python loops over individuals.

Scaling exponents and their sources:
    basal metabolism   ~ M^0.75   (Kleiber 1932)
    intake capacity    ~ M^0.75   (Peters 1983)
    locomotion cost    ~ M^0.70   (Taylor, Heglund & Maloiy 1982)
    heat exchange area ~ M^(2/3)  (square-cube law)
    fat storage        ~ M^1.0    -> fasting endurance ~ M^0.25 (Lindstedt & Boyce 1985)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvironmentConfig:
    name: str
    gravity_g: float          # 1.0 = Earth
    ambient_temp_c: float     # mean ambient temperature (deg C)
    food_abundance: float     # base food supply (per-agent tests) / carrying capacity per cell (simulation)
    seasonality_amp: float    # amplitude of the seasonal food cycle (>= 1 means food stops growing in winter)
    year_ticks: int = 100     # ticks per seasonal cycle
    temp_season_amp_c: float = 0.0  # winter-summer temperature swing (deg C); winter = cold AND scarce
    grass_abundance: float = 0.0    # carrying capacity of low-quality fibrous food per cell (open habitat)


@dataclass(frozen=True)
class PhysicsParams:
    """All calibratable constants in one place, so sweeps can vary them."""
    basal_coef: float = 1.2       # basal cost = basal_coef * M^0.75
    move_coef: float = 0.4        # locomotion cost = move_coef * g * M^0.70 * L^1.5
    k_thermal: float = 0.15       # heat-dissipation conductance
    c_heat: float = 0.8           # cost per unit of excess heat
    c_cold: float = 0.5           # cost per unit of heat deficit
    insulation_eff: float = 0.85  # max fraction of heat loss blocked by insulation
    fur_half_mass: float = 1.0    # body mass (kg) at which fur reaches half its max effect
    size_limits_fur: bool = True  # small bodies cannot carry thick fur (Scholander et al. 1950)
    storage_coef: float = 25.0    # max energy store = storage_coef * M
    core_temp_c: float = 37.0


DEFAULT_PHYSICS = PhysicsParams()


def season_factor(tick: int, env: EnvironmentConfig) -> float:
    """Seasonal multiplier on food, clipped at zero (a real winter if amp >= 1)."""
    return max(0.0, 1.0 + env.seasonality_amp * np.sin(2.0 * np.pi * tick / env.year_ticks))


def ambient_temp(tick: int, env: EnvironmentConfig) -> float:
    """Temperature this tick: colder in the same half of the year when food is scarce."""
    return env.ambient_temp_c + env.temp_season_amp_c * np.sin(2.0 * np.pi * tick / env.year_ticks)


def energy_costs(
    mass: np.ndarray,
    insulation: np.ndarray,
    locomotion: np.ndarray,
    env: EnvironmentConfig,
    physics: PhysicsParams = DEFAULT_PHYSICS,
    temp_c: float | None = None,
) -> np.ndarray:
    """Total energetic cost per tick for every individual."""
    if temp_c is None:
        temp_c = env.ambient_temp_c
    basal = physics.basal_coef * mass ** 0.75
    move = physics.move_coef * env.gravity_g * mass ** 0.70 * locomotion ** 1.5

    # Continuous heat balance: production ~M^0.75 vs dissipation through skin ~M^(2/3)
    heat_prod = basal
    effective_insulation = insulation
    if physics.size_limits_fur:
        # Fur thickness is limited by body size, so a 100 g animal cannot insulate like a bison.
        r = mass ** (1.0 / 3.0)
        effective_insulation = insulation * r / (r + physics.fur_half_mass ** (1.0 / 3.0))
    dissip = (
        physics.k_thermal
        * mass ** (2.0 / 3.0)
        * (1.0 - physics.insulation_eff * effective_insulation)
        * (physics.core_temp_c - temp_c)
    )
    heat_stress = np.maximum(0.0, heat_prod - dissip) * physics.c_heat
    cold_loss = np.maximum(0.0, dissip - heat_prod) * physics.c_cold
    return basal + move + heat_stress + cold_loss


def intake_capacity(mass: np.ndarray, locomotion: np.ndarray) -> np.ndarray:
    """Maximum intake per unit of food availability (~M^0.75, wider foraging with L^0.6)."""
    return locomotion ** 0.6 * mass ** 0.75


def grass_digestibility(mass: np.ndarray, half_mass: float = 50.0) -> np.ndarray:
    """Fraction of energy extracted from fibrous food.

    Gut capacity ~M^1 while metabolism ~M^0.75, so retention time ~M^0.25:
    big herbivores can live on poor forage (Jarman-Bell principle;
    Demment & Van Soest 1985).
    """
    r = mass ** 0.25
    return r / (r + half_mass ** 0.25)


def step_bioenergetics(
    mass: np.ndarray,
    insulation: np.ndarray,
    locomotion: np.ndarray,
    energy: np.ndarray,
    tick: int,
    env: EnvironmentConfig,
    core_temp_c: float = 37.0,
    k_thermal: float = 0.15,
    c_heat: float = 0.8,
    c_cold: float = 0.5,
    intake: np.ndarray | None = None,
    physics: PhysicsParams | None = None,
) -> np.ndarray:
    """Return updated energy after one tick.

    If ``intake`` is given (e.g. food actually obtained on the grid), it is used
    directly; otherwise intake is ``food_abundance * season * capacity``.
    The loose keyword constants are kept for backward compatibility; pass
    ``physics`` to control every constant.
    """
    if physics is None:
        physics = PhysicsParams(
            k_thermal=k_thermal, c_heat=c_heat, c_cold=c_cold, core_temp_c=core_temp_c
        )
    if intake is None:
        intake = env.food_abundance * season_factor(tick, env) * intake_capacity(mass, locomotion)

    costs = energy_costs(mass, insulation, locomotion, env, physics, ambient_temp(tick, env))
    return np.minimum(physics.storage_coef * mass, energy + intake - costs)
