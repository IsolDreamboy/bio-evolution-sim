"""Main evolutionary loop.

Each tick: food regrows -> agents move -> agents compete for food in their
cell -> energy balance -> background mortality -> starvation -> reproduction
with mutation. Body size, insulation and locomotion are never told what to do:
they change only through who survives and reproduces.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.engine import (
    DEFAULT_PHYSICS,
    EnvironmentConfig,
    PhysicsParams,
    intake_capacity,
    step_bioenergetics, 
)
from src.speciation import mutate_traits
from src.world import FoodGrid


@dataclass(frozen=True)
class SimConfig:
    grid_size: int = 1                # 1 = well-mixed world (see world.FoodGrid)
    world_area: float = 1024.0
    n_founders: int = 300
    founder_mass: float = 2.0
    founder_insulation: float = 0.3
    founder_locomotion: float = 1.0
    feed_coef: float = 3.0            # intake demand = feed_coef * L^0.6 * M^0.75
    regen_rate: float = 0.3
    repro_threshold: float = 20.0     # reproduce when energy >= threshold * M
    mortality_base: float = 0.004     # per-tick background death prob for a 1 kg animal
    mortality_exp: float = -0.25      # lifespan ~ M^0.25 (Calder 1984)
    mutation_rate: float = 0.15
    dispersal: int = 4                # offspring land up to this many cells away
    max_agents: int = 20000           # safety cap against runaway growth


@dataclass
class Population:
    mass: np.ndarray
    insulation: np.ndarray
    locomotion: np.ndarray
    energy: np.ndarray
    x: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.mass)

    def keep(self, mask: np.ndarray) -> None:
        for name in ("mass", "insulation", "locomotion", "energy", "x", "y"):
            setattr(self, name, getattr(self, name)[mask])

    def append(self, other: "Population") -> None:
        for name in ("mass", "insulation", "locomotion", "energy", "x", "y"):
            setattr(self, name, np.concatenate([getattr(self, name), getattr(other, name)]))


@dataclass
class YearRecord:
    year: int
    n: int
    median_mass: float
    p95_mass: float
    mean_insulation: float
    mean_locomotion: float


@dataclass
class SimResult:
    env: EnvironmentConfig
    seed: int
    history: list[YearRecord] = field(default_factory=list)
    final: Population | None = None
    extinct: bool = False


def found_population(cfg: SimConfig, rng: np.random.Generator) -> Population:
    n = cfg.n_founders
    mass = cfg.founder_mass * np.exp(rng.normal(0.0, 0.05, n))
    return Population(
        mass=mass,
        insulation=np.clip(cfg.founder_insulation + rng.normal(0, 0.02, n), 0, 1),
        locomotion=np.clip(cfg.founder_locomotion + rng.normal(0, 0.02, n), 0.1, 2.5),
        energy=10.0 * mass,
        x=rng.integers(0, cfg.grid_size, n),
        y=rng.integers(0, cfg.grid_size, n),
    )


def tick(
    pop: Population,
    world: FoodGrid,
    env: EnvironmentConfig,
    t: int,
    cfg: SimConfig,
    physics: PhysicsParams,
    rng: np.random.Generator,
) -> None:
    n = len(pop)
    world.regenerate(t)

    # Movement: each agent looks at one random neighbouring cell and moves there
    # if it has more food. More locomotion -> more chances to look (torus world).
    looks = rng.random(n) < 0.5 + 0.5 * np.clip(pop.locomotion / 2.5, 0.0, 1.0)
    cx = (pop.x + rng.integers(-1, 2, n)) % cfg.grid_size
    cy = (pop.y + rng.integers(-1, 2, n)) % cfg.grid_size
    better = world.food[cx, cy] > world.food[pop.x, pop.y]
    go = looks & better
    pop.x = np.where(go, cx, pop.x)
    pop.y = np.where(go, cy, pop.y)

    # Competition for food in each cell.
    demand = cfg.feed_coef * intake_capacity(pop.mass, pop.locomotion)
    intake = world.forage(pop.x, pop.y, demand, pop.mass)

    pop.energy = step_bioenergetics(
        pop.mass, pop.insulation, pop.locomotion, pop.energy, t, env,
        intake=intake, physics=physics,
    )

    # Background mortality (bigger animals live longer) + starvation.
    p_death = cfg.mortality_base * pop.mass ** cfg.mortality_exp
    alive = (pop.energy > 0.0) & (rng.random(n) >= p_death)
    pop.keep(alive)

    # Reproduction: parent splits its energy with one mutated offspring.
    parents = pop.energy >= cfg.repro_threshold * pop.mass
    if parents.any() and len(pop) < cfg.max_agents:
        idx = np.flatnonzero(parents)[: cfg.max_agents - len(pop)]
        m, i, l = mutate_traits(
            pop.mass[idx], pop.insulation[idx], pop.locomotion[idx], rng, cfg.mutation_rate
        )
        # The offspring gets an endowment proportional to ITS OWN mass, paid by the parent.
        # (Copying the parent's energy would secretly reward downward mass mutations.)
        endowment = 0.5 * cfg.repro_threshold * m
        pop.energy[idx] -= endowment
        # Natal dispersal: offspring settle up to `dispersal` cells away from the parent.
        d = cfg.dispersal
        ox = (pop.x[idx] + rng.integers(-d, d + 1, len(idx))) % cfg.grid_size
        oy = (pop.y[idx] + rng.integers(-d, d + 1, len(idx))) % cfg.grid_size
        pop.append(Population(m, i, l, endowment, ox, oy))


def run(
    env: EnvironmentConfig,
    years: int = 150,
    seed: int = 0,
    cfg: SimConfig = SimConfig(),
    physics: PhysicsParams = DEFAULT_PHYSICS,
    env_schedule=None,
) -> SimResult:
    """Run one simulation. ``env_schedule(year) -> EnvironmentConfig`` allows
    climate to change over time (e.g. gradual cooling)."""
    rng = np.random.default_rng(seed)
    pop = found_population(cfg, rng)
    world = FoodGrid(cfg.grid_size, env, cfg.regen_rate, cfg.world_area)
    result = SimResult(env=env, seed=seed)

    t = 0
    for year in range(years):
        current = env_schedule(year) if env_schedule else env
        world.env = current
        for _ in range(current.year_ticks):
            tick(pop, world, current, t, cfg, physics, rng)
            t += 1
            if len(pop) == 0:
                result.extinct = True
                result.final = pop
                return result
        result.history.append(
            YearRecord(
                year=year,
                n=len(pop),
                median_mass=float(np.median(pop.mass)),
                p95_mass=float(np.percentile(pop.mass, 95)),
                mean_insulation=float(pop.insulation.mean()),
                mean_locomotion=float(pop.locomotion.mean()),
            )
        )
    result.final = pop
    return result


def ramp_schedule(
    target: EnvironmentConfig,
    start_temp_c: float = 20.0,
    start_amp: float = 0.0,
    ramp_years: int = 60,
    start_food: float | None = None,
    start_grass: float | None = None,
    start_temp_amp_c: float = 0.0,
):
    """Climate moves linearly from a benign start to ``target`` over ``ramp_years``
    and then stays there. Lets founders adapt instead of dying in year one."""
    from dataclasses import replace

    f0 = target.food_abundance if start_food is None else start_food
    g0 = target.grass_abundance if start_grass is None else start_grass

    def lerp(a: float, b: float, f: float) -> float:
        return a + f * (b - a)

    def schedule(year: int) -> EnvironmentConfig:
        f = min(1.0, year / ramp_years) if ramp_years > 0 else 1.0
        return replace(
            target,
            ambient_temp_c=lerp(start_temp_c, target.ambient_temp_c, f),
            seasonality_amp=lerp(start_amp, target.seasonality_amp, f),
            temp_season_amp_c=lerp(start_temp_amp_c, target.temp_season_amp_c, f),
            food_abundance=lerp(f0, target.food_abundance, f),
            grass_abundance=lerp(g0, target.grass_abundance, f),
        )

    return schedule
