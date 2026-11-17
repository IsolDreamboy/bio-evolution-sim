import numpy as np


from src.engine import EnvironmentConfig, energy_costs, grass_digestibility, season_factor
from src.simulate import Population, SimConfig, run
from src.world import FoodGrid

ENV = EnvironmentConfig("test", 1.0, 20.0, 10.0, 0.0, 50)


def test_food_grid_conserves_food():
    world = FoodGrid(1, ENV, regen_rate=0.3, area=100.0)
    before = world.food.sum()
    x = np.zeros(5, dtype=int)
    y = np.zeros(5, dtype=int)
    demand = np.full(5, 10.0)
    intake = world.consume(x, y, demand)
    assert np.isclose(intake.sum() + world.food.sum(), before)


def test_food_is_split_when_scarce():
    world = FoodGrid(1, ENV, regen_rate=0.3, area=1.0)  # only 10 units of food
    intake = world.consume(np.zeros(4, int), np.zeros(4, int), np.array([10.0, 10.0, 20.0, 0.0]))
    assert np.isclose(intake.sum(), 10.0)
    assert np.isclose(intake[2], 2 * intake[0])  # proportional to demand


def test_hard_winter_stops_food_growth():
    env = EnvironmentConfig("winter", 1.0, 0.0, 10.0, 1.5, 100)
    assert season_factor(75, env) == 0.0  # three quarters through the year = midwinter


def test_insulation_only_helps_in_the_cold():
    m = np.array([50.0, 50.0])
    ins = np.array([0.0, 1.0])
    loc = np.ones(2)
    cold = EnvironmentConfig("cold", 1.0, -5.0, 10.0, 0.0)
    hot = EnvironmentConfig("hot", 1.0, 35.0, 10.0, 0.0)
    cost_cold = energy_costs(m, ins, loc, cold)
    cost_hot = energy_costs(m, ins, loc, hot)
    assert cost_cold[1] < cost_cold[0]   # fur saves energy in the cold
    assert cost_hot[1] > cost_hot[0]     # fur costs energy in the heat


def test_big_bodies_digest_fibre_better():
    d = grass_digestibility(np.array([1.0, 100.0, 1000.0]))
    assert d[0] < d[1] < d[2] < 1.0


def test_offspring_endowment_scales_with_own_mass():
    # A tiny world with abundant food: every agent reproduces quickly.
    cfg = SimConfig(n_founders=50, world_area=4096.0, mortality_base=0.0)
    result = run(ENV, years=2, seed=1, cfg=cfg)
    pop = result.final
    assert not result.extinct
    assert np.all(pop.energy <= 25.0 * pop.mass + 1e-9)


def test_run_is_reproducible():
    cfg = SimConfig(n_founders=100, world_area=512.0)
    a = run(ENV, years=5, seed=7, cfg=cfg)
    b = run(ENV, years=5, seed=7, cfg=cfg)
    assert [h.median_mass for h in a.history] == [h.median_mass for h in b.history]
