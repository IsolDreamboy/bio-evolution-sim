"""2D food field. Plants and invertebrates are not agents: they are the
regenerating resource that the vertebrate agents compete for."""
from __future__ import annotations

import numpy as np

from src.engine import EnvironmentConfig, grass_digestibility, season_factor


class FoodGrid:
    def __init__(self, size: int, env: EnvironmentConfig, regen_rate: float = 0.05, area: float = 1024.0):
        """``food_abundance`` is food per unit area; ``area`` is the world's total area.

        size=1 gives a well-mixed world (everyone competes for one pool). Finer grids
        add space, but a fixed cell size caps the intake of large animals unless
        home range grows with body size (planned for v2), so the experiments use size=1.
        """
        self.size = size
        self.env = env
        self.regen_rate = regen_rate
        self.cell_area = area / (size * size)
        self.capacity = env.food_abundance * self.cell_area
        self.food = np.full((size, size), self.capacity, dtype=float)
        self.grass_capacity = env.grass_abundance * self.cell_area
        self.grass = np.full((size, size), self.grass_capacity, dtype=float)

    def regenerate(self, tick: int) -> None:
        """Food relaxes toward a seasonal target: capacity * season.

        In a hard winter (season -> 0) standing food decays away (snow cover,
        dormant plants), so animals must live on their fat reserves."""
        self.capacity = self.env.food_abundance * self.cell_area
        self.grass_capacity = self.env.grass_abundance * self.cell_area
        s = season_factor(tick, self.env)
        self.food += self.regen_rate * (self.capacity * s - self.food)
        self.grass += self.regen_rate * (self.grass_capacity * s - self.grass)
        np.clip(self.food, 0.0, None, out=self.food)
        np.clip(self.grass, 0.0, None, out=self.grass)

    @staticmethod
    def _split(layer: np.ndarray, cell: np.ndarray, demand: np.ndarray, size: int):
        total = np.zeros(size * size)
        np.add.at(total, cell, demand)
        available = layer.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(total > 0, np.minimum(1.0, available / total), 0.0)
        eaten = demand * frac[cell]
        return eaten, (available - total * frac).reshape(size, size)

    def forage(self, x, y, demand, mass):
        """Eat high-quality food first; fill the remaining demand with grass,
        which yields energy only in proportion to digestibility(M)."""
        cell = x * self.size + y
        eaten_q, self.food = self._split(self.food, cell, demand, self.size)
        if self.grass_capacity <= 0:
            return eaten_q
        remaining = demand - eaten_q
        eaten_g, self.grass = self._split(self.grass, cell, remaining, self.size)
        return eaten_q + eaten_g * grass_digestibility(mass)

    def consume(self, x: np.ndarray, y: np.ndarray, demand: np.ndarray) -> np.ndarray:
        """Split each cell's food among the agents in it, proportionally to demand.

        Returns the intake of every agent. Fully vectorised via np.add.at.
        """
        cell = x * self.size + y
        total = np.zeros(self.size * self.size)
        np.add.at(total, cell, demand)
        available = self.food.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(total > 0, np.minimum(1.0, available / total), 0.0)
        intake = demand * frac[cell]
        eaten = total * frac
        self.food = (available - eaten).reshape(self.size, self.size)
        return intake
