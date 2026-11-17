import itertools
import numpy as np
from src.engine import EnvironmentConfig, step_bioenergetics


def evaluate_lineages(env: EnvironmentConfig, k_thermal: float, c_heat: float, c_cold: float) -> float:
    masses = np.array([1.0, 10.0, 100.0, 1000.0, 5000.0])
    n = len(masses)
    insulation = np.full(n, 0.5)
    locomotion = np.full(n, 1.0)
    energy = 15.0 * masses
    repro_threshold = 22.0 * masses
    offspring = np.zeros(n)
    alive = np.ones(n, dtype=bool)

    for tick in range(env.year_ticks * 4):
        energy = step_bioenergetics(
            masses, insulation, locomotion, energy, tick, env,
            k_thermal=k_thermal, c_heat=c_heat, c_cold=c_cold,
        )
        alive &= energy > 0.0
        can_repro = alive & (energy >= repro_threshold)
        offspring[can_repro] += 1
        energy[can_repro] *= 0.5

    surviving_fitness = np.where(alive, offspring, -1.0)
    if np.max(surviving_fitness) < 0:
        return np.nan
    return float(masses[np.argmax(surviving_fitness)])


def run_sweep() -> None:
    hot  = EnvironmentConfig("Hot",  gravity_g=1.0, ambient_temp_c=28.0, food_abundance=2.0, seasonality_amp=0.0)
    mild = EnvironmentConfig("Mild", gravity_g=1.0, ambient_temp_c=16.0, food_abundance=1.9, seasonality_amp=0.4)
    cold = EnvironmentConfig("Cold", gravity_g=1.0, ambient_temp_c=2.0,  food_abundance=2.4, seasonality_amp=0.95)

    grid = list(itertools.product(
        [0.12, 0.15, 0.18],
        [0.6, 0.8, 1.0],
        [0.3, 0.5, 0.7],
    ))

    valid_shifts = 0
    for k_th, c_h, c_c in grid:
        m_hot  = evaluate_lineages(hot,  k_th, c_h, c_c)
        m_mild = evaluate_lineages(mild, k_th, c_h, c_c)
        m_cold = evaluate_lineages(cold, k_th, c_h, c_c)
        if not np.isnan([m_hot, m_mild, m_cold]).any() and (m_hot < m_mild < m_cold):
            valid_shifts += 1

    print(f"Robustez Física: {valid_shifts}/{len(grid)} cenários ({valid_shifts/len(grid):.1%})")


if __name__ == "__main__":
    run_sweep()
