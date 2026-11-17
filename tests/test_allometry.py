import numpy as np
from src.engine import EnvironmentConfig, step_bioenergetics


def simulate_surviving_lineage_fitness(env: EnvironmentConfig, years: int = 4) -> np.ndarray:
    """
    Avalia o sucesso reprodutivo apenas de linhagens que sobrevivem a todos os ciclos anuais.
    Evita o viés de estrategistas-r que reproduzem no verão e morrem no primeiro inverno.
    """
    masses = np.array([1.0, 10.0, 100.0, 1000.0, 5000.0])
    n = len(masses)
    insulation = np.full(n, 0.5)
    locomotion = np.full(n, 1.0)
    energy = 15.0 * masses
    repro_threshold = 22.0 * masses
    offspring_count = np.zeros(n)
    alive = np.ones(n, dtype=bool)

    for tick in range(env.year_ticks * years):
        energy = step_bioenergetics(masses, insulation, locomotion, energy, tick, env)
        alive &= energy > 0.0
        can_repro = alive & (energy >= repro_threshold)
        offspring_count[can_repro] += 1
        energy[can_repro] *= 0.5

    return np.where(alive, offspring_count, -1.0)


def test_optimal_mass_shifts_with_climate():
    masses = np.array([1.0, 10.0, 100.0, 1000.0, 5000.0])

    hot_stable = EnvironmentConfig("Hot Stable", gravity_g=1.0, ambient_temp_c=28.0, food_abundance=2.0, seasonality_amp=0.0)
    mild_mid   = EnvironmentConfig("Mild",       gravity_g=1.0, ambient_temp_c=16.0, food_abundance=1.9, seasonality_amp=0.4)
    cold_seas  = EnvironmentConfig("Cold Seas.", gravity_g=1.0, ambient_temp_c=2.0,  food_abundance=2.4, seasonality_amp=0.95)

    hot_fit  = simulate_surviving_lineage_fitness(hot_stable)
    mild_fit = simulate_surviving_lineage_fitness(mild_mid)
    cold_fit = simulate_surviving_lineage_fitness(cold_seas)

    assert np.max(hot_fit) >= 0 and np.max(mild_fit) >= 0 and np.max(cold_fit) >= 0, "Extinção total inesperada."

    best_hot  = masses[np.argmax(hot_fit)]
    best_mild = masses[np.argmax(mild_fit)]
    best_cold = masses[np.argmax(cold_fit)]

    assert best_hot < best_mild < best_cold, (
        f"Esperado nanismo no calor e gigantismo no frio, obtido: hot={best_hot}, mild={best_mild}, cold={best_cold}"
    )
