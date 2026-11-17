import numpy as np
from src.speciation import init_founder_population, mutate_traits, detect_emergent_species


def test_founder_population_is_a_single_species():
    pop = init_founder_population(n_agents=500, seed=42)
    result = detect_emergent_species(pop["mass"], pop["insulation"], pop["locomotion"])
    assert result["n_species"] == 1


def test_mutation_respects_trait_bounds():
    rng = np.random.default_rng(0)
    pop = init_founder_population(n_agents=500, seed=1)
    m, i, l = pop["mass"], pop["insulation"], pop["locomotion"]
    for _ in range(300):
        m, i, l = mutate_traits(m, i, l, rng, mutation_rate=1.0)
    assert m.min() >= 0.1 and m.max() <= 10000.0
    assert i.min() >= 0.0 and i.max() <= 1.0
    assert l.min() >= 0.1 and l.max() <= 2.5
