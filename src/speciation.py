import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def init_founder_population(n_agents: int = 500, seed: int = 42) -> dict[str, np.ndarray]:
    """
    Inicializa uma população fundadora homogênea (ancestral basal generalista ~2 kg).
    Nenhuma espécie pré-definida existe no tick 0.
    """
    rng = np.random.default_rng(seed)
    mass = np.full(n_agents, 2.0) * rng.normal(1.0, 0.01, size=n_agents)
    insulation = np.clip(np.full(n_agents, 0.3) + rng.normal(0.0, 0.01, size=n_agents), 0.0, 1.0)
    locomotion = np.clip(np.full(n_agents, 1.0) + rng.normal(0.0, 0.01, size=n_agents), 0.1, 2.0)
    energy = 15.0 * mass

    return {
        "mass": mass,
        "insulation": insulation,
        "locomotion": locomotion,
        "energy": energy,
    }


def mutate_traits(
    mass: np.ndarray,
    insulation: np.ndarray,
    locomotion: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aplica mutação estocástica vetorizada respeitando os limites físicos dos traços."""
    n = len(mass)
    mask_m = rng.random(n) < mutation_rate
    mask_i = rng.random(n) < mutation_rate
    mask_l = rng.random(n) < mutation_rate

    # Massa evolui em escala log-normal para preservar proporções relativas
    new_mass = np.where(mask_m, mass * np.exp(rng.normal(0.0, 0.08, size=n)), mass)
    new_mass = np.clip(new_mass, 0.1, 10000.0)

    new_insul = np.where(mask_i, insulation + rng.normal(0.0, 0.05, size=n), insulation)
    new_insul = np.clip(new_insul, 0.0, 1.0)

    new_locom = np.where(mask_l, locomotion + rng.normal(0.0, 0.08, size=n), locomotion)
    new_locom = np.clip(new_locom, 0.1, 2.5)

    return new_mass, new_insul, new_locom


def detect_emergent_species(
    mass: np.ndarray, insulation: np.ndarray, locomotion: np.ndarray
) -> dict:
    """Descobre espécies a posteriori agrupando clusters no espaço de traços fenotípicos."""
    if len(mass) < 15:
        return {"n_species": 0, "labels": np.array([]), "dominant_mass_median": np.nan}

    features = np.column_stack([np.log10(mass), insulation, locomotion])
    scaled = StandardScaler().fit_transform(features)

    clustering = DBSCAN(eps=0.45, min_samples=15).fit(scaled)
    labels = clustering.labels_
    unique_species = [lbl for lbl in np.unique(labels) if lbl != -1]

    dominant_mass = np.nan
    if unique_species:
        dominant_label = max(unique_species, key=lambda lbl: np.sum(labels == lbl))
        dominant_mass = float(np.median(mass[labels == dominant_label]))

    return {
        "n_species": len(unique_species),
        "labels": labels,
        "dominant_mass_median": dominant_mass,
    }
