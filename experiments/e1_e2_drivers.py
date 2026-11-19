"""E1: temperature sweep (no seasonality). E2: seasonality sweep (fixed mean temperature).

One driver changes at a time; everything else is held fixed.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from experiments.common import FIGURES, FOOD, Job, group_stats, run_many, write_csv

SEEDS = range(5)
TEMPS = np.linspace(0.0, 33.0, 12)
SEASONS = np.linspace(0.0, 0.9, 10)
E2_TEMP = 15.0
SWING_PER_AMP = 10.0  # a seasonality of 0.9 also means a +/-9 C winter-summer swing


def e1_jobs(seeds=SEEDS):
    return [Job("E1", float(t), 0.0, 0.0, FOOD, s) for t in TEMPS for s in seeds]


def e2_jobs(seeds=SEEDS):
    return [
        Job("E2", E2_TEMP, float(a), float(a * SWING_PER_AMP), FOOD, s)
        for a in SEASONS
        for s in seeds
    ]


def spearman(rows, key):
    ok = [r for r in rows if not r["extinct"]]
    rho, p = spearmanr([r[key] for r in ok], [r["median_mass"] for r in ok])
    return float(rho), float(p)


def plot(ax, rows, key, xlabel, title):
    x, m, sd, _ = group_stats(rows, key)
    ax.errorbar(x, 10 ** m, yerr=[10 ** m - 10 ** (m - sd), 10 ** (m + sd) - 10 ** m],
                fmt="o-", color="#1f6feb", ecolor="#8b949e", capsize=3)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Evolved median body mass (kg)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")


def main(seeds=SEEDS, tag: str = ""):
    """tag="" writes the paper outputs; any other tag writes separate validation files."""
    seeds = list(seeds)
    rows1 = run_many(e1_jobs(seeds))
    rows2 = run_many(e2_jobs(seeds))
    write_csv(rows1, f"e1_temperature{tag}")
    write_csv(rows2, f"e2_seasonality{tag}")

    rho1, p1 = spearman(rows1, "temp_c")
    rho2, p2 = spearman(rows2, "seasonality")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot(axes[0], rows1, "temp_c", "Mean temperature (°C)",
         f"E1 · temperature only (ρ = {rho1:.2f})")
    plot(axes[1], rows2, "seasonality", "Seasonality amplitude (food & temperature)",
         f"E2 · seasonality only at {E2_TEMP:.0f} °C (ρ = {rho2:.2f})")
    fig.suptitle(f"Body size evolves with climate — no climate rule in the code ({len(seeds)} seeds, mean ± sd)")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig1_drivers{tag}.png", dpi=150)

    print(f"E1 temperature: Spearman rho={rho1:.3f} p={p1:.2g}; extinct runs={sum(r['extinct'] for r in rows1)}")
    print(f"E2 seasonality: Spearman rho={rho2:.3f} p={p2:.2g}; extinct runs={sum(r['extinct'] for r in rows2)}")
    return {"e1_rho": rho1, "e1_p": p1, "e2_rho": rho2, "e2_p": p2}


if __name__ == "__main__":
    main()
