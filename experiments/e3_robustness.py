"""E3: robustness. Does the direction of each effect survive changes to the
calibrated constants and to food supply?

For every combination we run four climates:
    warm (28 C) vs cool (8 C), no seasonality        -> temperature effect
    stable vs seasonal (0.8), both at 15 C           -> seasonality effect
and count how often cool > warm and seasonal > stable.
Also produces a heat map of evolved body size over food x seasonality.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import FIGURES, Job, run_many, write_csv
from src.engine import DEFAULT_PHYSICS

YEARS = 200
K_THERMAL = [0.12, 0.15, 0.18]
C_HEAT = [0.6, 0.8, 1.0]
C_COLD = [0.3, 0.5, 0.7]
FOODS = [7.0, 10.0, 13.0]
SEED = 0


def climates(physics, food, tag):
    return [
        Job(f"{tag}|warm", 28.0, 0.0, 0.0, food, SEED, YEARS, physics),
        Job(f"{tag}|cool", 8.0, 0.0, 0.0, food, SEED, YEARS, physics),
        Job(f"{tag}|stable", 15.0, 0.0, 0.0, food, SEED, YEARS, physics),
        Job(f"{tag}|seasonal", 15.0, 0.8, 8.0, food, SEED, YEARS, physics),
    ]


def sweep():
    jobs = []
    for k, ch, cc, food in itertools.product(K_THERMAL, C_HEAT, C_COLD, FOODS):
        phys = replace(DEFAULT_PHYSICS, k_thermal=k, c_heat=ch, c_cold=cc)
        jobs += climates(phys, food, f"{k}/{ch}/{cc}/{food}")
    rows = run_many(jobs)
    write_csv(rows, "e3_robustness")

    by_tag = {}
    for r in rows:
        tag, clim = r["label"].split("|")
        by_tag.setdefault(tag, {})[clim] = r
    temp_ok = season_ok = both_ok = extinct = 0
    for d in by_tag.values():
        if any(d[c]["extinct"] for c in d):
            extinct += 1
            continue
        t = d["cool"]["median_mass"] > d["warm"]["median_mass"]
        s = d["seasonal"]["median_mass"] > d["stable"]["median_mass"]
        temp_ok += t
        season_ok += s
        both_ok += t and s
    n = len(by_tag)
    return {"combinations": n, "temperature_effect": temp_ok, "seasonality_effect": season_ok,
            "both": both_ok, "with_extinction": extinct}


def heatmap():
    foods = [5.0, 7.5, 10.0, 12.5, 15.0]
    seasons = [0.0, 0.2, 0.4, 0.6, 0.8]
    jobs = [Job("E3map", 10.0, a, a * 10.0, f, s, YEARS) for f in foods for a in seasons for s in (0, 1)]
    rows = run_many(jobs)
    write_csv(rows, "e3_heatmap")
    grid = np.full((len(foods), len(seasons)), np.nan)
    for i, f in enumerate(foods):
        for j, a in enumerate(seasons):
            vals = [r["median_mass"] for r in rows if r["food"] == f and r["seasonality"] == a and not r["extinct"]]
            if vals:
                grid[i, j] = np.mean(np.log10(vals))
    fig, ax = plt.subplots(figsize=(6, 4.6))
    im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(seasons)), [f"{a:.1f}" for a in seasons])
    ax.set_yticks(range(len(foods)), [f"{f:g}" for f in foods])
    ax.set_xlabel("Seasonality amplitude")
    ax.set_ylabel("Food supply per unit area")
    for i in range(len(foods)):
        for j in range(len(seasons)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{10 ** grid[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, label="log10 evolved median mass (kg)")
    ax.set_title("E3 · evolved body size (kg) at 10 °C, 2 seeds")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig2_heatmap.png", dpi=150)


def main():
    summary = sweep()
    heatmap()
    n = summary["combinations"]
    print(f"E3 robustness over {n} parameter combinations (k_thermal x c_heat x c_cold x food):")
    print(f"  cooler -> bigger:      {summary['temperature_effect']}/{n} ({summary['temperature_effect']/n:.0%})")
    print(f"  seasonal -> bigger:    {summary['seasonality_effect']}/{n} ({summary['seasonality_effect']/n:.0%})")
    print(f"  both:                  {summary['both']}/{n} ({summary['both']/n:.0%})")
    print(f"  combos with extinction:{summary['with_extinction']}")
    return summary


if __name__ == "__main__":
    main()
