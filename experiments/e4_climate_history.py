"""E4: body size through a changing climate, plus the counterfactual.

Years 0-400: gradual cooling and rising seasonality (a stylised Cenozoic trend).
Years 400-800: the same path in reverse (warming). If body size simply
tracks climate, it must rise and then fall again; if the model had size
growth hard-wired, it would not come back down.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import FIGURES, FOOD, SIM, YEAR_TICKS, write_csv
from src.engine import EnvironmentConfig
from src.simulate import run

HALF = 400
WARM = dict(temp=25.0, amp=0.0)
COLD = dict(temp=2.0, amp=0.9)
SEEDS = range(3)


def climate(year: int) -> tuple[float, float]:
    f = year / HALF if year <= HALF else max(0.0, 2.0 - year / HALF)
    temp = WARM["temp"] + f * (COLD["temp"] - WARM["temp"])
    amp = WARM["amp"] + f * (COLD["amp"] - WARM["amp"])
    return temp, amp


BASE = EnvironmentConfig("E4", 1.0, WARM["temp"], FOOD, 0.0, YEAR_TICKS)


def schedule(year: int) -> EnvironmentConfig:
    temp, amp = climate(year)
    return replace(BASE, ambient_temp_c=temp, seasonality_amp=amp, temp_season_amp_c=10.0 * amp)


def one(seed: int):
    r = run(BASE, years=2 * HALF, seed=seed, cfg=SIM, env_schedule=schedule)
    return [
        {"seed": seed, "year": h.year, "temp_c": climate(h.year)[0], "seasonality": climate(h.year)[1],
         "n": h.n, "median_mass": h.median_mass, "p95_mass": h.p95_mass, "insulation": h.mean_insulation}
        for h in r.history
    ], r.extinct


def main():
    with ProcessPoolExecutor() as pool:
        outs = list(pool.map(one, SEEDS))
    rows = [row for rs, _ in outs for row in rs]
    write_csv(rows, "e4_climate_history")

    years = np.arange(2 * HALF)
    med = np.array([[r["median_mass"] for r in rs] for rs, _ in outs])
    p95 = np.array([[r["p95_mass"] for r in rs] for rs, _ in outs])

    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.plot(years, np.exp(np.log(med).mean(0)), color="#1f6feb", label="median mass (geo-mean of 3 seeds)")
    ax.plot(years, np.exp(np.log(p95).mean(0)), color="#f78166", label="95th percentile mass")
    ax.set_yscale("log")
    ax.set_xlabel("Simulated year")
    ax.set_ylabel("Body mass (kg)")
    ax.axvline(HALF, color="#8b949e", ls="--", lw=1)
    ax2 = ax.twinx()
    ax2.plot(years, [climate(y)[0] for y in years], color="#8b949e", lw=1, alpha=0.7)
    ax2.set_ylabel("Mean temperature (°C)", color="#8b949e")
    ax.legend(loc="upper left")
    ax.set_title("E4 · cooling + seasonality (years 0–400), then the reverse (400–800)")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig3_climate_history.png", dpi=150)

    gm = np.exp(np.log(med).mean(0))
    print(f"E4 median mass: start={gm[:20].mean():.2f} kg, peak(cold, yrs 380-420)={gm[380:420].mean():.2f} kg, "
          f"end(warm again)={gm[-20:].mean():.2f} kg; extinct seeds={sum(e for _, e in outs)}")
    return {"start": float(gm[:20].mean()), "peak": float(gm[380:420].mean()), "end": float(gm[-20:].mean())}


if __name__ == "__main__":
    main()
