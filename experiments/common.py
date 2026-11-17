"""Shared helpers for experiments: parallel runs, summary statistics, CSV output."""
from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.engine import DEFAULT_PHYSICS, EnvironmentConfig, PhysicsParams
from src.simulate import SimConfig, ramp_schedule, run

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"

# Defaults used by every experiment unless stated otherwise.
YEARS = 300
RAMP_YEARS = 100
TAIL_YEARS = 50          # summary = mean over the last 50 years
YEAR_TICKS = 50
FOOD = 10.0
SIM = SimConfig(world_area=2048.0)


@dataclass(frozen=True)
class Job:
    label: str
    temp_c: float
    seasonality: float
    temp_swing_c: float
    food: float
    seed: int
    years: int = YEARS
    physics: PhysicsParams = DEFAULT_PHYSICS


def run_job(job: Job) -> dict:
    env = EnvironmentConfig(
        name=job.label,
        gravity_g=1.0,
        ambient_temp_c=job.temp_c,
        food_abundance=job.food,
        seasonality_amp=job.seasonality,
        year_ticks=YEAR_TICKS,
        temp_season_amp_c=job.temp_swing_c,
    )
    result = run(
        env,
        years=job.years,
        seed=job.seed,
        cfg=SIM,
        physics=job.physics,
        env_schedule=ramp_schedule(env, ramp_years=RAMP_YEARS),
    )
    tail = result.history[-TAIL_YEARS:] if not result.extinct else []
    row = {
        "label": job.label,
        "temp_c": job.temp_c,
        "seasonality": job.seasonality,
        "temp_swing_c": job.temp_swing_c,
        "food": job.food,
        "seed": job.seed,
        "extinct": result.extinct,
        "median_mass": float(np.mean([h.median_mass for h in tail])) if tail else np.nan,
        "p95_mass": float(np.mean([h.p95_mass for h in tail])) if tail else np.nan,
        "insulation": float(np.mean([h.mean_insulation for h in tail])) if tail else np.nan,
        "locomotion": float(np.mean([h.mean_locomotion for h in tail])) if tail else np.nan,
        "population": float(np.mean([h.n for h in tail])) if tail else 0.0,
    }
    row.update({f"phys_{k}": v for k, v in asdict(job.physics).items()})
    return row


def run_many(jobs: list[Job]) -> list[dict]:
    workers = max(1, (os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_job, jobs, chunksize=1))


def write_csv(rows: list[dict], name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def group_stats(rows: list[dict], key: str, value: str = "median_mass"):
    """Mean and standard deviation of ``value`` for each distinct ``key`` (log10 scale)."""
    keys = sorted({r[key] for r in rows})
    means, sds, n_ok = [], [], []
    for k in keys:
        vals = np.array([r[value] for r in rows if r[key] == k and not r["extinct"]])
        logs = np.log10(vals) if len(vals) else np.array([np.nan])
        means.append(float(np.mean(logs)))
        sds.append(float(np.std(logs)))
        n_ok.append(len(vals))
    return np.array(keys), np.array(means), np.array(sds), np.array(n_ok)
