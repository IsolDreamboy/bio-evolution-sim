"""Out-of-sample check: the main result must hold for seeds never used during development.

Slow (~30 s). Skip with:  pytest -m "not slow"
"""
import numpy as np
import pytest
from scipy.stats import spearmanr

from experiments.common import FOOD, Job, run_job

OUT_OF_SAMPLE_SEEDS = (100, 101, 102)
TEMPS = (0.0, 15.0, 30.0)


@pytest.mark.slow
def test_cold_selects_larger_bodies_on_unseen_seeds():
    rows = [run_job(Job("inv", t, 0.0, 0.0, FOOD, s)) for t in TEMPS for s in OUT_OF_SAMPLE_SEEDS]
    assert not any(r["extinct"] for r in rows)

    geo = [np.exp(np.mean([np.log(r["median_mass"]) for r in rows if r["temp_c"] == t])) for t in TEMPS]
    assert geo[0] > geo[1] > geo[2], f"expected mass to fall with temperature, got {geo}"

    rho, _ = spearmanr([r["temp_c"] for r in rows], [r["median_mass"] for r in rows])
    assert rho < -0.8, f"rho={rho:.2f}"
