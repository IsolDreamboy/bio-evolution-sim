"""Reproduce every result and figure: python -m experiments.run_all  (~10 min on 2 cores)."""
from experiments import e1_e2_drivers, e3_robustness, e4_climate_history

if __name__ == "__main__":
    e1_e2_drivers.main()
    e3_robustness.main()
    e4_climate_history.main()
