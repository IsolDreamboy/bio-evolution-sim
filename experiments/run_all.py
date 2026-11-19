"""Reproduce every result and figure.

    python -m experiments.run_all                                # paper outputs (seeds 0-4; E4 seeds 0-2)
    python -m experiments.run_all --seed-start 100 --num-seeds 20 # out-of-sample validation, separate files

Validation runs write files suffixed with _seeds<start>-<end> and never overwrite the paper outputs.
"""
import argparse

from experiments import e1_e2_drivers, e3_robustness, e4_climate_history


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-start", type=int, default=None, help="first seed (default: paper seeds)")
    ap.add_argument("--num-seeds", type=int, default=5, help="seeds per condition for E1/E2")
    ap.add_argument("--e4-seeds", type=int, default=None, help="seeds for E4 (default: 3 paper / 5 validation)")
    ap.add_argument("--skip-e3", action="store_true", help="skip the 324-run robustness sweep")
    args = ap.parse_args()

    if args.seed_start is None:
        e1_e2_drivers.main()
        if not args.skip_e3:
            e3_robustness.main()
        e4_climate_history.main()
        return

    s0, n = args.seed_start, args.num_seeds
    tag = f"_seeds{s0}-{s0 + n - 1}"
    e1_e2_drivers.main(range(s0, s0 + n), tag)
    if not args.skip_e3:
        e3_robustness.main()
    n4 = args.e4_seeds or 5
    e4_climate_history.main(range(s0, s0 + n4), f"_seeds{s0}-{s0 + n4 - 1}")


if __name__ == "__main__":
    main()
