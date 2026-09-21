"""Measure PyBullet environment throughput and write ``results/env_throughput.csv``.

Phase 0, Gate 0. Argparse wrapper only; the measurement lives in
:mod:`rld.bench.throughput`.

Usage::

    python scripts/env_throughput.py --out results/env_throughput.csv --steps 6000

Rows are the cross product of {DummyVecEnv at 1 worker, SubprocVecEnv at 1/8/16 workers}
and {rpm, vel} action spaces. ``steps`` is the timed environment-step count per row; each
row also discards a warmup. Runs headless (PyBullet DIRECT) and needs no GPU.
"""

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

from rld.bench.throughput import (
    ACTION_POOL_SIZE,
    BUDGET_STEPS,
    DEFAULT_WORKERS,
    VecClsName,
    environment_provenance,
    measure_throughput,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "env_throughput.csv",
        help="destination CSV (default: results/env_throughput.csv)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=6000,
        help="timed environment steps per row (default: 6000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(DEFAULT_WORKERS),
        help=f"SubprocVecEnv worker counts (default: {list(DEFAULT_WORKERS)})",
    )
    parser.add_argument(
        "--acts",
        nargs="+",
        choices=["rpm", "vel"],
        default=["rpm", "vel"],
        help="action spaces to measure (default: both)",
    )
    parser.add_argument("--seed", type=int, default=0, help="base seed (default: 0)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < ACTION_POOL_SIZE:
        print(f"note: --steps {args.steps} is below the {ACTION_POOL_SIZE}-batch action pool")

    provenance = environment_provenance(REPO_ROOT)
    rows: list[dict[str, object]] = []
    plan: list[tuple[VecClsName, int]] = [("dummy", 1)]
    plan += [("subproc", n) for n in args.workers]

    for act in args.acts:
        for vec_cls, n_envs in plan:
            result = measure_throughput(
                n_envs=n_envs,
                act=act,
                total_steps=args.steps,
                vec_cls=vec_cls,
                seed=args.seed,
            )
            print(
                f"{vec_cls:>7s} n={n_envs:<3d} act={act:<4s} "
                f"{result.steps_per_s:9.1f} steps/s  "
                f"({result.steps_per_s_per_env:8.1f}/env, "
                f"{result.est_hours_budget:6.2f} h for {BUDGET_STEPS / 1e6:.0f}M)"
            )
            rows.append({**asdict(result), **provenance})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
