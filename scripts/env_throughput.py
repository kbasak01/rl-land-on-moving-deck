"""Measure PyBullet environment throughput and write ``results/env_throughput.csv``.

Phase 0, Gate 0. Argparse wrapper only; the measurement lives in
:mod:`rld.bench.throughput`.

Usage::

    python scripts/env_throughput.py --out results/env_throughput.csv --steps 6000
    python scripts/env_throughput.py --env landing \
        --out results/env_throughput_landing.csv --steps 6000

Rows are the cross product of {DummyVecEnv at 1 worker, SubprocVecEnv at 1/8/16 workers}
and the selected action spaces. ``steps`` is the timed environment-step count per row; each
row also discards a warmup. Runs headless (PyBullet DIRECT) and needs no GPU.

``--env hover`` is Phase 0's ``HoverAviary`` measurement and a **ceiling**: one body, no
deck, no constraint, no motion bridge. ``--env landing`` is the Phase 2
``DeckLandingAviary`` with the deck plate driven every physics step and the deck-motion
bridge evaluated once per reset, and it is the number P3-D1's training budget is sized from
(plan Phase 2 note (e)). It writes a separate CSV so the committed Gate 0 artifact is left
exactly as it was; ``landing`` has one shared action space, so ``--acts`` is forced to
``vel`` for it.
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
    parser.add_argument(
        "--env",
        choices=["hover", "landing"],
        default="hover",
        help="environment to measure (default: hover, the Phase 0 ceiling)",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["random", "hold"],
        default=["random"],
        help=(
            "action streams to measure (default: random, Phase 0's contract). 'hold' sends "
            "a constant zero action, which in the landing env hovers to the time limit and "
            "so measures step cost rather than reset cost"
        ),
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
    # The landing environment has exactly one action space by construction (P2-D2), so
    # asking it for `rpm` would measure a space no method in this project can emit.
    acts = ["vel"] if args.env == "landing" else args.acts

    for policy in args.policies:
        for act in acts:
            for vec_cls, n_envs in plan:
                result = measure_throughput(
                    n_envs=n_envs,
                    act=act,
                    total_steps=args.steps,
                    vec_cls=vec_cls,
                    seed=args.seed,
                    env=args.env,
                    policy=policy,
                )
                print(
                    f"{args.env:>7s} {policy:>6s} {vec_cls:>7s} n={n_envs:<3d} act={act:<4s} "
                    f"{result.steps_per_s:9.1f} steps/s  "
                    f"({result.steps_per_s_per_env:8.1f}/env, "
                    f"{result.steps_per_episode:6.1f} steps/ep, "
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
