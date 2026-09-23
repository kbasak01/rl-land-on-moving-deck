"""Train one PPO/SAC run (Phase 5). Argparse wrapper around :mod:`rld.rl.train`.

Never run this in the Claude Code foreground shell: launch it through ``make train-bg``
(``nohup``, log in the run directory) or ``make sweep`` / ``make tune`` (the detached
scheduler), and watch ``artifacts/runs/**/status.json`` with ``/sweep-status``.

Usage::

    # reserve a fresh run directory and print its path (the LAST stdout line; pybullet
    # prints a build banner on import)
    python scripts/train.py --config configs/rl/ppo_smoke.yaml --seed 0 --prepare

    # train into a reserved directory (what make train-bg and the scheduler run)
    python scripts/train.py --config configs/rl/ppo_smoke.yaml --seed 0 --run-dir <dir>

A finished run of the same ``run_group`` and seed makes ``--prepare`` refuse (exit 3); a
failed one is kept and the new run takes the next ``<seed>_r<k>`` suffix.
"""

import argparse
import sys
from pathlib import Path

from rld.rl.config import RUNS_ROOT, load_train_config
from rld.rl.train import RunDirExistsError, prepare_run_dir, train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="configs/rl/<name>.yaml")
    parser.add_argument("--seed", type=int, required=True, help="training seed")
    parser.add_argument("--run-dir", type=Path, default=None, help="reserved run directory")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT, help="root of all runs")
    parser.add_argument(
        "--prepare", action="store_true", help="reserve a run directory, print it, exit"
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    if args.prepare:
        try:
            run_dir = prepare_run_dir(cfg, args.seed, args.runs_root)
        except RunDirExistsError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 3
        print(run_dir)
        return 0
    run_dir = args.run_dir or prepare_run_dir(cfg, args.seed, args.runs_root)
    print(f"training {cfg.method} ({cfg.algo}) seed {args.seed} -> {run_dir}", flush=True)
    train(cfg, args.seed, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
