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

    # resume a FAILED run in place from its latest resumable checkpoint
    # (make train-bg CFG=<cfg> SEED=<seed> RESUME=<run_dir>)
    python scripts/train.py --config <cfg> --seed <seed> --resume <run_dir> --prepare
    python scripts/train.py --config <cfg> --seed <seed> --resume <run_dir> --run-dir <run_dir>

A finished run of the same ``run_group`` and seed makes ``--prepare`` refuse (exit 3); a
failed one is kept and the new run takes the next ``<seed>_r<k>`` suffix. With
``--resume``, ``--prepare`` validates the resume (exit 3 with the reason if refused: a
live, finished or unreconciled run, a non-resumable or non-latest checkpoint, or a config
or seed that is not the run's) and prints the run directory.
"""

import argparse
import sys
from pathlib import Path

from rld.rl.config import RUNS_ROOT, load_train_config
from rld.rl.resume import ResumeError
from rld.rl.train import RunDirExistsError, prepare_resume, prepare_run_dir, train


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="configs/rl/<name>.yaml")
    parser.add_argument("--seed", type=int, required=True, help="training seed")
    parser.add_argument("--run-dir", type=Path, default=None, help="reserved run directory")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT, help="root of all runs")
    parser.add_argument(
        "--prepare", action="store_true", help="reserve a run directory, print it, exit"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="continue a failed run: its run directory or its latest checkpoint directory",
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    if args.resume is not None:
        try:
            run_dir, ckpt = prepare_resume(cfg, args.seed, args.resume)
        except ResumeError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 3
        if args.prepare:
            print(run_dir)
            return 0
        if args.run_dir is not None and args.run_dir.resolve() != run_dir:
            print(f"refused: --run-dir {args.run_dir} is not {run_dir}", file=sys.stderr)
            return 3
        print(f"resuming {cfg.method} ({cfg.algo}) seed {args.seed} in {run_dir}", flush=True)
        train(cfg, args.seed, run_dir, resume_from=ckpt)
        return 0
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
