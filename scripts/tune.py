"""Run the pre-registered hyperparameter search for one method (Phase 5). Argparse wrapper.

``make tune CFG=configs/rl/tune_ppo.yaml MAX_WORKERS=34``:

1. validates the search config (<= 20 trials, weight-only reward targets, every trial
   config parses) and writes ``<results_dir>/configs/trial_XX.yaml``;
2. queues one run per trial (the fixed ``trial_seed``) on the detached scheduler of
   :mod:`rld.rl.scheduler`, and returns immediately;
3. when every trial is terminal, the scheduler scores them on the tune pool and writes
   ``<results_dir>/trials.csv`` and ``selection.json`` (:func:`rld.rl.tuning.collect_trials`).

``--collect`` re-runs step 3 by hand. ``--dry-run`` prints the trial points only.
"""

import argparse
import sys
from pathlib import Path

from rld.rl.config import RUNS_ROOT
from rld.rl.scheduler import SWEEPS_DIR, detach_scheduler, new_sweep
from rld.rl.tuning import collect_trials, load_tune_config, materialize_trials, trial_points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="configs/rl/tune_<m>.yaml")
    parser.add_argument("--max-workers", type=int, default=34, help="global worker-slot cap")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--poll-s", type=float, default=15.0, help="poll interval, host seconds")
    parser.add_argument("--collect", action="store_true", help="score finished trials only")
    parser.add_argument("--dry-run", action="store_true", help="print the trial points only")
    args = parser.parse_args()

    cfg = load_tune_config(args.config)
    if args.dry_run:
        for index, point in enumerate(trial_points(cfg)):
            print(index, point or "(base config)")
        return 0
    if args.collect:
        rows = collect_trials(cfg, args.runs_root)
        print(f"wrote {cfg.results_dir / 'trials.csv'} ({len(rows)} trials)")
        return 0
    paths = materialize_trials(cfg)
    jobs = [(str(p), cfg.trial_seed) for p in paths]
    source = cfg.source
    sweeps_dir = args.runs_root / SWEEPS_DIR.name
    sweep = new_sweep(
        "tune",
        jobs,
        args.max_workers,
        on_complete=f"tune_collect:{source}",
        sweeps_dir=sweeps_dir,
        label=cfg.method,
    )
    pid = detach_scheduler(sweep, args.runs_root, args.poll_s)
    print(f"tune {cfg.method}: {len(jobs)} trials x {cfg.trial_steps} steps, seed {cfg.trial_seed}")
    print(f"  trial configs  {cfg.results_dir / 'configs'}")
    print(f"  sweep file     {sweep}")
    print(f"  scheduler pid  {pid}")
    print(f"  scheduler log  {sweep.with_suffix('.log')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
