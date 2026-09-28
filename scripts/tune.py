"""Run the pre-registered hyperparameter search for one method (Phase 5). Argparse wrapper.

``make tune CFG=configs/rl/tune_ppo.yaml MAX_WORKERS=34``:

1. validates the search config (<= 20 trials, weight-only reward targets, every trial
   config parses) and writes ``<results_dir>/configs/trial_XX.yaml`` (existing files must be
   byte-identical: a search is never changed after it started);
2. decides per trial what to do (:func:`rld.rl.tuning.plan_trials`): a trial with a
   ``done`` run is **skipped**; a trial whose latest run failed is **resumed** from its
   latest resumable checkpoint when the config says ``resume_failed: true`` (else re-trained
   fresh into ``<seed>_r<k>``); a trial never started is started. It **refuses** (exit 3)
   if a scheduler of this search is still alive, or a trial run is live or says
   ``running`` with a dead process (run ``python scripts/reconcile_runs.py`` first);
3. queues the remaining trials (the fixed ``trial_seed``) on the detached scheduler of
   :mod:`rld.rl.scheduler`, and returns immediately;
4. when every queued trial is terminal, the scheduler scores ALL trials on the tune pool,
   each from its latest ``done`` run, and writes ``<results_dir>/trials.csv`` and
   ``selection.json`` (:func:`rld.rl.tuning.collect_trials`).

So re-invoking ``make tune`` on the same config after an interruption finishes the search.
``--collect`` re-runs step 4 by hand. ``--dry-run`` prints the trial points only; ``--plan``
prints step 2's decision per trial without queueing anything.
"""

import argparse
import json
import sys
from pathlib import Path

from rld.rl.config import RUNS_ROOT
from rld.rl.procs import pid_alive
from rld.rl.scheduler import SWEEPS_DIR, detach_scheduler, new_sweep
from rld.rl.tuning import (
    collect_trials,
    load_tune_config,
    materialize_trials,
    plan_trials,
    trial_points,
)


def _live_schedulers(sweeps_dir: Path, source: Path) -> list[str]:
    """Return the sweep files of this search whose scheduler is still alive."""
    live = []
    for path in sorted(sweeps_dir.glob("*.json")):
        try:
            sweep = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if sweep.get("on_complete") != f"tune_collect:{source}":
            continue
        if sweep.get("state") in ("created", "running") and pid_alive(sweep.get("scheduler_pid")):
            live.append(path.name)
    return live


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="configs/rl/tune_<m>.yaml")
    parser.add_argument("--max-workers", type=int, default=34, help="global worker-slot cap")
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--poll-s", type=float, default=15.0, help="poll interval, host seconds")
    parser.add_argument("--collect", action="store_true", help="score finished trials only")
    parser.add_argument("--dry-run", action="store_true", help="print the trial points only")
    parser.add_argument("--plan", action="store_true", help="print what a launch would do")
    args = parser.parse_args()

    cfg = load_tune_config(args.config)
    if args.dry_run:
        for index, point in enumerate(trial_points(cfg)):
            print(index, point or "(base config)")
        return 0
    if args.collect:
        rows = collect_trials(cfg, args.runs_root)
        print(f"wrote {cfg.results_dir / 'trials.csv'} ({len(rows)} trials)")
        for row in rows:
            if row.get("superseded_runs"):
                print(
                    f"  trial {row['trial']:02d}: scored {row['run_dir'] or '-'}; "
                    f"superseded {row['superseded_runs']}"
                )
        return 0

    plans = plan_trials(cfg, args.runs_root)
    for plan in plans:
        where = "" if plan.run_dir is None else f" {plan.run_dir}"
        extra = "" if plan.resume_from is None else f" from {plan.resume_from.name}"
        print(f"trial {plan.trial:02d}: {plan.action}{where}{extra}")
    if args.plan:
        return 0
    sweeps_dir = args.runs_root / SWEEPS_DIR.name
    if live := _live_schedulers(sweeps_dir, cfg.source):
        print(f"refused: a scheduler of this search is still alive ({live})", file=sys.stderr)
        return 3
    blocked = [p for p in plans if p.action in ("live", "unreconciled")]
    if blocked:
        for p in blocked:
            print(f"refused: trial {p.trial:02d} is {p.action} ({p.run_dir})", file=sys.stderr)
        print("run `python scripts/reconcile_runs.py` for unreconciled runs", file=sys.stderr)
        return 3
    paths = materialize_trials(cfg)
    todo = [p for p in plans if p.action in ("fresh", "resume")]
    if not todo:
        print("every trial is done; nothing to queue (use --collect to score)")
        return 0
    jobs = [
        (str(paths[p.trial]), cfg.trial_seed, None if p.resume_from is None else str(p.resume_from))
        for p in todo
    ]
    sweep = new_sweep(
        "tune",
        jobs,
        args.max_workers,
        on_complete=f"tune_collect:{cfg.source}",
        sweeps_dir=sweeps_dir,
        label=cfg.method,
        poll_s=args.poll_s,
        meta={
            "tune_config": str(cfg.source),
            "trials_queued": [p.trial for p in todo],
            "trials_skipped_done": [p.trial for p in plans if p.action == "skip_done"],
        },
    )
    pid = detach_scheduler(sweep, args.runs_root, args.poll_s)
    n_resume = sum(p.action == "resume" for p in todo)
    print(
        f"tune {cfg.method}: {len(jobs)} of {cfg.trials} trials queued ({n_resume} resumed) x "
        f"{cfg.trial_steps} steps, seed {cfg.trial_seed}"
    )
    print(f"  trial configs  {cfg.results_dir / 'configs'}")
    print(f"  sweep file     {sweep}")
    print(f"  scheduler pid  {pid}")
    print(f"  scheduler log  {sweep.with_suffix('.log')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
