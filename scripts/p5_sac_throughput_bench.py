"""SAC learner throughput by width x torch_threads x device (scratch measurement, not a trial).

Why: SAC tuning trials ran at 14-58 env steps/s against 86 in the single-run device benchmark
(``scripts/p5_sac_device_bench.py``, width 256, one torch thread) on a host that was mostly
idle, so the suspected bottleneck is the single-threaded gradient step (UTD = 1).

What one cell does (``--cell W:T:DEV``), in a fresh process:

* the committed ``configs/rl/sac.yaml`` with only ``policy.width``, ``torch_threads``,
  ``device`` and ``sac.learning_starts`` overridden;
* **train pool only** at the first curriculum stage (SS3), the same ``EnvFactory`` /
  ``SubprocVecEnv`` / ``VecNormalize`` / SB3 SAC construction as :func:`rld.rl.train.train`,
  but no Monitor, no evaluation workers, no checkpoints and no ``status.json``: nothing
  touches the tune pool, ``results/episodes/`` or ``artifacts/runs/``;
* phase 1: ``learning_starts`` uniform-random steps (the warm-up, not timed);
* phase 2 (timed): ``--steps`` env steps of normal SAC training (UTD = 1), reported as
  post-warm-up env steps per host second, plus the process-tree CPU use over that window
  (average busy cores) and the learner process's own CPU use;
* phase 3 (timed): ``--grad-steps`` bare gradient steps on the filled buffer
  (``model.train``), reported as gradient steps per host second -- the learner's ceiling.

No score is read and no episode outcome is looked at. Output: one JSON line per cell,
appended to ``<out-root>/results.jsonl``.

Usage::

    # every cell, one at a time (each in its own process)
    python scripts/p5_sac_throughput_bench.py --out-root <scratch> --grid
    # one cell N times concurrently (the contention a sweep sees)
    python scripts/p5_sac_throughput_bench.py --out-root <scratch> --cell 256:2:cpu --concurrent 4

Units: steps are env control steps (1/30 s model scale each); rates per HOST second; CPU
use in cores (CPU seconds per host second).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

WIDTHS = (256, 512)
THREADS = (1, 2, 4)
DEVICES = ("cpu", "cuda")


def _cell_env() -> dict[str, str]:
    """Return the launch environment ``make train-bg`` gives a run."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "1")
    return env


def run_cell(
    width: int,
    threads: int,
    device: str,
    steps: int,
    learning_starts: int,
    grad_steps: int,
    out_dir: Path,
    tag: str,
) -> dict[str, Any]:
    """Measure one cell (see the module docstring).

    Args:
        width: Hidden-layer width (actor and critics).
        threads: ``torch.set_num_threads`` in the learner.
        device: ``"cpu"`` or ``"cuda"``.
        steps: Timed post-warm-up env steps.
        learning_starts: Warm-up (uniform-random) env steps.
        grad_steps: Bare gradient steps timed after training.
        out_dir: Scratch directory for this cell (TensorBoard only).
        tag: Free label copied into the result.

    Returns:
        The result record.
    """
    import torch
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    from rld.rl.config import RL_CONFIG_DIR, apply_overrides, load_yaml, train_config_from_dict
    from rld.rl.curriculum import Curriculum
    from rld.rl.procs import tree_cpu_seconds
    from rld.rl.train import _build_model, build_env_configs, forkserver_preload, training_pools
    from rld.rl.wrappers import EnvFactory

    raw = load_yaml(RL_CONFIG_DIR / "sac.yaml")
    over = {
        "method": "sac_throughput_bench",
        "run_group": "sac_throughput_bench",
        "device": device,
        "torch_threads": threads,
        "policy.width": width,
        "sac.learning_starts": learning_starts,
        "total_steps": learning_starts + steps,
    }
    cfg = train_config_from_dict(apply_overrides(raw, over), f"bench {width}:{threads}:{device}")
    assert cfg.sac is not None
    torch.set_num_threads(cfg.torch_threads)
    cfgs = build_env_configs(cfg)
    train_pool, _ = training_pools(cfg, cfgs)
    stage = Curriculum(cfg.curriculum).sampling_stage
    forkserver_preload()
    factories: list[Any] = [
        EnvFactory(cfgs, tuple(train_pool), cfg.pad, 0, rank, stage, None, cfg.prefetch_reset)
        for rank in range(cfg.n_envs)
    ]
    venv = VecNormalize(
        SubprocVecEnv(factories),
        training=True,
        norm_obs=cfg.normalize.norm_obs,
        norm_reward=cfg.normalize.norm_reward,
        clip_obs=cfg.normalize.clip_obs,
        clip_reward=cfg.normalize.clip_reward,
        gamma=cfg.gamma,
    )
    try:
        model = _build_model(cfg, venv, 0, out_dir)
        model.verbose = 0
        load0 = os.getloadavg()
        t0 = time.perf_counter()
        model.learn(total_timesteps=learning_starts, log_interval=10**9)
        warm_s = time.perf_counter() - t0
        steps0 = int(model.num_timesteps)
        tree0, self0 = tree_cpu_seconds(), time.process_time()
        t1 = time.perf_counter()
        model.learn(total_timesteps=steps, reset_num_timesteps=False, log_interval=10**9)
        train_s = time.perf_counter() - t1
        tree1, self1 = tree_cpu_seconds(), time.process_time()
        done = int(model.num_timesteps) - steps0
        load1 = os.getloadavg()
        t2 = time.perf_counter()
        model.train(gradient_steps=grad_steps, batch_size=cfg.sac.batch_size)
        if device == "cuda":
            torch.cuda.synchronize()
        grad_s = time.perf_counter() - t2
    finally:
        venv.close()
    return {
        "tag": tag,
        "width": width,
        "torch_threads": threads,
        "torch_get_num_threads": torch.get_num_threads(),
        "device": device,
        "batch_size": cfg.sac.batch_size,
        "n_envs": cfg.n_envs,
        "learning_starts": learning_starts,
        "warmup_steps": steps0,
        "warmup_s": round(warm_s, 2),
        "timed_steps": done,
        "timed_s": round(train_s, 2),
        "steps_per_s": round(done / train_s, 2),
        "tree_cores": round((tree1 - tree0) / train_s, 2),
        "learner_cores": round((self1 - self0) / train_s, 2),
        "grad_steps": grad_steps,
        "grad_steps_per_s": round(grad_steps / grad_s, 1),
        "loadavg_start": load0,
        "loadavg_end": load1,
        "pid": os.getpid(),
    }


def _spawn(args: argparse.Namespace, cell: str, tag: str, out_dir: Path) -> subprocess.Popen[bytes]:
    """Start one cell in its own process."""
    cmd = [
        sys.executable,
        __file__,
        "--out-root",
        str(args.out_root),
        "--cell",
        cell,
        "--steps",
        str(args.steps),
        "--learning-starts",
        str(args.learning_starts),
        "--grad-steps",
        str(args.grad_steps),
        "--tag",
        tag,
        "--cell-dir",
        str(out_dir),
    ]
    log = (out_dir / "cell.log").open("ab")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=_cell_env())


def main() -> int:
    """Parse arguments and run a grid, a concurrent group, or one cell."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", type=Path, required=True, help="scratch dir (never artifacts/)")
    ap.add_argument("--grid", action="store_true", help="every cell, one at a time")
    ap.add_argument("--cell", default=None, help="WIDTH:THREADS:DEVICE, e.g. 256:2:cpu")
    ap.add_argument("--concurrent", type=int, default=1, help="copies of --cell at once")
    ap.add_argument("--steps", type=int, default=15000, help="timed post-warm-up env steps")
    ap.add_argument("--learning-starts", type=int, default=2000)
    ap.add_argument("--grad-steps", type=int, default=2000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cell-dir", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    from rld.rl.config import RUNS_ROOT

    root = args.out_root.resolve()
    if root == RUNS_ROOT.resolve() or RUNS_ROOT.resolve() in root.parents:
        ap.error("--out-root must be a scratch directory, never under artifacts/runs")
    root.mkdir(parents=True, exist_ok=True)

    if args.cell_dir is not None:  # a single cell, in this process
        width, threads, device = args.cell.split(":")
        rec = run_cell(
            int(width),
            int(threads),
            device,
            args.steps,
            args.learning_starts,
            args.grad_steps,
            args.cell_dir,
            args.tag,
        )
        line = json.dumps(rec)
        print(line, flush=True)
        with (root / "results.jsonl").open("a") as handle:
            handle.write(line + "\n")
        return 0

    if args.grid:
        cells = [f"{w}:{t}:{d}" for d in DEVICES for w in WIDTHS for t in THREADS]
        for cell in cells:
            out_dir = root / f"alone_{cell.replace(':', '_')}"
            out_dir.mkdir(parents=True, exist_ok=False)
            print(f"{time.strftime('%H:%M:%S')} cell {cell} ...", flush=True)
            rc = _spawn(args, cell, "alone", out_dir).wait()
            print(f"{time.strftime('%H:%M:%S')} cell {cell} rc {rc}", flush=True)
        return 0

    if args.cell is None:
        ap.error("--grid or --cell is required")
    stamp = time.strftime("%H%M%S")
    procs = []
    for k in range(args.concurrent):
        out_dir = root / f"conc{args.concurrent}_{args.cell.replace(':', '_')}_{stamp}_{k}"
        out_dir.mkdir(parents=True, exist_ok=False)
        procs.append(_spawn(args, args.cell, f"concurrent{args.concurrent}", out_dir))
    codes = [p.wait() for p in procs]
    print(f"concurrent {args.cell} x{args.concurrent}: rc {codes}", flush=True)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
