"""A detached queue scheduler: start training runs as CPU worker slots free up.

``make sweep`` and ``make tune`` both create a *sweep file*
``artifacts/runs/_sweeps/<sweep_id>.json`` listing jobs ``(config, seed, cost)`` and start
one scheduler process for it, in its own session (``start_new_session``: the equivalent of
``nohup`` -- closing the terminal does not reach it). The scheduler polls every
``poll_s`` seconds and, strictly first-in first-out, starts the next queued job whenever

    (worker slots in use across **all** live runs and sweeps) + job cost <= max_workers.

A job's cost is :attr:`rld.rl.config.TrainConfig.workers` (its busiest env pool plus the
learner process). Slots in use are counted from every ``status.json`` in state ``running``
whose pid is alive, plus every job another live scheduler has started whose run has not
written its status yet; the count and the launch happen under one file lock, so two
schedulers cannot both fill the same free slots.

Each job is launched exactly as ``make train-bg`` launches one: a fresh run directory from
:func:`rld.rl.train.prepare_run_dir` (a finished run of the same group and seed makes the job
``refused``, never retrained), then ``scripts/train.py --run-dir`` in its own session with
stdout and stderr to ``<run_dir>/train.log``. The scheduler never kills anything; on
SIGTERM it marks the sweep ``cancelled`` and exits, leaving started runs to finish.

When every job is terminal the optional ``on_complete`` action runs (``tune_collect:<path>``
scores a tuning search) and the sweep is marked ``done``.

Units: ``poll_s`` is host seconds; ``cost`` and ``max_workers`` are CPU worker slots.
"""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

from rld.config import REPO_ROOT
from rld.rl.callbacks import atomic_write_json, utc_now
from rld.rl.config import RUNS_ROOT, load_train_config
from rld.rl.procs import pid_alive
from rld.rl.train import RunDirExistsError, prepare_run_dir, read_status

__all__ = [
    "SWEEPS_DIR",
    "TERMINAL",
    "Job",
    "detach_scheduler",
    "launch_train",
    "launchable",
    "new_sweep",
    "run_scheduler",
    "used_workers",
]

#: Where sweep files live.
SWEEPS_DIR: Path = RUNS_ROOT / "_sweeps"

#: Job states after which a job never changes again.
TERMINAL: tuple[str, ...] = ("done", "failed", "refused")

TRAIN_SCRIPT: Path = REPO_ROOT / "scripts" / "train.py"
SWEEP_SCRIPT: Path = REPO_ROOT / "scripts" / "sweep.py"


@dataclass
class Job:
    """One training run of a sweep.

    Attributes:
        config: Training config path (repository-relative or absolute).
        seed: Training seed.
        cost: Worker slots it occupies.
        state: ``queued``, ``running``, ``done``, ``failed`` or ``refused``.
        pid: The training process id once started.
        run_dir: Its run directory once started.
        started: UTC start time.
        ended: UTC end time.
        returncode: Exit code once finished.
        error: Why it was refused or failed, if known.
    """

    config: str
    seed: int
    cost: int
    state: str = "queued"
    pid: int | None = None
    run_dir: str | None = None
    started: str | None = None
    ended: str | None = None
    returncode: int | None = None
    error: str | None = None


@contextmanager
def _locked(sweeps_dir: Path) -> Iterator[None]:
    """Hold the global scheduling lock (``<sweeps_dir>/.lock``)."""
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    with (sweeps_dir / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def new_sweep(
    kind: str,
    jobs: Sequence[tuple[str, int]],
    max_workers: int,
    *,
    on_complete: str | None = None,
    sweeps_dir: Path = SWEEPS_DIR,
    label: str = "",
) -> Path:
    """Create a sweep file.

    Args:
        kind: ``"sweep"`` or ``"tune"``.
        jobs: ``(config path, seed)`` pairs, in launch order.
        max_workers: Global worker-slot cap.
        on_complete: Action when all jobs are terminal (``tune_collect:<path>``) or None.
        sweeps_dir: Directory of sweep files.
        label: Short label for the sweep id.

    Returns:
        The sweep file's path.

    Raises:
        ValueError: If a job alone exceeds ``max_workers`` or there are no jobs.
    """
    if not jobs:
        raise ValueError("a sweep needs at least one job")
    built = []
    for config, seed in jobs:
        cost = load_train_config(Path(config)).workers
        if cost > max_workers:
            raise ValueError(f"{config} needs {cost} worker slots > max_workers {max_workers}")
        built.append(Job(config=str(config), seed=int(seed), cost=cost))
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    sweep_id = f"{stamp}-{kind}{'-' + label if label else ''}"
    path = sweeps_dir / f"{sweep_id}.json"
    k = 2
    while path.exists():
        path = sweeps_dir / f"{sweep_id}-{k}.json"
        k += 1
    atomic_write_json(
        path,
        {
            "sweep_id": path.stem,
            "kind": kind,
            "state": "created",
            "max_workers": int(max_workers),
            "on_complete": on_complete,
            "created": utc_now(),
            "updated": utc_now(),
            "scheduler_pid": None,
            "log": str(path.with_suffix(".log")),
            "jobs": [asdict(j) for j in built],
        },
    )
    return path


def launchable(jobs: Sequence[Job], used: int, max_workers: int) -> list[int]:
    """Return the indices of queued jobs to start now, strictly first-in first-out.

    Args:
        jobs: The sweep's jobs.
        used: Worker slots in use globally.
        max_workers: The cap.

    Returns:
        Indices, in order; stops at the first queued job that does not fit.
    """
    out = []
    for index, job in enumerate(jobs):
        if job.state != "queued":
            continue
        if used + job.cost > max_workers:
            break
        out.append(index)
        used += job.cost
    return out


def used_workers(runs_root: Path = RUNS_ROOT, sweeps_dir: Path = SWEEPS_DIR) -> int:
    """Return the worker slots in use by every live run and every just-started job.

    Args:
        runs_root: Root of all runs.
        sweeps_dir: Directory of sweep files.

    Returns:
        Slots in use.
    """
    used = 0
    counted: set[str] = set()
    for status_path in runs_root.glob("**/status.json"):
        status = read_status(status_path.parent)
        if status is None or status.get("state") != "running" or not pid_alive(status.get("pid")):
            continue
        used += int(status.get("workers") or (int(status.get("n_envs") or 0) + 1))
        counted.add(str(status_path.parent.resolve()))
    for sweep_path in sweeps_dir.glob("*.json"):
        try:
            sweep = json.loads(sweep_path.read_text())
        except (OSError, ValueError):
            continue
        for job in sweep.get("jobs", []):
            if job.get("state") != "running" or not job.get("run_dir"):
                continue
            run_dir = str(Path(job["run_dir"]).resolve())
            if run_dir in counted or not pid_alive(job.get("pid")):
                continue
            used += int(job.get("cost", 0))
            counted.add(run_dir)
    return used


def launch_train(
    config: str, seed: int, runs_root: Path = RUNS_ROOT
) -> tuple[subprocess.Popen[bytes], Path]:
    """Start one training run in its own session, logging to its run directory.

    Args:
        config: Training config path.
        seed: Training seed.
        runs_root: Root of all runs.

    Returns:
        ``(process, run_dir)``.

    Raises:
        RunDirExistsError: If the run already finished or is live.
    """
    cfg = load_train_config(Path(config))
    run_dir = prepare_run_dir(cfg, seed, runs_root)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"  # train.log is tail-able while the run is live
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with (run_dir / "train.log").open("ab") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--config",
                str(cfg.source),
                "--seed",
                str(seed),
                "--run-dir",
                str(run_dir),
                "--runs-root",
                str(runs_root),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            env=env,
            start_new_session=True,
        )
    return proc, run_dir


def detach_scheduler(sweep_path: Path, runs_root: Path = RUNS_ROOT, poll_s: float = 15.0) -> int:
    """Start the scheduler for a sweep file in its own session.

    Args:
        sweep_path: The sweep file.
        runs_root: Root of all runs.
        poll_s: Poll interval, host seconds.

    Returns:
        The scheduler's pid.
    """
    log_path = sweep_path.with_suffix(".log")
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(SWEEP_SCRIPT),
                "--run-scheduler",
                str(sweep_path),
                "--runs-root",
                str(runs_root),
                "--poll-s",
                str(poll_s),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            start_new_session=True,
        )
    return int(proc.pid)


@dataclass
class _Sweep:
    """In-memory sweep file."""

    path: Path
    data: dict[str, Any]
    jobs: list[Job] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "_Sweep":
        """Read a sweep file."""
        data = json.loads(path.read_text())
        return cls(path, data, [Job(**j) for j in data["jobs"]])

    def save(self, **fields: Any) -> None:
        """Write it back atomically, with ``fields`` updated."""
        self.data.update(fields)
        self.data["jobs"] = [asdict(j) for j in self.jobs]
        self.data["updated"] = utc_now()
        atomic_write_json(self.path, self.data)


def _on_complete(action: str | None, runs_root: Path) -> None:
    """Run a sweep's completion action.

    Args:
        action: ``tune_collect:<tune config path>`` or ``None``.
        runs_root: Root of all runs.

    Raises:
        ValueError: On an unknown action.
    """
    if not action:
        return
    kind, _, arg = action.partition(":")
    if kind != "tune_collect":
        raise ValueError(f"unknown on_complete action {action!r}")
    from rld.rl.tuning import collect_trials, load_tune_config

    rows = collect_trials(load_tune_config(Path(arg)), runs_root)
    winner = [r["trial"] for r in rows if r.get("selected")]
    print(f"tuning collected: {len(rows)} trials, winner {winner}", flush=True)


def run_scheduler(
    sweep_path: Path,
    runs_root: Path = RUNS_ROOT,
    poll_s: float = 15.0,
    sweeps_dir: Path | None = None,
) -> None:
    """Run a sweep to completion (the detached process's body).

    Args:
        sweep_path: The sweep file.
        runs_root: Root of all runs.
        poll_s: Poll interval, host seconds.
        sweeps_dir: Directory of sweep files (default: the sweep file's directory).
    """
    sweeps_dir = sweeps_dir or sweep_path.parent
    sweep = _Sweep.load(sweep_path)
    procs: dict[int, subprocess.Popen[bytes]] = {}
    cancelled = False

    def _term(signum: int, frame: FrameType | None) -> None:
        del frame
        nonlocal cancelled
        cancelled = True
        print(f"scheduler received signal {signum}; runs already started keep running", flush=True)

    signal.signal(signal.SIGTERM, _term)
    max_workers = int(sweep.data["max_workers"])
    sweep.save(state="running", scheduler_pid=os.getpid())
    while not cancelled:
        with _locked(sweeps_dir):
            for index, job in enumerate(sweep.jobs):
                if job.state != "running":
                    continue
                proc = procs.get(index)
                code = proc.poll() if proc is not None else (None if pid_alive(job.pid) else -1)
                if code is None:
                    continue
                status = read_status(Path(job.run_dir or "."))
                job.returncode = int(code)
                job.ended = utc_now()
                ok = code == 0 and status is not None and status.get("state") == "done"
                job.state = "done" if ok else "failed"
                if not ok:
                    job.error = None if status is None else status.get("error")
                print(
                    f"job {index} {job.config} seed {job.seed}: {job.state} (rc {code})", flush=True
                )
            used = used_workers(runs_root, sweeps_dir)
            for index in launchable(sweep.jobs, used, max_workers):
                job = sweep.jobs[index]
                try:
                    proc, run_dir = launch_train(job.config, job.seed, runs_root)
                except RunDirExistsError as exc:
                    job.state, job.error, job.ended = "refused", str(exc), utc_now()
                    print(f"job {index} refused: {exc}", flush=True)
                    continue
                procs[index] = proc
                job.state, job.pid, job.run_dir, job.started = (
                    "running",
                    proc.pid,
                    str(run_dir),
                    utc_now(),
                )
                print(
                    f"job {index} {job.config} seed {job.seed}: "
                    f"started pid {proc.pid} -> {run_dir}",
                    flush=True,
                )
            sweep.save(used_workers_at_poll=used)
        if all(job.state in TERMINAL for job in sweep.jobs):
            break
        time.sleep(poll_s)
    if cancelled:
        sweep.save(state="cancelled")
        return
    try:
        _on_complete(sweep.data.get("on_complete"), runs_root)
    except Exception as exc:  # recorded, not swallowed silently
        sweep.save(state="done_with_errors", on_complete_error=f"{type(exc).__name__}: {exc}")
        raise
    sweep.save(state="done")
