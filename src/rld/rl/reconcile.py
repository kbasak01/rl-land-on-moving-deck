"""Reconcile run and sweep records left ``running`` by processes that no longer exist.

A host restart (or ``kill -9``) stops a training run without its ``failed`` handler running,
so its ``status.json`` keeps saying ``running`` with a dead pid, and a sweep file keeps
saying ``running`` with a dead scheduler. Slot accounting, ``prepare_run_dir`` and the
tuning scheduler all read those records, so they must be corrected -- and only corrected:
nothing is deleted, moved or re-run here.

Rules
-----
**Run** (``<runs_root>/**/status.json``) -- marked failed when all of:

* ``state == "running"``;
* the writer is not alive (:func:`rld.rl.procs.status_owner_alive`: the pid is dead, or it
  now belongs to a process that started after the status was last written);
* the status file is older than ``stale_intervals`` (3) evaluation intervals of host time,
  i.e. ``3 * max(eval_interval_steps / fps, status_interval_s)`` seconds, or ``3 *
  status_interval_s`` when the run never reported a rate. A live run rewrites its status at
  least every ``status_interval_s`` (heartbeat) and after every evaluation, so a status this
  old with no writer is not a run between two writes.

Change: ``state -> "failed"``, ``error -> "<reason>; pid dead since <status mtime>"``,
``reconciled_at -> <now>``. Every other field is left exactly as it was (in particular
``steps``, ``updated``, ``wall_s`` and ``cpu_hours`` stay the last values the run itself
wrote).

**Sweep** (``<runs_root>/_sweeps/*.json``) -- treated when ``state`` is ``running`` or
``created`` and the scheduler pid is dead (the sweep file is rewritten every poll, so the
same staleness rule is applied with the poll interval, 15 s when the file does not record
it). After the runs above are reconciled:

* a ``running`` job whose run is now ``failed`` -> ``failed`` with the run's ``error``;
* a ``running`` job whose run finished (``done``) but was never polled -> ``done``;
* a ``running`` job whose run is still alive is left ``running``; ``queued`` jobs stay
  ``queued`` (they never started);
* every changed job gets ``reconciled_at``; the sweep gets ``state -> "interrupted"``,
  ``interrupted_reason`` and ``reconciled_at``. Its ``on_complete`` action (tuning
  collection) did not run and is not run here.

Units: ages and intervals are host seconds; steps are env control steps.
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rld.rl.callbacks import atomic_write_json, utc_now
from rld.rl.config import RUNS_ROOT
from rld.rl.procs import pid_alive, status_owner_alive

__all__ = [
    "DEFAULT_REASON",
    "STALE_INTERVALS",
    "Change",
    "format_changes",
    "reconcile",
    "run_stale_after_s",
]

#: Evaluation intervals of silence after which a dead run is declared failed.
STALE_INTERVALS: int = 3

#: Default reason written into ``error``.
DEFAULT_REASON: str = "host terminated"

#: Scheduler poll interval assumed for sweep files that do not record one, host seconds.
_DEFAULT_POLL_S: float = 15.0


@dataclass(frozen=True)
class Change:
    """One field change of one record.

    Attributes:
        path: The ``status.json`` or sweep file.
        kind: ``"run"``, ``"job"`` or ``"sweep"``.
        item: Run directory, ``<sweep id>#<job index>``, or the sweep id.
        field: The field.
        old: Value before.
        new: Value after.
    """

    path: Path
    kind: str
    item: str
    field: str
    old: Any
    new: Any


def _mtime_iso(path: Path) -> str:
    """Return a file's modification time as UTC ISO-8601, seconds resolution."""
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")


def run_stale_after_s(status: dict[str, Any], stale_intervals: int = STALE_INTERVALS) -> float:
    """Return how long a run's status may go unwritten before a dead run is declared failed.

    Args:
        status: A parsed ``status.json``.
        stale_intervals: Evaluation intervals of silence.

    Returns:
        ``stale_intervals * max(eval_interval_steps / fps, status_interval_s)`` host
        seconds; ``stale_intervals * status_interval_s`` if no rate was ever reported.
    """
    heartbeat = float(status.get("status_interval_s") or 60.0)
    fps = float(status.get("fps") or 0.0)
    interval = status.get("eval_interval_steps")
    eval_s = float(interval) / fps if fps > 0.0 and interval else 0.0
    return stale_intervals * max(eval_s, heartbeat)


def _load(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, or ``None``."""
    try:
        data: Any = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _reconcile_runs(
    runs_root: Path, now: float, reason: str, stale_intervals: int, dry_run: bool
) -> list[Change]:
    """Mark dead ``running`` runs failed (see the module docstring)."""
    changes: list[Change] = []
    sweeps_dir = runs_root / "_sweeps"
    for path in sorted(runs_root.glob("**/status.json")):
        if sweeps_dir in path.parents:
            continue
        status = _load(path)
        if status is None or status.get("state") != "running":
            continue
        if status_owner_alive(status):
            continue
        age = now - path.stat().st_mtime
        if age < run_stale_after_s(status, stale_intervals):
            continue
        since = _mtime_iso(path)
        stamp = utc_now()
        new = {
            "state": "failed",
            "error": f"{reason}; pid dead since {since}",
            "reconciled_at": stamp,
        }
        item = str(path.parent)
        for key, value in new.items():
            changes.append(Change(path, "run", item, key, status.get(key), value))
        if not dry_run:
            status.update(new)
            atomic_write_json(path, status)
    return changes


def _reconcile_sweeps(
    runs_root: Path,
    now: float,
    reason: str,
    stale_intervals: int,
    dry_run: bool,
    new_errors: dict[str, str],
) -> list[Change]:
    """Close sweeps whose scheduler died (see the module docstring)."""
    changes: list[Change] = []
    for path in sorted((runs_root / "_sweeps").glob("*.json")):
        sweep = _load(path)
        if sweep is None or sweep.get("state") not in ("running", "created"):
            continue
        if pid_alive(sweep.get("scheduler_pid")):
            continue
        poll_s = float(sweep.get("poll_s") or _DEFAULT_POLL_S)
        if now - path.stat().st_mtime < stale_intervals * poll_s:
            continue
        sweep_id = str(sweep.get("sweep_id", path.stem))
        stamp = utc_now()
        for index, job in enumerate(sweep.get("jobs", [])):
            if job.get("state") != "running" or not job.get("run_dir"):
                continue
            run_dir = Path(job["run_dir"])
            status = _load(run_dir / "status.json") or {}
            run_state = status.get("state")
            if str(run_dir) in new_errors:  # reconciled just now (or would be, in a dry run)
                run_state = "failed"
                status = {**status, "error": new_errors[str(run_dir)]}
            if run_state == "failed":
                new = {"state": "failed", "error": status.get("error"), "reconciled_at": stamp}
            elif run_state == "done":
                new = {"state": "done", "reconciled_at": stamp}
            else:
                continue
            for key, value in new.items():
                changes.append(Change(path, "job", f"{sweep_id}#{index}", key, job.get(key), value))
            if not dry_run:
                job.update(new)
        since = _mtime_iso(path)
        new_sweep = {
            "state": "interrupted",
            "interrupted_reason": f"{reason}; scheduler pid {sweep.get('scheduler_pid')} "
            f"dead since {since}",
            "reconciled_at": stamp,
        }
        for key, value in new_sweep.items():
            changes.append(Change(path, "sweep", sweep_id, key, sweep.get(key), value))
        if not dry_run:
            sweep.update(new_sweep)
            atomic_write_json(path, sweep)
    return changes


def reconcile(
    runs_root: Path = RUNS_ROOT,
    *,
    dry_run: bool = False,
    reason: str = DEFAULT_REASON,
    stale_intervals: int = STALE_INTERVALS,
    now: float | None = None,
) -> list[Change]:
    """Reconcile every run and sweep record under ``runs_root``.

    Args:
        runs_root: Root of all runs (sweep files in ``<runs_root>/_sweeps``).
        dry_run: Report the changes without writing anything.
        reason: The cause written into ``error`` (``"host terminated"``).
        stale_intervals: Evaluation intervals (runs) or poll intervals (sweeps) of silence
            required before a record with a dead writer is changed.
        now: Current Unix time (tests); the host clock when ``None``.

    Returns:
        Every field change, runs first, in path order.
    """
    when = datetime.now(UTC).timestamp() if now is None else float(now)
    runs = _reconcile_runs(runs_root, when, reason, stale_intervals, dry_run)
    new_errors = {c.item: str(c.new) for c in runs if c.field == "error"}
    sweeps = _reconcile_sweeps(runs_root, when, reason, stale_intervals, dry_run, new_errors)
    return runs + sweeps


def format_changes(changes: Sequence[Change], runs_root: Path = RUNS_ROOT) -> str:
    """Render changes as a plain-text table.

    Args:
        changes: From :func:`reconcile`.
        runs_root: Paths are shown relative to it.

    Returns:
        One line per change (``"no changes"`` when empty).
    """
    if not changes:
        return "no changes"

    def rel(text: str) -> str:
        try:
            return os.path.relpath(text, runs_root)
        except ValueError:
            return text

    lines = []
    for c in changes:
        item = rel(c.item) if c.kind == "run" else c.item
        lines.append(f"{c.kind:5s}  {item}  {c.field}: {c.old!r} -> {c.new!r}")
    return "\n".join(lines)
