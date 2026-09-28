"""Phase 5 recovery: reconciling runs and sweeps whose process died (host restart).

* A ``running`` run with a dead writer, silent for > 3 evaluation intervals, becomes
  ``failed`` with the host-terminated error and ``reconciled_at``; no other field changes.
* A recent, live, reused-pid-but-recent, ``done`` or ``failed`` run is left alone; a
  reused pid (a process that started after the status was written) counts as dead.
* A sweep with a dead scheduler becomes ``interrupted``; its running jobs follow their
  runs; queued jobs stay queued; a live scheduler's sweep is untouched.
* ``dry_run`` writes nothing; a second pass changes nothing.

Units: ages are host seconds; steps are env control steps.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

from rld.rl.callbacks import atomic_write_json
from rld.rl.reconcile import format_changes, reconcile, run_stale_after_s
from rld.rl.scheduler import Job

DEAD_PID = 2**22 + 12345  # above the default pid_max: never a live process


def _status(pid: int, **fields: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "state": "running",
        "method": "tune_sac_t01",
        "seed": 0,
        "steps": 354400,
        "budget": 500000,
        "fps": 14.3,
        "eval_interval_steps": 50000,
        "status_interval_s": 60.0,
        "updated": "2026-09-24T16:14:49+00:00",
        "wall_s": 24784.0,
        "cpu_hours": 9.1,
        "pid": pid,
        "error": None,
        "resumed_from": None,
    }
    base.update(fields)
    return base


def _write(path: Path, payload: dict[str, Any], age_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    old = time.time() - age_s
    os.utime(path, (old, old))


def test_stale_threshold_is_three_eval_intervals() -> None:
    # 50 000 steps at 14.3 steps/s is ~58 min per evaluation interval.
    assert abs(run_stale_after_s(_status(1)) - 3 * 50000 / 14.3) < 1e-6
    assert run_stale_after_s(_status(1, fps=0.0)) == 3 * 60.0  # no rate: 3 heartbeats
    assert run_stale_after_s(_status(1, fps=1e6)) == 3 * 60.0  # never below 3 heartbeats


def test_reconcile_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    week = 7 * 86400.0
    dead_old = runs / "g" / "trial_01" / "0" / "status.json"
    dead_recent = runs / "g" / "trial_02" / "0" / "status.json"
    live = runs / "g" / "trial_03" / "0" / "status.json"
    reused = runs / "g" / "trial_04" / "0" / "status.json"
    done = runs / "g" / "trial_05" / "0" / "status.json"
    _write(dead_old, _status(DEAD_PID), week)
    _write(dead_recent, _status(DEAD_PID), 60.0)  # < 3 x 58 min: may be between writes
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    _write(live, _status(os.getpid(), updated=now_iso), week)
    # Our own pid, but the status was written long before this process started: a reused pid.
    _write(reused, _status(os.getpid(), updated="2000-01-01T00:00:00+00:00"), week)
    _write(done, _status(DEAD_PID, state="done"), week)
    before = {p: json.loads(p.read_text()) for p in (dead_old, dead_recent, live, reused, done)}

    dry = reconcile(runs, dry_run=True)
    assert {c.item for c in dry} == {str(dead_old.parent), str(reused.parent)}
    for p, payload in before.items():  # a dry run writes nothing
        assert json.loads(p.read_text()) == payload

    changes = reconcile(runs)
    assert [(c.field, c.new) for c in changes if c.item == str(dead_old.parent)][:1] == [
        ("state", "failed")
    ]
    after = json.loads(dead_old.read_text())
    assert after["state"] == "failed"
    assert after["error"].startswith("host terminated; pid dead since 20")
    assert after["reconciled_at"]
    untouched = {k: v for k, v in after.items() if k not in ("state", "error", "reconciled_at")}
    assert untouched == {
        k: v for k, v in before[dead_old].items() if k not in ("state", "error", "reconciled_at")
    }
    assert json.loads(reused.read_text())["state"] == "failed"
    for p in (dead_recent, live, done):
        assert json.loads(p.read_text()) == before[p]
    assert reconcile(runs) == []  # idempotent
    assert "state" in format_changes(changes, runs)


def test_reconcile_sweeps(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    sweeps = runs / "_sweeps"
    week = 7 * 86400.0
    dead_run = runs / "tune_sac" / "trial_01" / "0"
    done_run = runs / "tune_sac" / "trial_02" / "0"
    live_run = runs / "tune_sac" / "trial_03" / "0"
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    _write(dead_run / "status.json", _status(DEAD_PID), week)
    _write(done_run / "status.json", _status(DEAD_PID, state="done"), week)
    _write(live_run / "status.json", _status(os.getpid(), updated=now_iso), 0.0)
    jobs = [
        {"config": "c", "seed": 0, "cost": 5, "state": "done", "run_dir": str(tmp_path / "x")},
        {"config": "c", "seed": 0, "cost": 5, "state": "running", "run_dir": str(dead_run)},
        {"config": "c", "seed": 0, "cost": 5, "state": "running", "run_dir": str(done_run)},
        {"config": "c", "seed": 0, "cost": 5, "state": "running", "run_dir": str(live_run)},
        {"config": "c", "seed": 0, "cost": 5, "state": "queued", "run_dir": None},
    ]
    dead_sweep = sweeps / "a-tune-sac.json"
    live_sweep = sweeps / "b-tune-ppo.json"
    _write(
        dead_sweep,
        {"sweep_id": "a-tune-sac", "state": "running", "scheduler_pid": DEAD_PID, "jobs": jobs},
        week,
    )
    live_payload = {
        "sweep_id": "b-tune-ppo",
        "state": "running",
        "scheduler_pid": os.getpid(),
        "jobs": [dict(j) for j in jobs[1:2]],
    }
    _write(live_sweep, live_payload, week)

    reconcile(runs)
    sweep = json.loads(dead_sweep.read_text())
    assert sweep["state"] == "interrupted" and sweep["reconciled_at"]
    assert f"scheduler pid {DEAD_PID} dead since" in sweep["interrupted_reason"]
    states = [j["state"] for j in sweep["jobs"]]
    assert states == ["done", "failed", "done", "running", "queued"]
    assert sweep["jobs"][1]["error"].startswith("host terminated; pid dead since")
    assert "reconciled_at" not in sweep["jobs"][0] and "reconciled_at" in sweep["jobs"][1]
    assert json.loads(live_sweep.read_text()) == live_payload  # live scheduler: untouched
    # The scheduler can still read a reconciled sweep file, extra keys included.
    job = Job.from_dict(sweep["jobs"][1])
    assert job.reconciled_at and job.to_dict().items() >= sweep["jobs"][1].items()
    odd = Job.from_dict({**sweep["jobs"][4], "note": "kept"})
    assert odd.extra == {"note": "kept"} and odd.to_dict()["note"] == "kept"
    assert reconcile(runs) == []
