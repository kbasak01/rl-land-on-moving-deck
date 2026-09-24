"""Phase 5: ``status.json``, run-directory safety, a failed run, and the sweep scheduler.

* ``status.json`` carries every :data:`rld.rl.callbacks.STATUS_KEYS` field, is written
  atomically, and is valid JSON (NaN -> null).
* A finished run is never retrained or overwritten; a failed one is kept and the next run
  takes a suffix.
* An exception inside training leaves ``state: failed`` with the error, and the traceback on
  stderr (the run's log).
* The scheduler starts jobs first-in first-out within the global worker-slot cap, counting
  every live run.

Units: steps are env control steps; ``*_s`` fields host seconds; costs are worker slots.
"""

import json
import os
from pathlib import Path

import pytest

import rld.rl.train as train_mod
from _rl_helpers import tiny_config
from rld.rl.callbacks import STATUS_KEYS, StatusWriter, atomic_write_json
from rld.rl.config import RL_CONFIG_DIR
from rld.rl.curriculum import Curriculum
from rld.rl.scheduler import Job, launchable, new_sweep, used_workers
from rld.rl.train import RunDirExistsError, prepare_run_dir, read_status


def _static() -> dict[str, object]:
    return {
        "method": "ppo",
        "seed": 0,
        "run_id": "x",
        "run_group": "ppo",
        "run_dir": "/r",
        "algo": "ppo",
        "budget": 1000,
        "eval_interval_steps": 100,
        "n_envs": 16,
        "workers": 17,
        "device": "cpu",
        "git_sha": "abc",
        "git_dirty": False,
        "config": "c.yaml",
        "config_sha256": "0" * 64,
        "log": "/r/train.log",
        "resumed_from": None,
    }


def test_status_schema_and_atomic_write(tmp_path: Path) -> None:
    cfg = tiny_config("ppo")
    writer = StatusWriter(tmp_path / "status.json", _static())
    payload = writer.write(
        "running",
        steps=250,
        curriculum=Curriculum(cfg.curriculum),
        last_eval={"success": float("nan"), "outcome_fractions": {"crash": 1.0}},
        eval_env_steps=10,
    )
    loaded = json.loads((tmp_path / "status.json").read_text())
    assert set(STATUS_KEYS) <= set(loaded)
    assert loaded["state"] == "running" and loaded["steps"] == 250 and loaded["budget"] == 1000
    assert loaded["curriculum_stage"] == "SS3"
    assert loaded["last_eval"]["success"] is None  # NaN written as null
    assert loaded["pid"] == os.getpid() and loaded["cpu_hours"] > 0.0
    assert payload["state"] == "running"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_prepare_run_dir_refuses_finished_and_suffixes_failed(tmp_path: Path) -> None:
    cfg = tiny_config("ppo")
    first = prepare_run_dir(cfg, 0, tmp_path)
    assert first == tmp_path / cfg.run_group / "0"
    atomic_write_json(first / "status.json", {"state": "failed", "pid": 0})
    second = prepare_run_dir(cfg, 0, tmp_path)
    assert second.name == "0_r2" and first.exists()  # failed run kept
    atomic_write_json(second / "status.json", {"state": "running", "pid": os.getpid()})
    with pytest.raises(RunDirExistsError, match="live"):
        prepare_run_dir(cfg, 0, tmp_path)
    atomic_write_json(second / "status.json", {"state": "done", "pid": os.getpid()})
    with pytest.raises(RunDirExistsError, match="finished"):
        prepare_run_dir(cfg, 0, tmp_path)
    assert prepare_run_dir(cfg, 1, tmp_path).name == "1"


def test_failed_run_writes_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tiny_config("ppo")
    run_dir = prepare_run_dir(cfg, 0, tmp_path)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom in model construction")

    monkeypatch.setattr(train_mod, "_build_model", boom)
    with pytest.raises(RuntimeError, match="boom"):
        train_mod.train(cfg, 0, run_dir)
    status = read_status(run_dir)
    assert status is not None
    assert status["state"] == "failed"
    assert "boom in model construction" in status["error"]
    assert any("boom" in line for line in status["traceback_tail"])
    assert "Traceback" in capsys.readouterr().err
    with pytest.raises(FileExistsError):
        train_mod.train(cfg, 0, run_dir)  # a failed run dir is never reused
    assert prepare_run_dir(cfg, 0, tmp_path).name == "0_r2"


def test_launchable_is_fifo_within_the_cap() -> None:
    jobs = [Job("a", 0, 17), Job("a", 1, 17), Job("b", 0, 9), Job("a", 2, 17)]
    assert launchable(jobs, 0, 34) == [0, 1]
    assert launchable(jobs, 17, 34) == [0]
    assert launchable(jobs, 30, 34) == []  # FIFO: the 9-slot job does not jump the queue
    jobs[0].state = jobs[1].state = "running"
    assert launchable(jobs, 34 - 9, 34) == [2]


def test_used_workers_counts_live_runs_only(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    sweeps = runs / "_sweeps"
    for name, state, pid, workers in (
        ("live", "running", os.getpid(), 17),
        ("dead", "running", 2**22 + 12345, 17),
        ("done", "done", os.getpid(), 17),
    ):
        (runs / name / "0").mkdir(parents=True)
        atomic_write_json(
            runs / name / "0" / "status.json", {"state": state, "pid": pid, "workers": workers}
        )
    sweeps.mkdir(parents=True)
    starting = runs / "starting" / "0"
    starting.mkdir(parents=True)
    atomic_write_json(
        sweeps / "s.json",
        {
            "jobs": [
                {"state": "running", "run_dir": str(starting), "pid": os.getpid(), "cost": 9},
                {
                    "state": "running",
                    "run_dir": str(runs / "live" / "0"),
                    "pid": os.getpid(),
                    "cost": 17,
                },
            ]
        },
    )
    assert used_workers(runs, sweeps) == 17 + 9


def test_new_sweep_file(tmp_path: Path) -> None:
    cfg = str(RL_CONFIG_DIR / "ppo_smoke.yaml")
    path = new_sweep("sweep", [(cfg, 0), (cfg, 1)], 34, sweeps_dir=tmp_path, label="smoke")
    data = json.loads(path.read_text())
    assert data["kind"] == "sweep" and data["max_workers"] == 34
    assert [j["cost"] for j in data["jobs"]] == [8, 8]  # ceil(0.4 * 16) + 1
    assert all(j["state"] == "queued" for j in data["jobs"])
    with pytest.raises(ValueError, match="max_workers"):
        new_sweep("sweep", [(cfg, 0)], 7, sweeps_dir=tmp_path)
