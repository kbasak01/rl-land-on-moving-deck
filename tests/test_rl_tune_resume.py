"""Phase 5 recovery: the scheduler's global queue and slot cost, and re-invoking a search.

* The slot cost counts the learner's torch threads (a 4-thread SAC run costs 8).
* Two live schedulers share the cap as one FIFO by enqueue time: the SAC scheduler cannot
  take slots a queued PPO trial enqueued earlier is waiting for (the 2026-09-24 starvation).
* Queues of sweeps whose scheduler is dead are ignored.
* A re-invoked search skips done trials, resumes or re-trains failed ones (per
  ``resume_failed``), starts missing ones, and flags live / unreconciled ones.
* ``collect_trials`` scores the latest ``done`` run (``_r10`` after ``_r2``) and lists the
  runs it supersedes.

Units: costs are worker slots; steps are env control steps.
"""

import csv
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import yaml

from _rl_helpers import tiny_config
from rld.rl.callbacks import atomic_write_json
from rld.rl.config import RL_CONFIG_DIR, apply_overrides, load_yaml, train_config_from_dict
from rld.rl.scheduler import QueuedJob, global_launchable, new_sweep, queued_elsewhere
from rld.rl.tuning import collect_trials, load_tune_config, plan_trials, trial_run_dirs

DEAD_PID = 2**22 + 12345


def test_slot_cost_counts_torch_threads() -> None:
    raw = load_yaml(RL_CONFIG_DIR / "sac.yaml")
    one = train_config_from_dict(raw)
    four = train_config_from_dict(apply_overrides(raw, {"torch_threads": 4}))
    assert one.workers == math.ceil(0.4 * 8) + 1 == 5
    assert four.workers == math.ceil(0.4 * 8) + 4 == 8
    assert train_config_from_dict(load_yaml(RL_CONFIG_DIR / "ppo.yaml")).workers == 8
    assert tiny_config("sac", torch_threads=2).workers == 1 + 2


def test_global_fifo_prevents_starvation() -> None:
    # 2026-09-24: six SAC trials (5 slots each) held 30 of 34 slots; a PPO trial (8) was
    # queued first. Whenever a SAC trial ended, the SAC scheduler refilled the 5 slots.
    ppo = [QueuedJob("2026-09-24T08:22:42+00:00", "ppo", i, 8) for i in (16, 17)]
    sac = [QueuedJob("2026-09-24T08:22:45+00:00", "sac", i, 5) for i in range(10, 20)]
    queue = sac + ppo  # order of discovery does not matter
    assert global_launchable("sac", queue, 25, 34) == []  # 9 free: reserved for PPO's 8
    assert global_launchable("ppo", queue, 25, 34) == [16]
    assert global_launchable("sac", queue, 34 - 21, 34) == [10]  # 21 free: 8 + 8 + 5
    assert global_launchable("ppo", queue, 34 - 21, 34) == [16, 17]
    # Same enqueue time: sweep id, then index.
    tie = [QueuedJob("t", "b", 0, 5), QueuedJob("t", "a", 1, 5), QueuedJob("t", "a", 0, 5)]
    assert global_launchable("a", tie, 0, 10) == [0, 1]


def test_queued_elsewhere_ignores_dead_schedulers(tmp_path: Path) -> None:
    sweeps = tmp_path / "_sweeps"
    sweeps.mkdir()
    job = {"config": "c", "seed": 0, "cost": 8, "state": "queued"}
    for name, pid, state in (
        ("live", os.getpid(), "running"),
        ("dead", DEAD_PID, "running"),
        ("over", os.getpid(), "done"),
    ):
        atomic_write_json(
            sweeps / f"{name}.json",
            {
                "sweep_id": name,
                "state": state,
                "scheduler_pid": pid,
                "created": "t0",
                "jobs": [job],
            },
        )
    own = sweeps / "own.json"
    atomic_write_json(own, {"sweep_id": "own", "state": "running", "scheduler_pid": os.getpid()})
    found = queued_elsewhere(sweeps, own)
    assert found == [QueuedJob("t0", "live", 0, 8)]


def test_new_sweep_records_resume_and_enqueue_time(tmp_path: Path) -> None:
    cfg = str(RL_CONFIG_DIR / "ppo_smoke.yaml")
    path = new_sweep("tune", [(cfg, 0), (cfg, 0, "/runs/x/0")], 34, sweeps_dir=tmp_path, poll_s=5)
    data = json.loads(path.read_text())
    assert data["poll_s"] == 5.0
    assert [j["resume_from"] for j in data["jobs"]] == [None, "/runs/x/0"]
    assert all(j["enqueued"] == data["created"] for j in data["jobs"])


# --------------------------------------------------------------------------- tuning


def _tune_cfg(tmp_path: Path, **changes: Any) -> Any:
    raw = yaml.safe_load((RL_CONFIG_DIR / "tune_ppo.yaml").read_text())
    raw.update({"results_dir": str(tmp_path / "results"), "run_group": "tt", "trials": 5})
    raw.update(changes)
    path = tmp_path / "tune.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_tune_config(path)


def _run(root: Path, trial: int, name: str, **status: Any) -> Path:
    run_dir = root / "tt" / f"trial_{trial:02d}" / name
    run_dir.mkdir(parents=True)
    atomic_write_json(run_dir / "status.json", {"pid": DEAD_PID, **status})
    return run_dir


def _checkpoint(run_dir: Path, steps: int, resumable: bool) -> Path:
    ckpt = run_dir / "checkpoints" / f"step_{steps:010d}"
    ckpt.mkdir(parents=True)
    (ckpt / "model.zip").write_bytes(b"")
    if resumable:
        with (ckpt / "resume_state.pkl").open("wb") as handle:
            pickle.dump({"format": 1, "algo": "ppo", "steps": steps}, handle)
    return ckpt


def test_resume_failed_is_optional_and_boolean(tmp_path: Path) -> None:
    assert _tune_cfg(tmp_path).resume_failed is False
    assert _tune_cfg(tmp_path, resume_failed=True).resume_failed is True
    try:
        _tune_cfg(tmp_path, resume_failed="yes")
    except ValueError as exc:
        assert "resume_failed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a non-boolean resume_failed must be rejected")


def test_plan_trials(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _run(root, 0, "0", state="done")
    failed = _run(root, 1, "0", state="failed")
    _checkpoint(failed, 100, resumable=False)
    ckpt = _checkpoint(failed, 200, resumable=True)
    legacy = _run(root, 2, "0", state="failed")
    _checkpoint(legacy, 100, resumable=False)
    _run(root, 4, "0", state="running")  # dead writer, never reconciled

    plans = {p.trial: p for p in plan_trials(_tune_cfg(tmp_path), root)}
    assert [plans[i].action for i in range(5)] == [
        "skip_done",
        "fresh",
        "fresh",
        "fresh",
        "unreconciled",
    ]
    assert plans[1].run_dir == failed and plans[3].run_dir is None

    plans = {p.trial: p for p in plan_trials(_tune_cfg(tmp_path, resume_failed=True), root)}
    assert plans[1].action == "resume" and plans[1].resume_from == ckpt
    assert plans[2].action == "fresh"  # latest checkpoint holds the model only

    # A resumable checkpoint behind a newer model-only one is not offered.
    _checkpoint(failed, 300, resumable=False)
    plans = {p.trial: p for p in plan_trials(_tune_cfg(tmp_path, resume_failed=True), root)}
    assert plans[1].action == "fresh"

    # A live run blocks the trial.
    live = _run(root, 3, "0", state="running")
    atomic_write_json(live / "status.json", {"state": "running", "pid": os.getpid()})
    assert plan_trials(_tune_cfg(tmp_path), root)[3].action == "live"


def _episodes(run_dir: Path, success: int) -> None:
    fields = [
        "eval_index",
        "train_steps",
        "ss",
        "outcome",
        "touchdown_contact",
        "closing_speed_normal_m_s",
        "lateral_offset_m",
        "rel_tilt_deg",
        "td_t_episode_s",
        "detectors_disagree",
        "tunnelled",
    ]
    rows = []
    for index in (0, 1):
        for ss in ("SS3", "SS4", "SS5"):
            for k in range(2):
                ok = index == 1 and k < success
                rows.append(
                    [index, 1000 * (index + 1), ss, "success" if ok else "timeout", ok]
                    + ([0.3, 0.02, 3.0, 2.0] if ok else ["", "", "", ""])
                    + [False, False]
                )
    with (run_dir / "eval_episodes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def test_collect_scores_latest_done_and_lists_superseded(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    cfg = _tune_cfg(tmp_path)
    _run(root, 0, "0", state="failed")
    done = _run(root, 0, "0_r2", state="done", steps=2000)
    _episodes(done, success=2)
    _run(root, 0, "0_r10", state="failed")
    resumed = _run(root, 1, "0", state="done", steps=2000, resumes=[{"resume_steps": 1000}])
    _episodes(resumed, success=1)
    _run(root, 2, "0", state="failed")
    assert [p.name for p in trial_run_dirs(cfg, 0, root)] == ["0", "0_r2", "0_r10"]

    rows = collect_trials(cfg, root)
    by_trial = {r["trial"]: r for r in rows}
    assert by_trial[0]["run_dir"] == str(done)
    assert by_trial[0]["superseded_runs"] == "0:failed;0_r10:failed"
    assert by_trial[0]["mean_success"] == 1.0 and by_trial[0]["selected"]
    # Trial 0 is the base config, whose `reward: {}` flies the committed weights; its row
    # reports those (this used to be None and crashed the collection).
    from rld.envs.config import REWARD_CONFIG, load_reward

    committed = load_reward(REWARD_CONFIG)
    assert by_trial[0]["param_w_progress"] == committed.w_progress
    assert by_trial[0]["param_r_fail"] == committed.r_hard_landing
    assert by_trial[1]["resumed"] is True and by_trial[1]["resume_steps"] == "1000"
    assert by_trial[2]["state"] == "failed" and by_trial[3]["state"] == "missing"
    selection = json.loads((cfg.results_dir / "selection.json").read_text())
    assert selection["winner_trial"] == 0
    assert selection["superseded_runs"] == {"0": ["0:failed", "0_r10:failed"]}
    assert selection["resumed_trials"] == [1]
