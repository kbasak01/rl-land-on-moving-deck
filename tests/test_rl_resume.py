"""Phase 5 recovery: checkpoint resume, end to end, for a tiny PPO and a tiny SAC run.

Each run is killed after a checkpoint and after one more evaluation (a simulated host kill:
``SystemExit`` from inside ``learn``, so the run records ``failed``), then resumed from that
checkpoint in the same run directory. Checked:

* the resumed run reaches its budget with ``state: done``;
* the weights and ``VecNormalize`` statistics it restored are exactly the checkpoint's
  (SHA-256 digests recorded in ``status.json`` vs. digests of the checkpoint files loaded
  independently); the SAC replay buffer size is the saved one;
* ``status.json`` and ``provenance.json`` record ``resumed_from``, the resume time and step;
  ``provenance.json`` keeps every original field;
* ``evals.csv`` / ``eval_episodes.csv`` hold one trajectory (the abandoned branch's rows
  moved to ``*.before_resume1.csv``, byte-identical to the pre-resume files); monitor and
  TensorBoard files of the resumed segment are new files, nothing is overwritten;
* resume refusals: done run, live run, unreconciled run, non-latest or non-resumable
  checkpoint, other seed, other config.

Units: steps are env control steps (1/30 s model scale each).
"""

import csv
import json
import os
import pickle
from pathlib import Path
from typing import Any

import pytest
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize

import rld.rl.train as train_mod
from _rl_helpers import tiny_config
from rld.rl.callbacks import StatusCallback, atomic_write_json
from rld.rl.config import TrainConfig, config_to_dict
from rld.rl.curves import read_monitor
from rld.rl.resume import (
    ResumeError,
    normalizer_digest,
    params_digest,
    plan_resume,
    resolve_resume_target,
)
from rld.rl.train import load_policy, prepare_resume, prepare_run_dir, read_status, train


def _killing_status_callback(kill_at: int) -> type[StatusCallback]:
    class Kill(StatusCallback):
        """Heartbeat, then a simulated host kill once ``kill_at`` steps are reached."""

        def _on_step(self) -> bool:
            super()._on_step()
            if self.num_timesteps >= kill_at:
                raise SystemExit("simulated host kill")
            return True

    return Kill


def _evals(run_dir: Path, name: str = "evals.csv") -> list[dict[str, str]]:
    with (run_dir / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _kill_and_resume(
    cfg: TrainConfig, root: Path, kill_at: int, ckpt_steps: int, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Any], dict[str, bytes], dict[str, Any], dict[str, Any]]:
    run_dir = prepare_run_dir(cfg, 0, root)
    with monkeypatch.context() as patch:
        patch.setattr(train_mod, "StatusCallback", _killing_status_callback(kill_at))
        with pytest.raises(SystemExit, match="simulated host kill"):
            train(cfg, 0, run_dir)
    killed = read_status(run_dir)
    assert killed is not None and killed["state"] == "failed"
    ckpt = run_dir / "checkpoints" / f"step_{ckpt_steps:010d}"
    assert resolve_resume_target(run_dir) == (run_dir.resolve(), ckpt.resolve())
    before = {n: (run_dir / n).read_bytes() for n in ("evals.csv", "eval_episodes.csv")}
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert prepare_resume(cfg, 0, run_dir) == (run_dir.resolve(), ckpt.resolve())
    status = train(cfg, 0, run_dir, resume_from=ckpt)
    return run_dir, ckpt, killed, before, provenance, status


def _check_common(
    cfg: TrainConfig,
    run_dir: Path,
    ckpt: Path,
    killed: dict[str, Any],
    before: dict[str, bytes],
    provenance: dict[str, Any],
    status: dict[str, Any],
    algo_cls: Any,
    eval_steps: list[int],
) -> dict[str, Any]:
    on_disk = json.loads((run_dir / "status.json").read_text())
    assert on_disk["state"] == status["state"] == "done"
    assert on_disk["steps"] >= cfg.total_steps
    assert on_disk["resumed_from"] == str(ckpt.resolve())
    assert on_disk["resume_steps"] == int(ckpt.name[5:]) and on_disk["resumed_at"]
    assert on_disk["started"] == killed["started"] and on_disk["run_id"] == killed["run_id"]
    assert on_disk["wall_s"] > killed["wall_s"] and on_disk["cpu_hours"] > killed["cpu_hours"]
    (record,) = on_disk["resumes"]
    assert record["previous_state"] == "failed" and "simulated" in record["previous_error"]
    assert record["abandoned_steps"] == killed["steps"] - record["resume_steps"] > 0

    # The restored state is exactly the checkpoint's, loaded independently here.
    saved_model = algo_cls.load(str(ckpt / "model.zip"), device="cpu")
    with (ckpt / "vecnormalize.pkl").open("rb") as handle:
        saved_norm: VecNormalize = pickle.load(handle)
    restored = record["restored"]
    assert restored["params_sha256"] == params_digest(saved_model)
    assert restored["vecnormalize_sha256"] == normalizer_digest(saved_norm)
    assert restored["num_timesteps"] == record["resume_steps"]

    prov = json.loads((run_dir / "provenance.json").read_text())
    assert {k: prov[k] for k in provenance} == provenance  # original fields kept
    (prov_record,) = prov["resumes"]
    assert prov_record["resumed_from"] == str(ckpt.resolve()) and prov_record["environment"]
    assert prov_record["restored"] == restored

    # One learning curve; the abandoned branch archived byte for byte.
    evals = _evals(run_dir)
    assert [int(r["steps"]) for r in evals] == eval_steps
    assert [int(r["eval_index"]) for r in evals] == list(range(len(eval_steps)))
    for name in ("evals.csv", "eval_episodes.csv"):
        archive = run_dir / name.replace(".csv", ".before_resume1.csv")
        assert archive.read_bytes() == before[name]
    episodes = _evals(run_dir, "eval_episodes.csv")
    assert sorted({int(r["eval_index"]) for r in episodes}) == list(range(len(eval_steps)))
    monitors = sorted(p.name for p in (run_dir / "monitor").glob("*.monitor.csv"))
    ranks = range(cfg.n_envs)
    assert monitors == sorted(
        [f"{r}.monitor.csv" for r in ranks] + [f"{r}.resume1.monitor.csv" for r in ranks]
    )
    kept, every = read_monitor(run_dir), read_monitor(run_dir, include_abandoned=True)
    assert set(kept["segment"]) <= {0, 1} and not kept["abandoned"].any()
    assert len(every) >= len(kept) and kept["t_abs"].is_monotonic_increasing
    assert (run_dir / "tb" / "eval_resume1").is_dir()
    assert (run_dir / "tb" / f"{cfg.algo}_resume1_0").is_dir()
    assert (run_dir / "final" / "model.zip").exists()
    assert load_policy(run_dir).steps == on_disk["steps"]
    with pytest.raises(ResumeError, match="finished"):
        prepare_resume(cfg, 0, run_dir)
    return record


@pytest.mark.slow
@pytest.mark.pybullet
def test_ppo_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tiny_config(
        "ppo",
        total_steps=1024,
        n_envs=2,
        vec_env="subproc",
        ppo__n_steps=128,
        ppo__batch_size=64,
        ppo__n_epochs=1,
        eval__interval_steps=256,
        eval__n_envs=2,
        eval__episodes_per_ss=1,
        checkpoint_interval_steps=512,
    )
    run_dir, ckpt, killed, before, prov, status = _kill_and_resume(
        cfg, tmp_path, 768, 512, monkeypatch
    )
    assert killed["steps"] == 768
    _check_common(cfg, run_dir, ckpt, killed, before, prov, status, PPO, [256, 512, 768, 1024])
    names = sorted(p.name for p in (run_dir / "checkpoints").iterdir())
    assert names == ["step_0000000512", "step_0000001024"]
    meta = json.loads((ckpt / "checkpoint.json").read_text())
    assert meta["resumable"] is True and meta["replay_buffer"] is False


@pytest.mark.slow
@pytest.mark.pybullet
def test_sac_resume_keeps_the_replay_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tiny_config(
        "sac",
        total_steps=600,
        sac__learning_starts=100,
        sac__batch_size=32,
        eval__interval_steps=200,
        checkpoint_interval_steps=300,
    )
    run_dir, ckpt, killed, before, prov, status = _kill_and_resume(
        cfg, tmp_path, 400, 300, monkeypatch
    )
    record = _check_common(cfg, run_dir, ckpt, killed, before, prov, status, SAC, [200, 400, 600])
    meta = json.loads((ckpt / "checkpoint.json").read_text())
    assert meta["resumable"] is True and meta["replay_buffer"] is True
    with (ckpt / "replay_buffer.pkl").open("rb") as handle:
        saved_buffer = pickle.load(handle)
    # The checkpoint fires before the step's transition is stored: steps - n_envs.
    assert saved_buffer.size() == meta["replay_buffer_size"] == 300 - cfg.n_envs
    assert record["restored"]["replay_buffer_size"] == saved_buffer.size()


# --------------------------------------------------------------------------- refusals


def _fake_run(tmp_path: Path, cfg: TrainConfig, **status: Any) -> tuple[Path, Path, str]:
    import hashlib

    import yaml

    run_dir = (tmp_path / "run").resolve()
    ckpt = run_dir / "checkpoints" / "step_0000000512"
    ckpt.mkdir(parents=True)
    text = yaml.safe_dump(config_to_dict(cfg), sort_keys=False)
    (run_dir / "config.yaml").write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    (ckpt / "model.zip").write_bytes(b"")
    with (ckpt / "resume_state.pkl").open("wb") as handle:
        pickle.dump(
            {"format": 1, "algo": "ppo", "seed": 0, "config_sha256": sha, "steps": 512}, handle
        )
    atomic_write_json(run_dir / "status.json", {"state": "failed", "pid": 2**22 + 1, **status})
    return run_dir, ckpt, text


def test_plan_resume_refusals(tmp_path: Path) -> None:
    import hashlib

    cfg = tiny_config("ppo")
    run_dir, ckpt, text = _fake_run(tmp_path, cfg)
    sha = hashlib.sha256(text.encode()).hexdigest()

    def plan(**over: Any) -> Any:
        kwargs: dict[str, Any] = {
            "seed": 0,
            "algo": "ppo",
            "config_text": text,
            "config_sha256": sha,
            "status": read_status(run_dir),
        }
        kwargs.update(over)
        return plan_resume(run_dir, ckpt, **kwargs)

    assert plan().segment == 1 and plan().steps == 512
    assert plan(status={"state": "failed", "resumes": [{}]}).segment == 2
    with pytest.raises(ResumeError, match="finished"):
        plan(status={"state": "done"})
    with pytest.raises(ResumeError, match="live"):
        plan(status={"state": "running", "pid": os.getpid()})
    with pytest.raises(ResumeError, match="reconcile"):
        plan(status={"state": "running", "pid": 2**22 + 1})
    with pytest.raises(ResumeError, match="seed"):
        plan(seed=1)
    with pytest.raises(ResumeError, match="config"):
        plan(config_text=text + "#", config_sha256="0" * 64)
    newer = run_dir / "checkpoints" / "step_0000001024"
    newer.mkdir()
    (newer / "model.zip").write_bytes(b"")
    with pytest.raises(ResumeError, match="latest"):
        plan()
    with pytest.raises(ResumeError, match="not resumable"):
        plan_resume(
            run_dir,
            newer,
            seed=0,
            algo="ppo",
            config_text=text,
            config_sha256=sha,
            status=read_status(run_dir),
        )
