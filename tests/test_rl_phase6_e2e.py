"""Phase 6 plumbing: the method components reach the tune-pool evaluation and a real run.

* **Evaluation workers.** :class:`~rld.rl.callbacks.TunePoolEvaluator` in worker processes,
  built from :class:`~rld.rl.wrappers.EnvFactory` with the residual component and handed a
  zero-initialised network, returns exactly the episode rows of ``pid_feedforward`` flown on
  the same tune-pool episodes -- the evaluation workers compose through the same
  :func:`~rld.rl.residual.compose_residual` path as training.
* **A tiny run.** ``train`` in a temporary directory (2 subprocess workers, 1 024 steps) with
  residual, forecast block **and** sinusoid motion at once, so every component crosses the
  process boundary: status ``done``, checkpoints with ``vecnormalize.pkl``, the component
  digests in ``provenance.json``, a 31-entry policy that loads as
  :class:`~rld.rl.forecast_obs.ResidualForecastPolicy`. (Not a method: the combination only
  exercises the plumbing. The Phase 6 smoke and final runs are launched by ``make train-bg``.)

Units: steps are env control steps (1/30 s model scale each).
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from _rl_helpers import tiny_config
from conftest import BASE_SEED
from rld.rl.callbacks import TunePoolEvaluator
from rld.rl.config import ForecastObsConfig, ResidualConfig
from rld.rl.forecast_obs import ResidualForecastPolicy, forecaster_dir
from rld.rl.residual import build_base_controller
from rld.rl.train import (
    _build_model,
    build_env_configs,
    build_policy,
    eval_episodes,
    prepare_run_dir,
    run_needs_motion_feed,
    train,
    training_pools,
)
from rld.rl.wrappers import EnvFactory

pytestmark = [pytest.mark.slow, pytest.mark.pybullet]

RESIDUAL = ResidualConfig(alpha=0.3, base="pid_feedforward")
FORECAST = ForecastObsConfig("residual_interval", (1.0, 2.0, 3.0))


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return bool(a == b)


def test_eval_workers_fly_the_residual_path(tmp_path: Path) -> None:
    cfg = tiny_config(
        "ppo",
        eval__episodes_per_ss=2,
        residual={"alpha": 0.3, "base": "pid_feedforward"},
    )
    cfgs = build_env_configs(cfg)
    _, tune = training_pools(cfg, cfgs)
    episodes = eval_episodes(cfg, cfgs)
    factories = [
        EnvFactory(cfgs, tuple(tune), "aft", BASE_SEED, rank, None, None, True, residual=RESIDUAL)
        for rank in range(2)
    ]
    probe = VecNormalize(DummyVecEnv([factories[0]]), norm_obs=True, norm_reward=True)
    model = _build_model(cfg, probe, BASE_SEED, tmp_path)
    evaluator = TunePoolEvaluator(factories, episodes, "subproc")
    try:
        rows, _ = evaluator.evaluate(model, probe)
    finally:
        evaluator.close()
        probe.close()

    # Reference: pid_feedforward on the same queue, no residual wrapper at all.
    plain = EnvFactory(cfgs, tuple(tune), "aft", BASE_SEED, 0, None, None)()
    base = build_base_controller("pid_feedforward", cfgs)
    reference: list[dict[str, Any]] = []
    try:
        plain.get_wrapper_attr("load_episode_queue")(episodes)
        for ep in episodes:
            obs, info = plain.reset()
            base.reset(int(info["episode_seed"]))
            assert info["episode_seed"] == ep.episode_seed
            while True:
                obs, _, terminated, truncated, info = plain.step(base.act(obs))
                if terminated or truncated:
                    break
            reference.append(dict(info["episode_row"]))
    finally:
        plain.close()
    assert len(rows) == len(reference) == 6
    for got, want in zip(rows, reference, strict=True):
        differing = [k for k in want if not _same(got[k], want[k])]
        assert not differing, {k: (got[k], want[k]) for k in differing}


@pytest.mark.skipif(
    not (forecaster_dir("residual_interval") / "meta.json").is_file(),
    reason="no fitted residual_interval under artifacts/dmf (make dmf-forecasters)",
)
def test_tiny_run_with_every_component(tmp_path: Path) -> None:
    cfg = tiny_config(
        "ppo",
        total_steps=1024,
        n_envs=2,
        vec_env="subproc",
        ppo__n_steps=512,
        ppo__batch_size=256,
        ppo__n_epochs=1,
        eval__interval_steps=1024,
        eval__n_envs=2,
        eval__episodes_per_ss=1,
        checkpoint_interval_steps=1024,
        residual={"alpha": 0.3, "base": "pid_feedforward"},
        forecast_obs={"forecaster": "residual_interval", "leads_full_s": [1.0, 2.0, 3.0]},
        motion="sinusoid",
    )
    run_dir = prepare_run_dir(cfg, 0, tmp_path)
    status = train(cfg, 0, run_dir)
    assert status["state"] == "done" and status["steps"] >= 1024
    assert status["last_eval"]["n_episodes"] == 3
    for ckpt in [*sorted((run_dir / "checkpoints").iterdir()), run_dir / "final"]:
        assert (ckpt / "model.zip").exists() and (ckpt / "vecnormalize.pkl").exists()
    components = json.loads((run_dir / "provenance.json").read_text())["components"]
    assert components["motion"] == "sinusoid"
    assert set(components) >= {
        "residual_base_config_sha256",
        "forecaster_meta_sha256",
        "deck_stats_seeds_sha256",
    }
    text = (run_dir / "config.yaml").read_text()
    assert "residual:" in text and "forecast_obs:" in text and "motion: sinusoid" in text
    assert run_needs_motion_feed(run_dir) is True
    policy = build_policy(build_env_configs(cfg), run_dir=run_dir)
    assert isinstance(policy, ResidualForecastPolicy)
    assert policy.model.observation_space.shape == (31,)
    assert list((run_dir / "monitor").glob("*.monitor.csv"))
