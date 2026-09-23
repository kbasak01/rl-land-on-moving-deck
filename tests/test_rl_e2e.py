"""Phase 5: a tiny end-to-end PPO run (2 048 steps, 2 subprocess workers) and a tiny SAC run.

The run must leave: ``status.json`` in state ``done``; ``evals.csv`` with one row per
evaluation and outcome fractions summing to 1; ``eval_episodes.csv``; ``monitor/`` files;
TensorBoard events; every checkpoint and ``final/`` holding **both** ``model.zip`` and
``vecnormalize.pkl``. :func:`rld.rl.train.load_policy` must return a
:class:`rld.control.base.Controller` that reproduces the model's action through the
checkpoint's frozen statistics, rejects a ship-motion feed and a privileged context, and
plugs into ``rld.eval.runner.callable_spec``. The learning-curve plot must render.

Units: steps are env control steps (1/30 s model scale each).
"""

import csv
import functools
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import VecNormalize

from _rl_helpers import tiny_config
from rld.control.base import Controller
from rld.envs.touchdown import OUTCOMES
from rld.eval.envs import load_eval_configs
from rld.eval.runner import callable_spec
from rld.rl.config import TrainConfig
from rld.rl.curves import plot_learning_curves
from rld.rl.train import build_policy, load_policy, load_vecnormalize, prepare_run_dir, train

pytestmark = [pytest.mark.slow, pytest.mark.pybullet]


def _run(cfg: TrainConfig, root: Path) -> tuple[Path, dict[str, object]]:
    run_dir = prepare_run_dir(cfg, 0, root)
    status = train(cfg, 0, run_dir)
    return run_dir, status


@pytest.fixture(scope="module")
def ppo_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    cfg = tiny_config(
        "ppo",
        total_steps=2048,
        n_envs=2,
        vec_env="subproc",
        ppo__n_steps=512,
        ppo__batch_size=256,
        ppo__n_epochs=2,
        eval__interval_steps=1024,
        eval__n_envs=2,
        eval__episodes_per_ss=2,
        checkpoint_interval_steps=1024,
    )
    return _run(cfg, tmp_path_factory.mktemp("runs"))


def test_ppo_run_artifacts(ppo_run: tuple[Path, dict[str, object]]) -> None:
    run_dir, status = ppo_run
    on_disk = json.loads((run_dir / "status.json").read_text())
    assert on_disk["state"] == status["state"] == "done"
    assert on_disk["steps"] >= 2048 and on_disk["budget"] == 2048
    assert on_disk["fps"] > 0 and on_disk["eval_env_steps"] > 0
    assert on_disk["fps_excl_eval"] >= on_disk["fps"] and on_disk["eval_wall_s"] > 0
    assert on_disk["curriculum_stage"] == "SS3"
    fractions = on_disk["last_eval"]["outcome_fractions"]
    assert set(fractions) == set(OUTCOMES) and abs(sum(fractions.values()) - 1.0) < 1e-9
    print(
        f"\ne2e PPO: {on_disk['steps']} steps, {on_disk['fps']:.1f} steps/s overall "
        f"(evaluation included), {on_disk['fps_excl_eval']:.1f} steps/s excluding evaluation, "
        f"wall {on_disk['wall_s']:.1f} s"
    )

    with (run_dir / "evals.csv").open() as handle:
        evals = list(csv.DictReader(handle))
    assert [int(r["steps"]) for r in evals] == [1024, 2048]
    for row in evals:
        assert abs(sum(float(row[f"frac_{o}"]) for o in OUTCOMES) - 1.0) < 1e-9
        assert int(row["n_episodes"]) == 6
    with (run_dir / "eval_episodes.csv").open() as handle:
        episodes = list(csv.DictReader(handle))
    assert len(episodes) == 12 and {r["ss"] for r in episodes} == {"SS3", "SS4", "SS5"}

    checkpoints = sorted((run_dir / "checkpoints").iterdir())
    assert [c.name for c in checkpoints] == ["step_0000001024", "step_0000002048"]
    for ckpt in [*checkpoints, run_dir / "final"]:
        assert (ckpt / "model.zip").exists() and (ckpt / "vecnormalize.pkl").exists()
    assert list((run_dir / "monitor").glob("*.monitor.csv"))
    assert list((run_dir / "tb").rglob("events.out.tfevents.*"))
    for name in ("config.yaml", "provenance.json"):  # config_source.yaml only for a file
        assert (run_dir / name).exists()


def test_load_policy_is_a_controller_with_frozen_stats(
    ppo_run: tuple[Path, dict[str, object]],
) -> None:
    run_dir, _ = ppo_run
    policy = load_policy(run_dir)
    assert isinstance(policy, Controller)
    assert policy.privileged is False and policy.needs_motion_feed is False
    assert isinstance(policy.normalizer, VecNormalize)
    assert policy.normalizer.training is False and policy.normalizer.norm_reward is False
    saved = load_vecnormalize(run_dir / "final" / "vecnormalize.pkl")
    assert np.array_equal(policy.normalizer.obs_rms.mean, saved.obs_rms.mean)
    assert np.array_equal(policy.normalizer.obs_rms.var, saved.obs_rms.var)

    policy.reset(0)
    rng = np.random.default_rng(0)
    obs = rng.normal(size=policy.model.observation_space.shape).astype(np.float32)
    action = policy.act(obs)
    assert action.shape == (3,) and action.dtype == np.float32
    assert np.all(np.abs(action) <= 1.0)
    expected, _ = policy.model.predict(saved.normalize_obs(obs.reshape(1, -1)), deterministic=True)
    assert np.allclose(action, np.clip(expected.reshape(-1), -1.0, 1.0))
    before = policy.normalizer.obs_rms.mean.copy()
    for _ in range(5):
        policy.act(obs)
    assert np.array_equal(policy.normalizer.obs_rms.mean, before)

    with pytest.raises(ValueError, match="ShipMotionFeed"):
        policy.reset(0, motion_feed=object())
    with pytest.raises(ValueError, match="privileged"):
        policy.reset(0, context=object())  # type: ignore[arg-type]
    step = load_policy(run_dir, 1024)
    assert step.steps == 1024


def test_callable_spec_hook(ppo_run: tuple[Path, dict[str, object]]) -> None:
    run_dir, _ = ppo_run
    spec = callable_spec("ppo", functools.partial(build_policy, run_dir=run_dir), run_seed=0)
    spec = pickle.loads(pickle.dumps(spec))
    policy = spec.build(load_eval_configs())
    assert policy.privileged is False  # type: ignore[attr-defined]


def test_learning_curve_plot(ppo_run: tuple[Path, dict[str, object]], tmp_path: Path) -> None:
    run_dir, _ = ppo_run
    out = plot_learning_curves([run_dir], tmp_path / "curve.png")
    assert out.exists() and out.stat().st_size > 10_000
    both = plot_learning_curves([run_dir, run_dir], tmp_path / "curves.png")
    assert both.exists()


def test_sac_tiny_run(tmp_path: Path) -> None:
    cfg = tiny_config(
        "sac",
        total_steps=300,
        sac__learning_starts=100,
        sac__batch_size=32,
        eval__interval_steps=300,
        checkpoint_interval_steps=300,
    )
    run_dir, status = _run(cfg, tmp_path)
    assert status["state"] == "done"
    assert (run_dir / "final" / "model.zip").exists()
    assert (run_dir / "final" / "vecnormalize.pkl").exists()
    policy = load_policy(run_dir)
    assert policy.normalizer is not None and policy.normalizer.norm_reward is False
    obs = np.zeros(policy.model.observation_space.shape, dtype=np.float32)
    assert policy.act(obs).shape == (3,)
