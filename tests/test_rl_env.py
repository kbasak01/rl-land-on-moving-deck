"""Phase 5: the training env stack -- timeouts truncate, and VecNormalize freezes at eval.

* A hovering policy times out; through the pool wrapper, ``Monitor`` and ``DummyVecEnv`` the
  timeout arrives as ``TimeLimit.truncated = True`` (i.e. ``truncated`` and not
  ``terminated``), so PPO bootstraps through it instead of learning that hovering is free.
* ``VecNormalize`` statistics saved with a checkpoint reload bit-identically; the eval loader
  sets ``training=False`` and ``norm_reward=False``, after which stepping leaves every
  statistic unchanged and rewards come back raw; the in-training evaluator's frozen copy is
  independent of the training normaliser.

Units: steps are env control steps (1/30 s model scale each); rewards dimensionless.
"""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from _rl_helpers import tiny_config
from conftest import BASE_SEED
from rld.rl.callbacks import frozen_normalizer
from rld.rl.train import build_env_configs, load_vecnormalize, training_pools
from rld.rl.wrappers import EnvFactory


@pytest.fixture(scope="module")
def factory(tmp_path_factory: pytest.TempPathFactory) -> EnvFactory:
    cfg = tiny_config("ppo")
    cfgs = build_env_configs(cfg)
    train, _ = training_pools(cfg, cfgs)
    return EnvFactory(
        cfgs, tuple(train), "aft", BASE_SEED, 0, "SS3", tmp_path_factory.mktemp("monitor")
    )


@pytest.mark.pybullet
@pytest.mark.slow
def test_hover_times_out_as_truncation(factory: EnvFactory) -> None:
    venv = DummyVecEnv([factory])
    try:
        venv.seed(BASE_SEED)
        venv.reset()
        info: dict[str, object] = {}
        for _ in range(400):
            _, _, dones, infos = venv.step(np.zeros((1, 3), dtype=np.float32))
            if dones[0]:
                info = infos[0]
                break
        assert info, "episode did not end within 400 steps"
        assert info["outcome"] == "timeout"
        assert info["TimeLimit.truncated"] is True
        assert info["ss"] == "SS3"
        assert "episode" in info  # Monitor recorded it
    finally:
        venv.close()
    monitor = sorted(Path(str(factory.monitor_dir)).glob("*.monitor.csv"))
    assert monitor and "timeout" in monitor[0].read_text()


def _stats(vn: VecNormalize) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
    return (
        vn.obs_rms.mean.copy(),
        vn.obs_rms.var.copy(),
        float(vn.obs_rms.count),
        np.asarray(vn.ret_rms.mean).copy(),
        np.asarray(vn.ret_rms.var).copy(),
        float(vn.ret_rms.count),
    )


def _same(a: tuple[object, ...], b: tuple[object, ...]) -> bool:
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b, strict=True))


@pytest.mark.pybullet
@pytest.mark.slow
def test_vecnormalize_saved_reloaded_and_frozen(factory: EnvFactory, tmp_path: Path) -> None:
    rng = np.random.default_rng(BASE_SEED)
    # Its own monitor directory: the shared fixture's already holds 0.monitor.csv, and a
    # monitor file is never overwritten (EnvFactory refuses).
    own = replace(factory, monitor_dir=tmp_path / "monitor")
    train = VecNormalize(DummyVecEnv([own]), norm_obs=True, norm_reward=True, clip_obs=10.0)
    try:
        train.seed(BASE_SEED)
        train.reset()
        for _ in range(80):
            train.step(rng.uniform(-0.3, 0.3, size=(1, 3)).astype(np.float32))
        path = tmp_path / "vecnormalize.pkl"
        train.save(str(path))
        saved = _stats(train)
        assert saved[2] > 1.0  # statistics actually accumulated

        # The evaluator's frozen copy: independent, and using it changes nothing.
        frozen = frozen_normalizer(train)
        assert frozen.training is False and frozen.norm_reward is False
        before = copy.deepcopy(train.obs_rms)
        frozen.obs_rms.mean += 1.0
        assert np.array_equal(train.obs_rms.mean, before.mean)
    finally:
        train.close()

    loaded = load_vecnormalize(path, DummyVecEnv([replace(own, monitor_dir=None)]))
    try:
        assert loaded.training is False and loaded.norm_reward is False
        assert _same(_stats(loaded), saved)  # bit-identical reload
        loaded.seed(BASE_SEED + 1)
        obs = loaded.reset()
        for _ in range(60):
            obs, reward, _, _ = loaded.step(rng.uniform(-0.3, 0.3, size=(1, 3)).astype(np.float32))
            raw_reward = loaded.get_original_reward()
            assert np.array_equal(reward, raw_reward)  # reward not normalised at eval
            assert np.allclose(obs, loaded.normalize_obs(loaded.get_original_obs()))
        assert _same(_stats(loaded), saved)  # stats unchanged across eval steps
        resaved = tmp_path / "again.pkl"
        loaded.save(str(resaved))
    finally:
        loaded.close()

    detached = load_vecnormalize(path)
    assert detached.training is False and detached.norm_reward is False
    assert _same(_stats(detached), saved)
    probe = rng.normal(size=(4, detached.obs_rms.mean.size)).astype(np.float32)
    detached.normalize_obs(probe)
    assert _same(_stats(detached), saved)
    assert _same(_stats(load_vecnormalize(resaved)), saved)  # save -> load -> save -> load
