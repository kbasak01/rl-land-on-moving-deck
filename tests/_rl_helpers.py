"""Helpers shared by the ``tests/test_rl_*.py`` files: tiny configs and a fake landing env.

Units: steps are env control steps (1/30 s model scale each).
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from rld.config import load_yaml
from rld.rl.config import RL_CONFIG_DIR, TrainConfig, apply_overrides, train_config_from_dict


def tiny_raw(base: str = "ppo", **overrides: Any) -> dict[str, Any]:
    """Return a raw config derived from a committed one with test-sized overrides.

    Args:
        base: ``"ppo"`` or ``"sac"``.
        **overrides: Dotted keys with ``__`` for ``.`` (``ppo__n_steps=64``).

    Returns:
        The raw mapping.
    """
    raw = load_yaml(RL_CONFIG_DIR / f"{base}.yaml")
    defaults: dict[str, Any] = {
        "run_group": f"test_{base}",
        "method": f"test_{base}",
        "total_steps": 256,
        "n_envs": 1,
        "vec_env": "dummy",
        "eval.interval_steps": 256,
        "eval.n_envs": 1,
        "eval.episodes_per_ss": 1,
        "checkpoint_interval_steps": 256,
        "status_interval_s": 1.0,
        "log_interval": 1,
    }
    if base == "ppo":
        defaults.update({"ppo.n_steps": 128, "ppo.batch_size": 64, "ppo.n_epochs": 1})
    else:
        defaults.update({"sac.learning_starts": 64, "sac.batch_size": 32, "sac.buffer_size": 5000})
    defaults.update({k.replace("__", "."): v for k, v in overrides.items()})
    return apply_overrides(raw, defaults)


def tiny_config(base: str = "ppo", **overrides: Any) -> TrainConfig:
    """Return a parsed test-sized config (see :func:`tiny_raw`)."""
    return train_config_from_dict(tiny_raw(base, **overrides), f"tiny {base}")


class FakeRecord:
    """Minimal ``EpisodeRecord`` stand-in."""

    def as_row(self) -> dict[str, Any]:
        """Return a row with the fields the wrapper reads."""
        return {
            "outcome": "timeout",
            "termination_reason": "time_limit",
            "t0_model_s": 5.0,
            "closing_speed_normal_m_s": None,
            "max_penetration_m": 0.0,
        }


class FakeLanding(gym.Env[np.ndarray, np.ndarray]):
    """A three-step episode env with the landing env's ``set_motion`` hook."""

    observation_space = Box(-1.0, 1.0, (3,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, (3,), dtype=np.float32)

    def __init__(self) -> None:
        """Start empty."""
        self.motion: Any = None
        self.record = FakeRecord()
        self.INIT_XYZS = np.array([[0.1, 0.0, 1.8]])
        self.steps = 0
        self.seeds: list[int | None] = []

    def set_motion(self, motion: Any, pad: Any = None, pad_offset_m: Any = None) -> None:
        """Record the motion source."""
        del pad, pad_offset_m
        self.motion = motion

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset."""
        del options
        self.steps = 0
        self.seeds.append(seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Step; the third step truncates."""
        del action
        self.steps += 1
        return np.zeros(3, dtype=np.float32), 0.0, False, self.steps >= 3, {}


def run_dirs_under(root: Path, group: str) -> list[Path]:
    """Return the run directories of one group, sorted."""
    return sorted(p for p in (root / group).iterdir() if p.is_dir())


def phase6_config(name: str, **overrides: Any) -> TrainConfig:
    """Return a committed Phase 6 config with one in-process worker (network unchanged).

    Args:
        name: Config stem under ``configs/rl/`` (``"residual_ppo"``, ...).
        **overrides: Dotted keys with ``__`` for ``.``.

    Returns:
        The parsed config; the hyperparameters, reward and method components are the
        committed ones.
    """
    raw = load_yaml(RL_CONFIG_DIR / f"{name}.yaml")
    changes: dict[str, Any] = {
        "n_envs": 1,
        "vec_env": "dummy",
        "eval.n_envs": 1,
        "prefetch_reset": False,
    }
    changes.update({k.replace("__", "."): v for k, v in overrides.items()})
    return train_config_from_dict(apply_overrides(raw, changes), f"phase6 {name}")


def fresh_run_dir(cfg: TrainConfig, root: Path, seed: int = 0) -> Path:
    """Write a run directory holding the **untrained** model of ``cfg`` as its ``final/``.

    The model is built exactly as a fresh training run builds it (``_build_model``, so a
    residual run's ``action_net`` is zero-initialised) on one in-process worker with the
    method's wrappers, with fresh ``VecNormalize`` statistics. ``config.yaml`` and
    ``provenance.json`` (with the component digests) are written as ``train`` writes them.

    Args:
        cfg: The config.
        root: Parent directory; the run lands in ``root/<run_group>/<seed>``.
        seed: Training seed.

    Returns:
        The run directory.
    """
    import json

    import yaml
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from rld.rl.callbacks import save_checkpoint
    from rld.rl.config import config_to_dict
    from rld.rl.train import (
        _build_model,
        _factory_components,
        build_env_configs,
        component_provenance,
        training_pools,
    )
    from rld.rl.wrappers import EnvFactory

    run_dir = root / cfg.run_group / str(seed)
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_to_dict(cfg), sort_keys=False))
    (run_dir / "provenance.json").write_text(
        json.dumps({"seed": seed, "components": component_provenance(cfg)})
    )
    cfgs = build_env_configs(cfg)
    train, tune = training_pools(cfg, cfgs)
    components = _factory_components(cfg, train, tune)["train"]
    factory = EnvFactory(cfgs, tuple(train), cfg.pad, seed, 0, "SS3", None, False, **components)
    venv = VecNormalize(
        DummyVecEnv([factory]),
        training=True,
        norm_obs=cfg.normalize.norm_obs,
        norm_reward=cfg.normalize.norm_reward,
        clip_obs=cfg.normalize.clip_obs,
        clip_reward=cfg.normalize.clip_reward,
        gamma=cfg.gamma,
    )
    try:
        model = _build_model(cfg, venv, seed, run_dir)
        save_checkpoint(model, venv, run_dir / "final", {"steps": 0})
    finally:
        venv.close()
    return run_dir
