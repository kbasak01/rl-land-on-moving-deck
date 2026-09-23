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
