"""Helpers for ``tests/test_deploy_*.py``: a tiny PPO policy with a frozen normaliser.

The policy is the same SB3 ``ActorCriticPolicy`` family the real runs use (flatten extractor,
separate actor/critic MLPs, tanh, linear mean head), only narrower, with a frozen ``VecNormalize``
whose statistics are drawn from a seeded generator -- one entry gets a very small variance, as the
real deck-normal entry has (sigma about 0.004), so the fold is tested where float32 is tightest.

Units: observation entries are arbitrary (dimensionless here); actions normalised in [-1, 1].
"""

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from conftest import BASE_SEED
from rld.rl.policy import LearnedPolicy


class BoxEnv(gym.Env[np.ndarray, np.ndarray]):
    """A do-nothing env with a ``dim``-wide observation box and a 3-wide action box."""

    def __init__(self, dim: int) -> None:
        self.observation_space = Box(-np.inf, np.inf, (dim,), dtype=np.float32)
        self.action_space = Box(-1.0, 1.0, (3,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}


def tiny_policy(
    dim: int = 25, width: int = 32, seed: int = 0, bias: float = 0.0, name: str = "test_ppo"
) -> LearnedPolicy:
    """Return a :class:`LearnedPolicy` with random weights and a frozen, non-trivial normaliser.

    Args:
        dim: Input width.
        width: Hidden width (two tanh layers).
        seed: Offset on :data:`BASE_SEED`.
        bias: Added to the mean head's bias (a large value saturates the output clip).
        name: Policy label.

    Returns:
        The policy (CPU).
    """
    rng = np.random.default_rng(BASE_SEED + 800 + seed)
    venv = VecNormalize(
        DummyVecEnv([lambda: BoxEnv(dim)]), norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    venv.obs_rms.mean = rng.normal(0.0, 1.0, dim)
    var = rng.uniform(0.01, 2.0, dim)
    var[min(3, dim - 1)] = 0.004**2
    venv.obs_rms.mean[min(3, dim - 1)] = 0.998
    venv.obs_rms.var = var
    venv.training = False
    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs={
            "net_arch": {"pi": [width, width], "vf": [width, width]},
            "activation_fn": torch.nn.Tanh,
        },
        device="cpu",
        seed=BASE_SEED + seed,
        n_steps=64,
        batch_size=64,
    )
    with torch.no_grad():
        for p in model.policy.action_net.parameters():
            p.copy_(torch.from_numpy(rng.normal(0.0, 0.5, tuple(p.shape)).astype(np.float32)))
        model.policy.action_net.bias.add_(bias)
    return LearnedPolicy(name, model, venv, Path(), Path(), 0)
