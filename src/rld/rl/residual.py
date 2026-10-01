"""Residual RL on ``pid_feedforward``: ``clip(a_base(o) + alpha * pi(o), -1, 1)`` (P3-D1 s. 6).

One pure function composes the action, and both places that fly a residual policy call it:

* **training and the tune-pool evaluation** -- :class:`ResidualActionWrapper`, a gym wrapper
  inside every env worker (:class:`rld.rl.wrappers.EnvFactory`). It owns its base controller,
  resets it at every episode start and feeds it the **raw** environment observation, so the
  learner (and ``VecNormalize``) only ever sees ``pi``'s own action ``a_res``;
* **evaluation on the frozen lists** -- :class:`ResidualPolicy`, a
  :class:`~rld.rl.policy.LearnedPolicy` that owns its own base controller and is reset per
  episode by the runner.

The base
--------
``pid_feedforward`` with its committed ``configs/control/pid_feedforward.yaml`` gains, built
through the control registry from the same committed configs the environment is built from
(exactly as :mod:`rld.eval.runner` builds the ``pid_feedforward`` baseline), and ``reset``
with the episode seed at every episode, so its integrator and descent latch are per episode.
The base is not privileged and takes no ship-motion feed; the config loader refuses any other.

The policy input
----------------
``pi`` reads the **unchanged** environment observation ``o`` -- P3-D1's ``pi(o)``; the base
action is not appended. (With the forecast block, :mod:`rld.rl.forecast_obs`, ``o`` is the
31-entry observation; the base still reads the raw 25.)

Zero initialisation
-------------------
:func:`zero_init_action_net` zeroes the weight and bias of PPO's ``action_net`` -- the
actor's last layer, the Gaussian mean -- right after the model is built, and nothing else:
the value head and ``log_std`` keep their initialisation. The initial deterministic policy
therefore outputs ``a_res = 0`` exactly, and the composed action is ``a_base`` bit for bit
(``tests/test_rl_residual.py`` flies it through the evaluation runner against
``pid_feedforward``). Exploration is unchanged in normalised units: ``log_std_init`` is
inherited from ``ppo.yaml`` (std 0.061), so the executed action's exploration std is
``alpha * 0.061`` = 0.018 normalised units (0.027 m/s model scale).

Units: actions are the shared normalised velocity setpoint, dimensionless (multiply by
``v_max`` = 1.5 m/s model scale); ``alpha`` is in the same units (0.3 = 0.45 m/s).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import torch
from dmf.typedefs import FloatArray
from gymnasium.spaces import Box
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from rld.control.base import Controller, ControlSpec, PrivilegedContext
from rld.control.registry import make_controller
from rld.envs.landing_env import ACTION_DIM
from rld.eval.envs import EvalConfigs
from rld.rl.policy import LearnedPolicy

if TYPE_CHECKING:
    from rld.deck.forecast import ShipMotionFeed

__all__ = [
    "ResidualActionWrapper",
    "ResidualPolicy",
    "build_base_controller",
    "compose_residual",
    "zero_init_action_net",
]


def compose_residual(a_base: FloatArray, a_res: FloatArray, alpha: float) -> FloatArray:
    """Return the executed residual action, ``clip(a_base + alpha * a_res, -1, 1)``.

    The environment applies its own norm cap (P2-D2) to what this returns, exactly as it
    does to every other method's action.

    Args:
        a_base: The base controller's action, ``(3,)``, normalised units.
        a_res: The policy's residual action, ``(3,)``, normalised units in ``[-1, 1]``.
        alpha: Residual scale, normalised units (0.3).

    Returns:
        ``(3,)`` float32 in ``[-1, 1]^3``. Computed in float64 and rounded once; with
        ``a_res = 0`` it equals ``a_base`` (clipped to the box) bit for bit.

    Raises:
        ValueError: If an input has the wrong size or anything is non-finite.
    """
    base = np.asarray(a_base, dtype=np.float64).reshape(-1)
    res = np.asarray(a_res, dtype=np.float64).reshape(-1)
    if base.shape != (ACTION_DIM,) or res.shape != (ACTION_DIM,):
        raise ValueError(f"actions must be ({ACTION_DIM},), got {base.shape} and {res.shape}")
    if not (np.all(np.isfinite(base)) and np.all(np.isfinite(res)) and np.isfinite(alpha)):
        raise ValueError(f"non-finite residual composition: {base}, {res}, alpha {alpha}")
    out: FloatArray = np.clip(base + float(alpha) * res, -1.0, 1.0).astype(np.float32)
    return out


def build_base_controller(name: str, cfgs: EvalConfigs) -> Controller:
    """Build the residual base from the committed configs the environment was built from.

    The same construction as :func:`rld.eval.runner.controller_spec`'s builder, so the base
    inside a residual policy is the ``pid_feedforward`` baseline, gain for gain.

    Args:
        name: Registry name (``"pid_feedforward"``).
        cfgs: The environment's configs (landing, observation layout, Froude scale).

    Returns:
        A fresh controller; ``reset`` it before use.

    Raises:
        ValueError: If the controller is privileged or needs a ship-motion feed.
    """
    spec = ControlSpec(
        landing=cfgs.landing,
        observation=cfgs.observation,
        scale=cfgs.scaling.froude_scale(),
    )
    base = make_controller(name, spec)
    if base.privileged or base.needs_motion_feed:
        raise ValueError(f"residual base {name!r} must see only the observation")
    return base


def zero_init_action_net(model: BaseAlgorithm) -> None:
    """Zero the weight and bias of a PPO policy's ``action_net`` (the Gaussian mean head).

    Only that layer: the value head, the shared/actor/critic MLPs and ``log_std`` keep
    their initialisation. Done in place, so the optimiser's parameter references stay valid.

    Args:
        model: A freshly built SB3 PPO model.

    Raises:
        TypeError: If the policy has no linear ``action_net``.
    """
    net = getattr(model.policy, "action_net", None)
    if not isinstance(net, torch.nn.Linear):
        raise TypeError(f"{type(model.policy).__name__} has no linear action_net to zero")
    with torch.no_grad():
        net.weight.zero_()
        if net.bias is not None:
            net.bias.zero_()


class ResidualActionWrapper(gym.Wrapper[FloatArray, FloatArray, FloatArray, FloatArray]):
    """Training-side residual: the wrapped env executes ``compose_residual(a_base, a_res)``.

    Sits directly on the pool-sampling env (raw observations), below the forecast wrapper and
    ``Monitor``. The agent's action space is unchanged, ``Box(-1, 1, (3,))``: it is ``a_res``.

    Attributes:
        base: The base controller, owned by this wrapper.
        alpha: Residual scale, normalised units.
        last_base_action: ``a_base`` of the last step, ``(3,)`` float32 (tests, audits).
        last_action: The executed action of the last step, ``(3,)`` float32.
    """

    def __init__(
        self, env: gym.Env[FloatArray, FloatArray], base: Controller, alpha: float
    ) -> None:
        """Wrap ``env``.

        Args:
            env: The pool-sampling landing env (its observation is the raw one).
            base: The base controller (:func:`build_base_controller`).
            alpha: Residual scale, normalised units.
        """
        super().__init__(env)
        self.base = base
        self.alpha = float(alpha)
        self.action_space = Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        self._obs: FloatArray | None = None
        self.last_base_action: FloatArray | None = None
        self.last_action: FloatArray | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Reset the env, then the base with the episode seed.

        Args:
            seed: Passed through.
            options: Passed through.

        Returns:
            The env's ``(observation, info)``, unchanged.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        episode_seed = info.get("episode_seed", 0 if seed is None else seed)
        self.base.reset(int(episode_seed))
        self._obs = obs
        return obs, info

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict[str, Any]]:
        """Execute ``compose_residual(base.act(o), action, alpha)``.

        Args:
            action: The policy's residual ``a_res``, normalised units.

        Returns:
            The env's step, unchanged.

        Raises:
            RuntimeError: If called before ``reset``.
        """
        if self._obs is None:
            raise RuntimeError("ResidualActionWrapper.step before reset")
        a_base = np.asarray(self.base.act(self._obs))
        executed = compose_residual(a_base, action, self.alpha)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        self.last_base_action = a_base
        self.last_action = executed
        self._obs = obs
        return obs, float(reward), bool(terminated), bool(truncated), info


class ResidualPolicy(LearnedPolicy):
    """Evaluation-side residual: ``compose_residual(base.act(o), pi(o), alpha)``.

    Owns its base controller and resets it in :meth:`reset`, so the runner's per-episode
    reset gives the base per-episode state. Not privileged; no ship-motion feed.

    Attributes:
        base: The base controller.
        alpha: Residual scale, normalised units.
    """

    def __init__(
        self,
        name: str,
        model: BaseAlgorithm,
        normalizer: VecNormalize | None,
        run_dir: Path,
        checkpoint: Path,
        steps: int,
        *,
        base: Controller,
        alpha: float,
    ) -> None:
        """Hold the model, its frozen normaliser and the base.

        Args:
            name: Method label.
            model: The SB3 PPO model (CPU).
            normalizer: Frozen normaliser, or ``None``.
            run_dir: The run directory.
            checkpoint: The checkpoint directory.
            steps: Training env steps at the checkpoint.
            base: The base controller (:func:`build_base_controller`).
            alpha: Residual scale, normalised units.
        """
        super().__init__(name, model, normalizer, run_dir, checkpoint, steps)
        self._setup_residual(base, alpha)

    def _setup_residual(self, base: Controller, alpha: float) -> None:
        """Attach the base and ``alpha`` (shared with the combined residual-forecast class).

        Raises:
            ValueError: If the base is privileged or needs a feed.
        """
        if base.privileged or base.needs_motion_feed:
            raise ValueError(f"residual base {base.name!r} must see only the observation")
        self.base = base
        self.alpha = float(alpha)

    def reset(
        self,
        seed: int,
        context: PrivilegedContext | None = None,
        motion_feed: "ShipMotionFeed | None" = None,
    ) -> None:
        """Start an episode: check the flags, then reset the base with the episode seed.

        Args:
            seed: The episode seed.
            context: Must be ``None``.
            motion_feed: Must be ``None`` (a forecast subclass consumes it first).

        Raises:
            ValueError: If a privileged context or an unconsumed feed is passed.
        """
        super().reset(seed, context, motion_feed)
        self.base.reset(int(seed))

    def act(self, obs: FloatArray) -> FloatArray:
        """Return ``compose_residual(base.act(obs), pi(obs), alpha)``.

        Args:
            obs: The raw environment observation (the base reads it as is).

        Returns:
            ``(3,)`` float32 in ``[-1, 1]^3``, normalised units.
        """
        a_res = self.network_action(obs)
        a_base = np.asarray(self.base.act(obs))
        return compose_residual(a_base, a_res, self.alpha)
