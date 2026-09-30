"""The evaluation-time learned policy behind the :class:`rld.control.base.Controller` contract.

:class:`LearnedPolicy` is what :func:`rld.rl.train.load_policy` returns for a pure PPO or SAC
run (it is re-exported there, its historical home). It lives in its own module so that the
Phase 6 policies can subclass it without an import cycle through :mod:`rld.rl.train`:

* :class:`rld.rl.residual.ResidualPolicy` -- ``clip(a_base(o) + alpha * pi(o), -1, 1)``;
* :class:`rld.rl.forecast_obs.ForecastPolicy` -- ``pi`` reads the observation with the
  forecast block appended (``needs_motion_feed = True``);
* :class:`rld.rl.forecast_obs.ResidualForecastPolicy` -- both.

Two hooks carry the variation: :meth:`LearnedPolicy.policy_input` (the raw, unnormalised
network input built from the observation) and :meth:`LearnedPolicy.network_action` (the
network's deterministic, clipped action on it). A pure policy's ``act`` is its network
action; for a pure policy both hooks do exactly what ``act`` did before they existed, so its
actions are bit-identical.

Nothing in this module reads ``results/episodes/``.

Units: actions are the shared normalised velocity setpoint (``v_max`` = 1.5 m/s model scale,
norm cap), dimensionless; observations are the environment's (``rld.envs.observation``).
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from dmf.typedefs import FloatArray
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from rld.control.base import PrivilegedContext, reject_motion_feed

if TYPE_CHECKING:
    from rld.deck.forecast import ShipMotionFeed

__all__ = ["LearnedPolicy"]


class LearnedPolicy:
    """A trained policy behind the :class:`rld.control.base.Controller` contract.

    Stateless across steps (an MLP of the current observation), so ``reset`` clears
    nothing. Never privileged and never given a ship-motion feed.

    Attributes:
        name: Method label (the run's ``method``).
        privileged: Always False.
        needs_motion_feed: Always False for this class.
        run_dir: The run it came from.
        checkpoint: The checkpoint directory.
        steps: Training env steps at the checkpoint.
    """

    privileged: bool = False
    needs_motion_feed: bool = False

    def __init__(
        self,
        name: str,
        model: BaseAlgorithm,
        normalizer: VecNormalize | None,
        run_dir: Path,
        checkpoint: Path,
        steps: int,
    ) -> None:
        """Hold the model and its frozen normaliser.

        Args:
            name: Method label.
            model: The SB3 model (CPU).
            normalizer: Frozen normaliser, or ``None`` if the run had none.
            run_dir: The run directory.
            checkpoint: The checkpoint directory.
            steps: Training env steps at the checkpoint.
        """
        self.name = name
        self.model = model
        self.normalizer = normalizer
        self.run_dir = run_dir
        self.checkpoint = checkpoint
        self.steps = steps

    def reset(
        self,
        seed: int,
        context: PrivilegedContext | None = None,
        motion_feed: "ShipMotionFeed | None" = None,
    ) -> None:
        """Start an episode; nothing to clear.

        Args:
            seed: The episode seed (unused: the policy is deterministic).
            context: Must be ``None``.
            motion_feed: Must be ``None``.

        Raises:
            ValueError: If a privileged context or a ship-motion feed is passed.
        """
        del seed
        if context is not None:
            raise ValueError(f"{self.name} is a learned policy and is never privileged")
        reject_motion_feed(self.name, motion_feed)

    def policy_input(self, obs: FloatArray) -> FloatArray:
        """Return the raw (unnormalised) network input for one observation.

        Args:
            obs: The environment's observation vector.

        Returns:
            ``(n,)`` float32: the observation itself for this class.
        """
        return cast(FloatArray, np.asarray(obs, dtype=np.float32).reshape(-1))

    def network_action(self, obs: FloatArray) -> FloatArray:
        """Return the network's deterministic action for one observation.

        Args:
            obs: The environment's observation vector.

        Returns:
            A ``(3,)`` float32 action in ``[-1, 1]^3``: the mean action of the policy on
            :meth:`policy_input`, normalised with the frozen statistics (batch of 1).
        """
        x = self.policy_input(obs).reshape(1, -1)
        if self.normalizer is not None:
            x = np.asarray(self.normalizer.normalize_obs(x), dtype=np.float32)
        action, _ = self.model.predict(x, deterministic=True)
        out: FloatArray = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return out

    def act(self, obs: FloatArray) -> FloatArray:
        """Return the deterministic action for one observation.

        Args:
            obs: The raw observation vector.

        Returns:
            A ``(3,)`` float32 action in ``[-1, 1]^3`` (the shared normalised velocity
            setpoint).
        """
        return self.network_action(obs)
