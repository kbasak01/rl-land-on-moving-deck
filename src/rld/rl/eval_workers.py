"""Evaluation workers that fly their share of the tune-pool draw independently.

Why not a lockstep vector env
-----------------------------
The first evaluator stepped all evaluation workers together from the learner: each vector
step waited for the slowest worker, and every ~90 ms reset stalled all of them. Here each
worker receives the policy weights and the frozen normalisation statistics once per
evaluation (:class:`EvalTask`) and runs its own loop over its share of the episodes,
so a worker that resets never holds up another.

Same numbers as the lockstep evaluator
--------------------------------------
Worker ``i`` flies ``episodes[i::n]`` in order, through the same :class:`PoolSamplingEnv`
queue, exactly as it did under the lockstep evaluator, so the environment side is
unchanged. The action side is where it could differ. A float32 MLP forward pass on this
stack (SB3 on CPU torch) is **not** batch-size invariant: a batch of 1 and a batch of 16
differ by up to 3.6e-7 in the action (measured on the P5-D4 smoke checkpoints), and for
some layer shapes a row's output also depends on its **position** in the batch (a 64-wide
network at batch 3: 450 of 900 probes). What was measured to hold -- 0 mismatches over
widths 32-256, tanh and ReLU, batch sizes 1-32, any thread count -- is that a row's output
depends only on the batch size, its position and its own content, not on the other rows.
The lockstep evaluator forwarded a batch of ``n`` rows with worker ``i``'s observation in
row ``i``; so worker ``i`` here forwards ``n`` copies of its own observation and takes row
``i`` (:attr:`EvalTask.batch_width`, :attr:`EpisodeRunner.rank`). The episode rows are
then identical to the lockstep evaluator's; ``tests/test_rl_eval_workers.py`` checks every
field, and the P5-D4 smoke checkpoints were checked on the full 180-episode draw.

Note that :class:`rld.rl.train.LearnedPolicy` (the frozen-list evaluation path) forwards a
batch of 1, so its actions can differ from these in the last float32 bits.

``eval_env_steps`` now counts the steps actually flown. The lockstep count also included
the idle replays of workers that had already finished their share.

Units: steps are env control steps (1/30 s model scale each); actions are the shared
normalised velocity setpoint.
"""

import copy
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from dmf.typedefs import FloatArray
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecNormalize

from rld.control.tuning import TuningEpisode
from rld.rl.wrappers import EnvFactory

__all__ = [
    "MAX_EPISODE_STEPS",
    "EpisodeRunner",
    "EvalTask",
    "EvalWorkerFactory",
    "normalizer_from_state",
    "normalizer_state",
    "policy_from_payload",
    "policy_payload",
    "split_shares",
]

#: Guard against an episode that never ends, control steps. The landing env truncates at
#: its episode length (a few hundred steps), so this only fires on a bug.
MAX_EPISODE_STEPS: int = 100_000

#: ``VecNormalize`` attributes that belong to the vector env it wraps, not to its statistics.
_VENV_KEYS: tuple[str, ...] = ("venv", "class_attributes", "returns")


def normalizer_state(normalizer: VecNormalize) -> dict[str, Any]:
    """Return a detached, picklable copy of a normaliser's state (statistics and settings).

    Args:
        normalizer: A ``VecNormalize``, attached or detached.

    Returns:
        A deep copy of its attributes without the wrapped vector env.
    """
    return {
        key: copy.deepcopy(value)
        for key, value in normalizer.__dict__.items()
        if key not in _VENV_KEYS
    }


def normalizer_from_state(state: Mapping[str, Any]) -> VecNormalize:
    """Rebuild a frozen, detached normaliser from :func:`normalizer_state`.

    Args:
        state: The state.

    Returns:
        A ``VecNormalize`` with ``training=False`` (statistics never update) and
        ``norm_reward=False``; only ``normalize_obs`` may be called on it.
    """
    frozen: VecNormalize = VecNormalize.__new__(VecNormalize)
    frozen.__setstate__(dict(state))
    frozen.training = False
    frozen.norm_reward = False
    return frozen


def policy_payload(policy: BasePolicy) -> bytes:
    """Serialise a policy's class, constructor arguments and weights.

    Args:
        policy: An SB3 policy on the CPU.

    Returns:
        Bytes that :func:`policy_from_payload` turns back into an identical policy.
    """
    buffer = io.BytesIO()
    torch.save(
        {
            "cls": type(policy),
            "data": policy._get_constructor_parameters(),
            "state_dict": policy.state_dict(),
        },
        buffer,
    )
    return buffer.getvalue()


def policy_from_payload(payload: bytes) -> BasePolicy:
    """Rebuild a policy from :func:`policy_payload`, in evaluation mode, on the CPU.

    Args:
        payload: The serialised policy.

    Returns:
        The policy, with bit-identical weights.
    """
    saved = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    policy = cast(BasePolicy, saved["cls"](**saved["data"]))
    policy.load_state_dict(saved["state_dict"])
    policy.to("cpu")
    policy.set_training_mode(False)
    return policy


@dataclass(frozen=True)
class EvalTask:
    """One evaluation, as shipped to every worker.

    Attributes:
        policy: :func:`policy_payload` of the policy to evaluate.
        normalizer: :func:`normalizer_state` of the frozen observation statistics, or
            ``None`` if the run does not normalise observations.
        deterministic: Fly the mean action.
        batch_width: Rows per policy forward pass, i.e. the lockstep evaluator's batch
            size (module docstring).
        shares: ``shares[i]`` are the episodes worker ``i`` flies, in order.
    """

    policy: bytes
    normalizer: dict[str, Any] | None
    deterministic: bool
    batch_width: int
    shares: tuple[tuple[TuningEpisode, ...], ...]


class EpisodeRunner(gym.Wrapper[FloatArray, FloatArray, FloatArray, FloatArray]):
    """A worker-side loop that flies a list of queued episodes with a shipped policy.

    Wraps the evaluation worker's :class:`~rld.rl.wrappers.PoolSamplingEnv`; ``reset`` and
    ``step`` pass straight through, so the same worker can also serve a lockstep vector
    env.

    Attributes:
        rank: This worker's index; it flies ``task.shares[rank]``.
    """

    def __init__(
        self, env: gym.Env[FloatArray, FloatArray], rank: int, torch_threads: int | None
    ) -> None:
        """Wrap the worker's environment.

        Args:
            env: The pool-sampling environment (tune pool).
            rank: Worker index.
            torch_threads: ``torch.set_num_threads`` for this process, or ``None`` to leave
                it (an in-process worker must not change the learner's setting).
        """
        super().__init__(env)
        self.rank = int(rank)
        if torch_threads is not None:
            torch.set_num_threads(int(torch_threads))

    def fly(self, task: EvalTask) -> tuple[list[dict[str, Any]], int]:
        """Fly this worker's share of an evaluation.

        Args:
            task: The evaluation.

        Returns:
            ``(rows, steps)``: the ``episode_row`` of every episode in the share, in order,
            and the env steps flown.

        Raises:
            RuntimeError: If an episode does not end within :data:`MAX_EPISODE_STEPS` or
                the environment flew a different episode than the one queued.
        """
        share = task.shares[self.rank] if self.rank < len(task.shares) else ()
        if not share:
            return [], 0
        if not 0 <= self.rank < task.batch_width:
            raise ValueError(f"worker rank {self.rank} outside batch width {task.batch_width}")
        policy = policy_from_payload(task.policy)
        normalizer = None if task.normalizer is None else normalizer_from_state(task.normalizer)
        self.env.get_wrapper_attr("load_episode_queue")(list(share))
        rows: list[dict[str, Any]] = []
        steps = 0
        for episode in share:
            if not task.deterministic:
                torch.manual_seed(int(episode.episode_seed))
            obs, _ = self.env.reset()
            info: dict[str, Any] = {}
            for _ in range(MAX_EPISODE_STEPS):
                action = self._act(policy, normalizer, obs, task, self.rank)
                obs, _, terminated, truncated, info = self.env.step(action)
                steps += 1
                if terminated or truncated:
                    break
            else:
                raise RuntimeError(f"evaluation episode {episode.ss}/{episode.index} never ended")
            row = dict(info["episode_row"])
            if (row["ss"], int(row["index"])) != (episode.ss, int(episode.index)):
                raise RuntimeError(
                    f"worker {self.rank} flew {row['ss']}/{row['index']}, "
                    f"expected {episode.ss}/{episode.index}"
                )
            rows.append(row)
        return rows, steps

    @staticmethod
    def _act(
        policy: BasePolicy,
        normalizer: VecNormalize | None,
        obs: FloatArray,
        task: EvalTask,
        row: int,
    ) -> FloatArray:
        """Return one action, computed exactly as the lockstep evaluator computed it.

        Args:
            policy: The policy.
            normalizer: Frozen normaliser, or ``None``.
            obs: The raw observation.
            task: The evaluation (``deterministic`` and ``batch_width``).
            row: This worker's row in the lockstep batch (its rank).

        Returns:
            The ``(3,)`` float32 action.
        """
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if normalizer is not None:
            x = np.asarray(normalizer.normalize_obs(x), dtype=np.float32)
        batch = np.repeat(x, task.batch_width, axis=0)
        actions, _ = policy.predict(batch, deterministic=task.deterministic)
        out: FloatArray = np.asarray(actions)[row]
        return out


@dataclass(frozen=True)
class EvalWorkerFactory:
    """Picklable factory of one evaluation worker: the pool env inside an :class:`EpisodeRunner`.

    Attributes:
        inner: The worker's :class:`~rld.rl.wrappers.EnvFactory` (tune pool, no Monitor).
        torch_threads: Torch threads in the worker process; ``None`` for an in-process
            (``DummyVecEnv``) worker.
    """

    inner: EnvFactory
    torch_threads: int | None = 1

    def __call__(self) -> gym.Env[FloatArray, FloatArray]:
        """Build the worker's environment."""
        return EpisodeRunner(self.inner(), self.inner.rank, self.torch_threads)


def split_shares(
    episodes: Sequence[TuningEpisode], n_workers: int
) -> tuple[tuple[TuningEpisode, ...], ...]:
    """Return worker ``i``'s episodes as ``episodes[i::n_workers]`` (the lockstep split).

    Args:
        episodes: The evaluation draw, in order.
        n_workers: Number of workers.

    Returns:
        One tuple per worker.
    """
    return tuple(tuple(episodes[i::n_workers]) for i in range(n_workers))
