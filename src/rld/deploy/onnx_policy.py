"""The ONNX override of :meth:`rld.rl.policy.LearnedPolicy.network_action` (P8-D1 §7).

:class:`OnnxNetworkMixin` replaces **only** the network: ``network_action`` runs the exported graph
(ORT, CPU provider, one intra-op and one inter-op thread) on the raw :meth:`policy_input` -- the
normaliser is inside the graph. Everything else is the evaluated policy's own code:

* :class:`OnnxLearnedPolicy` -- ``ppo`` (``act`` = ``network_action``);
* :class:`OnnxResidualPolicy` -- ``compose_residual(base.act(o), pi(o), alpha)``;
* :class:`OnnxForecastPolicy` -- ``pi`` reads ``[o, forecast_block(feed)]``;
* :class:`OnnxResidualForecastPolicy` -- both (``residual_ppo_forecast``).

:func:`build_onnx_policy` is a picklable runner factory: it builds the policy exactly as e05/e06
built it (:func:`rld.rl.train.build_policy`), then re-classes it to its ONNX twin and attaches the
session, so the base controller, ``alpha``, the forecaster and the feed handling are the very
objects the PyTorch policy would use. The SB3 model stays attached but is never called.

Units: observations as :mod:`rld.envs.observation`; actions normalised, ``[-1, 1]^3``.
"""

from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from dmf.deploy.export_onnx import INPUT_NAME
from dmf.deploy.providers import preload_gpu_libraries, session_providers
from dmf.typedefs import FloatArray

from rld.eval.envs import EvalConfigs
from rld.rl.forecast_obs import ForecastPolicy, ResidualForecastPolicy
from rld.rl.policy import LearnedPolicy
from rld.rl.residual import ResidualPolicy

__all__ = [
    "ONNX_TWINS",
    "OnnxForecastPolicy",
    "OnnxLearnedPolicy",
    "OnnxNetworkMixin",
    "OnnxResidualForecastPolicy",
    "OnnxResidualPolicy",
    "attach_onnx",
    "build_onnx_policy",
    "cpu_session",
]


class _HasPolicyInput(Protocol):
    """What the mixin needs from its host class."""

    def policy_input(self, obs: FloatArray) -> FloatArray:
        """Return the raw network input."""
        ...


def cpu_session(onnx_path: Path, threads: int = 1) -> Any:
    """Open an ORT session on the CPU provider with pinned threads.

    Args:
        onnx_path: The graph.
        threads: ``intra_op_num_threads`` = ``inter_op_num_threads``.

    Returns:
        The ``onnxruntime.InferenceSession``.

    Raises:
        ValueError: If the session did not realize the CPU provider.
    """
    import onnxruntime as ort

    preload_gpu_libraries()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=session_providers(("CPUExecutionProvider",)),
    )
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise ValueError(f"{onnx_path}: realized {session.get_providers()}")
    return session


class OnnxNetworkMixin:
    """Run the policy network through an exported ONNX graph instead of SB3.

    Attributes:
        onnx_path: The graph.
        onnx_session: The ORT session (CPU, 1 thread).
    """

    onnx_path: Path
    onnx_session: Any

    def network_action(self: "_HasPolicyInput", obs: FloatArray) -> FloatArray:
        """Return the graph's deterministic action for one observation.

        Args:
            obs: The environment observation (raw, 25 entries).

        Returns:
            A ``(3,)`` float32 action in ``[-1, 1]^3``: the graph on :meth:`policy_input`
            (batch of 1), clipped as :meth:`LearnedPolicy.network_action` clips.
        """
        x = np.ascontiguousarray(self.policy_input(obs).reshape(1, -1), dtype=np.float32)
        session = cast(Any, self).onnx_session
        (out,) = session.run([], {INPUT_NAME: x})
        action: FloatArray = np.clip(np.asarray(out, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return action


class OnnxLearnedPolicy(OnnxNetworkMixin, LearnedPolicy):
    """``ppo`` (and any pure policy) with the ONNX network."""


class OnnxResidualPolicy(OnnxNetworkMixin, ResidualPolicy):
    """``residual_ppo`` with the ONNX network; base and composition unchanged."""


class OnnxForecastPolicy(OnnxNetworkMixin, ForecastPolicy):
    """``ppo_forecast`` with the ONNX network; forecaster and feed unchanged."""


class OnnxResidualForecastPolicy(OnnxNetworkMixin, ResidualForecastPolicy):
    """``residual_ppo_forecast`` with the ONNX network; base, forecaster, feed unchanged."""


#: The ONNX twin of each evaluation policy class. Exact class match: a subclass not listed here
#: is refused rather than given a parent's twin.
ONNX_TWINS: dict[type[LearnedPolicy], type[LearnedPolicy]] = {
    LearnedPolicy: OnnxLearnedPolicy,
    ResidualPolicy: OnnxResidualPolicy,
    ForecastPolicy: OnnxForecastPolicy,
    ResidualForecastPolicy: OnnxResidualForecastPolicy,
}


def attach_onnx(policy: LearnedPolicy, onnx_path: Path, threads: int = 1) -> LearnedPolicy:
    """Turn a built policy into its ONNX twin in place.

    Args:
        policy: A policy from :func:`rld.rl.train.build_policy`.
        onnx_path: Its exported graph (the same run and checkpoint).
        threads: ORT CPU threads.

    Returns:
        The same object, now an instance of its :data:`ONNX_TWINS` class.

    Raises:
        TypeError: If the policy's class has no twin.
        ValueError: If the graph's input width is not the policy's.
    """
    twin = ONNX_TWINS.get(type(policy))
    if twin is None:
        raise TypeError(f"no ONNX twin for {type(policy).__name__}")
    session = cpu_session(onnx_path, threads)
    width = session.get_inputs()[0].shape[1]
    expected = int((policy.model.observation_space.shape or (0,))[0])
    if width != expected:
        raise ValueError(f"{onnx_path}: input width {width} != policy input {expected}")
    policy.__class__ = twin
    host = cast(OnnxNetworkMixin, policy)
    host.onnx_path = Path(onnx_path)
    host.onnx_session = session
    return policy


def build_onnx_policy(
    cfgs: EvalConfigs, *, run_dir: Path, onnx_path: Path, ckpt: str = "final"
) -> LearnedPolicy:
    """Picklable runner factory: the e05/e06 policy with its network swapped for the graph.

    Args:
        cfgs: The evaluation configs (the residual base is built from them).
        run_dir: The run directory.
        onnx_path: The run's exported graph.
        ckpt: Checkpoint name.

    Returns:
        The ONNX twin policy.
    """
    from rld.rl.train import build_policy

    return attach_onnx(build_policy(cfgs, run_dir=run_dir, ckpt=ckpt), onnx_path)
