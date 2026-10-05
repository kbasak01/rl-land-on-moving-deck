"""ONNX export of a learned policy's deterministic actor with ``VecNormalize`` folded in (P8-D1 §4).

The graph
---------
``input`` (float32, ``[batch, D]``, the **raw** policy input: the 25-entry observation, or 31
with the forecast block) ->

    z = clip((x - mu) / sqrt(var + eps), -c, c)       # the frozen VecNormalize, folded
    a = clip(action_net(policy_net(z)), -1, 1)        # SB3's deterministic mean action

-> ``output`` (float32, ``[batch, 3]``, normalised action units in [-1, 1]). ``mu`` and ``var``
are the run's frozen ``obs_rms`` statistics, ``eps`` = ``VecNormalize.epsilon`` and ``c`` =
``clip_obs``. ``sqrt(var + eps)`` is computed in float64 and cast to float32 once. This is what
:meth:`rld.rl.policy.LearnedPolicy.network_action` computes: ``normalize_obs``, then SB3
``predict(deterministic=True)`` -- which clips the Gaussian mean to the ``[-1, 1]`` action box --
then a clip to ``[-1, 1]``.

What is **not** in the graph
----------------------------
For a residual method the ``pid_feedforward`` base controller and the ``alpha`` = 0.3 composition
(:func:`rld.rl.residual.compose_residual`); for a forecast method the DLinear-OLS forecaster and its
ship-motion feed, which build the 6-entry block. The base is stateful (integrator, descent latch)
and the forecaster is its own ONNX graph; both are costed in :mod:`rld.deploy.e2e`.

Export settings follow :func:`dmf.deploy.export_onnx.export_model`: opset 18, **dynamic batch axis
only**, constant folding, the TorchScript exporter (``dynamo=False``), and dmf's input/output
names. The export is byte-reproducible on this machine (P8-D1 preamble).

Units: observation entries as :mod:`rld.envs.observation` (metres, metres per second, radians,
model scale); actions normalised (multiply by ``v_max`` = 1.5 m/s model scale).
"""

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dmf.deploy.export_onnx import INPUT_NAME, OUTPUT_NAME, graph_dims
from dmf.typedefs import FloatArray
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.running_mean_std import RunningMeanStd
from torch import Tensor, nn

from rld.config import REPO_ROOT
from rld.envs.landing_env import ACTION_DIM
from rld.rl.policy import LearnedPolicy

__all__ = [
    "DEFAULT_ONNX_DIR",
    "EXPORTS",
    "OPSET",
    "ExportedGraph",
    "FoldedActor",
    "NormalizerConstants",
    "export_folded",
    "export_run",
    "folded_actor",
    "graph_name",
    "normalizer_constants",
    "sha256_file",
]

#: Where graphs and sidecars are written (gitignored, like every other artifact).
DEFAULT_ONNX_DIR: Path = REPO_ROOT / "artifacts" / "onnx"

#: ONNX opset (plan Phase 8 step 1).
OPSET: int = 18

#: The exported (method, seed) pairs (P8-D1 §3): seed 0 and the median seed 4 of each.
EXPORTS: tuple[tuple[str, int], ...] = (
    ("ppo", 0),
    ("ppo", 4),
    ("residual_ppo_forecast", 0),
    ("residual_ppo_forecast", 4),
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file's bytes, hex."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_name(method: str, seed: int) -> str:
    """Return the graph file name, ``<method>_s<seed>.onnx``."""
    return f"{method}_s{seed}.onnx"


@dataclass(frozen=True)
class NormalizerConstants:
    """The frozen ``VecNormalize`` constants folded into a graph.

    Attributes:
        mean: ``obs_rms.mean``, ``(D,)`` float64, observation units.
        var: ``obs_rms.var``, ``(D,)`` float64, observation units squared.
        epsilon: ``VecNormalize.epsilon``, observation units squared.
        clip_obs: ``VecNormalize.clip_obs``, dimensionless (normalised units).
    """

    mean: FloatArray
    var: FloatArray
    epsilon: float
    clip_obs: float

    @property
    def std(self) -> FloatArray:
        """``sqrt(var + epsilon)``, float64, observation units."""
        out: FloatArray = np.sqrt(np.asarray(self.var, dtype=np.float64) + self.epsilon)
        return out

    @property
    def dim(self) -> int:
        """Input width ``D``."""
        return int(np.asarray(self.mean).shape[0])

    def as_json(self) -> dict[str, Any]:
        """Return the constants as JSON-serialisable lists (float64, exact round trip)."""
        return {
            "mean": [float(v) for v in np.asarray(self.mean, dtype=np.float64)],
            "var": [float(v) for v in np.asarray(self.var, dtype=np.float64)],
            "epsilon": float(self.epsilon),
            "clip_obs": float(self.clip_obs),
        }


def normalizer_constants(policy: LearnedPolicy) -> NormalizerConstants:
    """Read the frozen normaliser of a loaded policy.

    Args:
        policy: A policy from :func:`rld.rl.train.build_policy`.

    Returns:
        The constants.

    Raises:
        ValueError: If the policy has no normaliser, it is still in training mode, or it does
            not normalise observations.
    """
    norm = policy.normalizer
    if norm is None:
        raise ValueError(f"{policy.name}: no VecNormalize to fold")
    if norm.training or not norm.norm_obs:
        raise ValueError(f"{policy.name}: normaliser must be frozen and normalise observations")
    rms: Any = norm.obs_rms  # SB3 annotates a dict (Dict spaces); a flat Box has one
    if not isinstance(rms, RunningMeanStd):
        raise TypeError(f"{policy.name}: expected one RunningMeanStd for a flat Box observation")
    return NormalizerConstants(
        mean=np.asarray(rms.mean, dtype=np.float64).copy(),
        var=np.asarray(rms.var, dtype=np.float64).copy(),
        epsilon=float(norm.epsilon),
        clip_obs=float(norm.clip_obs),
    )


class FoldedActor(nn.Module):
    """``clip(action_net(policy_net(clip((x - mu) / std, -c, c))), -1, 1)`` as one module.

    Attributes:
        mean: Buffer, ``(D,)`` float32, observation units.
        std: Buffer, ``(D,)`` float32, ``sqrt(var + eps)`` rounded once from float64.
        clip_obs: The normalised-observation clip ``c``, dimensionless.
        policy_net: A copy of SB3's ``mlp_extractor.policy_net``.
        action_net: A copy of SB3's ``action_net`` (the Gaussian mean head).
    """

    mean: Tensor
    std: Tensor

    def __init__(
        self, constants: NormalizerConstants, policy_net: nn.Module, action_net: nn.Module
    ) -> None:
        """Hold copies of the actor layers and the normaliser as float32 buffers.

        Args:
            constants: The frozen normaliser.
            policy_net: The actor MLP (copied, so the caller's model is never moved).
            action_net: The mean head (copied).
        """
        super().__init__()
        self.register_buffer("mean", torch.tensor(constants.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(constants.std, dtype=torch.float32))
        self.clip_obs = float(constants.clip_obs)
        self.policy_net = copy.deepcopy(policy_net).float()
        self.action_net = copy.deepcopy(action_net).float()
        self.eval()

    def forward(self, x: Tensor) -> Tensor:
        """Map raw policy inputs to deterministic actions.

        Args:
            x: ``(batch, D)`` float32, raw (unnormalised) policy input.

        Returns:
            ``(batch, 3)`` float32 actions in ``[-1, 1]``, normalised units.
        """
        z = torch.clamp((x - self.mean) / self.std, -self.clip_obs, self.clip_obs)
        return torch.clamp(self.action_net(self.policy_net(z)), -1.0, 1.0)


def folded_actor(policy: LearnedPolicy) -> FoldedActor:
    """Build the folded actor of a loaded PPO policy.

    Args:
        policy: A policy from :func:`rld.rl.train.build_policy` (PPO, ``MlpPolicy``).

    Returns:
        The module, in eval mode, on the CPU.

    Raises:
        TypeError: If the SB3 policy is not an ``ActorCriticPolicy`` with a flatten feature
            extractor, or squashes its output.
        ValueError: If the normaliser's width is not the policy's input width.
    """
    sb3 = policy.model.policy
    if not isinstance(sb3, ActorCriticPolicy):
        raise TypeError(f"{policy.name}: expected an ActorCriticPolicy, got {type(sb3).__name__}")
    if sb3.squash_output:
        raise TypeError(f"{policy.name}: a squashed policy is not this graph")
    if type(sb3.features_extractor).__name__ != "FlattenExtractor":
        raise TypeError(f"{policy.name}: expected a FlattenExtractor (identity on a flat Box)")
    constants = normalizer_constants(policy)
    width = int((sb3.observation_space.shape or (0,))[0])
    if constants.dim != width:
        raise ValueError(f"{policy.name}: normaliser width {constants.dim} != input {width}")
    return FoldedActor(constants, sb3.mlp_extractor.policy_net, sb3.action_net)


@dataclass(frozen=True)
class ExportedGraph:
    """One exported graph and what it was built from.

    Attributes:
        method: Method label.
        seed: Training seed.
        onnx_path: The ``.onnx`` file.
        sidecar_path: Its JSON sidecar.
        obs_dim: Input width ``D`` (25 or 31).
        onnx_sha256: SHA-256 of the graph bytes.
        model_sha256: SHA-256 of ``model.zip``.
        vecnormalize_sha256: SHA-256 of ``vecnormalize.pkl``.
    """

    method: str
    seed: int
    onnx_path: Path
    sidecar_path: Path
    obs_dim: int
    onnx_sha256: str
    model_sha256: str
    vecnormalize_sha256: str

    @property
    def architecture(self) -> str:
        """``mlp512x2-tanh-in<D>``: the latency unit (P8-D1 §3)."""
        return f"mlp512x2-tanh-in{self.obs_dim}"


def export_folded(module: FoldedActor, obs_dim: int, path: Path, opset: int = OPSET) -> Path:
    """Export a folded actor to ONNX with a dynamic batch axis only.

    Args:
        module: The folded actor (CPU).
        obs_dim: Input width ``D``.
        path: Destination ``.onnx``.
        opset: ONNX opset.

    Returns:
        The path written.

    Raises:
        ValueError: If the written graph does not declare ``[batch, D] -> [batch, 3]``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    module = module.to("cpu").eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            (torch.zeros(1, obs_dim, dtype=torch.float32),),
            str(path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=opset,
            do_constant_folding=True,
            # As dmf: the TorchScript exporter, pinned, because the dynamo path needs
            # onnxscript (not a pinned dependency) and would change under a torch upgrade.
            dynamo=False,
        )
    dims_in, dims_out = graph_dims(path)
    if dims_in != ("batch", obs_dim) or dims_out != ("batch", ACTION_DIM):
        raise ValueError(f"{path}: declares {dims_in} -> {dims_out}, expected batch-only dynamic")
    import onnx

    onnx.checker.check_model(onnx.load(str(path)))
    return path


def export_run(
    method: str,
    seed: int,
    *,
    run_root: Path = REPO_ROOT / "artifacts" / "runs",
    out_dir: Path = DEFAULT_ONNX_DIR,
    ckpt: str = "final",
) -> ExportedGraph:
    """Export one run's checkpoint and write its sidecar.

    The run is inspected with :func:`rld.eval.learned.inspect_run` (done, method and seed
    consistent, checkpoint hashed) and loaded with :func:`rld.rl.train.build_policy` from the
    committed configs, exactly as the evaluations loaded it.

    Args:
        method: Method label.
        seed: Training seed (the run directory is ``run_root/<method>/<seed>``).
        run_root: Root of the run tree.
        out_dir: Where the graph and sidecar go.
        ckpt: Checkpoint name.

    Returns:
        The exported graph.

    Raises:
        ValueError: If the run's recorded seed is not ``seed``.
    """
    from rld.eval.envs import load_eval_configs
    from rld.eval.learned import inspect_run
    from rld.rl.train import build_policy

    run_dir = (run_root / method / str(seed)).resolve()
    run = inspect_run(method, run_dir, ckpt)
    if run.seed != seed:
        raise ValueError(f"{run_dir}: recorded seed {run.seed} != {seed}")
    policy = build_policy(load_eval_configs(), run_dir=run_dir, ckpt=ckpt)
    module = folded_actor(policy)
    constants = normalizer_constants(policy)
    onnx_path = export_folded(module, constants.dim, out_dir / graph_name(method, seed))
    digests = dict(run.digests)
    graph = ExportedGraph(
        method=method,
        seed=seed,
        onnx_path=onnx_path,
        sidecar_path=onnx_path.with_suffix(".json"),
        obs_dim=constants.dim,
        onnx_sha256=sha256_file(onnx_path),
        model_sha256=digests["run_model_sha256"],
        vecnormalize_sha256=digests["run_vecnormalize_sha256"],
    )
    sidecar = {
        "method": method,
        "seed": seed,
        "ckpt": ckpt,
        "run_dir": run.label_dir(),
        "policy_class": type(policy).__name__,
        "architecture": graph.architecture,
        "obs_dim": graph.obs_dim,
        "action_dim": ACTION_DIM,
        "opset": OPSET,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "dynamic_axes": {"input": ["batch"], "output": ["batch"]},
        "onnx_sha256": graph.onnx_sha256,
        "model_sha256": graph.model_sha256,
        "vecnormalize_sha256": graph.vecnormalize_sha256,
        "run_config_sha256": digests["run_config_sha256"],
        "normalizer": constants.as_json(),
        "inside_graph": "clip((x - mean) / sqrt(var + epsilon), -clip_obs, clip_obs) -> "
        "mlp_extractor.policy_net -> action_net -> clip(-1, 1)",
        "outside_graph": "residual base pid_feedforward, compose_residual (alpha), "
        "DLinear-OLS forecaster and its ship-motion feed (forecast block)",
        "torch": str(torch.__version__),
    }
    graph.sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    return graph
