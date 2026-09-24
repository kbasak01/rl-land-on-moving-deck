"""Training configuration: ``configs/rl/*.yaml`` -> frozen dataclasses.

One YAML file describes one training run up to its seed. Every key is required unless noted
and every unknown key is an error, so a typo in a search-space target cannot silently train
the default. The one optional key, ``prefetch_reset``, is an engineering switch that cannot
change what is trained (see :attr:`TrainConfig.prefetch_reset`); it defaults so that run
directories written before it existed still load. The checks that encode protocol
decisions live here, at load time:

* PPO trains on the CPU (CLAUDE.md, "SB3 MLP on GPU is slower than CPU");
* reward normalisation is PPO-only (plan Phase 5 item 1);
* the curriculum's stages are a subset of SS3-SS5 and **never** SS6 (P3-D1 section 6);
* only reward **weights** may be overridden -- ``v_safe_m_s`` and ``gate_height_m`` shape the
  reward's structure and are fixed (``configs/env/reward.yaml``).

The P3-D1 budget numbers (10 M PPO, 2 M SAC, 5 seeds, <= 20 trials, 0.80 over >= 100
episodes) are *not* enforced here, because smoke and test configs legitimately differ;
``tests/test_rl_config.py`` pins them for the committed ``ppo.yaml``, ``sac.yaml`` and
``tune_*.yaml`` instead.

Units: steps are environment control steps (1/30 s model scale each); wall-clock intervals
are seconds of host time (not simulated time); learning rates, coefficients and weights are
dimensionless.
"""

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, cast

from rld.config import CONFIG_DIR, REPO_ROOT, load_yaml
from rld.control.tuning import TUNING_CONFIG
from rld.envs.config import RewardConfig

__all__ = [
    "ALGOS",
    "FORBIDDEN_SEA_STATES",
    "PREFETCH_RESET_DEFAULT",
    "REWARD_WEIGHT_KEYS",
    "RL_CONFIG_DIR",
    "RUNS_ROOT",
    "STRUCTURE_REWARD_KEYS",
    "TRAINING_SEA_STATES",
    "ENV_WORKER_CORES",
    "CurriculumConfig",
    "EvalConfig",
    "NormalizeConfig",
    "PPOConfig",
    "PolicyConfig",
    "SACConfig",
    "TrainConfig",
    "apply_overrides",
    "config_to_dict",
    "load_train_config",
    "train_config_from_dict",
]

#: Directory of the committed training configs.
RL_CONFIG_DIR: Path = CONFIG_DIR / "rl"

#: Root of every run directory, ``artifacts/runs/<run_group>/<seed>/`` (gitignored).
RUNS_ROOT: Path = REPO_ROOT / "artifacts" / "runs"

#: Supported algorithms.
ALGOS: tuple[str, ...] = ("ppo", "sac")

#: Sea states a policy may ever be trained on (the P3-D2 pool's sea states).
TRAINING_SEA_STATES: tuple[str, ...] = ("SS3", "SS4", "SS5")

#: Sea states no training distribution may ever contain (P3-D1 section 6: ``unseen_seastate``).
FORBIDDEN_SEA_STATES: tuple[str, ...] = ("SS6",)

#: Time-averaged CPU cores one env worker costs, including its share of the learner's
#: stepping loop, for the slot cost :attr:`TrainConfig.workers`. Measured 2026-09-24 on
#: a PPO run with 16 train and 16 eval workers, reset prefetch on, host otherwise idle
#: (process-tree CPU sampled every second): rollout collection 6.6 cores in total at HEAD
#: env code (5.2 with the lazy-trajectory env), the PPO update ~1 core for ~2 s per
#: rollout, evaluation 15.5-16.7 cores for 5-17 s every ``eval.interval_steps``;
#: time-averaged ~6.4 cores. 0.4 x 16 workers + 1 learner = 8 slots covers that, so
#: ``MAX_WORKERS = 34`` admits four PPO runs (~26 busy cores on average). Evaluation
#: bursts overlap only briefly; when two coincide the evaluations slow down, nothing else.
ENV_WORKER_CORES: float = 0.4

#: Default of the optional ``prefetch_reset`` key (engineering only; see
#: :attr:`TrainConfig.prefetch_reset`).
PREFETCH_RESET_DEFAULT: bool = True

#: Reward parameters that define the pre-registered structure, not a weight; never tuned.
STRUCTURE_REWARD_KEYS: tuple[str, ...] = ("v_safe_m_s", "gate_height_m")

#: Reward parameters a config may override: every :class:`RewardConfig` field but the
#: structure ones.
REWARD_WEIGHT_KEYS: tuple[str, ...] = tuple(
    f.name for f in fields(RewardConfig) if f.name not in STRUCTURE_REWARD_KEYS
)

Activation = Literal["tanh", "relu"]


@dataclass(frozen=True)
class PolicyConfig:
    """The MLP shared by actor and critic.

    Attributes:
        width: Units per hidden layer, dimensionless.
        depth: Hidden layers, dimensionless.
        activation: ``"tanh"`` or ``"relu"``.
    """

    width: int
    depth: int
    activation: Activation

    @property
    def net_arch(self) -> list[int]:
        """Return the hidden-layer sizes, ``[width] * depth``."""
        return [self.width] * self.depth


@dataclass(frozen=True)
class PPOConfig:
    """SB3 PPO hyperparameters.

    Attributes:
        learning_rate: Initial Adam learning rate, dimensionless.
        lr_schedule: ``"linear"`` (decays to 0 at the budget) or ``"constant"``.
        n_steps: Rollout length per environment, control steps.
        batch_size: Minibatch size, transitions.
        n_epochs: Passes over each rollout.
        gamma: Discount per control step (1/30 s model scale).
        gae_lambda: GAE lambda.
        clip_range: PPO clip parameter.
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value-loss coefficient.
        max_grad_norm: Gradient-norm clip.
        target_kl: Early-stop KL per update, or ``None``.
        log_std_init: Initial log standard deviation of the Gaussian policy, in normalised
            action units (the action is Box(-1, 1)); -1.0 is std 0.37. SB3's default of 0
            (std 1.0) flips the drone in 99.9 % of training episodes (P5-D4).
    """

    learning_rate: float
    lr_schedule: Literal["linear", "constant"]
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    target_kl: float | None
    log_std_init: float


@dataclass(frozen=True)
class SACConfig:
    """SB3 SAC hyperparameters.

    Attributes:
        learning_rate: Adam learning rate, dimensionless (constant).
        buffer_size: Replay capacity, transitions.
        batch_size: Minibatch size, transitions.
        tau: Polyak coefficient.
        gamma: Discount per control step (1/30 s model scale).
        learning_starts: Uniform-random transitions collected before the first update.
        train_freq: Vector-environment steps between updates.
        gradient_steps: Gradient steps per update; ``-1`` = one per transition collected,
            i.e. an update-to-data ratio of 1 whatever ``n_envs`` is.
        ent_coef: ``"auto"``, ``"auto_<init>"`` or a fixed float.
    """

    learning_rate: float
    buffer_size: int
    batch_size: int
    tau: float
    gamma: float
    learning_starts: int
    train_freq: int
    gradient_steps: int
    ent_coef: str | float


@dataclass(frozen=True)
class NormalizeConfig:
    """``VecNormalize`` settings.

    Attributes:
        norm_obs: Normalise observations with running statistics.
        norm_reward: Scale rewards by a running return std (PPO only).
        clip_obs: Clip normalised observations to ``[-clip_obs, clip_obs]``.
        clip_reward: Clip normalised rewards (only if ``norm_reward``).
    """

    norm_obs: bool
    norm_reward: bool
    clip_obs: float
    clip_reward: float


@dataclass(frozen=True)
class EvalConfig:
    """The periodic tune-pool evaluation (learning curves, curriculum, tuning score).

    Attributes:
        interval_steps: Training env steps between evaluations; one more runs at the end.
        n_envs: Evaluation workers (a separate vector env, idle while training runs).
        tuning_config: The P3-D3 tuning config whose ``tuning_seed`` /
            ``episodes_per_sea_state`` / ``pad`` fix the episode draw. With the committed
            ``configs/control/tuning.yaml`` this is the **identical** 180-episode tune-pool
            draw the classical baselines were tuned and re-checked on (P3-D4 table).
        episodes_per_ss: Override of ``episodes_per_sea_state`` (tests only). ``None``
            keeps the committed draw; any other value changes the draw.
        deterministic: Evaluate the deterministic policy (mean action).
    """

    interval_steps: int
    n_envs: int
    tuning_config: Path
    episodes_per_ss: int | None
    deterministic: bool


@dataclass(frozen=True)
class CurriculumConfig:
    """SS3 -> SS4 -> SS5 (P3-D1 section 6).

    Attributes:
        enabled: If False, every reset samples uniformly over all of ``stages``.
        stages: Sea states in order; a subset of :data:`TRAINING_SEA_STATES`.
        promote_success: Rolling tune-pool success needed to advance, dimensionless.
        window_episodes: Tune-pool episodes at the current stage the rolling success is
            taken over; promotion needs at least this many.
    """

    enabled: bool
    stages: tuple[str, ...]
    promote_success: float
    window_episodes: int


@dataclass(frozen=True)
class TrainConfig:
    """One training run, up to its seed.

    Attributes:
        method: Method label in every table, e.g. ``"ppo"``.
        run_group: Path under :data:`RUNS_ROOT`; the run lands in
            ``<RUNS_ROOT>/<run_group>/<seed>/``. Defaults to ``method``.
        algo: ``"ppo"`` or ``"sac"``.
        total_steps: Env-step budget, control steps (PPO overshoots by < one rollout).
        n_envs: Training workers.
        vec_env: ``"subproc"`` or ``"dummy"`` (tests).
        device: Torch device; PPO must be ``"cpu"``.
        torch_threads: ``torch.set_num_threads`` in the learner process.
        pad: Pad name; the primary arm is ``"aft"``.
        policy: The MLP.
        ppo: PPO block (``algo == "ppo"``), else ``None``.
        sac: SAC block (``algo == "sac"``), else ``None``.
        normalize: ``VecNormalize`` settings.
        eval: Periodic tune-pool evaluation.
        curriculum: Sea-state curriculum.
        checkpoint_interval_steps: Env steps between checkpoints (model + stats).
        status_interval_s: Host-clock seconds between ``status.json`` heartbeats.
        log_interval: SB3 ``learn(log_interval=...)``: rollouts (PPO) or episodes (SAC).
        reward: Reward-weight overrides of ``configs/env/reward.yaml``; empty = committed.
        prefetch_reset: Give every env worker (training and evaluation) a second landing
            environment and prepare the next episode on it in a background thread
            (:class:`rld.rl.wrappers.PoolSamplingEnv`). The episodes, observations, rewards
            and evaluation rows are bit-identical either way; only throughput, memory and
            CPU use change. Optional in the YAML (default :data:`PREFETCH_RESET_DEFAULT`).
        source: The YAML file this came from, or ``None``.
    """

    method: str
    run_group: str
    algo: Literal["ppo", "sac"]
    total_steps: int
    n_envs: int
    vec_env: Literal["subproc", "dummy"]
    device: str
    torch_threads: int
    pad: str
    policy: PolicyConfig
    ppo: PPOConfig | None
    sac: SACConfig | None
    normalize: NormalizeConfig
    eval: EvalConfig
    curriculum: CurriculumConfig
    checkpoint_interval_steps: int
    status_interval_s: float
    log_interval: int
    reward: Mapping[str, float] = field(default_factory=dict)
    prefetch_reset: bool = True
    source: Path | None = None

    @property
    def gamma(self) -> float:
        """Return the algorithm's discount, per control step."""
        if self.ppo is not None:
            return self.ppo.gamma
        assert self.sac is not None
        return self.sac.gamma

    @property
    def workers(self) -> int:
        """Return the scheduler slots the run occupies: its measured average core use.

        ``ceil(ENV_WORKER_CORES * busiest env pool) + 1`` for the learner; see
        :data:`ENV_WORKER_CORES` for the measurement. 8 for ``ppo.yaml`` (16 workers), 5 for
        ``sac.yaml`` (8 workers; SAC's usage was not measured, and its synchronous gradient
        steps leave its env workers idle more of the time, so this is conservative).
        """
        return math.ceil(ENV_WORKER_CORES * max(self.n_envs, self.eval.n_envs)) + 1


def _take(
    raw: Mapping[str, Any],
    keys: tuple[str, ...],
    where: str,
    optional: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return ``raw`` after checking it has exactly ``keys`` (plus any of ``optional``).

    Args:
        raw: A parsed YAML mapping.
        keys: Required keys.
        where: Location for the error message.
        optional: Keys that may be present or absent.

    Returns:
        ``raw`` as a plain dict.

    Raises:
        ValueError: On a missing or an unknown key.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    missing = sorted(set(keys) - set(raw))
    unknown = sorted(set(raw) - set(keys) - set(optional))
    if missing or unknown:
        raise ValueError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    return dict(raw)


def _resolve(path: str | Path) -> Path:
    """Return a repository-relative path as absolute; absolute paths pass through."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _rel(path: Path) -> str:
    """Return ``path`` relative to the repository root when it is inside it."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def train_config_from_dict(raw: Mapping[str, Any], where: str = "<dict>") -> TrainConfig:
    """Parse and validate one training config.

    Args:
        raw: The YAML mapping.
        where: Its origin, for error messages.

    Returns:
        The :class:`TrainConfig`.

    Raises:
        ValueError: On a missing or unknown key, or a protocol violation (see the module
            docstring).
    """
    top = _take(
        raw,
        (
            "method",
            "run_group",
            "algo",
            "total_steps",
            "n_envs",
            "vec_env",
            "device",
            "torch_threads",
            "pad",
            "policy",
            "ppo",
            "sac",
            "normalize",
            "eval",
            "curriculum",
            "checkpoint_interval_steps",
            "status_interval_s",
            "log_interval",
            "reward",
        ),
        where,
        optional=("prefetch_reset",),
    )
    algo = str(top["algo"])
    if algo not in ALGOS:
        raise ValueError(f"{where}: algo must be one of {ALGOS}, got {algo!r}")
    method = str(top["method"])
    run_group = str(top["run_group"] or method)
    if Path(run_group).is_absolute() or ".." in Path(run_group).parts or run_group.startswith("_"):
        raise ValueError(f"{where}: run_group must be a relative path, got {run_group!r}")

    pol = _take(top["policy"], ("width", "depth", "activation"), f"{where}: policy")
    if pol["activation"] not in ("tanh", "relu"):
        raise ValueError(f"{where}: policy.activation must be tanh or relu")
    policy = PolicyConfig(int(pol["width"]), int(pol["depth"]), pol["activation"])

    ppo: PPOConfig | None = None
    sac: SACConfig | None = None
    if algo == "ppo":
        if top["ppo"] is None:
            raise ValueError(f"{where}: algo ppo needs a ppo block")
        if top["sac"] is not None:
            raise ValueError(f"{where}: algo ppo must leave sac: null")
        p = _take(
            top["ppo"],
            tuple(f.name for f in fields(PPOConfig)),
            f"{where}: ppo",
        )
        if p["lr_schedule"] not in ("linear", "constant"):
            raise ValueError(f"{where}: ppo.lr_schedule must be linear or constant")
        ppo = PPOConfig(
            learning_rate=float(p["learning_rate"]),
            lr_schedule=p["lr_schedule"],
            n_steps=int(p["n_steps"]),
            batch_size=int(p["batch_size"]),
            n_epochs=int(p["n_epochs"]),
            gamma=float(p["gamma"]),
            gae_lambda=float(p["gae_lambda"]),
            clip_range=float(p["clip_range"]),
            ent_coef=float(p["ent_coef"]),
            vf_coef=float(p["vf_coef"]),
            max_grad_norm=float(p["max_grad_norm"]),
            target_kl=None if p["target_kl"] is None else float(p["target_kl"]),
            log_std_init=float(p["log_std_init"]),
        )
    else:
        if top["sac"] is None:
            raise ValueError(f"{where}: algo sac needs a sac block")
        if top["ppo"] is not None:
            raise ValueError(f"{where}: algo sac must leave ppo: null")
        s = _take(top["sac"], tuple(f.name for f in fields(SACConfig)), f"{where}: sac")
        ent = s["ent_coef"]
        sac = SACConfig(
            learning_rate=float(s["learning_rate"]),
            buffer_size=int(s["buffer_size"]),
            batch_size=int(s["batch_size"]),
            tau=float(s["tau"]),
            gamma=float(s["gamma"]),
            learning_starts=int(s["learning_starts"]),
            train_freq=int(s["train_freq"]),
            gradient_steps=int(s["gradient_steps"]),
            ent_coef=str(ent) if isinstance(ent, str) else float(ent),
        )

    n = _take(
        top["normalize"],
        ("norm_obs", "norm_reward", "clip_obs", "clip_reward"),
        f"{where}: normalize",
    )
    normalize = NormalizeConfig(
        bool(n["norm_obs"]), bool(n["norm_reward"]), float(n["clip_obs"]), float(n["clip_reward"])
    )

    e = _take(
        top["eval"],
        ("interval_steps", "n_envs", "tuning_config", "episodes_per_ss", "deterministic"),
        f"{where}: eval",
    )
    evalc = EvalConfig(
        interval_steps=int(e["interval_steps"]),
        n_envs=int(e["n_envs"]),
        tuning_config=_resolve(e["tuning_config"] or TUNING_CONFIG),
        episodes_per_ss=None if e["episodes_per_ss"] is None else int(e["episodes_per_ss"]),
        deterministic=bool(e["deterministic"]),
    )

    c = _take(
        top["curriculum"],
        ("enabled", "stages", "promote_success", "window_episodes"),
        f"{where}: curriculum",
    )
    curriculum = CurriculumConfig(
        enabled=bool(c["enabled"]),
        stages=tuple(str(s) for s in c["stages"]),
        promote_success=float(c["promote_success"]),
        window_episodes=int(c["window_episodes"]),
    )

    reward_raw = top["reward"] or {}
    if not isinstance(reward_raw, Mapping):
        raise ValueError(f"{where}: reward must be a mapping of weight overrides")
    reward = {str(k): float(v) for k, v in reward_raw.items()}

    cfg = TrainConfig(
        method=method,
        run_group=run_group,
        algo=cast(Literal["ppo", "sac"], algo),
        total_steps=int(top["total_steps"]),
        n_envs=int(top["n_envs"]),
        vec_env=top["vec_env"],
        device=str(top["device"]),
        torch_threads=int(top["torch_threads"]),
        pad=str(top["pad"]),
        policy=policy,
        ppo=ppo,
        sac=sac,
        normalize=normalize,
        eval=evalc,
        curriculum=curriculum,
        checkpoint_interval_steps=int(top["checkpoint_interval_steps"]),
        status_interval_s=float(top["status_interval_s"]),
        log_interval=int(top["log_interval"]),
        reward=reward,
        prefetch_reset=bool(top.get("prefetch_reset", PREFETCH_RESET_DEFAULT)),
    )
    _validate(cfg, where)
    return cfg


def _validate(cfg: TrainConfig, where: str) -> None:
    """Enforce the protocol rules on a parsed config.

    Args:
        cfg: The parsed config.
        where: Origin for error messages.

    Raises:
        ValueError: On any violation.
    """
    errors: list[str] = []
    if cfg.vec_env not in ("subproc", "dummy"):
        errors.append(f"vec_env must be subproc or dummy, got {cfg.vec_env!r}")
    if cfg.algo == "ppo" and cfg.device != "cpu":
        errors.append("PPO trains on the CPU (CLAUDE.md); set device: cpu")
    if cfg.algo == "sac" and cfg.normalize.norm_reward:
        errors.append("reward normalisation is PPO-only; set normalize.norm_reward: false")
    for name in ("total_steps", "n_envs", "torch_threads", "checkpoint_interval_steps"):
        if int(getattr(cfg, name)) < 1:
            errors.append(f"{name} must be positive")
    if cfg.eval.interval_steps < 1 or cfg.eval.n_envs < 1:
        errors.append("eval.interval_steps and eval.n_envs must be positive")
    if cfg.eval.episodes_per_ss is not None and cfg.eval.episodes_per_ss < 1:
        errors.append("eval.episodes_per_ss must be positive or null")
    if cfg.policy.width < 1 or cfg.policy.depth < 1:
        errors.append("policy width and depth must be positive")
    if cfg.ppo is not None and (cfg.ppo.n_steps * cfg.n_envs) % cfg.ppo.batch_size != 0:
        errors.append(
            f"ppo.batch_size {cfg.ppo.batch_size} must divide n_steps*n_envs "
            f"{cfg.ppo.n_steps * cfg.n_envs}"
        )
    stages = cfg.curriculum.stages
    if not stages or len(set(stages)) != len(stages):
        errors.append(f"curriculum.stages must be non-empty and distinct, got {stages}")
    if forbidden := sorted(set(stages) & set(FORBIDDEN_SEA_STATES)):
        errors.append(f"curriculum.stages contains {forbidden}: SS6 is never trained on")
    if outside := sorted(set(stages) - set(TRAINING_SEA_STATES)):
        errors.append(f"curriculum.stages {outside} are not training sea states")
    if not 0.0 < cfg.curriculum.promote_success <= 1.0:
        errors.append("curriculum.promote_success must be in (0, 1]")
    if cfg.curriculum.window_episodes < 1:
        errors.append("curriculum.window_episodes must be positive")
    if structural := sorted(set(cfg.reward) & set(STRUCTURE_REWARD_KEYS)):
        errors.append(f"reward {structural} are structure, not weights, and are fixed")
    if unknown := sorted(set(cfg.reward) - set(REWARD_WEIGHT_KEYS) - set(STRUCTURE_REWARD_KEYS)):
        errors.append(f"reward keys {unknown} are not reward weights")
    if negative := sorted(k for k, v in cfg.reward.items() if v < 0.0):
        errors.append(f"reward weights {negative} must be non-negative (penalties are magnitudes)")
    if errors:
        raise ValueError(f"{where}: " + "; ".join(errors))


def load_train_config(path: Path) -> TrainConfig:
    """Load one training config file.

    Args:
        path: ``configs/rl/<name>.yaml``, absolute or repository-relative.

    Returns:
        The validated :class:`TrainConfig`, with ``source`` set.
    """
    resolved = _resolve(path)
    cfg = train_config_from_dict(load_yaml(resolved), str(resolved))
    return replace(cfg, source=resolved)


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Return a YAML-serialisable dict that :func:`train_config_from_dict` parses back.

    Args:
        cfg: The config.

    Returns:
        A plain nested dict; paths repository-relative, tuples as lists, ``source``
        dropped.
    """
    out = asdict(cfg)
    out.pop("source", None)
    out["eval"]["tuning_config"] = _rel(cfg.eval.tuning_config)
    out["curriculum"]["stages"] = list(cfg.curriculum.stages)
    out["reward"] = dict(cfg.reward)
    return out


def apply_overrides(raw: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of a raw config with dotted-key overrides applied.

    Args:
        raw: A raw YAML mapping (before parsing).
        overrides: ``{"ppo.learning_rate": 1e-4, "reward.w_vz": 3.0, ...}``. A ``reward.*``
            key may add a weight absent from the base; any other key must already exist.

    Returns:
        The overridden mapping, still unparsed; :func:`train_config_from_dict` validates it.

    Raises:
        KeyError: If a non-``reward`` target does not exist in ``raw``.
    """
    out = copy.deepcopy(dict(raw))
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node: dict[str, Any] = out
        for part in parts[:-1]:
            if node.get(part) is None and part == "reward":
                node[part] = {}
            if not isinstance(node.get(part), dict):
                raise KeyError(f"override target {dotted!r}: {part!r} is not a mapping")
            node = node[part]
        leaf = parts[-1]
        if leaf not in node and parts[0] != "reward":
            raise KeyError(f"override target {dotted!r} does not exist in the base config")
        node[leaf] = value
    return out
