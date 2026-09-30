"""Training configuration: ``configs/rl/*.yaml`` -> frozen dataclasses.

One YAML file describes one training run up to its seed. Every key is required unless noted
and every unknown key is an error, so a typo in a search-space target cannot silently train
the default. Four keys are optional:

* ``prefetch_reset``, an engineering switch that cannot change what is trained (see
  :attr:`TrainConfig.prefetch_reset`);
* ``residual``, ``forecast_obs`` and ``motion``, the Phase 6 method components (P3-D1
  sections 5-6): residual RL on a classical base, the forecast observation block, and the
  deck-motion model the run trains on. Absent means off (``null``, ``null``, ``jonswap``),
  which is exactly what every run written before they existed trained with, so those run
  directories still load; :func:`config_to_dict` writes them only when they are not the
  default, so the serialised text (and its SHA-256) of a pure PPO or SAC config is unchanged.

The checks that encode protocol decisions live here, at load time:

* PPO trains on the CPU (CLAUDE.md, "SB3 MLP on GPU is slower than CPU");
* reward normalisation is PPO-only (plan Phase 5 item 1);
* the curriculum's stages are a subset of SS3-SS5 and **never** SS6 (P3-D1 section 6);
* only reward **weights** may be overridden -- ``v_safe_m_s`` and ``gate_height_m`` shape the
  reward's structure and are fixed (``configs/env/reward.yaml``);
* a residual base is a registered, non-privileged controller that takes no ship-motion feed,
  and residual RL is PPO-only (its zero-initialised last layer is PPO's ``action_net``).

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
    "FORECAST_GRID_FULL_HZ",
    "FORECAST_HORIZON_FULL_S",
    "MOTION_KINDS",
    "OPTIONAL_KEYS",
    "PREFETCH_RESET_DEFAULT",
    "REWARD_WEIGHT_KEYS",
    "RL_CONFIG_DIR",
    "RUNS_ROOT",
    "STRUCTURE_REWARD_KEYS",
    "TRAINING_SEA_STATES",
    "ENV_WORKER_CORES",
    "CurriculumConfig",
    "EvalConfig",
    "ForecastObsConfig",
    "MotionKind",
    "NormalizeConfig",
    "PPOConfig",
    "PolicyConfig",
    "ResidualConfig",
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
#: time-averaged ~6.4 cores. 0.4 x 16 workers + 1 learner thread = 8 slots covers that, so
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

#: Deck-motion models a run may train on: dmf's JONSWAP realizations (every method but
#: ``ppo_sinusoid``) or the matched single-frequency sinusoid (the H4 arm, plan D0.4).
MOTION_KINDS: tuple[str, ...] = ("jonswap", "sinusoid")

#: The forecaster's lead grid, hertz **full** scale (dmf's 10 Hz corpus rate): every lead of
#: the forecast observation block must be a multiple of 0.1 s full scale.
FORECAST_GRID_FULL_HZ: float = 10.0

#: The forecaster's horizon, seconds **full** scale (150 leads at 10 Hz = 3.0 s model).
FORECAST_HORIZON_FULL_S: float = 15.0

#: Top-level keys a config may omit (module docstring).
OPTIONAL_KEYS: tuple[str, ...] = ("prefetch_reset", "residual", "forecast_obs", "motion")

Activation = Literal["tanh", "relu"]
MotionKind = Literal["jonswap", "sinusoid"]


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
class ResidualConfig:
    """Residual RL on a classical base (P3-D1 section 6).

    The executed action is ``clip(a_base(o) + alpha * pi(o), -1, 1)`` in the shared
    normalised action space, after which the environment applies its own norm cap
    (:func:`rld.rl.residual.compose_residual`).

    Attributes:
        alpha: Residual scale, normalised action units (0.3 = 0.45 m/s model scale at
            ``v_max`` = 1.5 m/s).
        base: Registry name of the base controller (``"pid_feedforward"``), built from its
            committed config under ``configs/control/``.
    """

    alpha: float
    base: str


@dataclass(frozen=True)
class ForecastObsConfig:
    """The forecast observation block (P4-D4a "Deferred"; plan Phase 6).

    Attributes:
        forecaster: Name of the fitted dmf model directory under ``artifacts/dmf/``
            (``"residual_interval"``, the forecaster ``configs/control/gated_forecast.yaml``
            uses).
        leads_full_s: Forecast leads, seconds **full** scale (1, 2, 3 s = 0.2, 0.4, 0.6 s
            model at ``lam = 1/25``); each contributes the point pad ``z`` (metres model) and
            pad ``v_z`` (metres per second model), so the block has ``2 * len`` entries.
    """

    forecaster: str
    leads_full_s: tuple[float, ...]

    @property
    def block_size(self) -> int:
        """Return the number of observation entries the block appends, dimensionless."""
        return 2 * len(self.leads_full_s)


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
        residual: Residual RL on a classical base, or ``None`` (optional in the YAML).
        forecast_obs: The forecast observation block appended to the environment
            observation, or ``None`` (optional in the YAML).
        motion: The deck-motion model of every training **and** tune-pool evaluation
            episode: ``"jonswap"`` (dmf's realizations) or ``"sinusoid"`` (each episode a
            :class:`~rld.deck.sinusoid.SinusoidDeckMotion` matched to its realization,
            phases from the episode seed). Optional in the YAML (default ``"jonswap"``).
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
    residual: ResidualConfig | None = None
    forecast_obs: ForecastObsConfig | None = None
    motion: MotionKind = "jonswap"
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

        ``ceil(ENV_WORKER_CORES * busiest env pool)`` for the env workers, plus
        ``torch_threads`` for the learner (a learner with ``t`` intra-op threads keeps up
        to ``t`` cores busy during its gradient steps); see :data:`ENV_WORKER_CORES` for the
        measurement. With one torch thread: 8 for ``ppo.yaml`` (16 workers), 5 for
        ``sac.yaml`` (8 workers); a 4-thread SAC run costs 8.

        The Phase 6 components start no thread: the residual base is a few NumPy operations
        per step, and the forecast block's ONNX Runtime session runs with one intra-op and
        one inter-op thread, sequentially, on the stepping thread. They make each step
        dearer, not wider, so the formula is unchanged; the forecast block's per-step cost
        is recorded with the method (P6-D1).
        """
        pool = math.ceil(ENV_WORKER_CORES * max(self.n_envs, self.eval.n_envs))
        return pool + self.torch_threads


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
        optional=OPTIONAL_KEYS,
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

    residual: ResidualConfig | None = None
    if top.get("residual") is not None:
        r = _take(top["residual"], ("alpha", "base"), f"{where}: residual")
        residual = ResidualConfig(alpha=float(r["alpha"]), base=str(r["base"]))
    forecast_obs: ForecastObsConfig | None = None
    if top.get("forecast_obs") is not None:
        f = _take(top["forecast_obs"], ("forecaster", "leads_full_s"), f"{where}: forecast_obs")
        leads = f["leads_full_s"]
        if not isinstance(leads, list | tuple):
            raise ValueError(f"{where}: forecast_obs.leads_full_s must be a list")
        forecast_obs = ForecastObsConfig(
            forecaster=str(f["forecaster"]), leads_full_s=tuple(float(x) for x in leads)
        )
    motion = str(top.get("motion") or "jonswap")
    if motion not in MOTION_KINDS:
        raise ValueError(f"{where}: motion must be one of {MOTION_KINDS}, got {motion!r}")

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
        residual=residual,
        forecast_obs=forecast_obs,
        motion=cast(MotionKind, motion),
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
    errors += _residual_errors(cfg)
    errors += _forecast_errors(cfg.forecast_obs)
    if errors:
        raise ValueError(f"{where}: " + "; ".join(errors))


def _residual_errors(cfg: TrainConfig) -> list[str]:
    """Return what is wrong with a config's residual block (empty when fine or absent).

    Args:
        cfg: The parsed config.

    Returns:
        Error messages.
    """
    res = cfg.residual
    if res is None:
        return []
    from rld.control.registry import REGISTRY  # local: the registry imports every controller

    errors: list[str] = []
    if cfg.algo != "ppo":
        errors.append("residual RL is PPO-only (its zero-initialised layer is PPO's action_net)")
    if not (math.isfinite(res.alpha) and 0.0 < res.alpha <= 1.0):
        errors.append(f"residual.alpha must be in (0, 1] normalised units, got {res.alpha}")
    item = REGISTRY.get(res.base)
    if item is None:
        errors.append(f"residual.base {res.base!r} is not a registered controller")
    elif item.privileged or item.needs_motion_feed:
        errors.append(
            f"residual.base {res.base!r} is privileged or needs a motion feed; the base must "
            "see only the observation"
        )
    return errors


def _forecast_errors(fc: ForecastObsConfig | None) -> list[str]:
    """Return what is wrong with a forecast-observation block (empty when fine or absent).

    Args:
        fc: The block, or ``None``.

    Returns:
        Error messages.
    """
    if fc is None:
        return []
    errors: list[str] = []
    name = fc.forecaster
    if not name or Path(name).name != name or name.startswith((".", "_")):
        errors.append(f"forecast_obs.forecaster must be a model directory name, got {name!r}")
    leads = fc.leads_full_s
    if not leads or len(set(leads)) != len(leads):
        errors.append(f"forecast_obs.leads_full_s must be non-empty and distinct, got {leads}")
    for lead in leads:
        steps = lead * FORECAST_GRID_FULL_HZ
        if not (0.0 < lead <= FORECAST_HORIZON_FULL_S and abs(steps - round(steps)) < 1e-9):
            errors.append(
                f"forecast_obs lead {lead} s (full) is not on the 0.1 s grid inside "
                f"(0, {FORECAST_HORIZON_FULL_S}]"
            )
    return errors


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
        dropped, and ``residual`` / ``forecast_obs`` / ``motion`` omitted at their defaults.
    """
    out = asdict(cfg)
    out.pop("source", None)
    out["eval"]["tuning_config"] = _rel(cfg.eval.tuning_config)
    out["curriculum"]["stages"] = list(cfg.curriculum.stages)
    out["reward"] = dict(cfg.reward)
    # The Phase 6 keys are written only when set, so a config without them serialises to
    # the same text (and SHA-256) as before they existed (module docstring).
    if cfg.residual is None:
        out.pop("residual")
    if cfg.forecast_obs is None:
        out.pop("forecast_obs")
    else:
        out["forecast_obs"]["leads_full_s"] = list(cfg.forecast_obs.leads_full_s)
    if cfg.motion == "jonswap":
        out.pop("motion")
    return out


def apply_overrides(raw: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of a raw config with dotted-key overrides applied.

    Args:
        raw: A raw YAML mapping (before parsing).
        overrides: ``{"ppo.learning_rate": 1e-4, "reward.w_vz": 3.0, ...}``. A ``reward.*``
            key may add a weight absent from the base, and one of :data:`OPTIONAL_KEYS` may
            be added at the top level; any other key must already exist.

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
        optional_top = len(parts) == 1 and leaf in OPTIONAL_KEYS
        if leaf not in node and parts[0] != "reward" and not optional_top:
            raise KeyError(f"override target {dotted!r} does not exist in the base config")
        node[leaf] = value
    return out
