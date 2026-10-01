"""Train one PPO or SAC policy on the landing environment, and load it back for evaluation.

What a run does
---------------
1. Resolve the pools: :func:`rld.deck.splits.dev_pool` -> ``(train, tune)`` (P3-D2). Training
   workers hold **only** train-pool realizations at the curriculum's sea states (never
   SS6); evaluation workers hold **only** the tune pool. The two are asserted disjoint.
2. Build ``n_envs`` training workers (:class:`rld.rl.wrappers.EnvFactory` ->
   ``SubprocVecEnv``), wrapped in ``VecNormalize(norm_obs, norm_reward=PPO only,
   clip_obs=10)``. PPO runs on the CPU; SAC on ``cfg.device``.
3. Every ``eval.interval_steps``: the tune-pool evaluation, the curriculum update and a
   ``status.json`` write (:mod:`rld.rl.callbacks`). Every ``checkpoint_interval_steps``: a
   checkpoint with ``model.zip`` **and** ``vecnormalize.pkl``, plus the resume state
   (and the SAC replay buffer) that makes it resumable.
4. At the end: a final evaluation (unless one just ran), the ``final/`` checkpoint and
   ``status.json`` ``state: done``. On any exception -- including SIGTERM, turned into
   ``SystemExit`` -- the traceback goes to stderr (the run's ``train.log``) and
   ``status.json`` says ``state: failed``. A failed run directory is kept; it is never
   reused for a *fresh* run, but it may be **resumed** in place from its latest resumable
   checkpoint (``train(..., resume_from=...)``; every periodic checkpoint is resumable --
   :mod:`rld.rl.resume` documents what is saved, what a resume records, and what is not
   bit-exact). One process at a time holds ``<run_dir>/.run.lock``.

Run directories
---------------
``artifacts/runs/<run_group>/<seed>/`` (:func:`prepare_run_dir`). A finished (``done``) or live
run of the same group and seed makes the launch **refuse**; a failed or abandoned one is
kept and the new run takes the next free suffix, ``<seed>_r2``, ``<seed>_r3``, ... . Nothing
is ever deleted or overwritten.

Evaluation-time loading
-----------------------
:func:`load_policy` returns a :class:`LearnedPolicy` (or, for the Phase 6 methods, one of its
subclasses; :func:`policy_class`) satisfying :class:`rld.control.base.Controller`: never
privileged, and normalising observations with the checkpoint's own statistics loaded with
``training=False, norm_reward=False``. A pure or residual policy never takes a ship-motion
feed (it raises); a forecast policy requires one (``needs_motion_feed = True``;
:func:`run_needs_motion_feed` tells the caller which, so it can set ``callable_spec``'s flag).
:func:`build_policy` is the picklable factory ``rld.eval.runner.callable_spec`` takes (bind
``run_dir``/``ckpt`` with ``functools.partial``); it builds a residual base from the
evaluation's configs, exactly as the runner builds the ``pid_feedforward`` baseline.

Phase 6 components
------------------
``residual`` (:mod:`rld.rl.residual`), ``forecast_obs`` (:mod:`rld.rl.forecast_obs`) and
``motion`` (:mod:`rld.rl.motion`) are wired into every training **and** tune-pool
evaluation worker through :class:`rld.rl.wrappers.EnvFactory`, so learning curves and
curriculum promotion see the same method as training. A residual run's ``action_net`` is
zero-initialised right after the PPO model is built (never on resume).
``provenance.json`` records the SHA-256 of the residual base's committed config, of the
forecaster's ``meta.json`` and of the deck-stats file a sinusoid run reads; loading a
checkpoint refuses a base config or forecaster that has changed since training.

Nothing in this module reads ``results/episodes/``.

Units: steps are env control steps (1/30 s model scale each); ``*_s`` wall-clock fields are
host seconds; actions are the shared normalised velocity setpoint (``v_max`` = 1.5 m/s model
scale, norm cap).
"""

import fcntl
import hashlib
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import FrameType
from typing import Any

import torch
import yaml
from dmf.sim.generate import RealizationSpec
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from rld.config import REPO_ROOT
from rld.control.registry import entry
from rld.control.tuning import TuningEpisode, draw_tuning_episodes, load_tuning
from rld.deck.splits import dev_pool, key_for_spec
from rld.envs.observation import observation_space
from rld.eval.envs import EvalConfigs, load_eval_configs
from rld.provenance import environment_provenance
from rld.rl.callbacks import (
    REPLAY_BUFFER_FILE,
    CheckpointCallback,
    PeriodicEvalCallback,
    RunTracker,
    StatusCallback,
    StatusWriter,
    TunePoolEvaluator,
    atomic_write_json,
    frozen_normalizer,
    save_checkpoint,
    utc_now,
)
from rld.rl.config import (
    FORBIDDEN_SEA_STATES,
    RUNS_ROOT,
    TrainConfig,
    config_to_dict,
    train_config_from_dict,
)
from rld.rl.curriculum import Curriculum
from rld.rl.forecast_obs import (
    ForecastPolicy,
    ResidualForecastPolicy,
    forecaster_dir,
)
from rld.rl.motion import DECK_STATS_SEEDS, committed_rms_full, file_sha256
from rld.rl.policy import LearnedPolicy
from rld.rl.procs import status_owner_alive
from rld.rl.residual import ResidualPolicy, build_base_controller, zero_init_action_net
from rld.rl.resume import (
    ResumeError,
    ResumePlan,
    collect_resume_state,
    normalizer_digest,
    params_digest,
    plan_resume,
    resolve_resume_target,
    restore_rng,
    truncate_eval_logs,
)
from rld.rl.wrappers import EnvFactory

__all__ = [
    "LearnedPolicy",
    "LinearDecay",
    "RunDirExistsError",
    "build_env_configs",
    "build_policy",
    "checkpoint_dir",
    "eval_episodes",
    "forkserver_preload",
    "git_state",
    "load_policy",
    "load_vecnormalize",
    "policy_class",
    "policy_input_size",
    "prepare_resume",
    "prepare_run_dir",
    "read_status",
    "run_needs_motion_feed",
    "train",
    "training_pools",
]


class RunDirExistsError(FileExistsError):
    """A finished or live run with the same group and seed already exists."""


class LinearDecay:
    """Picklable linear learning-rate schedule: ``lr * progress_remaining``.

    Attributes:
        initial: Learning rate at the start, dimensionless.
    """

    def __init__(self, initial: float) -> None:
        """Hold the initial rate.

        Args:
            initial: Learning rate at progress 0.
        """
        self.initial = float(initial)

    def __call__(self, progress_remaining: float) -> float:
        """Return the rate at ``progress_remaining`` (1 at start, 0 at the budget)."""
        return self.initial * float(progress_remaining)


# --------------------------------------------------------------------------- run dirs


def git_state(repo_root: Path) -> dict[str, Any]:
    """Return the checkout's commit and whether it has uncommitted changes.

    Same fields and semantics as :func:`rld.eval.reproduce.git_state`, re-implemented here
    so that the training path does not import ``rld.eval.reproduce`` (whose report module
    imports the frozen-list reader ``rld.eval.episodes``; ``tests/test_rl_leakage.py``
    asserts the RL import graph never contains it).

    Args:
        repo_root: The repository root.

    Returns:
        ``{"git_sha": <HEAD or "unknown">, "git_dirty": bool | None, "git_dirty_paths":
        [porcelain lines, at most 50]}``; ``git_dirty`` is None when git is unavailable.
    """

    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout

    sha = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=normal")
    paths = [] if status is None else [line for line in status.splitlines() if line]
    return {
        "git_sha": "unknown" if sha is None else sha.strip(),
        "git_dirty": None if status is None else bool(paths),
        "git_dirty_paths": paths[:50],
    }


def read_status(run_dir: Path) -> dict[str, Any] | None:
    """Return a run's ``status.json``, or ``None`` if it has none or it is unreadable."""
    path = run_dir / "status.json"
    try:
        loaded: Any = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def prepare_run_dir(cfg: TrainConfig, seed: int, runs_root: Path = RUNS_ROOT) -> Path:
    """Create a fresh run directory for ``(cfg.run_group, seed)``.

    Args:
        cfg: The training config.
        seed: The training seed.
        runs_root: Root of all runs.

    Returns:
        The new, empty directory: ``<runs_root>/<run_group>/<seed>`` or, if that holds a
        failed or abandoned run, the first free ``<seed>_r<k>`` (k >= 2).

    Raises:
        RunDirExistsError: If any directory of this group and seed holds a ``done`` run or
            a ``running`` run whose process is alive.
    """
    parent = runs_root / cfg.run_group
    parent.mkdir(parents=True, exist_ok=True)
    names = [str(seed)] + [f"{seed}_r{k}" for k in range(2, 1000)]
    for name in names:
        candidate = parent / name
        if not candidate.exists():
            # Refuse if a later suffix is done or live (a gap left by a manual edit).
            for sibling in parent.glob(f"{seed}_r*"):
                _refuse_if_finished(sibling)
            try:
                candidate.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            return candidate
        _refuse_if_finished(candidate)
    raise RunDirExistsError(f"no free run directory for {cfg.run_group}/{seed}")


def prepare_resume(cfg: TrainConfig, seed: int, target: Path) -> tuple[Path, Path]:
    """Validate a resume without writing anything (``make train-bg RESUME=...``).

    Args:
        cfg: The training config given; must equal the run's own ``config.yaml``.
        seed: The training seed given; must be the run's.
        target: A failed run directory (its latest checkpoint) or that checkpoint.

    Returns:
        ``(run_dir, checkpoint)``.

    Raises:
        ResumeError: If the resume is not allowed (:func:`rld.rl.resume.plan_resume`).
    """
    run_dir, ckpt = resolve_resume_target(target)
    text = yaml.safe_dump(config_to_dict(cfg), sort_keys=False)
    plan_resume(
        run_dir,
        ckpt,
        seed=seed,
        algo=cfg.algo,
        config_text=text,
        config_sha256=hashlib.sha256(text.encode()).hexdigest(),
        status=read_status(run_dir),
    )
    return run_dir, ckpt


def _refuse_if_finished(run_dir: Path) -> None:
    """Raise if ``run_dir`` holds a finished or live run.

    Args:
        run_dir: An existing run directory.

    Raises:
        RunDirExistsError: On ``done``, or ``running`` with a live pid.
    """
    status = read_status(run_dir)
    if status is None:
        return
    if status.get("state") == "done":
        raise RunDirExistsError(f"{run_dir} holds a finished run; refusing to train it again")
    if status.get("state") == "running" and status_owner_alive(status):
        raise RunDirExistsError(f"{run_dir} holds a live run (pid {status.get('pid')})")


# --------------------------------------------------------------------------- pools and envs

#: Modules the ``forkserver`` imports once, before it forks any env worker.
FORKSERVER_PRELOAD: tuple[str, ...] = (
    "torch",
    "stable_baselines3",
    "rld.rl.wrappers",
    "rld.rl.eval_workers",
    "rld.rl.residual",
    "rld.rl.forecast_obs",
)


def forkserver_preload() -> None:
    """Have the ``forkserver`` import the heavy modules once, before forking any worker.

    ``SubprocVecEnv`` starts its workers from Python's ``forkserver``. Without a preload each
    worker imports torch, SB3, PyBullet and dmf itself; with it they fork from a server
    that already has them, sharing those pages copy-on-write. Measured at 16 workers with
    prefetch (2026-09-24): proportional set size 479 -> 141 MB per worker (7.7 -> 2.4 GB for
    the pool, forkserver included), start-up 4.2 -> 3.2 s, throughput unchanged. Takes
    effect only if the forkserver has not started yet in this process (a no-op afterwards).
    """
    multiprocessing.set_forkserver_preload(list(FORKSERVER_PRELOAD))


def build_env_configs(cfg: TrainConfig) -> EvalConfigs:
    """Load the committed env configs and apply the run's reward-weight overrides.

    Args:
        cfg: The training config.

    Returns:
        The configs every worker builds its environment from.
    """
    cfgs = load_eval_configs()
    if cfg.reward:
        cfgs = replace(cfgs, reward=replace(cfgs.reward, **dict(cfg.reward)))
    return cfgs


def training_pools(
    cfg: TrainConfig, cfgs: EvalConfigs
) -> tuple[list[RealizationSpec], list[RealizationSpec]]:
    """Return the (train, tune) realizations a run may use.

    Args:
        cfg: The training config.
        cfgs: The env configs (for dmf's grid).

    Returns:
        ``train`` restricted to the curriculum's sea states, and the full ``tune`` pool
        (the evaluation draw picks from it).

    Raises:
        RuntimeError: If the pools overlap, hold SS6, or a stage has no train realization.
    """
    train_all, tune = dev_pool(cfgs.sim)
    stages = set(cfg.curriculum.stages)
    train = [s for s in train_all if s.sea_state in stages]
    if {key_for_spec(s) for s in train} & {key_for_spec(s) for s in tune}:
        raise RuntimeError("train and tune pools overlap")
    held = {s.sea_state for s in train} | {s.sea_state for s in tune}
    if held & set(FORBIDDEN_SEA_STATES):
        raise RuntimeError(f"a development pool holds {sorted(held & set(FORBIDDEN_SEA_STATES))}")
    if missing := sorted(stages - {s.sea_state for s in train}):
        raise RuntimeError(f"no train-pool realization at {missing}")
    return train, tune


def eval_episodes(cfg: TrainConfig, cfgs: EvalConfigs) -> list[TuningEpisode]:
    """Return the fixed tune-pool evaluation episodes.

    Args:
        cfg: The training config.
        cfgs: The env configs.

    Returns:
        The P3-D3 draw from ``cfg.eval.tuning_config`` on the run's pad (60 per sea state
        with the committed file), or a smaller draw when ``eval.episodes_per_ss`` is set.
    """
    tcfg = load_tuning(cfg.eval.tuning_config)
    tcfg = replace(tcfg, pad=cfg.pad)
    if cfg.eval.episodes_per_ss is not None:
        tcfg = replace(tcfg, episodes_per_sea_state=cfg.eval.episodes_per_ss)
    return draw_tuning_episodes(cfgs.sim, tcfg)


def _vec(factories: list[EnvFactory], kind: str) -> VecEnv:
    """Build a vector env from factories."""
    fns: list[Any] = list(factories)
    return SubprocVecEnv(fns) if kind == "subproc" else DummyVecEnv(fns)


def _build_model(cfg: TrainConfig, venv: VecEnv, seed: int, run_dir: Path) -> BaseAlgorithm:
    """Instantiate the SB3 model.

    Args:
        cfg: The training config.
        venv: The (normalised) training vector env.
        seed: The training seed.
        run_dir: Run directory; TensorBoard goes to ``tb/``.

    Returns:
        The model; a residual run's ``action_net`` is zero-initialised
        (:func:`rld.rl.residual.zero_init_action_net`).
    """
    activation = torch.nn.Tanh if cfg.policy.activation == "tanh" else torch.nn.ReLU
    arch = cfg.policy.net_arch
    tb = str(run_dir / "tb")
    if cfg.ppo is not None:
        p = cfg.ppo
        lr: Any = LinearDecay(p.learning_rate) if p.lr_schedule == "linear" else p.learning_rate
        ppo = PPO(
            "MlpPolicy",
            venv,
            learning_rate=lr,
            n_steps=p.n_steps,
            batch_size=p.batch_size,
            n_epochs=p.n_epochs,
            gamma=p.gamma,
            gae_lambda=p.gae_lambda,
            clip_range=p.clip_range,
            ent_coef=p.ent_coef,
            vf_coef=p.vf_coef,
            max_grad_norm=p.max_grad_norm,
            target_kl=p.target_kl,
            policy_kwargs={
                "net_arch": {"pi": arch, "vf": arch},
                "activation_fn": activation,
                "log_std_init": p.log_std_init,
            },
            tensorboard_log=tb,
            verbose=1,
            seed=seed,
            device="cpu",
        )
        if cfg.residual is not None:
            # P3-D1 section 6: the initial policy is the base, a_res = 0 exactly.
            zero_init_action_net(ppo)
        return ppo
    assert cfg.sac is not None
    s = cfg.sac
    return SAC(
        "MlpPolicy",
        venv,
        learning_rate=s.learning_rate,
        buffer_size=s.buffer_size,
        batch_size=s.batch_size,
        tau=s.tau,
        gamma=s.gamma,
        learning_starts=s.learning_starts,
        train_freq=s.train_freq,
        gradient_steps=s.gradient_steps,
        ent_coef=s.ent_coef,
        policy_kwargs={"net_arch": {"pi": arch, "qf": arch}, "activation_fn": activation},
        tensorboard_log=tb,
        verbose=1,
        seed=seed,
        device=cfg.device,
    )


def _factory_components(
    cfg: TrainConfig,
    train_pool: list[RealizationSpec],
    tune_pool: list[RealizationSpec],
) -> dict[str, dict[str, Any]]:
    """Return the Phase 6 :class:`EnvFactory` keyword arguments for the two worker pools.

    Training and tune-pool evaluation workers get the same components, so the learning
    curves and the curriculum see the method being trained (a sinusoid run never meets a
    JONSWAP deck, P6-D1). Each pool's sinusoid table holds only its own realizations.

    Args:
        cfg: The training config.
        train_pool: The run's train realizations.
        tune_pool: The tune realizations.

    Returns:
        ``{"train": kwargs, "tune": kwargs}``.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, pool in (("train", train_pool), ("tune", tune_pool)):
        kwargs: dict[str, Any] = {
            "residual": cfg.residual,
            "forecast_obs": cfg.forecast_obs,
            "motion": cfg.motion,
        }
        if cfg.motion == "sinusoid":
            table = committed_rms_full(key_for_spec(s) for s in pool)
            kwargs["sinusoid_rms"] = tuple(table.items())
        out[name] = kwargs
    return out


def _sigterm(signum: int, frame: FrameType | None) -> None:
    """Turn SIGTERM into ``SystemExit`` so the run records ``failed`` before dying."""
    del frame
    raise SystemExit(f"terminated by signal {signum}")


def _repo_relative(path: Path) -> str:
    """Return ``path`` relative to the repository root when it is inside it, else as is."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def component_provenance(cfg: TrainConfig) -> dict[str, Any]:
    """Return the digests of what a Phase 6 run reads besides its config.

    Args:
        cfg: The training config.

    Returns:
        ``{"motion": ..., "residual_base_config", "residual_base_config_sha256",
        "forecaster_dir", "forecaster_meta_sha256", "deck_stats_seeds",
        "deck_stats_seeds_sha256"}`` -- only the keys the run's components use.
    """
    out: dict[str, Any] = {"motion": cfg.motion}
    if cfg.residual is not None:
        path = entry(cfg.residual.base).config_path
        out["residual_base_config"] = _repo_relative(path)
        out["residual_base_config_sha256"] = file_sha256(path)
    if cfg.forecast_obs is not None:
        model_dir = forecaster_dir(cfg.forecast_obs.forecaster)
        out["forecaster_dir"] = _repo_relative(model_dir)
        out["forecaster_meta_sha256"] = file_sha256(model_dir / "meta.json")
    if cfg.motion == "sinusoid":
        out["deck_stats_seeds"] = _repo_relative(DECK_STATS_SEEDS)
        out["deck_stats_seeds_sha256"] = file_sha256(DECK_STATS_SEEDS)
    return out


def _episodes_digest(episodes: list[TuningEpisode]) -> str:
    """Return a SHA-256 of the evaluation draw (realization keys and episode seeds)."""
    text = "\n".join(
        f"{e.ss},{e.index},{e.pad},{e.vessel},{e.heading_deg!r},{e.speed_kn!r},"
        f"{e.realization_seed},{e.episode_seed}"
        for e in episodes
    )
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- training


def _record_resume_provenance(run_dir: Path, plan: ResumePlan) -> None:
    """Append the resume record, with this process's environment, to ``provenance.json``.

    Every original field is kept; only the ``resumes`` list grows.
    """
    path = run_dir / "provenance.json"
    data = json.loads(path.read_text())
    data["resumes"] = [
        *(data.get("resumes") or []),
        {**plan.record, "environment": environment_provenance(REPO_ROOT)},
    ]
    atomic_write_json(path, data)


@contextmanager
def _run_lock(run_dir: Path) -> Iterator[None]:
    """Hold ``<run_dir>/.run.lock`` exclusively for the life of one training process.

    Two processes can never train into one run directory at once (e.g. a hand resume and a
    scheduler resume of the same failed run).

    Raises:
        RunDirExistsError: If another process holds it.
    """
    with (run_dir / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunDirExistsError(f"{run_dir} is being trained by another process") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_model(
    cfg: TrainConfig, plan: ResumePlan, venv: VecNormalize, run_dir: Path
) -> BaseAlgorithm:
    """Load a resumable checkpoint's model (and SAC replay buffer) onto the new envs.

    Args:
        cfg: The training config.
        plan: The validated resume.
        venv: The restored training ``VecNormalize``.
        run_dir: The run directory (TensorBoard under ``tb/``).

    Returns:
        The model, with ``_last_obs`` cleared so that ``learn`` resets the new envs.
    """
    algo_cls: Any = PPO if cfg.algo == "ppo" else SAC
    device = "cpu" if cfg.algo == "ppo" else cfg.device
    model: BaseAlgorithm = algo_cls.load(
        str(plan.checkpoint / "model.zip"), env=venv, device=device, force_reset=True
    )
    model.tensorboard_log = str(run_dir / "tb")
    if cfg.algo == "sac":
        model.load_replay_buffer(str(plan.checkpoint / REPLAY_BUFFER_FILE))  # type: ignore[attr-defined]
    model._last_obs = None  # SB3 saves it in model.zip; the envs are new
    return model


def train(
    cfg: TrainConfig, seed: int, run_dir: Path, *, resume_from: Path | None = None
) -> dict[str, Any]:
    """Train one run into an existing run directory, fresh or resumed from a checkpoint.

    Args:
        cfg: The training config.
        seed: The training seed.
        run_dir: From :func:`prepare_run_dir` (fresh: must hold no ``status.json``), or the
            failed run to continue (resume).
        resume_from: A resumable checkpoint of ``run_dir`` (its latest), or ``None`` for a
            fresh run. See :mod:`rld.rl.resume` for what is restored, what is recorded and
            what is not bit-exact.

    Returns:
        The final ``status.json`` payload (``state: done``).

    Raises:
        FileExistsError: If a fresh ``run_dir`` already holds a status file.
        ResumeError: If the resume is not allowed (the message says why).
        BaseException: Whatever training raised, after ``status.json`` says ``failed``.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory {run_dir} does not exist; use prepare_run_dir")
    if resume_from is None and (run_dir / "status.json").exists():
        raise FileExistsError(f"{run_dir} already holds a run; never reused")
    with _run_lock(run_dir):
        return _train_locked(cfg, seed, run_dir, resume_from)


def _train_locked(
    cfg: TrainConfig, seed: int, run_dir: Path, resume_from: Path | None
) -> dict[str, Any]:
    """Body of :func:`train`, under the run lock."""
    resolved = config_to_dict(cfg)
    config_text = yaml.safe_dump(resolved, sort_keys=False)
    config_sha = hashlib.sha256(config_text.encode()).hexdigest()
    plan: ResumePlan | None = None
    if resume_from is not None:
        plan = plan_resume(
            run_dir,
            Path(resume_from),
            seed=seed,
            algo=cfg.algo,
            config_text=config_text,
            config_sha256=config_sha,
            status=read_status(run_dir),
        )
    elif (run_dir / "status.json").exists():
        raise FileExistsError(f"{run_dir} already holds a run; never reused")
    previous = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, _sigterm)
    except ValueError:  # not the main thread (tests); SIGTERM handling is optional there
        previous = None

    torch.set_num_threads(cfg.torch_threads)
    git = git_state(REPO_ROOT)
    run_id = f"{cfg.run_group.replace('/', '-')}-s{seed}-{run_dir.name}"
    static: dict[str, Any] = {
        "method": cfg.method,
        "seed": int(seed),
        "run_id": run_id,
        "run_group": cfg.run_group,
        "run_dir": str(run_dir),
        "algo": cfg.algo,
        "budget": cfg.total_steps,
        "eval_interval_steps": cfg.eval.interval_steps,
        "status_interval_s": cfg.status_interval_s,
        "n_envs": cfg.n_envs,
        "workers": cfg.workers,
        "device": "cpu" if cfg.algo == "ppo" else cfg.device,
        "git_sha": git["git_sha"],
        "git_dirty": git["git_dirty"],
        "config": None if cfg.source is None else str(cfg.source),
        "config_sha256": config_sha,
        "log": str(run_dir / "train.log"),
        "resumed_from": None,
    }
    if plan is None:
        (run_dir / "config.yaml").write_text(config_text)
        if cfg.source is not None and cfg.source.exists():
            (run_dir / "config_source.yaml").write_bytes(cfg.source.read_bytes())
        status = StatusWriter(run_dir / "status.json", static)
        started_steps = 0
    else:
        prev = plan.previous
        # The run's identity is its first segment's; the commit it was started from stays
        # in git_sha, each resume's own commit is in its record.
        for key in ("run_id", "run_dir", "git_sha", "git_dirty", "config"):
            static[key] = prev.get(key, static[key])
        plan.record.update(
            {
                "segment": plan.segment,
                "resumed_at": utc_now(),
                "resumed_from": str(plan.checkpoint),
                "resume_steps": plan.steps,
                "previous_state": prev.get("state"),
                "previous_error": prev.get("error"),
                "previous_steps": prev.get("steps"),
                "previous_updated": prev.get("updated"),
                "previous_pid": prev.get("pid"),
                "previous_reconciled_at": prev.get("reconciled_at"),
                "abandoned_steps": max(int(prev.get("steps") or 0) - plan.steps, 0),
                "git_sha": git["git_sha"],
                "git_dirty": git["git_dirty"],
                "pid": os.getpid(),
            }
        )
        static.update(
            {
                "resumed_from": str(plan.checkpoint),
                "resumed_at": plan.record["resumed_at"],
                "resume_steps": plan.steps,
                "resumes": [*(prev.get("resumes") or []), plan.record],
            }
        )
        status = StatusWriter(
            run_dir / "status.json",
            static,
            started=prev.get("started"),
            wall_offset_s=float(prev.get("wall_s") or 0.0),
            cpu_offset_h=float(prev.get("cpu_hours") or 0.0),
            steps_offset=plan.steps,
        )
        started_steps = plan.steps
        print(
            f"resuming {run_dir} from {plan.checkpoint.name} (segment {plan.segment}, "
            f"{plan.record['abandoned_steps']} steps after the checkpoint abandoned)",
            flush=True,
        )
        if git["git_sha"] != prev.get("git_sha"):
            print(
                f"warning: resuming at commit {git['git_sha']} but the run started at "
                f"{prev.get('git_sha')}; recorded in resumes[-1]",
                flush=True,
            )

    cfgs = build_env_configs(cfg)
    tracker = RunTracker(status, Curriculum(cfg.curriculum))
    if plan is not None:
        tracker.curriculum.restore(plan.state["curriculum"])
        saved = plan.state["tracker"]
        tracker.last_eval = saved["last_eval"]
        tracker.eval_env_steps = int(saved["eval_env_steps"])
        tracker.eval_wall_s = float(saved["eval_wall_s"])
    tracker.write("running", started_steps)

    venv: VecNormalize | None = None
    evaluator: TunePoolEvaluator | None = None
    eval_cb: PeriodicEvalCallback | None = None
    model: BaseAlgorithm | None = None
    try:
        train_pool, tune_pool = training_pools(cfg, cfgs)
        episodes = eval_episodes(cfg, cfgs)
        tune_keys = {key_for_spec(s) for s in tune_pool}
        if any(key_for_spec(e.realization) not in tune_keys for e in episodes):
            raise RuntimeError("an evaluation episode is not a tune-pool realization")
        episodes_sha = _episodes_digest(episodes)
        if plan is None:
            atomic_write_json(
                run_dir / "provenance.json",
                {
                    **environment_provenance(REPO_ROOT),
                    **git,
                    "seed": int(seed),
                    "train_pool_n": len(train_pool),
                    "tune_pool_n": len(tune_pool),
                    "eval_episodes_n": len(episodes),
                    "eval_episodes_sha256": episodes_sha,
                    "eval_tuning_config": str(cfg.eval.tuning_config),
                    "reward": vars(cfgs.reward),
                    "components": component_provenance(cfg),
                },
            )
        else:
            original = json.loads((run_dir / "provenance.json").read_text())
            if original.get("eval_episodes_sha256") != episodes_sha:
                raise ResumeError(
                    "the evaluation draw differs from the one this run started with "
                    f"({original.get('eval_episodes_sha256')} != {episodes_sha})"
                )
            plan.record["eval_logs"] = truncate_eval_logs(
                run_dir, int(plan.state["eval_cb"]["n_evals"]), plan.segment
            )
        stage = tracker.curriculum.sampling_stage
        if cfg.vec_env == "subproc":
            forkserver_preload()
        prefetch = cfg.prefetch_reset
        suffix = "" if plan is None else f".resume{plan.segment}"
        components = _factory_components(cfg, train_pool, tune_pool)
        factories = [
            EnvFactory(
                cfgs,
                tuple(train_pool),
                cfg.pad,
                seed,
                rank,
                stage,
                run_dir / "monitor",
                prefetch,
                suffix,
                **components["train"],
            )
            for rank in range(cfg.n_envs)
        ]
        eval_factories = [
            EnvFactory(
                cfgs,
                tuple(tune_pool),
                cfg.pad,
                seed,
                rank,
                None,
                None,
                prefetch,
                **components["tune"],
            )
            for rank in range(cfg.eval.n_envs)
        ]
        if plan is None:
            venv = VecNormalize(
                _vec(factories, cfg.vec_env),
                training=True,
                norm_obs=cfg.normalize.norm_obs,
                norm_reward=cfg.normalize.norm_reward,
                clip_obs=cfg.normalize.clip_obs,
                clip_reward=cfg.normalize.clip_reward,
                gamma=cfg.gamma,
            )
        else:
            venv = VecNormalize.load(
                str(plan.checkpoint / "vecnormalize.pkl"), _vec(factories, cfg.vec_env)
            )
            venv.training = True
            venv.norm_reward = cfg.normalize.norm_reward
        evaluator = TunePoolEvaluator(
            eval_factories, episodes, cfg.vec_env, deterministic=cfg.eval.deterministic
        )
        if plan is None:
            model = _build_model(cfg, venv, seed, run_dir)
        else:
            model = _load_model(cfg, plan, venv, run_dir)
            samplers = plan.state["samplers"]
            if len(samplers) != cfg.n_envs:
                raise ResumeError(f"checkpoint has {len(samplers)} samplers, run {cfg.n_envs}")
            for rank, sampler in enumerate(samplers):
                venv.env_method("restore_sampler_state", sampler, indices=[rank])
            restore_rng(plan.state["rng"], model)
            restored: dict[str, Any] = {
                "params_sha256": params_digest(model),
                "vecnormalize_sha256": normalizer_digest(venv),
                "num_timesteps": int(model.num_timesteps),
            }
            buffer = getattr(model, "replay_buffer", None)
            if buffer is not None:
                restored["replay_buffer_size"] = int(buffer.size())
            plan.record["restored"] = restored
            _record_resume_provenance(run_dir, plan)
        train_venv = venv
        train_model = model

        def broadcast(new_stage: str | None) -> None:
            train_venv.env_method("set_stage", new_stage)

        def get_norm() -> VecNormalize:
            return train_venv

        eval_cb = PeriodicEvalCallback(
            evaluator,
            tracker,
            run_dir,
            cfg.method,
            cfg.eval.interval_steps,
            broadcast,
            get_norm,
            tb_name="eval" if plan is None else f"eval_resume{plan.segment}",
        )
        the_eval_cb = eval_cb

        def resume_state(next_ckpt: int) -> dict[str, Any]:
            return collect_resume_state(
                model=train_model,
                venv=train_venv,
                tracker=tracker,
                eval_cb=the_eval_cb,
                next_ckpt=next_ckpt,
                seed=seed,
                algo=cfg.algo,
                config_sha256=config_sha,
            )

        ckpt_cb = CheckpointCallback(
            run_dir, cfg.checkpoint_interval_steps, get_norm, tracker, resume_state
        )
        if plan is not None:
            eval_cb.restore_counters(plan.state["eval_cb"])
            ckpt_cb.next_ckpt = int(plan.state["next_ckpt"])
            tracker.write("running", started_steps)
        callbacks = CallbackList([eval_cb, ckpt_cb, StatusCallback(tracker, cfg.status_interval_s)])
        if plan is None:
            model.learn(
                total_timesteps=cfg.total_steps,
                callback=callbacks,
                log_interval=cfg.log_interval,
                tb_log_name=cfg.algo,
            )
        else:
            model.learn(
                total_timesteps=max(cfg.total_steps - int(model.num_timesteps), 0),
                callback=callbacks,
                log_interval=cfg.log_interval,
                tb_log_name=f"{cfg.algo}_resume{plan.segment}",
                reset_num_timesteps=False,
            )
        steps = int(model.num_timesteps)
        if eval_cb.last_eval_steps != steps:
            eval_cb.run_eval(steps)
        save_checkpoint(
            model,
            venv,
            run_dir / "final",
            {"steps": steps, "curriculum": tracker.curriculum.state()},
        )
        payload = status.write(
            "done",
            steps=steps,
            curriculum=tracker.curriculum,
            last_eval=tracker.last_eval,
            eval_env_steps=tracker.eval_env_steps,
            extra={
                "final_checkpoint": str(run_dir / "final"),
                "overshoot_steps": steps - cfg.total_steps,
                **tracker.timing(steps),
            },
        )
        print(f"run done: {run_dir} steps={steps}", flush=True)
        return payload
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        steps = started_steps if model is None else int(model.num_timesteps)
        status.write(
            "failed",
            steps=steps,
            curriculum=tracker.curriculum,
            last_eval=tracker.last_eval,
            eval_env_steps=tracker.eval_env_steps,
            error=f"{type(exc).__name__}: {exc}",
            extra={"traceback_tail": traceback.format_exc().splitlines()[-12:]},
        )
        raise
    finally:
        if eval_cb is not None:
            eval_cb.close()
        if evaluator is not None:
            evaluator.close()
        if venv is not None:
            venv.close()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


# --------------------------------------------------------------------------- loading


def checkpoint_dir(run_dir: Path, ckpt: str | int = "final") -> Path:
    """Resolve a checkpoint name.

    Args:
        run_dir: The run directory.
        ckpt: ``"final"``, a step count (int or digit string), or ``"step_<n>"``.

    Returns:
        The checkpoint directory.

    Raises:
        FileNotFoundError: If it does not exist.
    """
    name = str(ckpt)
    if name == "final":
        path = run_dir / "final"
    elif name.isdigit():
        path = run_dir / "checkpoints" / f"step_{int(name):010d}"
    else:
        path = run_dir / "checkpoints" / name
    if not (path / "model.zip").exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    return path


def load_vecnormalize(path: Path, venv: VecEnv | None = None) -> VecNormalize:
    """Load saved ``VecNormalize`` statistics for evaluation.

    Args:
        path: ``vecnormalize.pkl``.
        venv: A vector env to attach (``VecNormalize.load``), or ``None`` for a detached
            normaliser on which only ``normalize_obs`` may be called.

    Returns:
        The normaliser with ``training=False`` and ``norm_reward=False``: its statistics
        never update and rewards pass through raw.
    """
    if venv is not None:
        loaded = VecNormalize.load(str(path), venv)
        loaded.training = False
        loaded.norm_reward = False
        return loaded
    with path.open("rb") as handle:
        import pickle

        detached: VecNormalize = pickle.load(handle)
    return frozen_normalizer(detached)


def _upgrade_saved_run_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill keys added after a run was trained, with the value that run actually used.

    Only :func:`load_policy` calls this; a *training* config must still state every key.

    * ``ppo.log_std_init`` (added by P5-D4, commit ``cbdb7cf``). Earlier PPO runs, i.e. the
      first smoke run ``artifacts/runs/ppo_smoke/0``, left SB3 at its default of 0.0. The
      value only seeds a new model; a loaded model's ``log_std`` comes from ``model.zip``.

    Args:
        raw: A run directory's ``config.yaml`` mapping.

    Returns:
        The mapping, with missing post-hoc keys filled (a copy when anything changes).
    """
    ppo = raw.get("ppo")
    if raw.get("algo") == "ppo" and isinstance(ppo, dict) and "log_std_init" not in ppo:
        return {**raw, "ppo": {**ppo, "log_std_init": 0.0}}
    return raw


def _run_config(run_dir: Path) -> TrainConfig:
    """Return a run directory's own config (upgraded for keys added after it was trained)."""
    raw = _upgrade_saved_run_config(yaml.safe_load((run_dir / "config.yaml").read_text()))
    return train_config_from_dict(raw, str(run_dir / "config.yaml"))


def policy_class(cfg: TrainConfig) -> type[LearnedPolicy]:
    """Return the evaluation policy class of a run's config.

    Args:
        cfg: The run's config.

    Returns:
        :class:`LearnedPolicy` (``ppo``, ``sac``, ``ppo_sinusoid``),
        :class:`~rld.rl.residual.ResidualPolicy` (``residual_ppo``),
        :class:`~rld.rl.forecast_obs.ForecastPolicy` (``ppo_forecast``) or
        :class:`~rld.rl.forecast_obs.ResidualForecastPolicy` (``residual_ppo_forecast``).
    """
    if cfg.residual is not None and cfg.forecast_obs is not None:
        return ResidualForecastPolicy
    if cfg.residual is not None:
        return ResidualPolicy
    if cfg.forecast_obs is not None:
        return ForecastPolicy
    return LearnedPolicy


def run_needs_motion_feed(run_dir: Path) -> bool:
    """Return whether a run's policy consumes the past-only ship-motion feed.

    The value ``rld.eval.runner.callable_spec(..., needs_motion_feed=...)`` must be given
    for the run; the runner refuses a policy whose own flag disagrees.

    Args:
        run_dir: The run directory.

    Returns:
        True for a forecast-observation run.
    """
    return policy_class(_run_config(Path(run_dir))).needs_motion_feed


def policy_input_size(cfg: TrainConfig, cfgs: EvalConfigs) -> int:
    """Return the network input size of a run: the env observation plus any forecast block.

    Args:
        cfg: The run's config.
        cfgs: The evaluation configs (observation layout).

    Returns:
        25 for the committed layout, 31 with the 1/2/3 s forecast block.
    """
    base = int(observation_space(cfgs.observation).shape[0])
    return base + (0 if cfg.forecast_obs is None else cfg.forecast_obs.block_size)


def _check_components(run_dir: Path, cfg: TrainConfig) -> None:
    """Refuse a checkpoint whose residual base config or forecaster changed since training.

    Compared against ``provenance.json["components"]`` when the run recorded it (runs from
    before Phase 6 did not, and have no such component).

    Raises:
        ValueError: On a digest mismatch.
    """
    path = run_dir / "provenance.json"
    if not path.exists():
        return
    recorded = json.loads(path.read_text()).get("components")
    if not recorded:
        return
    live = component_provenance(cfg)
    for key in ("residual_base_config_sha256", "forecaster_meta_sha256"):
        if key in recorded and recorded[key] != live.get(key):
            raise ValueError(
                f"{run_dir}: {key} was {recorded[key]} at training, is {live.get(key)} now"
            )


def load_policy(
    run_dir: Path, ckpt: str | int = "final", *, cfgs: EvalConfigs | None = None
) -> LearnedPolicy:
    """Load a trained policy with its frozen normalisation statistics.

    Args:
        run_dir: The run directory.
        ckpt: See :func:`checkpoint_dir`.
        cfgs: The evaluation configs a residual base is built from; the committed configs
            when ``None``.

    Returns:
        The :func:`policy_class` instance (CPU).

    Raises:
        ValueError: If the residual base config or the forecaster changed since training.
    """
    run_dir = Path(run_dir)
    cfg = _run_config(run_dir)
    path = checkpoint_dir(run_dir, ckpt)
    algo_cls: Any = PPO if cfg.algo == "ppo" else SAC
    model: BaseAlgorithm = algo_cls.load(str(path / "model.zip"), device="cpu")
    stats = path / "vecnormalize.pkl"
    normalizer = load_vecnormalize(stats) if stats.exists() else None
    meta = json.loads((path / "checkpoint.json").read_text())
    steps = int(meta.get("steps", -1))
    _check_components(run_dir, cfg)
    cls = policy_class(cfg)
    if cls is LearnedPolicy:
        return LearnedPolicy(cfg.method, model, normalizer, run_dir, path, steps)
    env_cfgs = load_eval_configs() if cfgs is None else cfgs
    args = (cfg.method, model, normalizer, run_dir, path, steps)
    if cls is ResidualPolicy:
        assert cfg.residual is not None
        base = build_base_controller(cfg.residual.base, env_cfgs)
        return ResidualPolicy(*args, base=base, alpha=cfg.residual.alpha)
    assert cfg.forecast_obs is not None
    model_dir = forecaster_dir(cfg.forecast_obs.forecaster)
    leads = cfg.forecast_obs.leads_full_s
    if cls is ForecastPolicy:
        return ForecastPolicy(*args, forecaster_dir=model_dir, leads_full_s=leads)
    assert cfg.residual is not None
    return ResidualForecastPolicy(
        *args,
        base=build_base_controller(cfg.residual.base, env_cfgs),
        alpha=cfg.residual.alpha,
        forecaster_dir=model_dir,
        leads_full_s=leads,
    )


def build_policy(cfgs: EvalConfigs, *, run_dir: Path, ckpt: str | int = "final") -> LearnedPolicy:
    """Picklable evaluation factory for ``rld.eval.runner.callable_spec``.

    Args:
        cfgs: The evaluation configs; their observation layout (plus the run's forecast
            block, if any) must match the policy's input, and a residual base is built
            from them.
        run_dir: The run directory.
        ckpt: See :func:`checkpoint_dir`.

    Returns:
        The policy; ``needs_motion_feed`` is True exactly for a forecast run
        (:func:`run_needs_motion_feed`).

    Raises:
        ValueError: If the input size differs from the policy's (31 for a forecast run).
    """
    policy = load_policy(run_dir, ckpt, cfgs=cfgs)
    cfg = _run_config(Path(run_dir))
    expected = (policy_input_size(cfg, cfgs),)
    got = policy.model.observation_space.shape
    if expected != got:
        raise ValueError(f"observation shape {expected} != policy input {got}")
    return policy
