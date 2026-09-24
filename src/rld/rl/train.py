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
   checkpoint with ``model.zip`` **and** ``vecnormalize.pkl``.
4. At the end: a final evaluation (unless one just ran), the ``final/`` checkpoint and
   ``status.json`` ``state: done``. On any exception -- including SIGTERM, turned into
   ``SystemExit`` -- the traceback goes to stderr (the run's ``train.log``) and
   ``status.json`` says ``state: failed``. A failed run directory is kept, never reused.

Run directories
---------------
``artifacts/runs/<run_group>/<seed>/`` (:func:`prepare_run_dir`). A finished (``done``) or live
run of the same group and seed makes the launch **refuse**; a failed or abandoned one is
kept and the new run takes the next free suffix, ``<seed>_r2``, ``<seed>_r3``, ... . Nothing
is ever deleted or overwritten.

Evaluation-time loading
-----------------------
:func:`load_policy` returns a :class:`LearnedPolicy` satisfying
:class:`rld.control.base.Controller`: never privileged, never takes a ship-motion feed (both
raise), and normalises observations with the checkpoint's own statistics loaded with
``training=False, norm_reward=False``. :func:`build_policy` is the picklable factory
``rld.eval.runner.callable_spec`` takes (bind ``run_dir``/``ckpt`` with ``functools.partial``).

Nothing in this module reads ``results/episodes/``.

Units: steps are env control steps (1/30 s model scale each); ``*_s`` wall-clock fields are
host seconds; actions are the shared normalised velocity setpoint (``v_max`` = 1.5 m/s model
scale, norm cap).
"""

import hashlib
import json
import signal
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch
import yaml
from dmf.sim.generate import RealizationSpec
from dmf.typedefs import FloatArray
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from rld.config import REPO_ROOT
from rld.control.base import PrivilegedContext, reject_motion_feed
from rld.control.tuning import TuningEpisode, draw_tuning_episodes, load_tuning
from rld.deck.splits import dev_pool, key_for_spec
from rld.envs.observation import observation_space
from rld.eval.envs import EvalConfigs, load_eval_configs
from rld.provenance import environment_provenance
from rld.rl.callbacks import (
    CheckpointCallback,
    PeriodicEvalCallback,
    RunTracker,
    StatusCallback,
    StatusWriter,
    TunePoolEvaluator,
    atomic_write_json,
    frozen_normalizer,
    save_checkpoint,
)
from rld.rl.config import (
    FORBIDDEN_SEA_STATES,
    RUNS_ROOT,
    TrainConfig,
    config_to_dict,
    train_config_from_dict,
)
from rld.rl.curriculum import Curriculum
from rld.rl.procs import pid_alive
from rld.rl.wrappers import EnvFactory

__all__ = [
    "LearnedPolicy",
    "LinearDecay",
    "RunDirExistsError",
    "build_env_configs",
    "build_policy",
    "checkpoint_dir",
    "eval_episodes",
    "git_state",
    "load_policy",
    "load_vecnormalize",
    "prepare_run_dir",
    "read_status",
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
    if status.get("state") == "running" and pid_alive(status.get("pid")):
        raise RunDirExistsError(f"{run_dir} holds a live run (pid {status.get('pid')})")


# --------------------------------------------------------------------------- pools and envs


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
        The model.
    """
    activation = torch.nn.Tanh if cfg.policy.activation == "tanh" else torch.nn.ReLU
    arch = cfg.policy.net_arch
    tb = str(run_dir / "tb")
    if cfg.ppo is not None:
        p = cfg.ppo
        lr: Any = LinearDecay(p.learning_rate) if p.lr_schedule == "linear" else p.learning_rate
        return PPO(
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


def _sigterm(signum: int, frame: FrameType | None) -> None:
    """Turn SIGTERM into ``SystemExit`` so the run records ``failed`` before dying."""
    del frame
    raise SystemExit(f"terminated by signal {signum}")


def _episodes_digest(episodes: list[TuningEpisode]) -> str:
    """Return a SHA-256 of the evaluation draw (realization keys and episode seeds)."""
    text = "\n".join(
        f"{e.ss},{e.index},{e.pad},{e.vessel},{e.heading_deg!r},{e.speed_kn!r},"
        f"{e.realization_seed},{e.episode_seed}"
        for e in episodes
    )
    return hashlib.sha256(text.encode()).hexdigest()


# --------------------------------------------------------------------------- training


def train(cfg: TrainConfig, seed: int, run_dir: Path) -> dict[str, Any]:
    """Train one run into an existing, empty run directory.

    Args:
        cfg: The training config.
        seed: The training seed.
        run_dir: From :func:`prepare_run_dir`; must exist and hold no ``status.json``.

    Returns:
        The final ``status.json`` payload (``state: done``).

    Raises:
        FileExistsError: If ``run_dir`` already holds a status file.
        BaseException: Whatever training raised, after ``status.json`` says ``failed``.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory {run_dir} does not exist; use prepare_run_dir")
    if (run_dir / "status.json").exists():
        raise FileExistsError(f"{run_dir} already holds a run; never reused")
    previous = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGTERM, _sigterm)
    except ValueError:  # not the main thread (tests); SIGTERM handling is optional there
        previous = None

    torch.set_num_threads(cfg.torch_threads)
    resolved = config_to_dict(cfg)
    config_text = yaml.safe_dump(resolved, sort_keys=False)
    (run_dir / "config.yaml").write_text(config_text)
    if cfg.source is not None and cfg.source.exists():
        (run_dir / "config_source.yaml").write_bytes(cfg.source.read_bytes())
    git = git_state(REPO_ROOT)
    started_steps = 0
    run_id = f"{cfg.run_group.replace('/', '-')}-s{seed}-{run_dir.name}"

    cfgs = build_env_configs(cfg)
    status = StatusWriter(
        run_dir / "status.json",
        {
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
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "log": str(run_dir / "train.log"),
            "resumed_from": None,
        },
    )
    tracker = RunTracker(status, Curriculum(cfg.curriculum))
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
        atomic_write_json(
            run_dir / "provenance.json",
            {
                **environment_provenance(REPO_ROOT),
                **git,
                "seed": int(seed),
                "train_pool_n": len(train_pool),
                "tune_pool_n": len(tune_pool),
                "eval_episodes_n": len(episodes),
                "eval_episodes_sha256": _episodes_digest(episodes),
                "eval_tuning_config": str(cfg.eval.tuning_config),
                "reward": vars(cfgs.reward),
            },
        )
        stage = tracker.curriculum.sampling_stage
        factories = [
            EnvFactory(cfgs, tuple(train_pool), cfg.pad, seed, rank, stage, run_dir / "monitor")
            for rank in range(cfg.n_envs)
        ]
        eval_factories = [
            EnvFactory(cfgs, tuple(tune_pool), cfg.pad, seed, rank, None, None)
            for rank in range(cfg.eval.n_envs)
        ]
        venv = VecNormalize(
            _vec(factories, cfg.vec_env),
            training=True,
            norm_obs=cfg.normalize.norm_obs,
            norm_reward=cfg.normalize.norm_reward,
            clip_obs=cfg.normalize.clip_obs,
            clip_reward=cfg.normalize.clip_reward,
            gamma=cfg.gamma,
        )
        evaluator = TunePoolEvaluator(
            eval_factories, episodes, cfg.vec_env, deterministic=cfg.eval.deterministic
        )
        model = _build_model(cfg, venv, seed, run_dir)
        train_venv = venv

        def broadcast(new_stage: str | None) -> None:
            train_venv.env_method("set_stage", new_stage)

        def get_norm() -> VecNormalize:
            return train_venv

        eval_cb = PeriodicEvalCallback(
            evaluator, tracker, run_dir, cfg.method, cfg.eval.interval_steps, broadcast, get_norm
        )
        callbacks = CallbackList(
            [
                eval_cb,
                CheckpointCallback(run_dir, cfg.checkpoint_interval_steps, get_norm, tracker),
                StatusCallback(tracker, cfg.status_interval_s),
            ]
        )
        model.learn(
            total_timesteps=cfg.total_steps,
            callback=callbacks,
            log_interval=cfg.log_interval,
            tb_log_name=cfg.algo,
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
        steps = 0 if model is None else int(model.num_timesteps)
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


class LearnedPolicy:
    """A trained policy behind the :class:`rld.control.base.Controller` contract.

    Stateless across steps (an MLP of the current observation), so ``reset`` clears
    nothing. Never privileged and never given a ship-motion feed.

    Attributes:
        name: Method label (the run's ``method``).
        privileged: Always False.
        needs_motion_feed: Always False.
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
        motion_feed: Any = None,
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

    def act(self, obs: FloatArray) -> FloatArray:
        """Return the deterministic action for one observation.

        Args:
            obs: The raw observation vector.

        Returns:
            A ``(3,)`` float32 action in ``[-1, 1]^3`` (the shared normalised velocity
            setpoint).
        """
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self.normalizer is not None:
            x = np.asarray(self.normalizer.normalize_obs(x), dtype=np.float32)
        action, _ = self.model.predict(x, deterministic=True)
        out: FloatArray = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return out


def load_policy(run_dir: Path, ckpt: str | int = "final") -> LearnedPolicy:
    """Load a trained policy with its frozen normalisation statistics.

    Args:
        run_dir: The run directory.
        ckpt: See :func:`checkpoint_dir`.

    Returns:
        The :class:`LearnedPolicy` (CPU).
    """
    run_dir = Path(run_dir)
    raw = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg = train_config_from_dict(raw, str(run_dir / "config.yaml"))
    path = checkpoint_dir(run_dir, ckpt)
    algo_cls: Any = PPO if cfg.algo == "ppo" else SAC
    model: BaseAlgorithm = algo_cls.load(str(path / "model.zip"), device="cpu")
    stats = path / "vecnormalize.pkl"
    normalizer = load_vecnormalize(stats) if stats.exists() else None
    meta = json.loads((path / "checkpoint.json").read_text())
    return LearnedPolicy(cfg.method, model, normalizer, run_dir, path, int(meta.get("steps", -1)))


def build_policy(cfgs: EvalConfigs, *, run_dir: Path, ckpt: str | int = "final") -> LearnedPolicy:
    """Picklable evaluation factory for ``rld.eval.runner.callable_spec``.

    Args:
        cfgs: The evaluation configs; their observation layout must match the policy's.
        run_dir: The run directory.
        ckpt: See :func:`checkpoint_dir`.

    Returns:
        The policy.

    Raises:
        ValueError: If the observation size differs from the policy's input size.
    """
    policy = load_policy(run_dir, ckpt)
    expected = observation_space(cfgs.observation).shape
    got = policy.model.observation_space.shape
    if expected != got:
        raise ValueError(f"observation shape {expected} != policy input {got}")
    return policy
