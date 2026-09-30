"""Resumable checkpoints: what a checkpoint holds, how a run continues from one, and what differs.

What every periodic checkpoint holds
------------------------------------
``checkpoints/step_<n>/`` (written atomically by :func:`rld.rl.callbacks.save_checkpoint`):

* ``model.zip`` -- policy, critics, optimiser states, ``num_timesteps``, ``_n_updates``, the
  SAC entropy coefficient and its optimiser, the learning-rate schedule;
* ``vecnormalize.pkl`` -- observation (and, PPO, return) running statistics;
* ``replay_buffer.pkl`` -- SAC only, SB3 ``save_replay_buffer`` (the whole buffer object,
  ``pos`` and ``full`` included);
* ``resume_state.pkl`` -- :func:`collect_resume_state`: the curriculum's stage, promotion
  history and **window of success flags**; the evaluation callback's counters (evaluation
  index, next due step, last evaluated step); the next checkpoint step; the tracker's
  latest evaluation and evaluation totals; every training worker's **sampler RNG state**
  (:meth:`rld.rl.wrappers.PoolSamplingEnv.sampler_state`); the learner's Python, NumPy,
  torch (and CUDA) RNG states and the action-space RNG; the seed and config hash it belongs
  to;
* ``checkpoint.json`` -- human-readable summary (``resumable``, ``replay_buffer``,
  ``replay_buffer_size``).

The ``final/`` checkpoint is model-only (a finished run is never resumed). Checkpoints
written before this module existed are model-only and are **not** resumable.

Resuming (``train(..., resume_from=<checkpoint dir>)``)
------------------------------------------------------
The run continues in the **same run directory** (:func:`plan_resume` checks all of this):

* the run's ``status.json`` must say ``failed`` (a host kill leaves ``running`` with a dead
  pid: run ``scripts/reconcile_runs.py`` first); a ``done`` run is never resumed;
* the checkpoint must be the run's **latest** (a later one would be overwritten by the
  resumed branch -- refused instead), resumable, and of the same seed and config (the
  run's ``config.yaml`` must equal the config given, byte for byte);
* rows of ``evals.csv`` / ``eval_episodes.csv`` from evaluations after the checkpoint (the
  abandoned branch) are removed from those files so the learning curve stays one
  trajectory; the files are first copied, byte for byte, to
  ``evals.before_resume<k>.csv`` / ``eval_episodes.before_resume<k>.csv`` (nothing is
  lost);
* training workers write new monitor files ``monitor/<rank>.resume<k>.monitor.csv``, SB3
  TensorBoard goes to ``tb/<algo>_resume<k>_0`` and evaluation TensorBoard to
  ``tb/eval_resume<k>``; no earlier file is overwritten;
* ``status.json`` keeps the run's original ``started``, sets ``resumed_from``,
  ``resumed_at`` and ``resume_steps``, and appends a record to ``resumes`` (previous state,
  error, steps and pid; steps abandoned; the resuming commit; digests of the restored
  weights and normalisation statistics; the restored replay-buffer size). ``wall_s`` and
  ``cpu_hours`` accumulate across segments -- the abandoned branch's compute is real
  compute and is counted; ``fps`` is the current segment's rate;
* ``provenance.json`` keeps every original field and gains the same record under
  ``resumes``, with the resuming process's environment.

What is NOT bit-exact (a resumed run is a valid continuation, not a replay)
--------------------------------------------------------------------------
1. **The in-flight episodes are dropped.** At the checkpoint every training worker is
   mid-episode; a resumed worker starts at the *next* episode of its own stream (its
   sampler RNG is restored to the state before any prefetched draw). The per-worker
   *sequence* of (realization, episode seed) is the uninterrupted one minus the episode in
   flight, but the ``SubprocVecEnv`` workers' episode boundaries are re-aligned (all
   reset together), so the interleaving of transitions into the rollout / replay buffer
   differs from then on.
2. **VecNormalize** updates its observation statistics with the ``n_envs`` reset
   observations of the resumed start (an extra update the uninterrupted run does not
   make), and PPO's running discounted returns restart at zero.
3. **PPO** loses the partial rollout collected before the checkpoint (a checkpoint fires
   mid-rollout); those steps count in ``num_timesteps`` but never enter an update, and the
   rollout boundaries shift by that offset. The linear learning-rate schedule is a function
   of ``num_timesteps`` and continues exactly.
4. **SAC**: the checkpoint fires inside ``collect_rollouts`` after the env step and before
   ``_store_transition``, so the saved buffer holds ``steps - n_envs`` transitions -- the
   checkpoint step's transitions are lost. Gradient steps (UTD = 1) continue from the saved
   ``_n_updates``.
5. **RNG streams.** The learner's torch/NumPy/Python RNGs are restored, but the draws they
   feed (actions, minibatch indices) now see different data (points 1-4), so they diverge
   anyway. Worker-side prefetch threads and the evaluation workers hold no state that
   matters: every evaluation episode is re-seeded explicitly.
6. **Timing and float order**: throughput, the host-time columns and any thread-count
   dependent float reductions differ, as between any two runs.

Evaluation of a given set of weights is unaffected: the tune-pool evaluator flies a fixed,
explicitly seeded episode list.

Units: steps are env control steps (1/30 s model scale each); ``*_s`` host seconds.
"""

import csv
import hashlib
import os
import pickle
import random
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from rld.rl.callbacks import (
    REPLAY_BUFFER_FILE,
    RESUME_STATE_FILE,
    PeriodicEvalCallback,
    RunTracker,
    utc_now,
)
from rld.rl.procs import status_owner_alive

__all__ = [
    "RESUME_FORMAT",
    "ResumeError",
    "ResumePlan",
    "capture_rng",
    "checkpoint_steps",
    "collect_resume_state",
    "is_resumable",
    "latest_checkpoint",
    "latest_resumable_checkpoint",
    "normalizer_digest",
    "params_digest",
    "plan_resume",
    "resolve_resume_target",
    "restore_rng",
    "truncate_eval_logs",
]

#: Version of ``resume_state.pkl``.
RESUME_FORMAT: int = 1

#: Evaluation logs truncated at the checkpoint on resume, keyed by their index column.
EVAL_LOGS: tuple[str, ...] = ("evals.csv", "eval_episodes.csv")


class ResumeError(RuntimeError):
    """A run cannot be resumed from the given checkpoint (the message says why)."""


# --------------------------------------------------------------------------- checkpoints


def checkpoint_steps(ckpt: Path) -> int:
    """Return the step count in a ``step_<n>`` checkpoint name.

    Raises:
        ValueError: If the name is not ``step_<digits>``.
    """
    name = ckpt.name
    if not name.startswith("step_") or not name[5:].isdigit():
        raise ValueError(f"{ckpt} is not a periodic checkpoint (step_<n>)")
    return int(name[5:])


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the run's periodic checkpoint with the most steps, or ``None``.

    Hidden directories (``.step_*.tmp``: a save interrupted by a kill) are ignored.
    """
    root = run_dir / "checkpoints"
    if not root.is_dir():
        return None
    found = [
        p
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("step_") and p.name[5:].isdigit()
    ]
    return max(found, key=checkpoint_steps) if found else None


def is_resumable(ckpt: Path) -> bool:
    """Return whether a checkpoint directory holds everything a resume needs."""
    if not (ckpt / "model.zip").exists() or not (ckpt / RESUME_STATE_FILE).exists():
        return False
    try:
        with (ckpt / RESUME_STATE_FILE).open("rb") as handle:
            state = pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        return False
    if state.get("algo") == "sac" and not (ckpt / REPLAY_BUFFER_FILE).exists():
        return False
    return int(state.get("format", -1)) == RESUME_FORMAT


def latest_resumable_checkpoint(run_dir: Path) -> Path | None:
    """Return the latest checkpoint if it is resumable, else ``None``.

    Only the latest checkpoint may be resumed from (:func:`plan_resume`), so an older
    resumable checkpoint behind a newer model-only one is not offered.
    """
    ckpt = latest_checkpoint(run_dir)
    return ckpt if ckpt is not None and is_resumable(ckpt) else None


def resolve_resume_target(path: Path) -> tuple[Path, Path]:
    """Return ``(run_dir, checkpoint)`` for a run directory or a checkpoint directory.

    Args:
        path: A run directory (its latest checkpoint is used) or a
            ``<run_dir>/checkpoints/step_<n>`` directory.

    Returns:
        The run directory and the checkpoint directory.

    Raises:
        ResumeError: If no checkpoint can be found.
    """
    path = Path(path).resolve()
    if (path / "model.zip").exists():
        if path.parent.name != "checkpoints":
            raise ResumeError(f"{path} is not a periodic checkpoint (checkpoints/step_<n>)")
        return path.parent.parent, path
    ckpt = latest_checkpoint(path)
    if ckpt is None:
        raise ResumeError(f"{path} has no periodic checkpoint to resume from")
    return path, ckpt


# --------------------------------------------------------------------------- digests


def params_digest(model: BaseAlgorithm) -> str:
    """Return a SHA-256 of the policy's parameters and buffers (names, dtypes, shapes, bytes).

    ``policy.state_dict()`` holds the actor and the critic(s), their target networks for
    SAC, and ``log_std`` for PPO; two models with equal digests have equal weights.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        digest.update(f"{name}|{array.dtype}|{array.shape}|".encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def normalizer_digest(normalizer: VecNormalize) -> str:
    """Return a SHA-256 of a ``VecNormalize``'s running statistics and clip settings."""
    digest = hashlib.sha256()
    for rms_name in ("obs_rms", "ret_rms"):
        rms = getattr(normalizer, rms_name)
        for attr in ("mean", "var"):
            array = np.asarray(getattr(rms, attr), dtype=np.float64)
            digest.update(f"{rms_name}.{attr}|{array.shape}|".encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(f"{rms_name}.count|{float(rms.count)!r}|".encode())
    digest.update(f"clip|{normalizer.clip_obs!r}|{normalizer.clip_reward!r}".encode())
    return digest.hexdigest()


# --------------------------------------------------------------------------- RNG


def capture_rng(model: BaseAlgorithm) -> dict[str, Any]:
    """Return the learner-process RNG states (Python, NumPy global, torch, CUDA, action space)."""
    out: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "action_space": dict(model.action_space.np_random.bit_generator.state),
    }
    # Only if CUDA is already in use: never create a CUDA context in a CPU learner.
    if torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        out["torch_cuda"] = torch.cuda.get_rng_state_all()
    return out


def restore_rng(state: Mapping[str, Any], model: BaseAlgorithm) -> None:
    """Restore :func:`capture_rng` (CUDA only if both sides have it)."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    model.action_space.np_random.bit_generator.state = dict(state["action_space"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# --------------------------------------------------------------------------- state


def collect_resume_state(
    *,
    model: BaseAlgorithm,
    venv: VecNormalize,
    tracker: RunTracker,
    eval_cb: PeriodicEvalCallback,
    next_ckpt: int,
    seed: int,
    algo: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Return the ``resume_state.pkl`` payload at the current step (module docstring)."""
    return {
        "format": RESUME_FORMAT,
        "seed": int(seed),
        "algo": algo,
        "config_sha256": config_sha256,
        "steps": int(model.num_timesteps),
        "n_envs": int(venv.num_envs),
        "curriculum": tracker.curriculum.snapshot(),
        "tracker": {
            "last_eval": tracker.last_eval,
            "eval_env_steps": int(tracker.eval_env_steps),
            "eval_wall_s": float(tracker.eval_wall_s),
        },
        "eval_cb": eval_cb.counters(),
        "next_ckpt": int(next_ckpt),
        "rng": capture_rng(model),
        "samplers": venv.env_method("sampler_state"),
        "saved": utc_now(),
    }


def truncate_eval_logs(run_dir: Path, n_evals: int, segment: int) -> list[dict[str, Any]]:
    """Drop evaluation rows of the abandoned branch (``eval_index >= n_evals``), keeping a copy.

    Each affected file is first copied byte for byte to ``<stem>.before_resume<k>.csv``,
    then rewritten atomically with the kept rows (same header, same cell text, same CSV
    dialect as :mod:`csv`'s default the writer used).

    Args:
        run_dir: The run directory.
        n_evals: Evaluations the checkpoint had completed (its ``eval_cb.n_evals``).
        segment: The resume number ``k``.

    Returns:
        One record per file: ``file``, ``kept``, ``abandoned``, ``archive`` (``None`` when
        nothing was abandoned and the file was left untouched).

    Raises:
        ResumeError: If an archive of that name already exists.
    """
    out: list[dict[str, Any]] = []
    for name in EVAL_LOGS:
        path = run_dir / name
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            continue
        header, body = rows[0], rows[1:]
        col = header.index("eval_index")
        kept = [r for r in body if int(r[col]) < n_evals]
        record: dict[str, Any] = {
            "file": name,
            "kept": len(kept),
            "abandoned": len(body) - len(kept),
            "archive": None,
        }
        if len(kept) != len(body):
            archive = run_dir / f"{path.stem}.before_resume{segment}.csv"
            if archive.exists():
                raise ResumeError(f"{archive} exists; refusing to overwrite it")
            shutil.copy2(path, archive)
            tmp = run_dir / f".{name}.{os.getpid()}.tmp"
            with tmp.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(kept)
            tmp.replace(path)
            record["archive"] = archive.name
        out.append(record)
    return out


@dataclass
class ResumePlan:
    """A validated resume, before anything is written.

    Attributes:
        run_dir: The run directory (continued in place).
        checkpoint: The checkpoint directory.
        state: The loaded ``resume_state.pkl``.
        previous: The run's ``status.json`` before the resume.
        segment: The resume number ``k`` (1 for the first resume).
        steps: Training steps at the checkpoint.
        record: The record appended to ``status.json`` / ``provenance.json`` ``resumes``
            (filled in further as the resume proceeds).
    """

    run_dir: Path
    checkpoint: Path
    state: dict[str, Any]
    previous: dict[str, Any]
    segment: int
    steps: int
    record: dict[str, Any] = field(default_factory=dict)


def plan_resume(
    run_dir: Path,
    checkpoint: Path,
    *,
    seed: int,
    algo: str,
    config_text: str,
    config_sha256: str,
    status: Mapping[str, Any] | None,
) -> ResumePlan:
    """Validate a resume and load its state (module docstring, "Resuming").

    Args:
        run_dir: The run directory.
        checkpoint: The checkpoint to resume from.
        seed: The training seed given.
        algo: ``"ppo"`` or ``"sac"``.
        config_text: The config as ``train`` would write it to ``config.yaml``.
        config_sha256: Its SHA-256.
        status: The run's current ``status.json``.

    Returns:
        The plan.

    Raises:
        ResumeError: On any violation, with the reason.
    """
    run_dir, checkpoint = Path(run_dir).resolve(), Path(checkpoint).resolve()
    if checkpoint.parent != run_dir / "checkpoints":
        raise ResumeError(f"{checkpoint} is not a checkpoint of {run_dir}")
    if status is None:
        raise ResumeError(f"{run_dir} has no readable status.json")
    state_name = status.get("state")
    if state_name == "done":
        raise ResumeError(f"{run_dir} finished (state done); a finished run is never resumed")
    if state_name == "running":
        if status_owner_alive(status):
            raise ResumeError(f"{run_dir} is live (pid {status.get('pid')})")
        raise ResumeError(
            f"{run_dir} says running but its process is gone; run "
            "`python scripts/reconcile_runs.py` first so the record says failed"
        )
    if state_name != "failed":
        raise ResumeError(f"{run_dir} is in state {state_name!r}; only failed runs are resumed")
    if (run_dir / "final" / "model.zip").exists():
        raise ResumeError(f"{run_dir}/final exists: training finished; not resumable")
    latest = latest_checkpoint(run_dir)
    if latest is None or latest != checkpoint:
        raise ResumeError(
            f"{checkpoint.name} is not the run's latest checkpoint ({latest}); resuming from "
            "an older one would collide with the later checkpoint directories"
        )
    if not is_resumable(checkpoint):
        raise ResumeError(
            f"{checkpoint} is not resumable (no {RESUME_STATE_FILE}"
            f"{' or ' + REPLAY_BUFFER_FILE if algo == 'sac' else ''}; checkpoints written "
            "before resumable checkpoints existed hold the model only)"
        )
    with (checkpoint / RESUME_STATE_FILE).open("rb") as handle:
        state: dict[str, Any] = pickle.load(handle)
    if int(state["seed"]) != int(seed):
        raise ResumeError(f"checkpoint seed {state['seed']} != given seed {seed}")
    if state["algo"] != algo:
        raise ResumeError(f"checkpoint algo {state['algo']} != {algo}")
    saved_config = (
        (run_dir / "config.yaml").read_text() if (run_dir / "config.yaml").exists() else ""
    )
    if saved_config != config_text or state["config_sha256"] != config_sha256:
        raise ResumeError(
            f"the given config differs from {run_dir}/config.yaml; a run is resumed with its "
            "own config only"
        )
    segment = len(status.get("resumes") or []) + 1
    return ResumePlan(
        run_dir=run_dir,
        checkpoint=checkpoint,
        state=state,
        previous=dict(status),
        segment=segment,
        steps=int(state["steps"]),
    )
