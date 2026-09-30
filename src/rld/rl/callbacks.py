"""In-training evaluation, checkpoints with their normalisation statistics, and ``status.json``.

Tune-pool evaluation
--------------------
:class:`TunePoolEvaluator` flies a **fixed** list of tune-pool episodes -- by default the
P3-D3 draw from ``configs/control/tuning.yaml`` (seed 20260923, 60 per sea state, SS3-SS5,
aft pad), the same 180 episodes every classical baseline was tuned and re-checked on
(P3-D4 table) -- with the deterministic policy, on a separate set of worker processes that
are idle while training runs. Observations are normalised with a **frozen copy** of the
training ``VecNormalize`` statistics (the state a saved checkpoint holds, with
``training=False`` and ``norm_reward=False``), so evaluation can neither update the
training statistics nor see a normalised reward. Each worker flies its share of the draw
on its own, without lockstep (:mod:`rld.rl.eval_workers`); the episode rows are identical
to the lockstep evaluator's, which is kept as :meth:`TunePoolEvaluator.evaluate_lockstep`
for that comparison. Evaluation steps are not counted against the training budget; they
are reported separately as ``eval_env_steps``.

:class:`PeriodicEvalCallback` runs it every ``eval.interval_steps`` training steps, writes one
row to ``evals.csv`` (success per sea state and pooled, the six outcome fractions, p95
closing speed of the touched-down episodes, curriculum state) and every episode to
``eval_episodes.csv`` (``eval_index`` and ``train_steps`` -- the training step count of the
evaluation -- then the episode row, whose own ``steps`` is the episode length), feeds the
current stage's outcomes to the curriculum and, on a promotion, broadcasts the new stage to
the training workers through ``env_method``.

Checkpoints
-----------
:func:`save_checkpoint` writes ``model.zip`` **and** ``vecnormalize.pkl`` into one directory,
built under a temporary name and renamed into place, so a checkpoint without its statistics
cannot exist. :func:`frozen_normalizer` is the one eval-time loader: ``training=False``,
``norm_reward=False``.

``status.json``
---------------
:class:`StatusWriter` writes the run's status atomically (temporary file + ``os.replace``),
at start, after every evaluation, every ``status_interval_s`` of host time, and at the end
(``done``) or on an exception (``failed``). :data:`STATUS_KEYS` is its schema.

Units: ``steps`` are training env steps (control steps, 1/30 s model scale each); every
``*_s`` field and ``fps`` refer to **host wall-clock** time, not simulated time; closing
speeds are metres per second model scale.
"""

import copy
import csv
import json
import os
import pickle
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from rld.control.tuning import TuningEpisode, summarise_trial
from rld.envs.touchdown import OUTCOMES
from rld.rl.curriculum import Curriculum
from rld.rl.eval_workers import (
    EvalTask,
    EvalWorkerFactory,
    normalizer_from_state,
    normalizer_state,
    policy_payload,
    split_shares,
)
from rld.rl.procs import tree_cpu_seconds
from rld.rl.wrappers import EnvFactory

__all__ = [
    "REPLAY_BUFFER_FILE",
    "RESUME_STATE_FILE",
    "STATUS_KEYS",
    "CheckpointCallback",
    "PeriodicEvalCallback",
    "RunTracker",
    "StatusCallback",
    "StatusWriter",
    "TunePoolEvaluator",
    "atomic_write_json",
    "copy_obs_rms",
    "frozen_normalizer",
    "save_checkpoint",
    "summarise_eval",
    "utc_now",
]

#: Keys every ``status.json`` carries (the ``/sweep-status`` table reads them).
STATUS_KEYS: tuple[str, ...] = (
    "state",
    "method",
    "seed",
    "run_id",
    "run_group",
    "run_dir",
    "algo",
    "steps",
    "budget",
    "fps",
    "eta_s",
    "curriculum_stage",
    "curriculum",
    "last_eval",
    "eval_interval_steps",
    "status_interval_s",
    "eval_env_steps",
    "n_envs",
    "workers",
    "device",
    "started",
    "updated",
    "wall_s",
    "cpu_hours",
    "pid",
    "host",
    "git_sha",
    "git_dirty",
    "config",
    "config_sha256",
    "log",
    "resumed_from",
    "error",
)

RunState = Literal["running", "done", "failed"]


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string, seconds resolution."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON so that a reader never sees a half-written file.

    Args:
        path: Destination.
        payload: JSON-serialisable mapping (NaN is written as ``null``).
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def _json_safe(value: Any) -> Any:
    """Return ``value`` with NaN/inf floats replaced by ``None`` and numpy scalars unwrapped."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# --------------------------------------------------------------------------- normalisation


def frozen_normalizer(normalizer: VecNormalize) -> VecNormalize:
    """Return a detached, frozen copy of a ``VecNormalize`` for evaluation.

    The copy is what a pickle round trip gives -- the exact object a checkpoint's
    ``vecnormalize.pkl`` holds, without its ``venv`` -- with ``training=False`` (statistics
    never update) and ``norm_reward=False`` (rewards are reported raw). Only
    :meth:`VecNormalize.normalize_obs` may be called on it.

    Args:
        normalizer: The training normaliser.

    Returns:
        The frozen copy; the original is not touched.
    """
    # The pickle state by hand: VecNormalize.__getstate__ fails on an already detached
    # object (it deletes attributes that only set_venv creates).
    return normalizer_from_state(normalizer_state(normalizer))


#: File names inside a checkpoint directory.
RESUME_STATE_FILE: str = "resume_state.pkl"
REPLAY_BUFFER_FILE: str = "replay_buffer.pkl"


def save_checkpoint(
    model: BaseAlgorithm,
    normalizer: VecNormalize | None,
    directory: Path,
    meta: Mapping[str, Any],
    resume_state: Mapping[str, Any] | None = None,
) -> Path:
    """Save the model and its normalisation statistics together, atomically.

    With ``resume_state`` the checkpoint is also **resumable**
    (:func:`rld.rl.resume.resume_checkpoint`): the pickled training state goes to
    ``resume_state.pkl`` and, for an off-policy model, the replay buffer to
    ``replay_buffer.pkl`` (SB3 ``save_replay_buffer``). ``checkpoint.json`` records which.

    Args:
        model: The SB3 model.
        normalizer: The training ``VecNormalize`` (``None`` only if normalisation is off).
        directory: Final checkpoint directory; must not exist yet.
        meta: Extra fields for ``checkpoint.json`` (steps, curriculum stage, ...).
        resume_state: Everything else a run needs to continue (curriculum window, RNG
            states, callback counters, ...), or ``None`` for a model-only checkpoint.

    Returns:
        ``directory``.

    Raises:
        FileExistsError: If ``directory`` already exists (a checkpoint is never overwritten).
    """
    if directory.exists():
        raise FileExistsError(f"checkpoint {directory} exists; refusing to overwrite")
    # The pid keeps a half-written directory left by a killed process from blocking the
    # resumed run when it reaches the same step (that leftover is kept, never deleted).
    tmp = directory.with_name(f".{directory.name}.{os.getpid()}.tmp")
    tmp.mkdir(parents=True, exist_ok=False)
    model.save(str(tmp / "model.zip"))
    if normalizer is not None:
        normalizer.save(str(tmp / "vecnormalize.pkl"))
    extra: dict[str, Any] = {"resumable": resume_state is not None, "replay_buffer": False}
    if resume_state is not None:
        with (tmp / RESUME_STATE_FILE).open("wb") as handle:
            pickle.dump(dict(resume_state), handle, protocol=pickle.HIGHEST_PROTOCOL)
        if getattr(model, "replay_buffer", None) is not None:
            model.save_replay_buffer(str(tmp / REPLAY_BUFFER_FILE))  # type: ignore[attr-defined]
            extra["replay_buffer"] = True
            extra["replay_buffer_size"] = int(model.replay_buffer.size())  # type: ignore[attr-defined]
    stamp = {"saved": utc_now(), "saved_unix": time.time()}
    atomic_write_json(tmp / "checkpoint.json", {**meta, **extra, **stamp})
    tmp.replace(directory)
    return directory


# --------------------------------------------------------------------------- evaluation


def summarise_eval(
    rows: Sequence[Mapping[str, Any]], sea_states: Sequence[str], method: str, index: int
) -> dict[str, Any]:
    """Reduce one evaluation's episodes to its ``evals.csv`` fields.

    The per-sea-state block and the objective are :func:`rld.control.tuning.summarise_trial`
    verbatim -- the P3-D3 definitions the baselines' tune-pool numbers use -- so an RL
    tune-pool number and a baseline tune-pool number are the same statistic.

    Args:
        rows: Episode rows (``episode_row`` of the terminal info).
        sea_states: Sea-state order.
        method: Method label.
        index: Evaluation index.

    Returns:
        ``success`` (mean over sea states), pooled outcome fractions ``frac_<outcome>``,
        ``pooled_td_closing_p95_m_s`` (m/s model, touched-down episodes only),
        ``mean_return``, ``n_episodes``, then the per-sea-state fields.
    """
    trial = summarise_trial(method, index, {}, rows, sea_states).row
    n = len(rows)
    out: dict[str, Any] = {
        "success": trial["mean_success"],
        "n_episodes": n,
        **{f"frac_{o}": sum(r["outcome"] == o for r in rows) / n for o in OUTCOMES},
        "pooled_td_closing_p95_m_s": trial["pooled_td_closing_p95_m_s"],
        "mean_return": float(np.mean([float(r["return"]) for r in rows])),
    }
    for key, value in trial.items():
        if key not in ("controller", "trial", "mean_success", "pooled_td_closing_p95_m_s"):
            out[key] = value
    return out


class TunePoolEvaluator:
    """Fly a fixed tune-pool episode list with the deterministic policy.

    Attributes:
        episodes: The fixed episodes, in draw order.
        sea_states: Their sea states, in first-seen order.
    """

    def __init__(
        self,
        factories: Sequence[EnvFactory],
        episodes: Sequence[TuningEpisode],
        vec_env: Literal["subproc", "dummy"],
        deterministic: bool = True,
    ) -> None:
        """Hold the factories; the workers are started at the first evaluation.

        Args:
            factories: One factory per evaluation worker, each built on the tune pool; worker
                ``i`` must have ``rank == i``.
            episodes: The fixed episode list.
            vec_env: ``"subproc"`` (one process per worker) or ``"dummy"`` (in-process).
            deterministic: Use the mean action.

        Raises:
            ValueError: If there are no episodes or no factories, or the ranks are not
                ``0 .. n-1``.
        """
        if not episodes or not factories:
            raise ValueError("an evaluator needs episodes and at least one worker")
        self.episodes = list(episodes)
        self.sea_states = list(dict.fromkeys(ep.ss for ep in self.episodes))
        self._factories = list(factories)[: len(self.episodes)]
        if [f.rank for f in self._factories] != list(range(len(self._factories))):
            raise ValueError("evaluation worker ranks must be 0 .. n-1 in order")
        self._vec_env = vec_env
        self._deterministic = deterministic
        self._venv: VecEnv | None = None

    @property
    def n_workers(self) -> int:
        """Return the number of evaluation workers (the lockstep batch width)."""
        return len(self._factories)

    def _ensure(self) -> VecEnv:
        """Return the evaluation workers' vector env, building it on first use."""
        if self._venv is None:
            threads = 1 if self._vec_env == "subproc" else None
            fns: list[Callable[[], Any]] = [EvalWorkerFactory(f, threads) for f in self._factories]
            self._venv = SubprocVecEnv(fns) if self._vec_env == "subproc" else DummyVecEnv(fns)
        return self._venv

    def _ordered(self, found: Mapping[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the rows in draw order."""
        return [found[(ep.ss, ep.index)] for ep in self.episodes]

    def evaluate(
        self, model: BaseAlgorithm, normalizer: VecNormalize | None
    ) -> tuple[list[dict[str, Any]], int]:
        """Fly every episode once; each worker flies its share independently.

        Args:
            model: The policy.
            normalizer: The training normaliser; a frozen copy is shipped, the original is
                never updated.

        Returns:
            ``(rows, eval_env_steps)``: one row per episode in draw order, and the number of
            environment steps flown.
        """
        venv = self._ensure()
        n = self.n_workers
        task = EvalTask(
            policy=policy_payload(model.policy),
            normalizer=None if normalizer is None else normalizer_state(normalizer),
            deterministic=self._deterministic,
            batch_width=n,
            shares=split_shares(self.episodes, n),
        )
        results: list[tuple[list[dict[str, Any]], int]] = venv.env_method("fly", task)
        found: dict[tuple[str, int], dict[str, Any]] = {}
        steps = 0
        for rows, worker_steps in results:
            steps += int(worker_steps)
            for row in rows:
                found[(str(row["ss"]), int(row["index"]))] = row
        return self._ordered(found), steps

    def evaluate_lockstep(
        self, model: BaseAlgorithm, normalizer: VecNormalize | None
    ) -> tuple[list[dict[str, Any]], int]:
        """Fly every episode once with all workers in lockstep (the original evaluator).

        Kept as the reference :meth:`evaluate` is proven identical to
        (``tests/test_rl_eval_workers.py``); training uses :meth:`evaluate`.

        Args:
            model: The policy.
            normalizer: The training normaliser; a frozen copy is used.

        Returns:
            ``(rows, eval_env_steps)``, the step count including idle replays.

        Raises:
            RuntimeError: If an episode never ends.
        """
        venv = self._ensure()
        frozen = None if normalizer is None else frozen_normalizer(normalizer)
        n = venv.num_envs
        for i in range(n):
            venv.env_method("load_episode_queue", self.episodes[i::n], indices=[i])
        obs = np.asarray(venv.reset(), dtype=np.float32)
        found: dict[tuple[str, int], dict[str, Any]] = {}
        steps = 0
        max_steps = 100_000 * max(1, len(self.episodes))
        while len(found) < len(self.episodes):
            x = obs if frozen is None else np.asarray(frozen.normalize_obs(obs), dtype=np.float32)
            actions, _ = model.predict(x, deterministic=self._deterministic)
            raw_obs, _, dones, infos = venv.step(actions)
            obs = np.asarray(raw_obs, dtype=np.float32)
            steps += n
            for k in np.flatnonzero(dones):
                info = infos[int(k)]
                if info.get("eval_idle", False):
                    continue
                row = dict(info["episode_row"])
                found[(str(row["ss"]), int(row["index"]))] = row
            if steps > max_steps:
                raise RuntimeError("evaluation did not finish; an episode never ended")
        return self._ordered(found), steps

    def close(self) -> None:
        """Close the evaluation workers."""
        if self._venv is not None:
            self._venv.close()
            self._venv = None


# --------------------------------------------------------------------------- status


class StatusWriter:
    """Write ``status.json`` for one run.

    A resumed run (:mod:`rld.rl.resume`) continues the same file: ``started`` stays the
    original start, ``wall_s`` and ``cpu_hours`` are the earlier segments' totals (as last
    recorded by them) plus this segment's, and ``fps`` / ``eta_s`` are this segment's rate.
    ``resumes`` lists every resume.

    Attributes:
        path: ``<run_dir>/status.json``.
        static: Fields fixed for the run's lifetime.
        started: UTC start of the run (of its first segment).
        started_monotonic: Host monotonic clock at the start of this segment, seconds.
        wall_offset_s: Host seconds of earlier segments.
        cpu_offset_h: CPU hours of earlier segments.
        steps_offset: Training steps this segment started from.
    """

    def __init__(
        self,
        path: Path,
        static: Mapping[str, Any],
        *,
        started: str | None = None,
        wall_offset_s: float = 0.0,
        cpu_offset_h: float = 0.0,
        steps_offset: int = 0,
    ) -> None:
        """Hold the fixed fields.

        Args:
            path: Destination file.
            static: Fields fixed for the run: method, seed, run_id, budget, ... .
            started: The run's original UTC start (resumed runs); now when ``None``.
            wall_offset_s: Host seconds already spent by earlier segments.
            cpu_offset_h: CPU hours already spent by earlier segments.
            steps_offset: Training steps at the start of this segment.
        """
        self.path = path
        self.static = dict(static)
        self.started = utc_now() if started is None else started
        self.started_monotonic = time.monotonic()
        self.wall_offset_s = float(wall_offset_s)
        self.cpu_offset_h = float(cpu_offset_h)
        self.steps_offset = int(steps_offset)
        self.last: dict[str, Any] = {}

    def cpu_hours(self) -> float:
        """Return CPU time of this process and its live children (the env workers), hours.

        Includes earlier segments of a resumed run (:attr:`cpu_offset_h`).
        """
        return self.cpu_offset_h + tree_cpu_seconds() / 3600.0

    def write(
        self,
        state: RunState,
        *,
        steps: int,
        curriculum: Curriculum | None,
        last_eval: Mapping[str, Any] | None,
        eval_env_steps: int,
        error: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write the status file.

        Args:
            state: ``running``, ``done`` or ``failed``.
            steps: Training env steps so far.
            curriculum: The curriculum, or ``None`` before it exists.
            last_eval: The latest evaluation summary, or ``None``.
            eval_env_steps: Evaluation env steps so far (not budgeted).
            error: One-line error for a failed run; the traceback is in the log.
            extra: Additional fields.

        Returns:
            The payload written.
        """
        wall = time.monotonic() - self.started_monotonic
        budget = int(self.static["budget"])
        segment_steps = steps - self.steps_offset
        fps = segment_steps / wall if wall > 0 and segment_steps > 0 else 0.0
        eta = (budget - steps) / fps if fps > 0 and state == "running" else None
        payload: dict[str, Any] = {
            "state": state,
            **{k: self.static.get(k) for k in ("method", "seed", "run_id", "run_group")},
            "run_dir": self.static.get("run_dir"),
            "algo": self.static.get("algo"),
            "steps": int(steps),
            "budget": budget,
            "fps": fps,
            "eta_s": eta,
            "curriculum_stage": None if curriculum is None else curriculum.stage,
            "curriculum": None if curriculum is None else curriculum.state(),
            "last_eval": None if last_eval is None else dict(last_eval),
            "eval_interval_steps": self.static.get("eval_interval_steps"),
            "status_interval_s": self.static.get("status_interval_s"),
            "eval_env_steps": int(eval_env_steps),
            "n_envs": self.static.get("n_envs"),
            "workers": self.static.get("workers"),
            "device": self.static.get("device"),
            "started": self.started,
            "updated": utc_now(),
            "wall_s": self.wall_offset_s + wall,
            "cpu_hours": self.cpu_hours(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "git_sha": self.static.get("git_sha"),
            "git_dirty": self.static.get("git_dirty"),
            "config": self.static.get("config"),
            "config_sha256": self.static.get("config_sha256"),
            "log": self.static.get("log"),
            "resumed_from": self.static.get("resumed_from"),
            "error": error,
            **{
                k: self.static[k]
                for k in ("resumed_at", "resume_steps", "resumes")
                if k in self.static
            },
            **dict(extra or {}),
        }
        atomic_write_json(self.path, payload)
        self.last = payload
        return payload


class RunTracker:
    """Shared mutable state of one run: status writer, curriculum, latest evaluation.

    Attributes:
        status: The status writer.
        curriculum: The curriculum.
        last_eval: Latest evaluation summary for ``status.json``.
        eval_env_steps: Evaluation env steps so far.
    """

    def __init__(self, status: StatusWriter, curriculum: Curriculum) -> None:
        """Hold the shared state.

        Args:
            status: The status writer.
            curriculum: The curriculum.
        """
        self.status = status
        self.curriculum = curriculum
        self.last_eval: dict[str, Any] | None = None
        self.eval_env_steps = 0
        #: Host seconds in evaluations, all segments of a resumed run.
        self.eval_wall_s = 0.0
        #: Host seconds in evaluations, this segment only.
        self.segment_eval_wall_s = 0.0

    def add_eval_time(self, seconds: float) -> None:
        """Account host seconds spent in one evaluation."""
        self.eval_wall_s += seconds
        self.segment_eval_wall_s += seconds

    def write(self, state: RunState, steps: int, error: str | None = None) -> None:
        """Write ``status.json`` with the current shared state.

        Args:
            state: Run state.
            steps: Training env steps so far.
            error: One-line error for a failed run.
        """
        self.status.write(
            state,
            steps=steps,
            curriculum=self.curriculum,
            last_eval=self.last_eval,
            eval_env_steps=self.eval_env_steps,
            error=error,
            extra=self.timing(steps),
        )

    def timing(self, steps: int) -> dict[str, float]:
        """Return host time spent evaluating and the training rate without it.

        Args:
            steps: Training env steps so far.

        Returns:
            ``eval_wall_s`` (host seconds in evaluations, all segments) and
            ``fps_excl_eval`` (this segment's training env steps per host second of
            non-evaluation time; learner updates included).
        """
        wall = time.monotonic() - self.status.started_monotonic
        train_wall = wall - self.segment_eval_wall_s
        done = steps - self.status.steps_offset
        return {
            "eval_wall_s": self.eval_wall_s,
            "fps_excl_eval": done / train_wall if train_wall > 0 and done > 0 else 0.0,
        }


def _append_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append rows to a CSV, writing the header when the file is new.

    Args:
        path: The CSV file.
        rows: Rows with identical keys.
    """
    if not rows:
        return
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)


class PeriodicEvalCallback(BaseCallback):
    """Evaluate on the tune pool every ``interval`` training steps; drive the curriculum."""

    def __init__(
        self,
        evaluator: TunePoolEvaluator,
        tracker: RunTracker,
        run_dir: Path,
        method: str,
        interval_steps: int,
        broadcast: Callable[[str | None], None],
        normalizer: Callable[[], VecNormalize | None],
        tb_name: str = "eval",
    ) -> None:
        """Wire the evaluation.

        Args:
            evaluator: The tune-pool evaluator.
            tracker: Shared run state.
            run_dir: The run directory (``evals.csv``, ``eval_episodes.csv``, ``tb/eval``).
            method: Method label.
            interval_steps: Training env steps between evaluations.
            broadcast: Sends a new sampling stage to every training worker.
            normalizer: Returns the training normaliser at call time.
            tb_name: TensorBoard subdirectory under ``tb/`` (a resumed segment writes its
                own, ``eval_resume<k>``, so abandoned and resumed points never interleave).
        """
        super().__init__(verbose=0)
        self.evaluator = evaluator
        self.tracker = tracker
        self.run_dir = run_dir
        self.method = method
        self.interval = int(interval_steps)
        self.broadcast = broadcast
        self.get_normalizer = normalizer
        self.next_eval = self.interval
        self.n_evals = 0
        self.last_eval_steps = -1
        self.tb_name = tb_name
        self._tb: Any = None

    def counters(self) -> dict[str, int]:
        """Return the counters a resumed run restores with :meth:`restore_counters`."""
        return {
            "n_evals": self.n_evals,
            "next_eval": self.next_eval,
            "last_eval_steps": self.last_eval_steps,
        }

    def restore_counters(self, counters: Mapping[str, int]) -> None:
        """Restore :meth:`counters` (evaluation index, next due step, last evaluated step)."""
        self.n_evals = int(counters["n_evals"])
        self.next_eval = int(counters["next_eval"])
        self.last_eval_steps = int(counters["last_eval_steps"])

    def _on_step(self) -> bool:
        """Run an evaluation whenever the step counter crosses the next multiple."""
        if self.num_timesteps >= self.next_eval:
            self.run_eval(self.num_timesteps)
            while self.next_eval <= self.num_timesteps:
                self.next_eval += self.interval
        return True

    def _writer(self) -> Any:
        """Return the TensorBoard writer for ``tb/eval``, created on first use."""
        if self._tb is None:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb" / self.tb_name))
        return self._tb

    def run_eval(self, steps: int) -> dict[str, Any]:
        """Evaluate once, log it, and update the curriculum.

        Args:
            steps: Training env steps at this evaluation.

        Returns:
            The ``evals.csv`` row.
        """
        assert self.model is not None
        tracker = self.tracker
        cur = tracker.curriculum
        t0 = time.monotonic()
        rows, eval_steps = self.evaluator.evaluate(self.model, self.get_normalizer())
        eval_s = time.monotonic() - t0
        tracker.eval_env_steps += eval_steps
        tracker.add_eval_time(eval_s)
        stage = cur.stage
        stage_success = [r["outcome"] == "success" for r in rows if r["ss"] == stage]
        promoted = cur.observe(stage_success, steps)
        if promoted:
            self.broadcast(cur.sampling_stage)
        summary = summarise_eval(rows, self.evaluator.sea_states, self.method, self.n_evals)
        row: dict[str, Any] = {
            "eval_index": self.n_evals,
            "steps": int(steps),
            "wall_s": time.monotonic() - tracker.status.started_monotonic,
            "eval_s": eval_s,
            "eval_env_steps": eval_steps,
            "stage_evaluated": stage,
            "stage_after": cur.stage,
            "promoted": promoted,
            "window_n": len(cur.window),
            "window_success": cur.window_success,
            **summary,
        }
        _append_csv(self.run_dir / "evals.csv", [row])
        _append_csv(
            self.run_dir / "eval_episodes.csv",
            [{"eval_index": self.n_evals, "train_steps": int(steps), **r} for r in rows],
        )
        tb = self._writer()
        tb.add_scalar("eval/success", summary["success"], steps)
        for ss in self.evaluator.sea_states:
            tb.add_scalar(f"eval/success_{ss}", summary[f"{ss}_frac_success"], steps)
        for outcome in OUTCOMES:
            tb.add_scalar(f"eval/frac_{outcome}", summary[f"frac_{outcome}"], steps)
        p95 = summary["pooled_td_closing_p95_m_s"]
        if np.isfinite(p95):
            tb.add_scalar("eval/td_closing_p95_m_s", p95, steps)
        tb.add_scalar("eval/mean_return", summary["mean_return"], steps)
        tb.add_scalar("curriculum/stage_index", cur.index, steps)
        tb.flush()
        tracker.last_eval = {
            "eval_index": self.n_evals,
            "steps": int(steps),
            "success": summary["success"],
            "success_by_ss": {
                ss: summary[f"{ss}_frac_success"] for ss in self.evaluator.sea_states
            },
            "outcome_fractions": {o: summary[f"frac_{o}"] for o in OUTCOMES},
            "td_closing_p95_m_s": p95,
            "n_episodes": summary["n_episodes"],
        }
        self.n_evals += 1
        self.last_eval_steps = int(steps)
        tracker.write("running", steps)
        return row

    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self._tb is not None:
            self._tb.close()
            self._tb = None


class CheckpointCallback(BaseCallback):
    """Save ``model.zip`` + ``vecnormalize.pkl`` (+ resume state) every ``interval`` steps.

    In a :class:`~stable_baselines3.common.callbacks.CallbackList` it must come **after**
    :class:`PeriodicEvalCallback`, so that a checkpoint taken at an evaluation step already
    holds that evaluation's curriculum update.
    """

    def __init__(
        self,
        run_dir: Path,
        interval_steps: int,
        normalizer: Callable[[], VecNormalize | None],
        tracker: RunTracker,
        resume_state: Callable[[int], Mapping[str, Any]] | None = None,
    ) -> None:
        """Wire the checkpoints.

        Args:
            run_dir: The run directory; checkpoints go to ``checkpoints/step_<steps>``.
            interval_steps: Training env steps between checkpoints.
            normalizer: Returns the training normaliser at call time.
            tracker: Shared run state (for the stage recorded in ``checkpoint.json``).
            resume_state: Called with the next checkpoint's step; returns the state that
                makes the checkpoint resumable (:func:`save_checkpoint`), or ``None`` for
                model-only checkpoints.
        """
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.interval = int(interval_steps)
        self.get_normalizer = normalizer
        self.tracker = tracker
        self.get_resume_state = resume_state
        self.next_ckpt = self.interval

    def _on_step(self) -> bool:
        """Checkpoint whenever the step counter crosses the next multiple."""
        if self.num_timesteps >= self.next_ckpt:
            assert self.model is not None
            while self.next_ckpt <= self.num_timesteps:
                self.next_ckpt += self.interval
            state = None if self.get_resume_state is None else self.get_resume_state(self.next_ckpt)
            save_checkpoint(
                self.model,
                self.get_normalizer(),
                self.run_dir / "checkpoints" / f"step_{self.num_timesteps:010d}",
                {"steps": self.num_timesteps, "curriculum": self.tracker.curriculum.state()},
                state,
            )
        return True


class StatusCallback(BaseCallback):
    """Heartbeat ``status.json`` every ``interval_s`` seconds of host time."""

    def __init__(self, tracker: RunTracker, interval_s: float) -> None:
        """Wire the heartbeat.

        Args:
            tracker: Shared run state.
            interval_s: Host-clock seconds between writes.
        """
        super().__init__(verbose=0)
        self.tracker = tracker
        self.interval_s = float(interval_s)
        self._last = time.monotonic()

    def _on_step(self) -> bool:
        """Write the status when the heartbeat interval has elapsed."""
        now = time.monotonic()
        if now - self._last >= self.interval_s:
            self.tracker.write("running", self.num_timesteps)
            self._last = now
        return True


def copy_obs_rms(normalizer: VecNormalize) -> Any:
    """Return a deep copy of a normaliser's observation statistics (for tests and checks)."""
    return copy.deepcopy(normalizer.obs_rms)
