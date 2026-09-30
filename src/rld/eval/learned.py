r"""Learned policies on the frozen episode lists: per-seed cells, seed spread, rliable aggregates.

Phase 5's ``results/e05/``: the ``final/`` checkpoint of every training seed of a learned
method (``ppo``, ``sac``), flown on a committed list through the same chunked, parallel
runner the classical baselines went through (:func:`rld.eval.runner.run_matrix`), with the
baselines printed beside them -- **carried** from their committed runs, never re-flown.

What is flown
-------------
Each run is a :class:`~rld.eval.runner.PolicySpec` from
:func:`~rld.eval.runner.callable_spec` with the picklable factory
``functools.partial(rld.rl.train.build_policy, run_dir=<run>, ckpt=<ckpt>)``: the batch-1
:class:`~rld.rl.train.LearnedPolicy`, deterministic actions, ``VecNormalize`` statistics
frozen (``training=False``, ``norm_reward=False``), never privileged and never given a
ship-motion feed. The seed label (``run_seed``) is the run's **training seed**, read from
its ``provenance.json`` and cross-checked against ``status.json``; the method label must be
the run's own ``method``. A run that is not ``done`` is refused. Every run's config,
``model.zip``, ``vecnormalize.pkl`` and ``checkpoint.json`` are hashed before the flight and
again after it; a change aborts before anything is written.

What is written (``out_dir``)
-----------------------------
* ``episodes.csv`` -- one row per (method, seed, episode), :data:`EPISODE_COLUMNS`.
* ``summary.csv`` -- one row per (method, seed, pad, regime, sea state): every P3-D1 §3
  metric (:func:`rld.eval.report.summarise`), then the run's provenance
  (:data:`RUN_COLUMNS`), then the deterministic environment provenance.
* ``seeds.csv`` -- the seed-to-seed spread per (method, pad, regime, sea state)
  (:func:`seed_spread`).
* ``aggregate.csv`` -- rliable IQM and optimality gap of success across seeds per cell, and
  the IQM of per-seed p95 closing speed, each with a stratified-bootstrap 95 % CI
  (:func:`aggregate_rows`; P3-D1 §4: 2 000 replicates, seed 20260926, runs resampled within
  the task, tasks never resampled). One task per (regime, sea state): success is never pooled
  across sea states.
* ``carried_summary_<source>.csv`` -- the baselines' summary rows for the flown cells,
  **line-for-line copies** of ``results/<source>/summary.csv`` under that file's own header
  (:func:`carry_baselines` checks every line against its source, the source's provenance
  against the live one, that the source flew the identical listed episodes, and that its
  summary is reproduced by re-summarising its own ``episodes.csv``).
* ``success_vs_seastate.md`` -- rendered from the CSVs above only (:func:`render_learned`);
  :func:`main` ``--render-only`` re-renders it, ``--check`` re-derives every CSV and the
  markdown from ``episodes.csv`` and compares bytes.
* ``run_info.json`` -- the non-deterministic facts: git state, host, worker count, timings,
  the run table with its digests, list hashes and the carry checks.

Entry point: ``python -m rld.eval.learned`` (``scripts/`` is outside the eval-auditor's
write scope; a thin ``scripts/eval_learned.py`` wrapper may call :func:`main`).

Units: speeds metres per second model scale, lengths metres model scale, times seconds model
scale, angles degrees; rates dimensionless; training budgets in environment control steps
(1/30 s model scale each).
"""

import argparse
import functools
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rld.config import REPO_ROOT
from rld.control.registry import REGISTRY
from rld.envs.config import LANDING_CONFIG, NOISE_CONFIG, OBSERVATION_CONFIG, SUCCESS_CONFIG
from rld.envs.touchdown import OUTCOMES, TERMINATION_REASONS
from rld.eval.envs import EvalConfigs, load_eval_configs
from rld.eval.episodes import (
    DEFAULT_GENERATOR_SEED,
    EPISODES_DIR,
    MANIFEST_NAME,
    ListedEpisode,
    content_sha256,
    file_sha256,
    read_list,
    read_manifest,
)
from rld.eval.metrics import CELL_METRIC_COLUMNS, as_bool, as_float
from rld.eval.report import (
    CAVEATS,
    FORBIDDEN_RENDERED_PHRASES,
    deterministic_provenance,
    method_label,
    read_rows,
    summarise,
    summary_columns,
    write_rows,
)
from rld.eval.reproduce import git_state
from rld.eval.runner import (
    DEFAULT_CHUNK,
    DEFAULT_WORKERS,
    EPISODE_COLUMNS,
    PolicySpec,
    callable_spec,
    run_matrix,
)
from rld.eval.stats import (
    DEFAULT_BOOTSTRAP_SEED,
    STRATIFIED_REPS,
    aggregate_estimates,
    iqm,
    stratified_bootstrap_ci,
)
from rld.provenance import environment_provenance
from rld.rl.train import build_policy

__all__ = [
    "AGGREGATE_COLUMNS",
    "DEFAULT_CARRY",
    "RUN_COLUMNS",
    "LearnedRun",
    "aggregate_rows",
    "carry_baselines",
    "evaluate_runs",
    "inspect_run",
    "learned_spec",
    "live_provenance",
    "main",
    "parse_learned",
    "render_learned",
    "seed_spread",
    "summarise_learned",
    "verify_lists",
    "write_learned",
]

#: Per-run provenance columns of ``summary.csv`` (deterministic: paths and file digests).
RUN_COLUMNS: tuple[str, ...] = (
    "run_dir",
    "run_algo",
    "run_total_steps",
    "run_ckpt",
    "run_ckpt_steps",
    "run_train_git_sha",
    "run_config_sha256",
    "run_config_source_sha256",
    "run_model_sha256",
    "run_vecnormalize_sha256",
    "run_checkpoint_json_sha256",
)

#: ``aggregate.csv`` columns.
AGGREGATE_COLUMNS: tuple[str, ...] = (
    "method",
    "pad",
    "regime",
    "ss",
    "metric",
    "statistic",
    "reported",
    "point",
    "ci_lo",
    "ci_hi",
    "confidence",
    "n_runs",
    "n_tasks",
    "seeds",
    "reps",
    "bootstrap_seed",
    "resampling",
    "note",
)

#: Baselines carried beside the learned rows by default: ``{source dir under results/:
#: methods}`` (CLAUDE.md non-negotiable 4). Carried, never re-flown.
DEFAULT_CARRY: dict[str, tuple[str, ...]] = {
    "e01": (
        "pid_track_descend",
        "pid_feedforward",
        "pid_feedforward_lowvz",
        "gated",
        "oracle_gated",
    ),
    "e01_lowvz_cut": ("pid_feedforward_lowvz_cut",),
}

#: File-name prefix of a carried summary; the rest of the stem is the source dir's name.
CARRIED_PREFIX: str = "carried_summary_"

#: Provenance keys that vary between runs of identical results; they go to run_info.json.
VOLATILE_KEYS: tuple[str, ...] = ("omp_num_threads", "cpu_count", "host", "timestamp_utc")

#: The success metric's column and the primary secondary metric's column (P3-D1 §3).
SUCCESS_COLUMN: str = "success_rate"
P95_COLUMN: str = "rel_vz_normal_p95_m_s"

#: How the stratified bootstrap resamples (P3-D1 §4), written into every aggregate row.
_RESAMPLING: str = "stratified: runs resampled within the task; tasks never resampled"

#: The markdown file.
MARKDOWN_NAME: str = "success_vs_seastate.md"

RESULTS_ROOT = (REPO_ROOT / "results").resolve()


# --------------------------------------------------------------------------- runs


@dataclass(frozen=True)
class LearnedRun:
    """One finished training run and the checkpoint of it that is evaluated.

    Attributes:
        method: Method label (the run's ``method``), e.g. ``"ppo"``.
        seed: Training seed, from ``provenance.json`` (checked against ``status.json``).
        run_dir: The run directory, absolute.
        ckpt: Checkpoint name (``"final"``, a step count or ``"step_<n>"``).
        algo: ``"ppo"`` or ``"sac"``.
        total_steps: Configured training budget, env control steps.
        ckpt_steps: Env control steps at the checkpoint (``checkpoint.json``).
        train_git_sha: Commit the run was trained at.
        train_git_dirty_paths: Dirty paths when it was trained.
        resumed_from: ``status.json``'s ``resumed_from`` (``None`` for a run never resumed).
        digests: ``{column: SHA-256}`` for :data:`RUN_COLUMNS`' hashed files.
    """

    method: str
    seed: int
    run_dir: Path
    ckpt: str
    algo: str
    total_steps: int
    ckpt_steps: int
    train_git_sha: str
    train_git_dirty_paths: tuple[str, ...]
    resumed_from: str | None
    digests: tuple[tuple[str, str], ...]

    def label_dir(self) -> str:
        """Return the run directory relative to the repository root when inside it."""
        path = self.run_dir.resolve()
        root = REPO_ROOT.resolve()
        return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)

    def columns(self) -> dict[str, str]:
        """Return this run's :data:`RUN_COLUMNS` values, as text."""
        return {
            "run_dir": self.label_dir(),
            "run_algo": self.algo,
            "run_total_steps": str(self.total_steps),
            "run_ckpt": self.ckpt,
            "run_ckpt_steps": str(self.ckpt_steps),
            "run_train_git_sha": self.train_git_sha,
            **dict(self.digests),
        }


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_path(run_dir: Path, ckpt: str) -> Path:
    """Resolve a checkpoint directory exactly as :func:`rld.rl.train.checkpoint_dir` does."""
    if ckpt == "final":
        return run_dir / "final"
    if ckpt.isdigit():
        return run_dir / "checkpoints" / f"step_{int(ckpt):010d}"
    return run_dir / "checkpoints" / ckpt


def _run_digests(run_dir: Path, ckpt: str) -> tuple[tuple[str, str], ...]:
    """Return the SHA-256 of every file a run's evaluation depends on.

    Args:
        run_dir: The run directory.
        ckpt: The checkpoint name.

    Returns:
        ``((column, digest), ...)`` in :data:`RUN_COLUMNS` order; ``"absent"`` for a
        ``vecnormalize.pkl`` the run does not have.
    """
    path = _checkpoint_path(run_dir, ckpt)
    stats = path / "vecnormalize.pkl"
    source = run_dir / "config_source.yaml"
    return (
        ("run_config_sha256", _sha256(run_dir / "config.yaml")),
        ("run_config_source_sha256", _sha256(source) if source.exists() else "absent"),
        ("run_model_sha256", _sha256(path / "model.zip")),
        ("run_vecnormalize_sha256", _sha256(stats) if stats.exists() else "absent"),
        ("run_checkpoint_json_sha256", _sha256(path / "checkpoint.json")),
    )


def inspect_run(method: str, run_dir: Path, ckpt: str = "final") -> LearnedRun:
    """Read a finished run's identity and hash its evaluated checkpoint.

    Args:
        method: The label it will carry in every table; must be the run's own ``method``.
        run_dir: The run directory.
        ckpt: Checkpoint name.

    Returns:
        The :class:`LearnedRun`.

    Raises:
        ValueError: If the run is not ``done``, its method is not ``method``, its seed
            differs between ``provenance.json`` and ``status.json``, or ``status.json``'s
            ``config_sha256`` is not the SHA-256 of its ``config.yaml``.
        FileNotFoundError: If the checkpoint or a run file is missing.
    """
    run_dir = Path(run_dir).resolve()
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    prov = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    raw = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    path = _checkpoint_path(run_dir, ckpt)
    if not (path / "model.zip").exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    meta = json.loads((path / "checkpoint.json").read_text(encoding="utf-8"))
    if status.get("state") != "done":
        raise ValueError(f"{run_dir}: state {status.get('state')!r}, not 'done'")
    if raw.get("method") != method or status.get("method") != method:
        raise ValueError(
            f"{run_dir}: run method {raw.get('method')!r} / {status.get('method')!r} is not the "
            f"label {method!r}"
        )
    seed = int(prov["seed"])
    if int(status["seed"]) != seed:
        raise ValueError(f"{run_dir}: seed {status['seed']} in status.json != {seed} in provenance")
    digests = _run_digests(run_dir, ckpt)
    if status.get("config_sha256") not in (None, dict(digests)["run_config_sha256"]):
        raise ValueError(f"{run_dir}: status.json config_sha256 is not config.yaml's SHA-256")
    return LearnedRun(
        method=method,
        seed=seed,
        run_dir=run_dir,
        ckpt=ckpt,
        algo=str(raw["algo"]),
        total_steps=int(raw["total_steps"]),
        ckpt_steps=int(meta.get("steps", -1)),
        train_git_sha=str(prov.get("git_sha", "unknown")),
        train_git_dirty_paths=tuple(str(p) for p in prov.get("git_dirty_paths", [])),
        resumed_from=status.get("resumed_from"),
        digests=digests,
    )


def parse_learned(items: Sequence[str]) -> list[tuple[str, tuple[Path, ...]]]:
    """Parse ``NAME=RUN_DIR[,RUN_DIR...]`` items, keeping their order.

    Args:
        items: The ``--learned`` arguments.

    Returns:
        ``[(method, (run_dir, ...)), ...]``.

    Raises:
        SystemExit: On a malformed item, a repeated method, or a registry controller's name
            (a learned method must not be mistaken for a classical one).
    """
    out: list[tuple[str, tuple[Path, ...]]] = []
    for item in items:
        name, sep, dirs = item.partition("=")
        paths = tuple(Path(d) for d in dirs.split(",") if d)
        if not sep or not name or not paths:
            raise SystemExit(f"--learned {item!r}: expected NAME=RUN_DIR[,RUN_DIR...]")
        if name in REGISTRY:
            raise SystemExit(f"--learned {item!r}: {name!r} is a registry controller")
        if name in [m for m, _ in out]:
            raise SystemExit(f"--learned: method {name!r} given twice")
        out.append((name, paths))
    return out


def order_runs(runs: Sequence[LearnedRun]) -> list[LearnedRun]:
    """Return runs ordered by method (first appearance), then seed; refuse a repeated seed.

    Raises:
        ValueError: If a (method, seed) pair appears twice.
    """
    methods = list(dict.fromkeys(r.method for r in runs))
    seen: set[tuple[str, int]] = set()
    for run in runs:
        if (run.method, run.seed) in seen:
            raise ValueError(f"{run.method} seed {run.seed} given twice")
        seen.add((run.method, run.seed))
    return sorted(runs, key=lambda r: (methods.index(r.method), r.seed))


def learned_spec(run: LearnedRun) -> PolicySpec:
    """Return the runner spec of one run: the batch-1 ``LearnedPolicy`` of its checkpoint.

    Args:
        run: The run.

    Returns:
        ``callable_spec(method, partial(build_policy, run_dir=..., ckpt=...), run_seed=seed)``:
        not privileged, no motion feed.
    """
    return callable_spec(
        run.method,
        functools.partial(build_policy, run_dir=run.run_dir, ckpt=run.ckpt),
        run_seed=run.seed,
    )


def evaluate_runs(
    runs: Sequence[LearnedRun],
    episodes: Sequence[ListedEpisode],
    cfgs: EvalConfigs,
    *,
    workers: int = DEFAULT_WORKERS,
    chunk: int = DEFAULT_CHUNK,
) -> list[dict[str, Any]]:
    """Fly every run on the same listed episodes through one worker pool.

    Args:
        runs: The runs, in output order (:func:`order_runs`).
        episodes: Listed episodes, in committed order.
        cfgs: The committed configs.
        workers: Processes; the rows do not depend on it.
        chunk: Episodes per task; the rows do not depend on it.

    Returns:
        ``len(runs) * len(episodes)`` rows (:data:`EPISODE_COLUMNS`), run after run.

    Raises:
        RuntimeError: If any run's checkpoint files changed during the flight.
    """
    before = [_run_digests(r.run_dir, r.ckpt) for r in runs]
    rows = run_matrix(episodes, [learned_spec(r) for r in runs], cfgs, workers=workers, chunk=chunk)
    after = [_run_digests(r.run_dir, r.ckpt) for r in runs]
    if before != after or any(b != r.digests for b, r in zip(before, runs, strict=True)):
        raise RuntimeError("a run's config or checkpoint changed during the evaluation")
    return rows


# --------------------------------------------------------------------------- tables


def live_provenance(cfgs: EvalConfigs, episodes_dir: Path = EPISODES_DIR) -> dict[str, str]:
    """Return the deterministic provenance columns, exactly as ``eval_baselines.py`` builds them.

    Args:
        cfgs: The committed configs.
        episodes_dir: Where the lists and the manifest live.

    Returns:
        Versions, submodule SHAs, config and manifest SHA-256s, generator seed, ``v_max`` and
        the platform driver, as text.
    """
    env = environment_provenance(REPO_ROOT)
    return deterministic_provenance(
        {k: v for k, v in env.items() if k not in VOLATILE_KEYS},
        {
            "success_yaml_sha256": SUCCESS_CONFIG,
            "landing_yaml_sha256": LANDING_CONFIG,
            "observation_yaml_sha256": OBSERVATION_CONFIG,
            "noise_yaml_sha256": NOISE_CONFIG,
            "episodes_manifest_sha256": episodes_dir / MANIFEST_NAME,
        },
        {
            "generator_seed": str(DEFAULT_GENERATOR_SEED),
            "v_max_m_s": repr(cfgs.landing.v_max_m_s),
            "driver": cfgs.landing.platform.driver,
        },
    )


def summarise_learned(
    rows: Sequence[Mapping[str, Any]],
    run_columns: Mapping[tuple[str, int], Mapping[str, str]],
    provenance: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Reduce learned episode rows to per-(method, seed, cell) summary rows.

    Args:
        rows: Episode rows (in memory or read back as text).
        run_columns: ``{(method, seed): RUN_COLUMNS values}``.
        provenance: Deterministic provenance columns.

    Returns:
        ``(records, columns)``: :func:`rld.eval.report.summarise`'s records with the run's
        :data:`RUN_COLUMNS` inserted before the provenance, and the column order.

    Raises:
        KeyError: If a (method, seed) in the rows has no run columns.
    """
    methods = list(dict.fromkeys(str(r["method"]) for r in rows))
    records = summarise(rows, methods, None, None)
    out: list[dict[str, Any]] = []
    for rec in records:
        out.append({**rec, **run_columns[(str(rec["method"]), int(rec["run_seed"]))], **provenance})
    return out, summary_columns([*RUN_COLUMNS, *provenance])


def _cells(summary: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], list[Any]]:
    """Group summary records by (method, pad, regime, ss), in first-appearance order."""
    groups: dict[tuple[str, str, str, str], list[Any]] = {}
    for rec in summary:
        key = (str(rec["method"]), str(rec["pad"]), str(rec["regime"]), str(rec["ss"]))
        groups.setdefault(key, []).append(rec)
    return groups


def _all_seeds(summary: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return every run seed in the summary, sorted."""
    return sorted({int(rec["run_seed"]) for rec in summary})


def seed_spread(summary: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return the seed-to-seed spread of success and p95 closing speed per cell.

    Per (method, pad, regime, sea state): each seed's success rate, k and p95 closing speed
    along the deck normal (NaN for a seed that is absent or never touched down), then, over
    the seeds, min, max, mean and sample SD (``ddof = 1``) of each. A p95 statistic is over
    the seeds with a finite p95 only, and ``p95_n_seeds_finite`` says how many.

    Args:
        summary: Summary records (text or values).

    Returns:
        ``(records, columns)``.
    """
    seeds = _all_seeds(summary)
    columns = [
        "method",
        "pad",
        "regime",
        "ss",
        "n_seeds",
        "seeds",
        "n_episodes_per_seed",
        *(f"success_rate_seed{s}" for s in seeds),
        *(f"n_success_seed{s}" for s in seeds),
        *(f"{P95_COLUMN}_seed{s}" for s in seeds),
        "success_min",
        "success_max",
        "success_mean",
        "success_sd",
        "p95_n_seeds_finite",
        "p95_min_m_s",
        "p95_max_m_s",
        "p95_mean_m_s",
        "p95_sd_m_s",
    ]
    out: list[dict[str, Any]] = []
    nan = float("nan")
    for (method, pad, regime, ss), recs in _cells(summary).items():
        by_seed = {int(r["run_seed"]): r for r in recs}
        succ = np.array([as_float(by_seed[s][SUCCESS_COLUMN]) for s in sorted(by_seed)])
        p95 = np.array([as_float(by_seed[s][P95_COLUMN]) for s in sorted(by_seed)])
        finite = p95[np.isfinite(p95)]
        n_eps = sorted({int(r["n_episodes"]) for r in recs})
        record: dict[str, Any] = {
            "method": method,
            "pad": pad,
            "regime": regime,
            "ss": ss,
            "n_seeds": len(by_seed),
            "seeds": "|".join(str(s) for s in sorted(by_seed)),
            "n_episodes_per_seed": "|".join(str(n) for n in n_eps),
        }
        for s in seeds:
            rec = by_seed.get(s)
            record[f"success_rate_seed{s}"] = nan if rec is None else as_float(rec[SUCCESS_COLUMN])
            record[f"n_success_seed{s}"] = "" if rec is None else int(rec["n_success"])
            record[f"{P95_COLUMN}_seed{s}"] = nan if rec is None else as_float(rec[P95_COLUMN])
        record.update(
            success_min=float(succ.min()),
            success_max=float(succ.max()),
            success_mean=float(succ.mean()),
            success_sd=float(succ.std(ddof=1)) if succ.size > 1 else nan,
            p95_n_seeds_finite=int(finite.size),
            p95_min_m_s=float(finite.min()) if finite.size else nan,
            p95_max_m_s=float(finite.max()) if finite.size else nan,
            p95_mean_m_s=float(finite.mean()) if finite.size else nan,
            p95_sd_m_s=float(finite.std(ddof=1)) if finite.size > 1 else nan,
        )
        out.append(record)
    return out, columns


def aggregate_rows(
    summary: Sequence[Mapping[str, Any]],
    *,
    reps: int = STRATIFIED_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Return rliable aggregates across training seeds, one task per cell (P3-D1 §4).

    Per (method, pad, regime, sea state), with the cell as the single task and its seeds as
    the runs:

    * ``success_rate`` / ``iqm`` and ``success_rate`` / ``optimality_gap`` (gamma = 1),
      from :func:`rld.eval.stats.aggregate_estimates`, sharing one set of replicates;
    * ``rel_vz_normal_p95_m_s`` / ``iqm``: the IQM of the per-seed p95 closing speeds, from
      :func:`rld.eval.stats.stratified_bootstrap_ci` with the same seed. It is descriptive
      and is **not** P3-D1 §4's pooled relative-p95 contrast statistic (H1a/H3). It is not
      reported (``reported = False``, reason in ``note``) when any seed has no touchdown,
      because a missing p95 is not a number to trim.

    With 5 runs rliable's IQM is the mean of the middle 3 (25 % trimmed from each end).

    Args:
        summary: Summary records.
        reps: Bootstrap replicates (P3-D1: 2 000).
        seed: Bootstrap seed (P3-D1: 20260926).

    Returns:
        Records with :data:`AGGREGATE_COLUMNS`.
    """
    out: list[dict[str, Any]] = []
    nan = float("nan")
    for (method, pad, regime, ss), recs in _cells(summary).items():
        ordered = sorted(recs, key=lambda r: int(r["run_seed"]))
        seeds = "|".join(str(int(r["run_seed"])) for r in ordered)
        base = {
            "method": method,
            "pad": pad,
            "regime": regime,
            "ss": ss,
            "confidence": 0.95,
            "n_runs": len(ordered),
            "n_tasks": 1,
            "seeds": seeds,
            "reps": reps,
            "bootstrap_seed": seed,
            "resampling": _RESAMPLING,
        }
        success = np.array([[as_float(r[SUCCESS_COLUMN])] for r in ordered])
        estimates = aggregate_estimates(success, reps=reps, seed=seed)
        for stat in ("iqm", "optimality_gap"):
            est = estimates[stat]
            out.append(
                {
                    **base,
                    "metric": SUCCESS_COLUMN,
                    "statistic": stat,
                    "reported": True,
                    "point": est.point,
                    "ci_lo": est.lo,
                    "ci_hi": est.hi,
                    "note": "success rate in [0, 1]" + ("; gamma = 1" if stat != "iqm" else ""),
                }
            )
        p95 = np.array([[as_float(r[P95_COLUMN])] for r in ordered])
        if np.all(np.isfinite(p95)):
            est = stratified_bootstrap_ci(p95, iqm, reps=reps, seed=seed)
            p95_row = {
                "reported": True,
                "point": est.point,
                "ci_lo": est.lo,
                "ci_hi": est.hi,
                "note": (
                    "m/s model scale; IQM across seeds of each seed's p95 over its touched-down "
                    "episodes; descriptive, not the P3-D1 §4 pooled relative-p95 statistic"
                ),
            }
        else:
            missing = [
                str(int(r["run_seed"]))
                for r in ordered
                if not math.isfinite(as_float(r[P95_COLUMN]))
            ]
            p95_row = {
                "reported": False,
                "point": nan,
                "ci_lo": nan,
                "ci_hi": nan,
                "note": "not reported: no touchdown (no p95) for seed(s) " + "|".join(missing),
            }
        out.append({**base, "metric": P95_COLUMN, "statistic": "iqm", **p95_row})
    return [{c: rec[c] for c in AGGREGATE_COLUMNS} for rec in out]


# --------------------------------------------------------------------------- carry


def verify_lists(episodes_dir: Path = EPISODES_DIR) -> list[dict[str, str]]:
    """Check every list against ``MANIFEST.csv`` (file and content SHA-256).

    Args:
        episodes_dir: Where the lists and the manifest live.

    Returns:
        The manifest rows, every one verified.

    Raises:
        SystemExit: On any mismatch: a result on a list other than the frozen one is not a
            result.
    """
    rows = read_manifest(episodes_dir / MANIFEST_NAME)
    for row in rows:
        path = episodes_dir / row["file"]
        if file_sha256(path) != row["file_sha256"]:
            raise SystemExit(f"{path}: file SHA-256 differs from MANIFEST.csv")
        if content_sha256(read_list(path)) != row["content_sha256"]:
            raise SystemExit(f"{path}: content SHA-256 differs from MANIFEST.csv")
    return rows


def _pad(record: Mapping[str, str]) -> str:
    """Return a record's pad (a summary written before the column existed is all-aft)."""
    return record.get("pad") or "aft"


def _text(value: Any) -> str:
    """Return a value exactly as :func:`rld.eval.report.write_rows` writes it."""
    return repr(float(value)) if isinstance(value, float) else str(value)


_LIST_TEXT_COLUMNS: tuple[str, ...] = ("regime", "ss", "vessel")
_LIST_NUM_COLUMNS: tuple[str, ...] = (
    "index",
    "heading_deg",
    "speed_kn",
    "realization_seed",
    "episode_seed",
    "t0_model_s",
    "init_x_m",
    "init_y_m",
    "init_z_m",
)


def _source_episodes_match(
    rows: Sequence[Mapping[str, str]], episodes: Sequence[ListedEpisode]
) -> bool:
    """Return whether a method's source rows are exactly the listed episodes, in order."""
    if len(rows) != len(episodes):
        return False
    for row, ep in zip(rows, episodes, strict=True):
        if any(row[c] != str(getattr(ep, c)) for c in _LIST_TEXT_COLUMNS):
            return False
        if any(float(row[c]) != float(getattr(ep, c)) for c in _LIST_NUM_COLUMNS):
            return False
        if as_bool(row["in_training_distribution"]) != ep.in_training_distribution:
            return False
    return True


def carry_baselines(
    out_dir: Path,
    sources: Mapping[Path, Sequence[str]],
    episodes: Sequence[ListedEpisode],
    provenance: Mapping[str, str],
    *,
    pad: str = "aft",
    write: bool = True,
) -> dict[str, Any]:
    """Copy baselines' summary rows for the flown cells, verbatim, and verify them.

    For every source directory, ``carried_summary_<name>.csv`` gets the source
    ``summary.csv``'s header line and, in source order, every line whose method is carried,
    whose pad is ``pad`` and whose (regime, sea state) cell is one of ``episodes``' cells.
    Nothing is re-flown and nothing is reformatted. Checks, each recorded:

    1. every written line is byte-identical to a line of the source (by construction, and
       re-read to confirm);
    2. the source's run-wide provenance columns equal ``provenance`` (same package versions,
       configs, list manifest), or its rows would be mislabelled -- a difference raises;
    3. the source's ``episodes.csv`` rows of each carried method on these cells are exactly
       the listed episodes (same regime, SS, index, realization, episode seed, ``t0`` and
       initial position), so the baselines flew the identical list;
    4. re-summarising those source episode rows reproduces every metric column of the
       source summary line as text.

    Args:
        out_dir: Where to write.
        sources: ``{source dir: methods}``, in output order.
        episodes: The listed episodes this run flew.
        provenance: The live deterministic provenance.
        pad: The pad of the carried rows (the frozen aft pad).
        write: Write the files (``False``: check only).

    Returns:
        ``{source name: {file, source SHA-256s, methods, rows, checks}}``.

    Raises:
        SystemExit: If a method is missing from a source, or any check fails.
    """
    cells = list(dict.fromkeys((e.regime, e.ss) for e in episodes))
    info: dict[str, Any] = {}
    for source, methods in sources.items():
        summary_path = source / "summary.csv"
        episodes_path = source / "episodes.csv"
        lines = summary_path.read_text(encoding="utf-8").splitlines(keepends=True)
        records = read_rows(summary_path)
        if len(records) != len(lines) - 1:
            raise SystemExit(f"{summary_path}: a record spans several lines; cannot copy lines")
        missing = sorted(set(methods) - {r["method"] for r in records})
        if missing:
            raise SystemExit(f"{summary_path}: no rows for {missing}")
        for column, value in provenance.items():
            if records[0].get(column) != value:
                raise SystemExit(
                    f"{summary_path}: {column} {records[0].get(column)!r} != live {value!r}"
                )
        keep = [
            i
            for i, r in enumerate(records)
            if r["method"] in methods and _pad(r) == pad and (r["regime"], r["ss"]) in cells
        ]
        chosen = [lines[0], *(lines[i + 1] for i in keep)]
        ep_rows = [
            r
            for r in read_rows(episodes_path)
            if r["method"] in methods and _pad(r) == pad and (r["regime"], r["ss"]) in cells
        ]
        identical_list: dict[str, bool] = {}
        resummarised: dict[str, bool] = {}
        for method in methods:
            mine = [r for r in ep_rows if r["method"] == method]
            identical_list[method] = _source_episodes_match(mine, episodes)
            recomputed = summarise(mine, [method])
            source_recs = [records[i] for i in keep if records[i]["method"] == method]
            resummarised[method] = len(recomputed) == len(source_recs) and all(
                all(_text(new[c]) == old[c] for c in CELL_METRIC_COLUMNS)
                for new, old in zip(recomputed, source_recs, strict=True)
            )
        if not all(identical_list.values()) or not all(resummarised.values()):
            raise SystemExit(
                f"{source}: carried rows fail a check: identical list {identical_list}, "
                f"re-summarised {resummarised}"
            )
        name = source.name
        target = out_dir / f"{CARRIED_PREFIX}{name}.csv"
        if write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("".join(chosen), encoding="utf-8")
        source_lines = set(lines)
        written = target.read_text(encoding="utf-8").splitlines(keepends=True) if write else chosen
        info[name] = {
            "file": target.name,
            "source_summary": _rel(summary_path),
            "source_summary_sha256": _sha256(summary_path),
            "source_episodes": _rel(episodes_path),
            "source_episodes_sha256": _sha256(episodes_path),
            "methods": list(methods),
            "rows": len(chosen) - 1,
            "lines_byte_identical_to_source": all(line in source_lines for line in written)
            and written == chosen,
            "provenance_equal_to_live": True,
            "source_flew_identical_listed_episodes": identical_list,
            "source_summary_reproduced_from_its_episodes": resummarised,
            "note": "copied from the source run, not re-flown",
        }
    return info


def _rel(path: Path) -> str:
    """Return a path relative to the repository root when inside it."""
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


# --------------------------------------------------------------------------- write


def write_learned(
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    runs: Sequence[LearnedRun],
    provenance: Mapping[str, str],
    title: str,
) -> dict[str, Any]:
    """Write ``episodes.csv``, ``summary.csv``, ``seeds.csv``, ``aggregate.csv``, the markdown.

    The carried baseline files must already be in ``out_dir`` (:func:`carry_baselines`).

    Args:
        out_dir: The output directory.
        rows: Learned episode rows.
        runs: The runs the rows came from.
        provenance: Deterministic provenance columns.
        title: The markdown title.

    Returns:
        ``{file name: SHA-256}`` of every file written.
    """
    run_cols = {(r.method, r.seed): r.columns() for r in runs}
    summary, columns = summarise_learned(rows, run_cols, provenance)
    spread, spread_cols = seed_spread(summary)
    write_rows(out_dir / "episodes.csv", rows, EPISODE_COLUMNS)
    write_rows(out_dir / "summary.csv", summary, columns)
    write_rows(out_dir / "seeds.csv", spread, spread_cols)
    write_rows(out_dir / "aggregate.csv", aggregate_rows(summary), AGGREGATE_COLUMNS)
    (out_dir / MARKDOWN_NAME).write_text(render_learned(out_dir, title), encoding="utf-8")
    names = ("episodes.csv", "summary.csv", "seeds.csv", "aggregate.csv", MARKDOWN_NAME)
    return {name: _sha256(out_dir / name) for name in names}


def rederive(out_dir: Path, title: str) -> dict[str, bool]:
    """Recompute every derived file from ``episodes.csv`` in memory and compare bytes.

    ``summary.csv``'s run and provenance columns are taken from the committed summary (they
    are inputs, not derived); every metric, ``seeds.csv``, ``aggregate.csv`` and the
    markdown are recomputed.

    Args:
        out_dir: A written e05-style directory.
        title: The markdown title.

    Returns:
        ``{file name: byte-identical}``.
    """
    rows = read_rows(out_dir / "episodes.csv")
    committed = read_rows(out_dir / "summary.csv")
    header = (out_dir / "summary.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    extra = header[header.index(RUN_COLUMNS[0]) :]
    run_cols = {
        (r["method"], int(r["run_seed"])): {c: r[c] for c in RUN_COLUMNS} for r in committed
    }
    prov = {c: committed[0][c] for c in extra if c not in RUN_COLUMNS}
    summary, columns = summarise_learned(rows, run_cols, prov)
    spread, spread_cols = seed_spread(summary)
    scratch: dict[str, str] = {}
    for name, recs, cols in (
        ("summary.csv", summary, columns),
        ("seeds.csv", spread, spread_cols),
        ("aggregate.csv", aggregate_rows(summary), list(AGGREGATE_COLUMNS)),
    ):
        scratch[name] = _csv_text(recs, cols)
    result = {
        name: text == (out_dir / name).read_text(encoding="utf-8") for name, text in scratch.items()
    }
    result[MARKDOWN_NAME] = render_learned(out_dir, title) == (out_dir / MARKDOWN_NAME).read_text(
        encoding="utf-8"
    )
    return result


def _csv_text(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    """Return the CSV text :func:`rld.eval.report.write_rows` would write."""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for rec in records:
        writer.writerow([_text(rec[c]) if isinstance(rec[c], float) else rec[c] for c in columns])
    return buffer.getvalue()


# --------------------------------------------------------------------------- render


def _pct(value: float) -> str:
    """Format a rate in ``[0, 1]`` as a percentage with one decimal, or ``–`` for NaN."""
    return "–" if value != value else f"{100.0 * value:.1f}"


def _num(value: float, digits: int) -> str:
    """Format a number with fixed decimals, or ``–`` for NaN."""
    return "–" if value != value else f"{value:.{digits}f}"


def _success(rec: Mapping[str, str]) -> str:
    """Return ``"rate [lo, hi] (k/N)"`` in percent."""
    return (
        f"{_pct(as_float(rec['success_rate']))} [{_pct(as_float(rec['success_wilson_lo']))}, "
        f"{_pct(as_float(rec['success_wilson_hi']))}] ({rec['n_success']}/{rec['n_episodes']})"
    )


def _steps(value: str) -> str:
    """Format an integer step count with thin grouping, e.g. ``10 000 000``."""
    return f"{int(value):,}".replace(",", " ")


def _carried(out_dir: Path) -> list[tuple[str, dict[str, str]]]:
    """Return ``[(source name, record), ...]`` from the carried files, registry order."""
    found: list[tuple[str, dict[str, str]]] = []
    for path in sorted(out_dir.glob(f"{CARRIED_PREFIX}*.csv")):
        source = path.stem.removeprefix(CARRIED_PREFIX)
        found += [(source, rec) for rec in read_rows(path)]
    order = list(REGISTRY)

    def rank(item: tuple[str, dict[str, str]]) -> int:
        method = item[1]["method"]
        return order.index(method) if method in order else len(order)

    return sorted(found, key=rank)


def _learned_label(method: str, recs: Sequence[Mapping[str, str]]) -> str:
    """Return a learned method's label with its budget, read from the summary."""
    budgets = sorted({int(r["run_total_steps"]) for r in recs})
    text = " / ".join(_steps(str(b)) for b in budgets)
    return f"{method} (learned; {text} env steps per seed)"


def _baseline_label(method: str, privileged: bool, source: str) -> str:
    """Return a carried baseline's label: registry label, 'carried', its source."""
    return f"{method_label(method, privileged)} [baseline, carried from `results/{source}`]"


def render_learned(out_dir: Path, title: str) -> str:
    """Render ``success_vs_seastate.md`` from the CSVs in ``out_dir`` and nothing else.

    Reads ``summary.csv``, ``seeds.csv``, ``aggregate.csv`` and every
    ``carried_summary_*.csv``. Sections: caveats and notes; the headline success table per
    sea state (every learned seed with its Wilson CI and k/N, the IQM row with its
    stratified-bootstrap CI, the seed range, then each carried baseline); the aggregates;
    the seed-to-seed spread; the full outcome breakdown and touchdown audit for every
    (method, seed, sea state) including the baselines; termination-reason counts.

    Args:
        out_dir: The directory holding the CSVs.
        title: The document title.

    Returns:
        The markdown text.

    Raises:
        ValueError: If the text contains a forbidden phrase (P3-D4).
    """
    summary = read_rows(out_dir / "summary.csv")
    spread = read_rows(out_dir / "seeds.csv")
    agg = read_rows(out_dir / "aggregate.csv")
    carried = _carried(out_dir)
    methods = list(dict.fromkeys(r["method"] for r in summary))
    regimes = list(dict.fromkeys(r["regime"] for r in summary))
    pads = list(dict.fromkeys(r["pad"] for r in summary))
    base_methods = list(dict.fromkeys(r["method"] for _, r in carried))
    by_method = {m: [r for r in summary if r["method"] == m] for m in methods}
    agg_by = {
        (r["method"], r["pad"], r["regime"], r["ss"], r["metric"], r["statistic"]): r for r in agg
    }
    spread_by = {(r["method"], r["pad"], r["regime"], r["ss"]): r for r in spread}
    carried_by = {(r["method"], _pad(r), r["regime"], r["ss"]): (s, r) for s, r in carried}

    lines: list[str] = [f"# {title}", ""]
    lines += [f"- {c}" for c in CAVEATS]
    budgets = "; ".join(
        f"`{m}` {' / '.join(_steps(b) for b in sorted({r['run_total_steps'] for r in recs}))} "
        f"env steps per seed (checkpoint `{recs[0]['run_ckpt']}`, "
        f"{' / '.join(_steps(s) for s in sorted({r['run_ckpt_steps'] for r in recs}))} steps)"
        for m, recs in by_method.items()
    )
    lines += [
        "- Success = all four frozen criteria (`configs/env/success.yaml`); rates in %, "
        "Wilson 95 % CI in brackets, then k/N. Success is never pooled across sea states.",
        "- Learned rows: each training seed's checkpoint flown as the batch-1 `LearnedPolicy` "
        "(deterministic action, `VecNormalize` statistics frozen, no ship-motion feed, not "
        "privileged) on the identical listed episodes. `seed` is the training seed.",
        f"- **Unequal training budgets:** {budgets}. P3-D1 §5 sets them so; every PPO-vs-SAC "
        "reading carries that confound.",
        "- `IQM` is rliable's interquartile mean of the per-seed success rates of the cell "
        "(with 5 seeds, the mean of the middle 3); its CI is the stratified bootstrap of "
        "P3-D1 §4 (2 000 replicates, seed 20260926, seeds resampled within the cell, cells "
        "never resampled). One cell is one task: nothing is pooled across sea states. "
        "That CI reflects seed-to-seed variation only, not episode sampling: when every seed "
        "scores the same it collapses to a point (e.g. [100.0, 100.0]), and each seed's Wilson "
        "CI is then the episode-level uncertainty.",
        "- Baselines are **carried, not re-run**: their rows are line-for-line copies of the "
        "committed summaries named in each label (`carried_summary_*.csv`). They are "
        "deterministic and have one run (`seed` `–`).",
        "- **`id` SS6 is outside every method's training distribution**: no learned method "
        "trained on SS6 (curriculum SS3 → SS5) and the baselines were tuned on SS3–SS5; the "
        "`in-dist` column is 0.000 there. At SS3–SS5, `in-dist` < 1 because 90 deg-heading "
        "episodes are outside the development pool (P3-D2).",
        "- `oracle_gated` is privileged: it reads the true future deck motion. It is a "
        "commit-timing oracle, not a bound on success, and never a deployable result.",
        "- **Measurement caveats from the reward-hacking audit (P5-D14; `results/audit/`)**, "
        "both frozen definitions applied to every method alike. (1) The recorded closing "
        "speed is read after the first contact substep's solver impulse, about 7 % below the "
        "speed one substep earlier; in a re-flight sample 15 of 445 `sac` successes (3.4 %) "
        "arrived above 0.5 m/s one substep before contact (`ppo` 0 of 500). (2) Up to 41 `sac` "
        "successes (1 / 3 / 13 / 24 at SS3–SS6, per 1 000 seed-episodes) had an unloaded "
        "stretch while tunnelled, and whether they would have lost contact for more than the "
        "50 ms grace without the overlap is not shown (`ppo`: 2 at SS6; "
        "`pid_feedforward_lowvz_cut`: 1 at SS6). Both can only raise "
        "`sac`'s success, by at most a few points at SS5/SS6.",
        "- No hypothesis (H1–H5) is scored here and no method contrast is tested; the spread "
        "and aggregates are descriptive.",
        "- Rendered from `summary.csv`, `seeds.csv`, `aggregate.csv` and "
        "`carried_summary_*.csv` by `rld.eval.learned`; do not edit by hand.",
        "",
    ]
    for pad in pads:
        for regime in regimes:
            sea_states = list(
                dict.fromkeys(r["ss"] for r in summary if r["regime"] == regime and r["pad"] == pad)
            )
            where = f"{regime}, {pad} pad"
            lines += [f"## {where}: success per sea state", ""]
            lines += ["| method | seed | " + " | ".join(sea_states) + " |"]
            lines += ["|---|---|" + "---|" * len(sea_states)]
            for method in methods:
                recs = [r for r in by_method[method] if r["regime"] == regime and r["pad"] == pad]
                label = _learned_label(method, recs)
                by_ss_seed = {(r["ss"], r["run_seed"]): r for r in recs}
                seeds = list(dict.fromkeys(r["run_seed"] for r in recs))
                cells = []
                for ss in sea_states:
                    a = agg_by.get((method, pad, regime, ss, SUCCESS_COLUMN, "iqm"))
                    cells.append(
                        "missing"
                        if a is None
                        else f"**{_pct(as_float(a['point']))}** [{_pct(as_float(a['ci_lo']))}, "
                        f"{_pct(as_float(a['ci_hi']))}]"
                    )
                lines.append(f"| {label} | IQM ({len(seeds)} seeds) | " + " | ".join(cells) + " |")
                for seed in seeds:
                    row = [
                        _success(by_ss_seed[(ss, seed)]) if (ss, seed) in by_ss_seed else "missing"
                        for ss in sea_states
                    ]
                    lines.append(f"| {method} | {seed} | " + " | ".join(row) + " |")
                rng = []
                for ss in sea_states:
                    s = spread_by.get((method, pad, regime, ss))
                    lo_hi = "missing"
                    if s is not None:
                        lo_hi = f"{_pct(as_float(s['success_min']))}–"
                        lo_hi += _pct(as_float(s["success_max"]))
                    rng.append(lo_hi)
                lines.append(f"| {method} | min–max over seeds | " + " | ".join(rng) + " |")
            for method in base_methods:
                row = []
                label = method
                for ss in sea_states:
                    hit = carried_by.get((method, pad, regime, ss))
                    if hit is None:
                        row.append("missing")
                        continue
                    source, rec = hit
                    label = _baseline_label(method, as_bool(rec["privileged"]), source)
                    row.append(_success(rec))
                lines.append(f"| {label} | – | " + " | ".join(row) + " |")
            lines.append("")

            lines += [f"### {where}: aggregates across training seeds (`aggregate.csv`)", ""]
            lines += [
                "| method | SS | runs | IQM success % [95 % CI] | optimality gap, points "
                "[95 % CI] | IQM of per-seed p95 closing speed, m/s [95 % CI] |",
                "|---|---|---|---|---|---|",
            ]
            for method in methods:
                for ss in sea_states:
                    i = agg_by.get((method, pad, regime, ss, SUCCESS_COLUMN, "iqm"))
                    g = agg_by.get((method, pad, regime, ss, SUCCESS_COLUMN, "optimality_gap"))
                    p = agg_by.get((method, pad, regime, ss, P95_COLUMN, "iqm"))
                    if i is None or g is None or p is None:
                        lines.append(f"| {method} | {ss} | missing | | | |")
                        continue
                    p_text = (
                        f"{_num(as_float(p['point']), 3)} [{_num(as_float(p['ci_lo']), 3)}, "
                        f"{_num(as_float(p['ci_hi']), 3)}]"
                        if as_bool(p["reported"])
                        else p["note"]
                    )
                    lines.append(
                        f"| {method} | {ss} | {i['n_runs']} | {_pct(as_float(i['point']))} "
                        f"[{_pct(as_float(i['ci_lo']))}, {_pct(as_float(i['ci_hi']))}] | "
                        f"{_pct(as_float(g['point']))} [{_pct(as_float(g['ci_lo']))}, "
                        f"{_pct(as_float(g['ci_hi']))}] | {p_text} |"
                    )
            lines += [
                "",
                "The p95 IQM is descriptive: each seed's p95 is over its touched-down episodes "
                "only, and it is not P3-D1 §4's pooled relative-p95 contrast statistic.",
                "",
            ]

            seed_cols = sorted(
                {
                    c.removeprefix("success_rate_seed")
                    for c in spread[0]
                    if c.startswith("success_rate_seed")
                },
                key=int,
            )
            lines += [f"### {where}: seed-to-seed spread (`seeds.csv`)", ""]
            lines += [
                "| method | SS | success % by seed ("
                + ", ".join(seed_cols)
                + ") | min | max | mean | SD | p95 closing speed m/s by seed | min | max | mean "
                "| SD |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|",
            ]
            for method in methods:
                for ss in sea_states:
                    s = spread_by.get((method, pad, regime, ss))
                    if s is None:
                        lines.append(f"| {method} | {ss} | missing |||||||||| ")
                        continue
                    by_seed = " / ".join(
                        _pct(as_float(s[f"success_rate_seed{k}"])) for k in seed_cols
                    )
                    p95s = " / ".join(
                        _num(as_float(s[f"{P95_COLUMN}_seed{k}"]), 3) for k in seed_cols
                    )
                    lines.append(
                        f"| {method} | {ss} | {by_seed} | {_pct(as_float(s['success_min']))} | "
                        f"{_pct(as_float(s['success_max']))} | {_pct(as_float(s['success_mean']))} "
                        f"| {_pct(as_float(s['success_sd']))} | {p95s} | "
                        f"{_num(as_float(s['p95_min_m_s']), 3)} | "
                        f"{_num(as_float(s['p95_max_m_s']), 3)} "
                        f"| {_num(as_float(s['p95_mean_m_s']), 3)} | "
                        f"{_num(as_float(s['p95_sd_m_s']), 3)} |"
                    )
            lines += [
                "",
                "SD is the sample SD across seeds (ddof = 1), in points for success. A p95 "
                "statistic is over the seeds with a touchdown (`p95_n_seeds_finite` in the CSV).",
                "",
            ]

            lines += [f"### {where}: outcome breakdown and touchdown audit", ""]
            cols = [
                "method",
                "seed",
                "SS",
                "N",
                *OUTCOMES,
                "v_n p50 / p95 (m/s)",
                "v_z p50 / p95 (m/s)",
                "lat p50 / p95 (m)",
                "tilt p95 (deg)",
                "t_td p50 (s)",
                "effort",
                "jerk",
                "disagree",
                "tunnel",
                "max depth (mm)",
                "in-dist",
            ]
            lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
            entries: list[tuple[str, str, Mapping[str, str]]] = []
            for method in methods:
                recs = [r for r in by_method[method] if r["regime"] == regime and r["pad"] == pad]
                for seed in dict.fromkeys(r["run_seed"] for r in recs):
                    for ss in sea_states:
                        for r in recs:
                            if r["run_seed"] == seed and r["ss"] == ss:
                                entries.append((method, seed, r))
            for method in base_methods:
                for ss in sea_states:
                    hit = carried_by.get((method, pad, regime, ss))
                    if hit is not None:
                        entries.append(
                            (
                                _baseline_label(method, as_bool(hit[1]["privileged"]), hit[0]),
                                "–",
                                hit[1],
                            )
                        )
            for name, seed, erec in entries:
                values = [
                    name,
                    seed,
                    erec["ss"],
                    erec["n_episodes"],
                    *(_num(as_float(erec[f"frac_{o}"]), 3) for o in OUTCOMES),
                    f"{_num(as_float(erec['rel_vz_normal_p50_m_s']), 3)} / "
                    f"{_num(as_float(erec['rel_vz_normal_p95_m_s']), 3)}",
                    f"{_num(as_float(erec['rel_vz_world_z_p50_m_s']), 3)} / "
                    f"{_num(as_float(erec['rel_vz_world_z_p95_m_s']), 3)}",
                    f"{_num(as_float(erec['lateral_p50_m']), 3)} / "
                    f"{_num(as_float(erec['lateral_p95_m']), 3)}",
                    _num(as_float(erec["rel_tilt_p95_deg"]), 1),
                    _num(as_float(erec["time_to_touchdown_p50_s"]), 2),
                    _num(as_float(erec["effort_mean_sq"]), 3),
                    _num(as_float(erec["action_jerk_mean"]), 3),
                    f"{erec['disagreement_n']}/{erec['n_episodes']}",
                    erec["tunnelling_n"],
                    _num(-1000.0 * as_float(erec["max_penetration_m"]), 2),
                    _num(as_float(erec["frac_in_training_distribution"]), 3),
                ]
                lines.append("| " + " | ".join(str(v) for v in values) + " |")
            lines += [
                "",
                "`v_n` is the touchdown closing speed along the deck normal (criterion 1), "
                "`v_z` the same along world z, both over touched-down episodes only; `lat` the "
                "lateral offset from the pad centre; `tilt` the drone-versus-deck-normal tilt; "
                "`t_td` the time to first contact; `effort` the mean ||a||^2 and `jerk` the mean "
                "||a_t - a_(t-1)|| in normalised action units; `disagree` the analytic-vs-contact "
                "touchdown detector disagreements; `tunnel` the episodes whose penetration "
                "exceeded 5 mm; `max depth` the deepest penetration in the cell, in mm "
                "(the negated most-negative separation, `max_penetration_m` in the CSV).",
                "",
                f"### {where}: termination reasons (counts)",
                "",
            ]
            tcols = ["method", "seed", "SS", "N", *TERMINATION_REASONS]
            lines += ["| " + " | ".join(tcols) + " |", "|" + "---|" * len(tcols)]
            for name, seed, erec in entries:
                values = [
                    name,
                    seed,
                    erec["ss"],
                    erec["n_episodes"],
                    *(erec[f"n_reason_{t}"] for t in TERMINATION_REASONS),
                ]
                lines.append("| " + " | ".join(str(v) for v in values) + " |")
            lines.append("")
    text = "\n".join(lines)
    lowered = text.lower()
    for phrase in FORBIDDEN_RENDERED_PHRASES:
        if phrase in lowered:
            raise ValueError(f"rendered output contains the forbidden phrase {phrase!r}")
    return text


# --------------------------------------------------------------------------- main

DEFAULT_TITLE = (
    "e05 — PPO and SAC final checkpoints on the frozen `id` list (aft pad): success versus "
    "sea state, with the classical baselines carried beside them"
)


def _parse_carry(items: Sequence[str] | None) -> dict[Path, tuple[str, ...]]:
    """Parse ``SOURCE_DIR=METHOD[,METHOD...]``; ``None`` gives :data:`DEFAULT_CARRY`."""
    if items is None:
        return {RESULTS_ROOT / name: methods for name, methods in DEFAULT_CARRY.items()}
    out: dict[Path, tuple[str, ...]] = {}
    for item in items:
        src, sep, methods = item.partition("=")
        names = tuple(m for m in methods.split(",") if m)
        if not sep or not names:
            raise SystemExit(f"--carry {item!r}: expected SOURCE_DIR=METHOD[,METHOD...]")
        out[Path(src).resolve()] = names
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line (see the module docstring)."""
    parser = argparse.ArgumentParser(
        prog="python -m rld.eval.learned",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--learned", nargs="+", default=[], metavar="NAME=RUN_DIR[,RUN_DIR...]")
    parser.add_argument("--ckpt", default="final", help="checkpoint of every run (default final)")
    parser.add_argument("--list", default="id", help="the committed list to fly (default id)")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_ROOT / "e05")
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument(
        "--carry",
        nargs="+",
        default=None,
        metavar="SOURCE_DIR=METHOD[,METHOD...]",
        help="baselines to carry (default: results/e01 five, results/e01_lowvz_cut lowvz_cut)",
    )
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--render-only", action="store_true", help="re-render the markdown")
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-derive summary/seeds/aggregate/markdown from episodes.csv; exit 1 on a diff",
    )
    parser.add_argument(
        "--per-cell", type=int, default=None, help="scratch only: first K episodes per cell"
    )
    return parser.parse_args(argv)


def _first_per_cell(episodes: Sequence[ListedEpisode], k: int) -> list[ListedEpisode]:
    """Return the first ``k`` listed episodes of every (regime, sea state) cell."""
    seen: dict[tuple[str, str], int] = {}
    out: list[ListedEpisode] = []
    for ep in episodes:
        cell = (ep.regime, ep.ss)
        if seen.get(cell, 0) < k:
            seen[cell] = seen.get(cell, 0) + 1
            out.append(ep)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate learned runs on a committed list and write the e05 tables.

    Args:
        argv: Arguments (``None``: ``sys.argv``).

    Returns:
        Process exit status.
    """
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    if args.render_only:
        (out_dir / MARKDOWN_NAME).write_text(render_learned(out_dir, args.title), encoding="utf-8")
        return 0
    if args.check:
        result = rederive(out_dir, args.title)
        print(json.dumps(result, indent=2))
        return 0 if all(result.values()) else 1
    started_utc = environment_provenance(REPO_ROOT)["timestamp_utc"]
    git = git_state(REPO_ROOT)
    committed = out_dir.resolve().is_relative_to(RESULTS_ROOT)
    if committed and args.per_cell is not None:
        raise SystemExit("--per-cell is scratch-only, not under results/")
    learned = parse_learned(args.learned)
    if not learned:
        raise SystemExit("nothing to evaluate: give --learned NAME=RUN_DIR[,RUN_DIR...]")
    # Torch inside every spawned worker: one thread (the batch-1 MLP gains nothing from more,
    # and 24 workers x all cores would oversubscribe). Children inherit this environment.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    # 1. The frozen lists, before anything else.
    manifest = verify_lists(args.episodes_dir)
    entry = next((m for m in manifest if m["file"] == f"{args.list}.parquet"), None)
    if entry is None:
        raise SystemExit(f"list {args.list!r} is not in the manifest")
    episodes = read_list(args.episodes_dir / entry["file"])
    if args.per_cell is not None:
        episodes = _first_per_cell(episodes, args.per_cell)

    # 2. Runs and carried sources, checked before anything is flown.
    runs = order_runs(
        [inspect_run(name, path, args.ckpt) for name, paths in learned for path in paths]
    )
    cfgs = load_eval_configs()
    provenance = live_provenance(cfgs, args.episodes_dir)
    carry = _parse_carry(args.carry)
    carry_baselines(out_dir, carry, episodes, provenance, write=False)
    for run in runs:
        print(
            f"{run.method} seed {run.seed}: {run.label_dir()} ckpt {run.ckpt} "
            f"({run.ckpt_steps} steps), trained at {run.train_git_sha[:7]}",
            flush=True,
        )
    print(f"flying {len(runs)} runs x {len(episodes)} episodes, {args.workers} workers", flush=True)

    # 3. Fly.
    t0 = time.perf_counter()
    rows = evaluate_runs(runs, episodes, cfgs, workers=args.workers, chunk=args.chunk)
    wall_s = time.perf_counter() - t0
    after = live_provenance(cfgs, args.episodes_dir)
    if after != provenance:
        raise SystemExit("configs or lists changed during the run; not writing")

    # 4. Write.
    out_dir.mkdir(parents=True, exist_ok=True)
    carried = carry_baselines(out_dir, carry, episodes, provenance)
    hashes = write_learned(out_dir, rows, runs, provenance, args.title)
    for name in (f"{CARRIED_PREFIX}{src.name}.csv" for src in carry):
        hashes[name] = _sha256(out_dir / name)
    check = rederive(out_dir, args.title)
    env = environment_provenance(REPO_ROOT)
    run_info = {
        "started_utc": started_utc,
        **{k: env[k] for k in VOLATILE_KEYS},
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "workers": args.workers,
        "chunk": args.chunk,
        "wall_s": round(wall_s, 1),
        "episodes": len(rows),
        **git,
        "list": {
            "name": args.list,
            "file": _rel(args.episodes_dir / entry["file"]),
            "rows": len(episodes),
            "per_cell": args.per_cell,
            "file_sha256": entry["file_sha256"],
            "content_sha256": entry["content_sha256"],
            "manifest_sha256": _sha256(args.episodes_dir / MANIFEST_NAME),
            "manifest_check": f"{2 * len(manifest)}/{2 * len(manifest)} hashes OK",
        },
        "runs": [
            {
                "method": r.method,
                "seed": r.seed,
                "run_dir": r.label_dir(),
                "ckpt": r.ckpt,
                "ckpt_steps": r.ckpt_steps,
                "algo": r.algo,
                "total_steps": r.total_steps,
                "train_git_sha": r.train_git_sha,
                "train_git_dirty_paths": list(r.train_git_dirty_paths),
                "resumed_from": r.resumed_from,
                **dict(r.digests),
            }
            for r in runs
        ],
        "policy_path": (
            "rld.eval.runner.callable_spec(method, functools.partial("
            "rld.rl.train.build_policy, run_dir=<run>, ckpt=<ckpt>), run_seed=<training seed>)"
        ),
        "carried": carried,
        "rederived_byte_identical": check,
        "files_sha256": hashes,
        "bootstrap": {"reps": STRATIFIED_REPS, "seed": DEFAULT_BOOTSTRAP_SEED},
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    for rec in read_rows(out_dir / "summary.csv"):
        print(
            f"{rec['method']:>4s} s{rec['run_seed']} {rec['regime']} {rec['ss']}: "
            f"success={_success(rec)} crash={rec['frac_crash']} off={rec['frac_off_pad']} "
            f"hard={rec['frac_hard_landing']} bounce={rec['frac_bounce']} "
            f"timeout={rec['frac_timeout']} tun={rec['tunnelling_n']} dis={rec['disagreement_n']}"
        )
    print(f"{len(rows)} episode rows -> {out_dir}; {wall_s:.1f} s wall, {args.workers} workers")
    print(f"re-derived byte-identical: {check}")
    return 0 if all(check.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
