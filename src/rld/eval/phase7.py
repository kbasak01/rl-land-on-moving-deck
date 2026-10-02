r"""Phase 7: fly the frozen matrix and the shift and ablation arms; verify what is written.

Authority: P7-D1 (``docs/protocol.md``), on top of P3-D1 and P3-D4. The arms and their
conditions are data in :mod:`rld.eval.arms`; this module flies a condition, writes it, and
re-verifies it. Hypotheses are scored by :mod:`rld.eval.hypotheses` from what is written here;
``results/results.md`` is rendered by :func:`rld.eval.report.render_results`.

What one condition writes (``results/e07/<arm>[/<condition>]/``)
---------------------------------------------------------------
* ``episodes.csv`` -- every flown (method, seed, pad, episode), learned runs first, then the
  baselines flown in this condition; :data:`~rld.eval.runner.EPISODE_COLUMNS` **unchanged**
  (an arm's condition is never an episode column).
* ``summary.csv`` -- learned runs, one row per (method, seed, pad, regime, sea state):
  :func:`rld.eval.learned.summarise_learned` (P3-D1 §3 metrics, run provenance, deterministic
  environment provenance), then :data:`ARM_COLUMNS`, the condition: motion, noise sigmas and
  latency in ms and in control steps, lambda, the start check, the hashes of the RMS table
  and of the list manifest. ``eval_deck_motion`` is the motion actually flown.
* ``seeds.csv`` / ``aggregate.csv`` -- :func:`rld.eval.learned.seed_spread` and
  :func:`rld.eval.learned.aggregate_rows` (P3-D1 §4: IQM and optimality gap, 2 000-replicate
  stratified bootstrap, seed 20260926) over the cells that were flown.
* ``baselines_summary.csv`` -- baselines flown here (:func:`rld.eval.report.summarise`, with
  each controller's YAML and resolved-config SHA-256), then :data:`ARM_COLUMNS`.
* ``carried_summary_<source>.csv`` -- baselines **carried line for line** from a committed
  run on the identical lists (:func:`rld.eval.learned.carry_baselines`: byte-identical lines,
  live provenance, identical listed episodes, summary re-derived from the source episodes).
* ``run_info.json`` -- the non-deterministic facts: git state, worker count, chunk, wall
  time, the run table with checkpoint digests before and after the flight, list hashes, the
  carry checks, any reproduction check, the worker-count re-flight checks, and
  ``"complete": true`` written last.

Resumable and verified
----------------------
A condition whose ``run_info.json`` says ``complete`` is **verified, not re-flown**:
:func:`verify_condition` re-derives every CSV above from ``episodes.csv`` (the run, controller
and provenance columns are read from the committed files, the condition columns are recomputed
from :mod:`rld.eval.arms`), re-carries the carried files into a scratch directory, and
compares bytes. ``--check`` does exactly that for every requested condition and writes
nothing. A directory without ``complete`` is a crashed flight and is flown again.

Pre-flight and provenance
-------------------------
Every list directory flown is checked against its manifest (file and content SHA-256) first;
every learned run is inspected (``done``, method, seed) and its policy built once and checked
(:func:`rld.eval.learned.check_policies`); every run's ``config.yaml``, ``model.zip``,
``vecnormalize.pkl`` and ``checkpoint.json`` is hashed before and after each condition and
before and after the whole invocation (``results/e07/run_info.json``, one entry appended per
invocation). ``make eval`` runs ``scripts/make_episodes.py --check`` before this module.

Worker-count check (P3-D4 carry item)
-------------------------------------
``--refly KEY --refly-workers W --refly-per-cell K`` re-flies the first ``K`` listed episodes
per cell of every spec of a written condition at ``W`` workers, compares every row with the
committed line, and appends the result to that condition's ``run_info.json``
(``worker_count_checks``).

Entry point: ``scripts/eval_phase7.py`` (argparse wrapper) -> :func:`main`.

Units: speeds metres per second model scale, lengths metres model scale, times seconds model
scale; sigmas metres and metres per second model scale; latency milliseconds model scale.
"""

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from rld.config import REPO_ROOT
from rld.envs.noise import PerceptionNoise
from rld.eval.arms import (
    ARMS,
    BASELINES,
    E07_DIR,
    EPISODES_MSS_DIR,
    LEARNED_METHODS,
    LEARNED_SEEDS,
    RESULTS_DIR,
    RUNS_ROOT,
    Condition,
    conditions,
    run_dir,
)
from rld.eval.controller_config import PROVENANCE_COLUMNS, controller_provenance_columns
from rld.eval.envs import (
    EvalConfigs,
    MotionKind,
    load_eval_configs,
    motion_for,
    with_lambda,
    with_noise,
)
from rld.eval.episodes import MANIFEST_NAME, ListedEpisode, read_list
from rld.eval.learned import (
    AGGREGATE_COLUMNS,
    RUN_COLUMNS,
    RUN_POLICY_COLUMNS,
    LearnedRun,
    _csv_text,
    _run_digests,
    _sha256,
    aggregate_rows,
    carry_baselines,
    check_policies,
    inspect_run,
    learned_spec,
    live_provenance,
    seed_spread,
    summarise_learned,
    verify_lists,
)
from rld.eval.report import SkipRecord, read_rows, summarise, summary_columns, write_rows
from rld.eval.reproduce import compare_to_reference, git_state
from rld.eval.runner import (
    DEFAULT_CHUNK,
    DEFAULT_WORKERS,
    EPISODE_COLUMNS,
    EvalArm,
    PolicySpec,
    StartCheck,
    controller_spec,
    run_arms,
    split_runnable,
)
from rld.provenance import environment_provenance
from rld.rl.motion import DECK_STATS_SEEDS

__all__ = [
    "ARM_COLUMNS",
    "ConditionUnavailableError",
    "arm_columns",
    "condition_configs",
    "condition_episodes",
    "derive_condition",
    "fly_condition",
    "inspect_runs",
    "lambda_feasibility",
    "main",
    "refly_check",
    "verify_condition",
]

#: The condition columns appended to every Phase 7 summary row (never to an episode row).
ARM_COLUMNS: tuple[str, ...] = (
    "arm",
    "condition",
    "arm_motion",
    "start_check",
    "lam_inverse",
    "noise_enabled",
    "noise_position_sigma_m",
    "noise_velocity_sigma_m_s",
    "noise_latency_ms_configured",
    "noise_latency_steps",
    "noise_latency_ms_effective",
    "noise_hold_freq_hz",
    "sinusoid_rms_source_sha256",
    "episode_lists_manifest_sha256",
)

#: Provenance keys that vary between runs of identical results; they go to run_info.json.
VOLATILE_KEYS: tuple[str, ...] = ("omp_num_threads", "cpu_count", "host", "timestamp_utc")

#: The written files of a condition, in write order (carried files are added per source).
DERIVED_FILES: tuple[str, ...] = (
    "summary.csv",
    "seeds.csv",
    "aggregate.csv",
    "baselines_summary.csv",
)


class ConditionUnavailableError(RuntimeError):
    """An optional condition cannot be flown yet (its lists or its motion source are missing)."""


# --------------------------------------------------------------------------- condition setup


def condition_configs(cond: Condition, cfgs: EvalConfigs) -> EvalConfigs:
    """Return the configs a condition flies under (noise or lambda applied; else ``cfgs``).

    Args:
        cond: The condition.
        cfgs: The committed configs.

    Returns:
        :func:`~rld.eval.envs.with_noise` / :func:`~rld.eval.envs.with_lambda` of ``cfgs``,
        or ``cfgs`` itself for a clean, project-lambda condition.
    """
    out = cfgs
    if cond.noise is not None:
        out = with_noise(
            out,
            position_sigma_m=cond.noise.sigma_p_m,
            velocity_sigma_m_s=cond.noise.sigma_v_m_s,
            latency_ms=cond.noise.latency_ms,
        )
    if cond.lam_inverse != cfgs.scaling.lam_inverse:
        out = with_lambda(out, cond.lam_inverse)
    return out


def _latency_steps(cfgs: EvalConfigs) -> int:
    """Return the control steps the environment's noise model quantises the latency to."""
    noise = PerceptionNoise(cfgs.noise, np.random.default_rng(0), float(cfgs.landing.ctrl_freq_hz))
    return int(noise.latency_steps)


def arm_columns(cond: Condition, cfgs_arm: EvalConfigs, regime: str) -> dict[str, str]:
    """Return the :data:`ARM_COLUMNS` of one summary row, as text.

    Args:
        cond: The condition.
        cfgs_arm: The configs it flies under (:func:`condition_configs`).
        regime: The row's regime (the MSS arm's motion depends on it).

    Returns:
        ``{column: text}``. The latency is given as configured, as whole control steps
        (asserted equal to the condition's stated steps) and as the effective milliseconds.

    Raises:
        RuntimeError: If the environment would quantise the latency to another step count
            than P7-D1 states for the condition.
    """
    noise = cfgs_arm.noise
    steps = _latency_steps(cfgs_arm) if noise.enabled else 0
    if cond.noise is not None and steps != cond.noise.latency_steps:
        raise RuntimeError(
            f"{cond.key}: latency {noise.latency_ms} ms quantises to {steps} steps, "
            f"P7-D1 states {cond.noise.latency_steps}"
        )
    motion = cond.motion_for_regime(regime)
    manifest = cond.episodes_dir / MANIFEST_NAME
    return {
        "arm": cond.arm,
        "condition": cond.name or cond.arm,
        "arm_motion": motion,
        "start_check": cond.start_check,
        "lam_inverse": repr(float(cfgs_arm.scaling.lam_inverse)),
        "noise_enabled": str(bool(noise.enabled)),
        "noise_position_sigma_m": repr(float(noise.position_sigma_m)),
        "noise_velocity_sigma_m_s": repr(float(noise.velocity_sigma_m_s)),
        "noise_latency_ms_configured": repr(float(noise.latency_ms)),
        "noise_latency_steps": str(steps),
        "noise_latency_ms_effective": repr(1000.0 * steps / float(cfgs_arm.landing.ctrl_freq_hz)),
        "noise_hold_freq_hz": repr(float(noise.hold_freq_hz)),
        "sinusoid_rms_source_sha256": _sha256(DECK_STATS_SEEDS) if motion == "sinusoid" else "",
        "episode_lists_manifest_sha256": _sha256(manifest) if manifest.is_file() else "",
    }


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


def condition_episodes(
    cond: Condition, per_cell: int | None = None
) -> tuple[list[ListedEpisode], dict[str, Any]]:
    """Verify a condition's list directory against its manifest and read its lists.

    Args:
        cond: The condition.
        per_cell: Scratch only: the first ``per_cell`` episodes of every cell.

    Returns:
        ``(episodes in list then committed order, list facts for run_info.json)``.

    Raises:
        ConditionUnavailableError: If an optional condition's manifest is absent.
        SystemExit: On a manifest mismatch or a list missing from the manifest.
    """
    manifest_path = cond.episodes_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        if cond.optional:
            raise ConditionUnavailableError(f"{cond.key}: no list manifest at {manifest_path}")
        raise SystemExit(f"{cond.key}: no list manifest at {manifest_path}")
    manifest = verify_lists(cond.episodes_dir)
    by_file = {row["file"]: row for row in manifest}
    episodes: list[ListedEpisode] = []
    lists: list[dict[str, Any]] = []
    for name in cond.lists:
        entry = by_file.get(f"{name}.parquet")
        if entry is None:
            raise SystemExit(f"{cond.key}: list {name!r} is not in {manifest_path}")
        rows = read_list(cond.episodes_dir / entry["file"])
        episodes += rows
        lists.append(
            {
                "name": name,
                "file": _rel(cond.episodes_dir / entry["file"]),
                "rows": len(rows),
                "file_sha256": entry["file_sha256"],
                "content_sha256": entry["content_sha256"],
            }
        )
    if per_cell is not None:
        episodes = _first_per_cell(episodes, per_cell)
    facts = {
        "manifest": _rel(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_check": f"{2 * len(manifest)}/{2 * len(manifest)} hashes OK",
        "lists": lists,
        "per_cell": per_cell,
        "episodes": len(episodes),
    }
    return episodes, facts


def _rel(path: Path) -> str:
    """Return a path relative to the repository root when inside it."""
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    return str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)


def inspect_runs(
    methods: Sequence[str],
    seeds: Sequence[int],
    cfgs: EvalConfigs,
    runs_root: Path = RUNS_ROOT,
) -> tuple[list[LearnedRun], list[dict[str, Any]]]:
    """Inspect and policy-check every (method, seed) run's ``final/`` checkpoint.

    Args:
        methods: Learned methods, in table order.
        seeds: Training seeds.
        cfgs: The committed configs (policy input layout).
        runs_root: ``artifacts/runs``.

    Returns:
        ``(runs in method then seed order, per-run policy facts)``.
    """
    runs = [inspect_run(m, run_dir(m, s, runs_root), "final") for m in methods for s in seeds]
    for run in runs:
        if run.resumed_from is not None:
            print(f"note: {run.method} seed {run.seed} was resumed from {run.resumed_from}")
    return check_policies(runs, cfgs)


def _cond_runs(cond: Condition, runs: Sequence[LearnedRun]) -> list[LearnedRun]:
    """Return the runs a condition flies, in method then seed order."""
    order = {m: i for i, m in enumerate(cond.learned)}
    chosen = [r for r in runs if r.method in order]
    return sorted(chosen, key=lambda r: (order[r.method], r.seed))


def _feed_skips(
    episodes: Sequence[ListedEpisode], spec: PolicySpec, pad: str | None
) -> tuple[list[ListedEpisode], list[SkipRecord]]:
    """Return a spec's runnable episodes and its skip records on ``pad``."""
    runnable, skips = split_runnable(episodes, spec, pad)
    records = [
        SkipRecord(
            spec.method,
            spec.privileged,
            spec.run_seed,
            s.pad,
            s.episode.regime,
            s.episode.ss,
            s.reason,
        )
        for s in skips
    ]
    return runnable, records


def build_arms(
    cond: Condition,
    specs: Sequence[PolicySpec],
    episodes: Sequence[ListedEpisode],
    cfgs_arm: EvalConfigs,
) -> tuple[list[EvalArm], list[SkipRecord]]:
    """Return the runner arms of a condition and the episodes each spec cannot fly.

    One :class:`~rld.eval.runner.EvalArm` per (pad, spec, regime): pads outermost, then the
    specs in the order given, then the lists in the condition's order, so the rows come back
    pad by pad, spec by spec, in committed list order.

    Args:
        cond: The condition.
        specs: What to fly, in output order (learned runs, then baselines).
        episodes: The condition's listed episodes.
        cfgs_arm: The condition's configs (set as every arm's ``cfgs_override``).

    Returns:
        ``(arms, skip records)``.
    """
    regimes = list(dict.fromkeys(ep.regime for ep in episodes))
    arms: list[EvalArm] = []
    skipped: list[SkipRecord] = []
    for pad in cond.pads:
        for spec in specs:
            runnable, skips = _feed_skips(episodes, spec, pad)
            skipped += skips
            for regime in regimes:
                mine = [ep for ep in runnable if ep.regime == regime]
                if not mine:
                    continue
                arms.append(
                    EvalArm(
                        spec,
                        pad,
                        mine,
                        motion_kind=cast(MotionKind, cond.motion_for_regime(regime)),
                        cfgs_override=cfgs_arm,
                        start_check=cast(StartCheck, cond.start_check),
                    )
                )
    return arms, skipped


def _merge_arms(arms: Sequence[EvalArm]) -> list[EvalArm]:
    """Return arms with consecutive same-(spec, pad, motion) arms merged into one.

    A JONSWAP condition flies every list under one motion, so one arm per (spec, pad) keeps
    the chunks contiguous across list boundaries exactly as e01-e06 cut them (the rows do not
    depend on the chunking, but this keeps the task count and order the same as before).
    """
    out: list[EvalArm] = []
    for arm in arms:
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.spec == arm.spec
            and prev.pad_override == arm.pad_override
            and prev.motion_kind == arm.motion_kind
        ):
            out[-1] = dataclasses.replace(prev, episodes=[*prev.episodes, *arm.episodes])
        else:
            out.append(arm)
    return out


# --------------------------------------------------------------------------- derive / write


def _split_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split episode rows into learned-method rows and baseline rows."""
    learned = [r for r in rows if str(r["method"]) in LEARNED_METHODS]
    base = [r for r in rows if str(r["method"]) not in LEARNED_METHODS]
    return learned, base


def derive_condition(
    cond: Condition,
    cfgs_arm: EvalConfigs,
    rows: Sequence[Mapping[str, Any]],
    run_cols: Mapping[tuple[str, int], Mapping[str, str]],
    base_prov: Mapping[str, Mapping[str, str]],
    provenance: Mapping[str, str],
    skipped: Sequence[SkipRecord],
) -> dict[str, str]:
    """Return the text of every derived CSV of a condition, from its episode rows.

    Args:
        cond: The condition.
        cfgs_arm: Its configs.
        rows: The condition's ``episodes.csv`` rows, read back as text.
        run_cols: ``{(method, seed): run + policy columns}`` of the learned runs.
        base_prov: ``{baseline: PROVENANCE_COLUMNS values}`` of the flown baselines.
        provenance: The deterministic environment provenance columns.
        skipped: Skip records of learned runs (forecast runs on the static list).

    Returns:
        ``{file name: CSV text}`` for :data:`DERIVED_FILES` (``baselines_summary.csv`` only
        when baselines were flown).
    """
    learned_rows, base_rows = _split_rows(rows)
    by_regime: dict[str, dict[str, str]] = {}

    def cols(regime: str) -> dict[str, str]:
        if regime not in by_regime:
            by_regime[regime] = arm_columns(cond, cfgs_arm, regime)
        return by_regime[regime]

    out: dict[str, str] = {}
    summary, columns = summarise_learned(learned_rows, run_cols, provenance, skipped)
    summary = [{**rec, **cols(str(rec["regime"]))} for rec in summary]
    out["summary.csv"] = _csv_text(summary, [*columns, *ARM_COLUMNS])
    flown = [rec for rec in summary if int(rec["n_episodes"]) > 0]
    spread, spread_cols = seed_spread(flown)
    out["seeds.csv"] = _csv_text(spread, spread_cols)
    out["aggregate.csv"] = _csv_text(aggregate_rows(flown), list(AGGREGATE_COLUMNS))
    if base_rows:
        order = [m for m in BASELINES if m in {str(r["method"]) for r in base_rows}]
        base = summarise(base_rows, order, base_prov, provenance)
        base = [{**rec, **cols(str(rec["regime"]))} for rec in base]
        base_columns = summary_columns([*PROVENANCE_COLUMNS, *provenance])
        out["baselines_summary.csv"] = _csv_text(base, [*base_columns, *ARM_COLUMNS])
    return out


def _carry(
    cond: Condition,
    out_dir: Path,
    episodes: Sequence[ListedEpisode],
    provenance: Mapping[str, str],
    results_dir: Path,
) -> dict[str, Any]:
    """Carry a condition's baselines line for line into ``out_dir``; return the checks."""
    info: dict[str, Any] = {}
    pads = list(dict.fromkeys(pad for _, _, pad in cond.carry))
    for pad in pads:
        sources = {results_dir / name: methods for name, methods, p in cond.carry if p == pad}
        info.update(carry_baselines(out_dir, sources, episodes, provenance, pad=pad))
    return info


def _run_table(runs: Sequence[LearnedRun]) -> list[dict[str, Any]]:
    """Return the run facts recorded in run_info.json."""
    return [
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
            "policy_class": r.policy_class,
            "needs_motion_feed": r.needs_motion_feed,
            "training_motion": r.training_motion,
        }
        for r in runs
    ]


def _digests(runs: Sequence[LearnedRun]) -> dict[str, dict[str, str]]:
    """Return the live checkpoint digests of every run, keyed ``method/seed``."""
    return {f"{r.method}/{r.seed}": dict(_run_digests(r.run_dir, r.ckpt)) for r in runs}


def _check_digests(runs: Sequence[LearnedRun], when: str) -> dict[str, dict[str, str]]:
    """Return the live digests, refusing any that differ from the inspected ones."""
    live = _digests(runs)
    changed = [
        f"{r.method}/{r.seed}" for r in runs if live[f"{r.method}/{r.seed}"] != dict(r.digests)
    ]
    if changed:
        raise RuntimeError(f"checkpoint files changed ({when}): {changed}")
    return live


def fly_condition(
    cond: Condition,
    runs: Sequence[LearnedRun],
    cfgs: EvalConfigs,
    *,
    out_root: Path = E07_DIR,
    results_dir: Path = RESULTS_DIR,
    workers: int = DEFAULT_WORKERS,
    chunk: int = DEFAULT_CHUNK,
    per_cell: int | None = None,
    baselines: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fly one condition and write its directory (see the module docstring).

    Args:
        cond: The condition.
        runs: Every inspected and checked learned run; the condition's are selected.
        cfgs: The committed configs.
        out_root: Output root (``results/e07``; a scratch directory for smoke flights).
        results_dir: Where the carried sources live (``results/``).
        workers: Worker processes; the rows do not depend on it.
        chunk: Episodes per task; the rows do not depend on it.
        per_cell: Scratch only: first ``per_cell`` episodes per cell (carrying is skipped,
            because a carried source covers whole lists).
        baselines: Scratch only: a subset of the condition's flown baselines.

    Returns:
        The run_info facts written.

    Raises:
        RuntimeError: If a checkpoint or a config changed during the flight.
    """
    started_utc = environment_provenance(REPO_ROOT)["timestamp_utc"]
    git = git_state(REPO_ROOT)
    out_dir = cond.out_dir(out_root)
    cfgs_arm = condition_configs(cond, cfgs)
    episodes, list_facts = condition_episodes(cond, per_cell)
    provenance = live_provenance(cfgs)
    my_runs = _cond_runs(cond, runs)
    base_names = [b for b in cond.baselines if baselines is None or b in baselines]
    base_specs = [controller_spec(name) for name in base_names]
    base_prov = controller_provenance_columns(base_names)
    learned_specs = [learned_spec(r) for r in my_runs]
    arms, skipped = build_arms(cond, [*learned_specs, *base_specs], episodes, cfgs_arm)
    arms = _merge_arms(arms)
    carry_on = per_cell is None and bool(cond.carry)
    if carry_on:  # check the carried sources before anything is flown
        with tempfile.TemporaryDirectory() as tmp:
            _carry(cond, Path(tmp), episodes, provenance, results_dir)
    for arm in {a.motion_kind: a for a in arms}.values():  # a missing source fails here, early
        first = arm.episodes[0]
        motion_for(
            cfgs_arm,
            first.vessel,
            first.ss,
            first.heading_deg,
            first.speed_kn,
            first.realization_seed,
            kind=arm.motion_kind,
            episode_seed=first.episode_seed,
        )
    n_eps = sum(len(a.episodes) for a in arms)
    print(
        f"[{cond.key}] flying {len(my_runs)} learned runs + {len(base_specs)} baselines, "
        f"{n_eps} episodes, {workers} workers",
        flush=True,
    )
    before = _check_digests(my_runs, "before the flight")
    t0 = time.perf_counter()
    rows = run_arms(arms, cfgs_arm, workers=workers, chunk=chunk)
    wall_s = time.perf_counter() - t0
    after = _check_digests(my_runs, "after the flight")
    if live_provenance(cfgs) != provenance or controller_provenance_columns(base_names) != (
        base_prov
    ):
        raise RuntimeError(f"{cond.key}: configs or lists changed during the flight; not writing")

    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / "run_info.json"
    if stale.exists():
        stale.unlink()
    write_rows(out_dir / "episodes.csv", rows, EPISODE_COLUMNS)
    text_rows = read_rows(out_dir / "episodes.csv")
    run_cols = {
        (r.method, r.seed): {**r.columns(), "eval_deck_motion": cond.motion} for r in my_runs
    }
    learned_skips = [s for s in skipped if s.method in LEARNED_METHODS]
    texts = derive_condition(
        cond, cfgs_arm, text_rows, run_cols, base_prov, provenance, learned_skips
    )
    for name, text in texts.items():
        (out_dir / name).write_text(text, encoding="utf-8")
    carried = _carry(cond, out_dir, episodes, provenance, results_dir) if carry_on else None
    hashes = {name: _sha256(out_dir / name) for name in ("episodes.csv", *texts)}
    if carried:
        for src in carried.values():
            hashes[src["file"]] = _sha256(out_dir / src["file"])
    check = verify_condition(
        cond, out_root=out_root, results_dir=results_dir, require=False, per_cell=per_cell
    )
    reference = _reference_checks(cond, rows, results_dir)
    env = environment_provenance(REPO_ROOT)
    info: dict[str, Any] = {
        "condition": cond.key,
        "started_utc": started_utc,
        **{k: env[k] for k in VOLATILE_KEYS},
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "workers": workers,
        "chunk": chunk,
        "wall_s": round(wall_s, 1),
        "episodes_flown": len(rows),
        **git,
        "list": list_facts,
        "arm_columns_by_regime": {
            regime: arm_columns(cond, cfgs_arm, regime)
            for regime in dict.fromkeys(ep.regime for ep in episodes)
        },
        "learned_methods": list(dict.fromkeys(r.method for r in my_runs)),
        "seeds": sorted({r.seed for r in my_runs}),
        "baselines_flown": base_names,
        "runs": _run_table(my_runs),
        "checkpoint_digests_before": before,
        "checkpoint_digests_after": after,
        "skipped_episodes": len(skipped),
        "skip_reasons": sorted({s.reason for s in skipped}),
        "carried": carried if carry_on else f"not carried (per_cell={per_cell}: scratch subset)",
        "reference_checks": reference,
        "lambda_t0": _t0_facts(cond, rows, episodes),
        "rederived_byte_identical": check,
        "files_sha256": hashes,
        "worker_count_checks": [],
        "complete": True,
    }
    stale.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    bad = [k for k, ok in check.items() if not ok]
    ref_bad = [k for k, v in reference.items() if v.get("applicable") and not v["all_identical"]]
    refs = {k: v.get("all_identical") for k, v in reference.items()}
    print(
        f"[{cond.key}] {len(rows)} rows in {wall_s:.1f} s; re-derived byte-identical: "
        f"{not bad} {bad or ''}; reference checks {refs}",
        flush=True,
    )
    if ref_bad:
        print(f"REFERENCE MISMATCH in {cond.key}: {ref_bad}", file=sys.stderr)
    return info


def _reference_checks(
    cond: Condition, rows: Sequence[Mapping[str, Any]], results_dir: Path
) -> dict[str, Any]:
    """Compare the matrix arm's ``id`` rows with e05 / e06, line for line.

    Only the matrix arm has committed counterparts: e05 (``ppo``, ``sac``) and e06 (the four
    Phase 6 methods) flew the identical ``id`` episodes (aft, JONSWAP) with the same
    checkpoints. The other lists were never flown by a learned method before Phase 7.
    """
    if cond.arm != "matrix":
        return {}
    out: dict[str, Any] = {}
    id_rows = [r for r in rows if str(r["regime"]) == "id"]
    for name in ("e05", "e06"):
        ref = results_dir / name / "episodes.csv"
        if ref.is_file():
            res = compare_to_reference(id_rows, ref, pad="aft")
            res["reference"] = _rel(ref)
            out[name] = res
    return out


def _t0_facts(
    cond: Condition, rows: Sequence[Mapping[str, Any]], episodes: Sequence[ListedEpisode]
) -> dict[str, Any] | None:
    """For the lambda arm: how many flown starts differ from the listed (1/25) ``t0``."""
    if cond.start_check != "lambda":
        return None
    listed = {(e.regime, e.ss, e.index): e.t0_model_s for e in episodes}
    differ = sum(
        float(r["t0_model_s"]) != listed[(str(r["regime"]), str(r["ss"]), int(r["index"]))]
        for r in rows
    )
    return {
        "rows": len(rows),
        "t0_differs_from_listed": int(differ),
        "note": "t0 re-drawn by the environment at this lambda (P7-D1 §3); recorded per row in "
        "episodes.csv t0_model_s",
    }


# --------------------------------------------------------------------------- verify


def _committed_run_cols(
    summary: Sequence[Mapping[str, str]],
) -> dict[tuple[str, int], dict[str, str]]:
    """Return ``{(method, seed): run + policy columns}`` from a committed learned summary."""
    names = [c for c in (*RUN_COLUMNS, *RUN_POLICY_COLUMNS) if summary and c in summary[0]]
    return {(r["method"], int(r["run_seed"])): {c: r[c] for c in names} for r in summary}


def _committed_provenance(path: Path, first_after: Sequence[str]) -> dict[str, str]:
    """Return the provenance columns of a committed summary: after the run/controller columns."""
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    first = read_rows(path)[0]
    start = max(header.index(c) for c in first_after if c in header) + 1
    end = header.index(ARM_COLUMNS[0])
    return {c: first[c] for c in header[start:end]}


def verify_condition(
    cond: Condition,
    *,
    out_root: Path = E07_DIR,
    results_dir: Path = RESULTS_DIR,
    require: bool = True,
    per_cell: int | None = None,
) -> dict[str, bool]:
    """Re-derive a written condition from its ``episodes.csv`` and compare every file's bytes.

    Args:
        cond: The condition.
        out_root: Output root.
        results_dir: Where carried sources live.
        require: Require ``run_info.json`` to say ``complete`` (``False`` while writing).
        per_cell: The scratch subset while writing (``require=False``); otherwise read from
            ``run_info.json``.

    Returns:
        ``{file name: byte-identical}``; ``{"complete": False}`` for an unwritten condition.
    """
    out_dir = cond.out_dir(out_root)
    info_path = out_dir / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    if require and not info.get("complete"):
        return {"complete": False}
    if info:
        per_cell = info.get("list", {}).get("per_cell")
    cfgs = load_eval_configs()
    cfgs_arm = condition_configs(cond, cfgs)
    episodes, _ = condition_episodes(cond, per_cell)
    rows = read_rows(out_dir / "episodes.csv")
    header = (out_dir / "episodes.csv").read_text(encoding="utf-8").split("\n", 1)[0]
    result: dict[str, bool] = {"episodes.csv header": header == ",".join(EPISODE_COLUMNS)}
    summary = read_rows(out_dir / "summary.csv")
    run_cols = _committed_run_cols(summary)
    provenance = _committed_provenance(out_dir / "summary.csv", [*RUN_COLUMNS, *RUN_POLICY_COLUMNS])
    base_path = out_dir / "baselines_summary.csv"
    base_prov: dict[str, dict[str, str]] = {}
    if base_path.is_file():
        for rec in read_rows(base_path):
            base_prov.setdefault(rec["method"], {c: rec[c] for c in PROVENANCE_COLUMNS})
    skipped: list[SkipRecord] = []
    for (method, seed), cols in run_cols.items():
        if cols.get("run_needs_motion_feed") == "True":
            stub = PolicySpec(method, False, _no_build, seed, needs_motion_feed=True)
            for pad in cond.pads:
                skipped += _feed_skips(episodes, stub, pad)[1]
    texts = derive_condition(cond, cfgs_arm, rows, run_cols, base_prov, provenance, skipped)
    for name, text in texts.items():
        path = out_dir / name
        result[name] = path.is_file() and text == path.read_text(encoding="utf-8")
    if base_path.is_file() and "baselines_summary.csv" not in texts:
        result["baselines_summary.csv"] = False
    if cond.carry and per_cell is None:
        with tempfile.TemporaryDirectory() as tmp:
            carried = _carry(cond, Path(tmp), episodes, provenance, results_dir)
            for src in carried.values():
                name = src["file"]
                result[name] = (out_dir / name).is_file() and (Path(tmp) / name).read_bytes() == (
                    out_dir / name
                ).read_bytes()
    runnable = {
        (method, seed): len(_feed_skips(episodes, _stub(method, seed, cols), None)[0])
        for (method, seed), cols in run_cols.items()
    }
    expected = len(cond.pads) * sum(runnable.values())
    expected += len(cond.pads) * len(episodes) * len(base_prov)
    result["episodes.csv rows"] = len(rows) == expected
    return result


def _no_build(cfgs: EvalConfigs) -> Any:
    """A spec factory that is never called (skip bookkeeping only)."""
    del cfgs
    raise RuntimeError("not a flyable spec")


def _stub(method: str, seed: int, cols: Mapping[str, str]) -> PolicySpec:
    """Return a non-flyable spec carrying a run's feed flag (for skip bookkeeping)."""
    feed = cols.get("run_needs_motion_feed") == "True"
    return PolicySpec(method, False, _no_build, seed, needs_motion_feed=feed)


# --------------------------------------------------------------------------- re-fly


def refly_check(
    cond: Condition,
    runs: Sequence[LearnedRun],
    cfgs: EvalConfigs,
    *,
    out_root: Path = E07_DIR,
    workers: int = 7,
    chunk: int = DEFAULT_CHUNK,
    per_cell: int = 10,
) -> dict[str, Any]:
    """Re-fly a slice of a written condition at another worker count; record byte identity.

    Args:
        cond: A written (complete) condition.
        runs: Inspected runs.
        cfgs: The committed configs.
        out_root: Output root.
        workers: The other worker count.
        chunk: Episodes per task.
        per_cell: First ``per_cell`` listed episodes per cell, for every spec.

    Returns:
        The check, also appended to the condition's ``run_info.json``.

    Raises:
        SystemExit: If the condition is not written.
    """
    out_dir = cond.out_dir(out_root)
    info_path = out_dir / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    if not info.get("complete"):
        raise SystemExit(f"{cond.key}: not written; nothing to re-fly against")
    cfgs_arm = condition_configs(cond, cfgs)
    full, _ = condition_episodes(cond, info["list"]["per_cell"])
    episodes = _first_per_cell(full, per_cell)
    my_runs = [r for r in _cond_runs(cond, runs) if r.seed in info["seeds"]]
    specs = [*(learned_spec(r) for r in my_runs), *map(controller_spec, info["baselines_flown"])]
    arms, _ = build_arms(cond, specs, episodes, cfgs_arm)
    t0 = time.perf_counter()
    rows = run_arms(_merge_arms(arms), cfgs_arm, workers=workers, chunk=chunk)
    wall_s = time.perf_counter() - t0
    by_pad = {}
    for pad in dict.fromkeys(("aft" if p is None else p) for p in cond.pads):
        by_pad[pad] = compare_to_reference(rows, out_dir / "episodes.csv", pad=pad)
        by_pad[pad]["reference"] = _rel(out_dir / "episodes.csv")
    check = {
        "timestamp_utc": environment_provenance(REPO_ROOT)["timestamp_utc"],
        **git_state(REPO_ROOT),
        "workers": workers,
        "chunk": chunk,
        "flight_workers": info["workers"],
        "per_cell": per_cell,
        "rows": len(rows),
        "wall_s": round(wall_s, 1),
        "byte_identical": all(v["all_identical"] for v in by_pad.values()),
        "by_pad": by_pad,
    }
    info.setdefault("worker_count_checks", []).append(check)
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"[{cond.key}] re-flight at {workers} workers: byte-identical {check['byte_identical']}")
    return check


# --------------------------------------------------------------------------- lambda feasibility

#: ``results/e07/lambda/feasibility.csv`` columns (the P1-D1 rule at each lambda).
FEASIBILITY_FILE: str = "feasibility.csv"


def _feasibility_cell(task: tuple[float, float]) -> list[Any]:
    """Return Phase 1's cell rows for the rule's cell at one (1 / lambda, speed) (picklable)."""
    from rld.deck.stats import compute_cell

    lam_inv, speed = task
    cfgs = load_eval_configs()
    scaling = dataclasses.replace(cfgs.scaling, lam_inverse=float(lam_inv))
    rows, _ = compute_cell(
        ("frigate", "SS6", 180.0, float(speed)),
        cfgs.sim,
        scaling,
        cfgs.pads,
        float(cfgs.motion.physics_freq_hz),
    )
    return list(rows)


def lambda_feasibility(
    out_dir: Path,
    lam_inverses: Sequence[float] = (15.0, 25.0, 40.0),
    *,
    workers: int = DEFAULT_WORKERS,
) -> Path:
    """Write the P1-D1 feasibility rule's SS6 deck-point v_z p99 at each lambda (context only).

    The rule's cell (frigate, SS6, head seas, aft pad, worst speed) is recomputed at each
    lambda with Phase 1's own :func:`rld.deck.stats.compute_cell` and
    :func:`rld.deck.stats.feasibility_rows` (gate reference only). The 1/25 row must
    reproduce ``results/deck_feasibility.csv``'s gate row; the committed value is recorded
    beside every row. P7-D1 §3: this is a sensitivity point; the project lambda stays 1/25.

    Args:
        out_dir: ``results/e07/lambda``.
        lam_inverses: ``1 / lambda`` values.
        workers: Processes over the (lambda, speed) cells; the CSV does not depend on it.

    Returns:
        The CSV path.
    """
    import multiprocessing as mp

    from rld.deck.stats import feasibility_rows, rows_as_dicts

    cfgs = load_eval_configs()
    speeds = sorted({float(s) for s in cfgs.sim.speeds_kn})
    tasks = [(float(lam), speed) for lam in lam_inverses for speed in speeds]
    with mp.get_context("spawn").Pool(processes=max(1, min(workers, len(tasks)))) as pool:
        cells = pool.map(_feasibility_cell, tasks, chunksize=1)
    committed = read_rows(RESULTS_DIR / "deck_feasibility.csv")
    gate = next(r for r in committed if r["is_gate"] == "True")
    records: list[dict[str, Any]] = []
    for lam_inv in lam_inverses:
        scaling = dataclasses.replace(cfgs.scaling, lam_inverse=float(lam_inv))
        rows = [
            r for (lam, _), cell in zip(tasks, cells, strict=True) if lam == lam_inv for r in cell
        ]
        for rec in rows_as_dicts(feasibility_rows(rows, scaling, reference="gate")):
            records.append(
                {
                    "lam_inverse": repr(float(lam_inv)),
                    **rec,
                    "committed_lam25_vz_p99_model_m_s": gate["vz_p99_model_m_s"],
                    "note": "sensitivity context only (P7-D1 §3); the project lambda stays 1/25 "
                    "(P1-D1, P1-D3)",
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FEASIBILITY_FILE
    write_rows(path, records, list(records[0]))
    return path


# --------------------------------------------------------------------------- main


def _conditions(arm: str, episodes_mss_dir: Path) -> list[Condition]:
    """Return the conditions of an ``--arm`` choice (``all``: every arm, flight order)."""
    names = ARMS if arm == "all" else (arm,)
    return [c for name in names for c in conditions(name, episodes_mss_dir)]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line (see the module docstring)."""
    parser = argparse.ArgumentParser(
        prog="scripts/eval_phase7.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arm", choices=[*ARMS, "all", "hypotheses"], default="all")
    parser.add_argument("--check", action="store_true", help="verify written outputs; no flight")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--out-root", type=Path, default=E07_DIR)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--episodes-mss-dir", type=Path, default=EPISODES_MSS_DIR)
    parser.add_argument(
        "--condition", default=None, help="fly or check only this condition key, e.g. noise/…"
    )
    refly = parser.add_argument_group("worker-count re-flight of a written condition")
    refly.add_argument("--refly", default=None, metavar="KEY", help="condition key to re-fly")
    refly.add_argument("--refly-workers", type=int, default=7)
    refly.add_argument("--refly-per-cell", type=int, default=10)
    scratch = parser.add_argument_group("scratch only (refused under results/)")
    scratch.add_argument("--per-cell", type=int, default=None)
    scratch.add_argument("--methods", nargs="+", default=None, help="learned methods subset")
    scratch.add_argument("--seeds", nargs="+", type=int, default=None, help="seeds subset")
    scratch.add_argument("--baselines", nargs="+", default=None, help="flown baselines subset")
    return parser.parse_args(argv)


def _invocation_log(out_root: Path, entry: Mapping[str, Any]) -> None:
    """Append one invocation's facts to ``<out_root>/run_info.json``."""
    path = out_root / "run_info.json"
    doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"invocations": []}
    doc["invocations"].append(dict(entry))
    out_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Fly, verify or re-fly Phase 7 conditions, then (``all`` / ``hypotheses``) score them.

    Args:
        argv: Arguments (``None``: ``sys.argv``).

    Returns:
        Process exit status: 0 when everything requested is written and byte-identical.
    """
    args = parse_args(argv)
    out_root: Path = args.out_root
    committed = out_root.resolve().is_relative_to(RESULTS_DIR.resolve())
    scratch = (args.per_cell, args.methods, args.seeds, args.baselines)
    if committed and any(v is not None for v in scratch):
        raise SystemExit("--per-cell / --methods / --seeds / --baselines are scratch-only")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    conds = [] if args.arm == "hypotheses" else _conditions(args.arm, args.episodes_mss_dir)
    if args.condition is not None:
        conds = [c for c in conds if c.key == args.condition]
        if not conds:
            raise SystemExit(f"no condition {args.condition!r} in arm {args.arm!r}")

    if args.check:
        status = 0
        for cond in conds:
            try:
                result = verify_condition(cond, out_root=out_root)
            except ConditionUnavailableError as exc:
                print(f"[{cond.key}] not flown (optional): {exc}")
                continue
            ok = all(result.values())
            if not ok and cond.optional and result == {"complete": False}:
                print(f"[{cond.key}] not flown (optional)")
                continue
            print(f"[{cond.key}] {'OK' if ok else 'MISMATCH'} {json.dumps(result)}")
            status |= 0 if ok else 1
        if any(c.arm == "lambda" for c in conds) and args.condition is None:
            feas = out_root / "lambda" / FEASIBILITY_FILE
            with tempfile.TemporaryDirectory() as tmp:
                fresh = lambda_feasibility(Path(tmp), workers=args.workers)
                same = feas.is_file() and fresh.read_bytes() == feas.read_bytes()
            print(f"[lambda/feasibility] {'OK' if same else 'MISMATCH or missing'}")
            status |= 0 if same else 1
        if args.arm in ("all", "hypotheses") and args.condition is None:
            from rld.eval.hypotheses import check_hypotheses

            status |= 0 if check_hypotheses(out_root, allow_subset=not committed) else 1
        return status

    started = environment_provenance(REPO_ROOT)["timestamp_utc"]
    t_start = time.perf_counter()
    cfgs = load_eval_configs()
    methods = list(args.methods or LEARNED_METHODS)
    seeds = list(args.seeds if args.seeds is not None else LEARNED_SEEDS)
    needed = [m for m in methods if any(m in c.learned for c in conds)]
    runs, policy_facts = inspect_runs(needed, seeds, cfgs, args.runs_root) if needed else ([], [])
    if args.refly is not None:
        target = next((c for c in conds if c.key == args.refly), None)
        if target is None:
            raise SystemExit(f"--refly {args.refly!r}: no such condition in arm {args.arm!r}")
        check = refly_check(
            target,
            runs,
            cfgs,
            out_root=out_root,
            workers=args.refly_workers,
            chunk=args.chunk,
            per_cell=args.refly_per_cell,
        )
        return 0 if check["byte_identical"] else 1
    digests_before = _check_digests(runs, "at start")
    status = 0
    done: dict[str, str] = {}
    for cond in conds:
        try:
            result = verify_condition(cond, out_root=out_root)
            if result.get("complete", True) and all(result.values()):
                done[cond.key] = "verified (already written; not re-flown)"
                print(f"[{cond.key}] already written and byte-identical; not re-flown")
                continue
            if result != {"complete": False}:
                print(f"[{cond.key}] written but NOT byte-identical: {result}", file=sys.stderr)
                done[cond.key] = f"MISMATCH {result}"
                status = 1
                continue
            info = fly_condition(
                cond,
                runs,
                cfgs,
                out_root=out_root,
                workers=args.workers,
                chunk=args.chunk,
                per_cell=args.per_cell,
                baselines=args.baselines,
            )
        except ConditionUnavailableError as exc:
            done[cond.key] = f"not flown (optional): {exc}"
            print(f"[{cond.key}] not flown (optional): {exc}")
            continue
        except NotImplementedError as exc:
            if not cond.optional:
                raise
            done[cond.key] = f"not flown (optional): {exc}"
            print(f"[{cond.key}] not flown (optional): {exc}")
            continue
        ok = all(info["rederived_byte_identical"].values())
        ref_ok = all(
            v["all_identical"] for v in info["reference_checks"].values() if v.get("applicable")
        )
        done[cond.key] = f"flown in {info['wall_s']} s; byte-identical {ok}; reference {ref_ok}"
        status |= 0 if ok and ref_ok else 1
    if any(c.arm == "lambda" for c in conds) and args.condition is None:
        feas = out_root / "lambda" / FEASIBILITY_FILE
        if feas.is_file():
            done["lambda/feasibility"] = "present (re-computed and compared by --check)"
        else:
            lambda_feasibility(feas.parent, workers=args.workers)
            done["lambda/feasibility"] = "written"
            print(f"[lambda] feasibility context -> {feas}")
    digests_after = _check_digests(runs, "at the end")
    hyp: dict[str, Any] | None = None
    if args.arm in ("all", "hypotheses") and args.condition is None:
        from rld.eval.hypotheses import write_hypotheses

        hyp = write_hypotheses(out_root, allow_subset=not committed)
    _invocation_log(
        out_root,
        {
            "started_utc": started,
            **git_state(REPO_ROOT),
            "argv": list(sys.argv[1:] if argv is None else argv),
            "workers": args.workers,
            "chunk": args.chunk,
            "wall_s": round(time.perf_counter() - t_start, 1),
            "conditions": done,
            "policy_check": policy_facts,
            "checkpoint_digests_before": digests_before,
            "checkpoint_digests_after": digests_after,
            "checkpoints_unchanged": digests_before == digests_after,
            "hypotheses": hyp,
        },
    )
    return status
