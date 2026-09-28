"""The pre-registered hyperparameter search for PPO and SAC (P3-D1 section 5).

Procedure (``configs/rl/tune_<method>.yaml``)
---------------------------------------------
* **Budget.** At most :data:`MAX_TRIALS` (20) trials per method, each trained for
  ``trial_steps`` env steps (PPO 2 M, SAC 0.5 M) on **one fixed training seed**
  (``trial_seed``). Trials train on the P3-D2 train pool exactly as a final run does
  (curriculum included).
* **Points.** Trial 0 is the base config unchanged. Trials ``1..n-1`` are the first ``n-1``
  points of one scrambled Sobol sequence (``sobol_seed``) of dimension = number of search
  parameters, mapped per parameter: ``linear`` and ``log`` ranges, or a ``choices`` list
  (``floor(u * k)``). A parameter may set several config keys at once (``targets``), e.g.
  one failure-penalty magnitude shared by ``r_hard_landing``, ``r_off_pad`` and ``r_bounce``.
  Reward targets may only be weights; ``v_safe_m_s`` and ``gate_height_m`` are structure
  and are rejected (:mod:`rld.rl.config`).
* **Score.** Each trial's **final** checkpoint is scored on the fixed P3-D3 tune-pool draw
  (180 episodes, SS3-SS5) that every periodic evaluation flies: mean success over the
  sea states, ties broken by the lower pooled p95 touchdown closing speed, then the lower
  trial index -- :func:`rld.control.tuning.summarise_trial` and
  :func:`rld.control.tuning.select_trial`, the rule the baselines were selected by. A
  failed trial is listed with ``state = failed`` and cannot win; it is never dropped.

* **Re-invoking a search** (``make tune`` again on the same config, e.g. after a host
  restart; :func:`plan_trials`): a trial with a ``done`` run is skipped; a trial whose
  latest run ``failed`` is re-run -- resumed in place from that run's latest resumable
  checkpoint if the search config says ``resume_failed: true`` and one exists
  (:mod:`rld.rl.resume`), otherwise trained afresh into the next ``<seed>_r<k>``
  directory; a trial never started is started. A live or unreconciled (``running`` with a
  dead writer) trial run makes the re-invocation refuse. ``resume_failed`` is optional and
  defaults to ``false``: a fresh re-run is the protocol-clean choice (one uninterrupted
  training pass per trial); a resumed trial is flagged in ``trials.csv``.
* **Collection** scores each trial from its **latest** ``done`` run (run directories
  ordered by their numeric ``_r<k>`` suffix) and lists every other run directory of the
  trial -- failed or abandoned attempts it supersedes -- in ``superseded_runs``.

Nothing here reads ``results/episodes/``; the tune pool is the only scoring data.

Outputs: ``<results_dir>/configs/trial_XX.yaml`` (the materialised trial configs, written
before any trial starts), ``<results_dir>/trials.csv`` and ``<results_dir>/selection.json``.

Units: steps are env control steps (1/30 s model scale each); closing speeds metres per
second model scale; every hyperparameter in its SB3 unit.
"""

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from scipy.stats import qmc

from rld.config import REPO_ROOT, load_yaml
from rld.control.tuning import TrialResult, select_trial, summarise_trial
from rld.rl.config import RUNS_ROOT, apply_overrides, load_train_config, train_config_from_dict

__all__ = [
    "MAX_TRIALS",
    "SearchParam",
    "TrialPlan",
    "TuneConfig",
    "collect_trials",
    "load_tune_config",
    "materialize_trials",
    "plan_trials",
    "trial_points",
    "trial_raw_configs",
    "trial_run_dirs",
]

#: P3-D1 section 5: at most 20 tuning trials per method.
MAX_TRIALS: int = 20


@dataclass(frozen=True)
class SearchParam:
    """One search dimension.

    Attributes:
        name: Label (a ``param_<name>`` column in ``trials.csv``).
        targets: Dotted config keys it sets, e.g. ``("ppo.learning_rate",)``.
        kind: ``"linear"``, ``"log"`` or ``"choice"``.
        low: Range low (linear/log).
        high: Range high (linear/log).
        choices: Allowed values (choice).
    """

    name: str
    targets: tuple[str, ...]
    kind: Literal["linear", "log", "choice"]
    low: float = 0.0
    high: float = 0.0
    choices: tuple[Any, ...] = ()

    def value(self, u: float) -> Any:
        """Map a unit-interval coordinate to a parameter value.

        Args:
            u: Sobol coordinate in ``[0, 1)``.

        Returns:
            The value in the parameter's unit.
        """
        if self.kind == "choice":
            return self.choices[min(int(u * len(self.choices)), len(self.choices) - 1)]
        if self.kind == "log":
            return float(
                math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
            )
        return float(self.low + u * (self.high - self.low))


@dataclass(frozen=True)
class TuneConfig:
    """``configs/rl/tune_<method>.yaml``.

    Attributes:
        method: Tuned method, e.g. ``"ppo"``.
        base_config: The final-run config whose hyperparameters are searched.
        trials: Trials including trial 0 drawn by this search. ``trials +
            trials_used_before_search`` is at most :data:`MAX_TRIALS`.
        trials_used_before_search: Tune-pool training runs already charged to this method's
            trial budget before the search (P5-D4: the PPO smoke run counts as one).
        sobol_seed: Seed of the scrambled Sobol sequence.
        trial_seed: The training seed every trial uses.
        trial_steps: Env-step budget per trial.
        run_group: Runs go to ``artifacts/runs/<run_group>/trial_XX/<trial_seed>/``.
        results_dir: Where configs, ``trials.csv`` and ``selection.json`` go.
        params: The search dimensions, in Sobol coordinate order.
        source: The YAML file.
        resume_failed: On re-invocation, resume a failed trial from its latest resumable
            checkpoint instead of re-training it from scratch (optional key, default
            False).
    """

    method: str
    base_config: Path
    trials: int
    trials_used_before_search: int
    sobol_seed: int
    trial_seed: int
    trial_steps: int
    run_group: str
    results_dir: Path
    params: tuple[SearchParam, ...]
    source: Path
    resume_failed: bool = False


def _rel(path: Path) -> str:
    """Return ``path`` relative to the repository root when inside it, else as given."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _abs(path: str | Path) -> Path:
    """Return a repository-relative path as absolute."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_tune_config(path: Path) -> TuneConfig:
    """Load and validate a search config (every trial config is built and parsed).

    Args:
        path: ``configs/rl/tune_<method>.yaml``.

    Returns:
        The :class:`TuneConfig`.

    Raises:
        ValueError: On a missing key, more than :data:`MAX_TRIALS` trials, a malformed
            dimension, or any trial config that fails validation.
    """
    path = _abs(path)
    raw = load_yaml(path)
    needed = (
        "method",
        "base_config",
        "trials",
        "trials_used_before_search",
        "sobol_seed",
        "trial_seed",
        "trial_steps",
        "run_group",
        "results_dir",
        "search_space",
    )
    optional = ("resume_failed",)
    if missing := sorted(set(needed) - set(raw)):
        raise ValueError(f"{path}: missing {missing} (is this a tune_*.yaml?)")
    if unknown := sorted(set(raw) - set(needed) - set(optional)):
        raise ValueError(f"{path}: unknown keys {unknown}")
    resume_failed = raw.get("resume_failed", False)
    if not isinstance(resume_failed, bool):
        raise ValueError(f"{path}: resume_failed must be true or false")
    params: list[SearchParam] = []
    for name, spec in dict(raw["search_space"]).items():
        spec = dict(spec)
        targets = tuple(str(t) for t in spec.pop("targets"))
        if not targets:
            raise ValueError(f"{path}: {name} has no targets")
        if "choices" in spec:
            choices = tuple(spec.pop("choices"))
            if not choices or spec:
                raise ValueError(f"{path}: {name}: choices must be alone and non-empty")
            params.append(SearchParam(str(name), targets, "choice", choices=choices))
            continue
        kind = str(spec.pop("scale", "linear"))
        low, high = float(spec.pop("low")), float(spec.pop("high"))
        if spec or kind not in ("linear", "log") or not low <= high:
            raise ValueError(f"{path}: {name}: bad range {low, high, kind} or keys {list(spec)}")
        if kind == "log" and low <= 0.0:
            raise ValueError(f"{path}: {name}: a log range needs low > 0")
        params.append(
            SearchParam(str(name), targets, "log" if kind == "log" else "linear", low, high)
        )
    cfg = TuneConfig(
        method=str(raw["method"]),
        base_config=_abs(raw["base_config"]),
        trials=int(raw["trials"]),
        trials_used_before_search=int(raw["trials_used_before_search"]),
        sobol_seed=int(raw["sobol_seed"]),
        trial_seed=int(raw["trial_seed"]),
        trial_steps=int(raw["trial_steps"]),
        run_group=str(raw["run_group"]),
        results_dir=_abs(raw["results_dir"]),
        params=tuple(params),
        source=path,
        resume_failed=resume_failed,
    )
    if not 1 <= cfg.trials <= MAX_TRIALS:
        raise ValueError(f"{path}: trials must be in [1, {MAX_TRIALS}] (P3-D1), got {cfg.trials}")
    if cfg.trials_used_before_search < 0:
        raise ValueError(f"{path}: trials_used_before_search must be >= 0")
    if cfg.trials + cfg.trials_used_before_search > MAX_TRIALS:
        raise ValueError(
            f"{path}: trials ({cfg.trials}) + trials_used_before_search "
            f"({cfg.trials_used_before_search}) exceeds {MAX_TRIALS} (P3-D1)"
        )
    if cfg.trial_steps < 1:
        raise ValueError(f"{path}: trial_steps must be positive")
    for index, trial_raw in trial_raw_configs(cfg):
        train_config_from_dict(trial_raw, f"{path} trial {index}")
    return cfg


def trial_points(cfg: TuneConfig) -> list[dict[str, Any]]:
    """Return each trial's parameter values by parameter name.

    Args:
        cfg: The search config.

    Returns:
        ``cfg.trials`` dicts. Trial 0 is ``{}`` (the base config); the others map every
        parameter name to its value.
    """
    points: list[dict[str, Any]] = [{}]
    if cfg.trials > 1:
        m = int(np.ceil(np.log2(cfg.trials - 1))) if cfg.trials > 2 else 0
        sampler = qmc.Sobol(d=len(cfg.params), scramble=True, seed=cfg.sobol_seed)
        unit = sampler.random_base2(m=m)[: cfg.trials - 1]
        for row in unit:
            points.append({p.name: p.value(float(u)) for p, u in zip(cfg.params, row, strict=True)})
    return points


def _get(raw: Mapping[str, Any], dotted: str) -> Any:
    """Return a dotted key's value from a raw config, or ``None`` if absent."""
    node: Any = raw
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def _base_value(base: Mapping[str, Any], dotted: str) -> Any:
    """Return the value a base config trains with for one search target.

    A ``reward.<weight>`` the base config does not override (``reward: {}``) is the
    committed ``configs/env/reward.yaml`` weight -- the value trial 0 actually flies with.
    """
    value = _get(base, dotted)
    if value is None and dotted.startswith("reward."):
        from rld.envs.config import REWARD_CONFIG, load_reward

        value = getattr(load_reward(REWARD_CONFIG), dotted.split(".", 1)[1])
    return value


def trial_raw_configs(cfg: TuneConfig) -> list[tuple[int, dict[str, Any]]]:
    """Return every trial's full raw training config.

    Each is the base config with the trial's overrides, ``total_steps = trial_steps``,
    ``run_group = <run_group>/trial_XX`` and ``method = tune_<method>_tXX``.

    Args:
        cfg: The search config.

    Returns:
        ``(trial index, raw config)`` pairs.
    """
    base = load_yaml(cfg.base_config)
    out = []
    for index, point in enumerate(trial_points(cfg)):
        overrides: dict[str, Any] = {}
        for param in cfg.params:
            if param.name in point:
                for target in param.targets:
                    overrides[target] = point[param.name]
        overrides["total_steps"] = cfg.trial_steps
        overrides["run_group"] = f"{cfg.run_group}/trial_{index:02d}"
        overrides["method"] = f"tune_{cfg.method}_t{index:02d}"
        out.append((index, apply_overrides(base, overrides)))
    return out


def materialize_trials(cfg: TuneConfig) -> list[Path]:
    """Write every trial config to ``<results_dir>/configs/trial_XX.yaml``.

    Existing files must be byte-identical (a search is never silently changed after it
    started).

    Args:
        cfg: The search config.

    Returns:
        The trial config paths, in trial order.

    Raises:
        FileExistsError: If a trial config exists with different content.
    """
    out_dir = cfg.results_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, raw in trial_raw_configs(cfg):
        header = (
            f"# Trial {index} of {_rel(cfg.source)} (generated; do not edit).\n"
            f"# Base: {_rel(cfg.base_config)}; Sobol seed {cfg.sobol_seed}.\n"
        )
        text = header + yaml.safe_dump(raw, sort_keys=False)
        path = out_dir / f"trial_{index:02d}.yaml"
        if path.exists() and path.read_text() != text:
            raise FileExistsError(
                f"{path} exists with different content; refusing to change a search"
            )
        path.write_text(text)
        paths.append(path)
    return paths


def _attempt(name: str, seed: str) -> int | None:
    """Return the attempt number of a run directory name: ``<seed>`` -> 1, ``<seed>_r<k>`` -> k."""
    if name == seed:
        return 1
    prefix = f"{seed}_r"
    if name.startswith(prefix) and name[len(prefix) :].isdigit():
        return int(name[len(prefix) :])
    return None


def trial_run_dirs(cfg: TuneConfig, index: int, runs_root: Path = RUNS_ROOT) -> list[Path]:
    """Return every run directory of one trial, oldest attempt first.

    The original ``<seed>`` and each rerun ``<seed>_r<k>``, ordered by ``k`` numerically
    (``_r10`` after ``_r9``, which a string sort gets wrong).
    """
    parent = runs_root / cfg.run_group / f"trial_{index:02d}"
    seed = str(cfg.trial_seed)
    found = [(k, p) for p in parent.glob(f"{seed}*") if (k := _attempt(p.name, seed)) is not None]
    return [p for _, p in sorted(found)]


@dataclass(frozen=True)
class TrialPlan:
    """What a (re-)invoked search does with one trial.

    Attributes:
        trial: Trial index.
        action: ``"skip_done"`` (a run is done), ``"fresh"`` (never started, or re-train a
            failed trial from scratch into the next ``_r<k>``), ``"resume"`` (continue the
            latest failed run from its latest resumable checkpoint), ``"live"`` (a run is
            still training) or ``"unreconciled"`` (a run says ``running`` but its process
            is gone).
        run_dir: The run that decided it (``None`` for a never-started trial).
        resume_from: The checkpoint for ``"resume"``, else ``None``.
    """

    trial: int
    action: str
    run_dir: Path | None
    resume_from: Path | None = None


def _status(run_dir: Path) -> dict[str, Any]:
    """Return a run's ``status.json`` or ``{}``."""
    try:
        loaded: Any = json.loads((run_dir / "status.json").read_text())
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def plan_trials(cfg: TuneConfig, runs_root: Path = RUNS_ROOT) -> list[TrialPlan]:
    """Decide, per trial, what re-invoking the search does (module docstring).

    Args:
        cfg: The search config (``resume_failed`` decides between resume and fresh).
        runs_root: Root of all runs.

    Returns:
        One plan per trial, in trial order.
    """
    from rld.rl.procs import status_owner_alive
    from rld.rl.resume import latest_resumable_checkpoint

    plans = []
    for index in range(cfg.trials):
        dirs = trial_run_dirs(cfg, index, runs_root)
        statuses = [(d, _status(d)) for d in dirs]
        done = [d for d, s in statuses if s.get("state") == "done"]
        running = [(d, s) for d, s in statuses if s.get("state") == "running"]
        if done:
            plans.append(TrialPlan(index, "skip_done", done[-1]))
            continue
        if live := [d for d, s in running if status_owner_alive(s)]:
            plans.append(TrialPlan(index, "live", live[-1]))
            continue
        if running:
            plans.append(TrialPlan(index, "unreconciled", running[-1][0]))
            continue
        failed = [d for d, s in statuses if s.get("state") == "failed"]
        if not failed:
            plans.append(TrialPlan(index, "fresh", None))
            continue
        latest = failed[-1]
        ckpt = latest_resumable_checkpoint(latest) if cfg.resume_failed else None
        if ckpt is not None:
            plans.append(TrialPlan(index, "resume", latest, ckpt))
        else:
            plans.append(TrialPlan(index, "fresh", latest))
    return plans


def _final_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Return the final evaluation's episode rows of a run (typed for summarise_trial)."""
    with (run_dir / "eval_episodes.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    last = max(int(r["eval_index"]) for r in rows)
    out = []
    for r in rows:
        if int(r["eval_index"]) != last:
            continue
        out.append(
            {
                **r,
                "touchdown_contact": r["touchdown_contact"] == "True",
                "detectors_disagree": r["detectors_disagree"] == "True",
                "tunnelled": r["tunnelled"] == "True",
                "closing_speed_normal_m_s": _num(r["closing_speed_normal_m_s"]),
                "lateral_offset_m": _num(r["lateral_offset_m"]),
                "rel_tilt_deg": _num(r["rel_tilt_deg"]),
                "td_t_episode_s": _num(r["td_t_episode_s"]),
            }
        )
    return out


def _num(text: str) -> float | None:
    """Parse a CSV number; empty means ``None``."""
    return None if text in ("", "None") else float(text)


def collect_trials(cfg: TuneConfig, runs_root: Path = RUNS_ROOT) -> list[dict[str, Any]]:
    """Score every trial and write ``trials.csv`` and ``selection.json``.

    Args:
        cfg: The search config.
        runs_root: Root of all runs.

    Returns:
        One row per trial (every trial, including failed or missing ones).
    """
    base = load_yaml(cfg.base_config)
    points = trial_points(cfg)
    sea_states = list(load_train_config(cfg.base_config).curriculum.stages)
    rows: list[dict[str, Any]] = []
    scored: list[tuple[int, TrialResult]] = []
    superseded_by_trial: dict[int, list[str]] = {}
    for index, point in enumerate(points):
        values = {p.name: point.get(p.name, _base_value(base, p.targets[0])) for p in cfg.params}
        dirs = trial_run_dirs(cfg, index, runs_root)
        statuses = {d: _status(d) for d in dirs}
        done = [d for d in dirs if statuses[d].get("state") == "done"]
        row: dict[str, Any] = {"trial": index, **{f"param_{k}": v for k, v in values.items()}}
        chosen = done[-1] if done else (dirs[-1] if dirs else None)
        superseded = [
            f"{d.name}:{statuses[d].get('state', 'no_status')}" for d in dirs if d != chosen
        ]
        superseded_by_trial[index] = superseded
        if not done:
            state = "missing" if not dirs else "failed"
            row.update(
                {
                    "state": state,
                    "run_dir": str(dirs[-1]) if dirs else "",
                    "superseded_runs": ";".join(superseded),
                    "mean_success": float("nan"),
                }
            )
            rows.append(row)
            continue
        run_dir = done[-1]
        status = statuses[run_dir]
        result = summarise_trial(cfg.method, index, values, _final_rows(run_dir), sea_states)
        scored.append((index, result))
        resumes = status.get("resumes") or []
        row.update(
            {
                "state": "done",
                "run_dir": str(run_dir),
                "superseded_runs": ";".join(superseded),
                "resumed": bool(resumes),
                "resume_steps": ";".join(str(r.get("resume_steps")) for r in resumes),
                "steps": status.get("steps"),
                "wall_s": status.get("wall_s"),
                "cpu_hours": status.get("cpu_hours"),
                "git_sha": status.get("git_sha"),
                **{
                    k: v
                    for k, v in result.row.items()
                    if k not in ("controller", "trial") and not k.startswith("param_")
                },
            }
        )
        rows.append(row)
    winner: int | None = None
    if scored:
        winner = scored[select_trial([r for _, r in scored])][0]
    for row in rows:
        row["selected"] = row["trial"] == winner
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames += [k for k in row if k not in fieldnames]
    with (cfg.results_dir / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    selection = {
        "method": cfg.method,
        "rule": "max mean tune-pool success over SS3-SS5 (final checkpoint), then min pooled "
        "p95 touchdown closing speed, then min trial index; failed trials cannot win",
        "winner_trial": winner,
        "winner_params": None
        if winner is None
        else {k[6:]: v for k, v in rows[winner].items() if k.startswith("param_")},
        "n_trials": len(rows),
        "trials_used_before_search": cfg.trials_used_before_search,
        "n_done": len(scored),
        "tune_config": _rel(cfg.source),
        "scored_from": "each trial's latest done run directory",
        "superseded_runs": {str(k): v for k, v in superseded_by_trial.items() if v},
        "resumed_trials": [r["trial"] for r in rows if r.get("resumed")],
    }
    (cfg.results_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, default=str) + "\n"
    )
    return rows
