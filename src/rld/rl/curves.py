"""Learning curves from run directories: one run, or mean +- std across seeds.

Three panels, all against training env steps:

1. tune-pool success (``evals.csv``: pooled over SS3-SS5, and per sea state for a single
   run), with the curriculum's promotions marked for a single run;
2. training episode return (``monitor/*.monitor.csv``, binned);
3. p95 touchdown closing speed of the evaluation's touched-down episodes.

Across seeds, each run's evaluation curve is linearly interpolated onto the first run's
evaluation steps (identical when the runs share ``eval.interval_steps``) up to the shortest
run, and the band is mean +- one standard deviation across seeds (``ddof = 1``); the number
of seeds is printed in the legend. Nothing is smoothed beyond the return binning.

Units: steps are env control steps (1/30 s model scale each); closing speed metres per
second model scale; return and success dimensionless.
"""

import json
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

__all__ = ["aggregate", "binned_return", "plot_learning_curves", "read_evals", "read_monitor"]


def read_evals(run_dir: Path) -> pd.DataFrame:
    """Read a run's ``evals.csv``.

    Args:
        run_dir: The run directory.

    Returns:
        One row per evaluation, sorted by ``steps``.
    """
    return pd.read_csv(run_dir / "evals.csv").sort_values("steps").reset_index(drop=True)


def _monitor_segment(name: str) -> int:
    """Return the resume segment of a monitor file: ``<rank>.resume<k>.monitor.csv`` -> k."""
    parts = name.split(".")
    tag = parts[1] if len(parts) > 3 else ""
    return int(tag[len("resume") :]) if tag.startswith("resume") and tag[6:].isdigit() else 0


def _resume_cutoffs(run_dir: Path) -> dict[int, float]:
    """Return, per resume segment ``k``, the Unix time its checkpoint was saved.

    Training episodes of earlier segments that ended after that time belong to the branch
    the resume abandoned.
    """
    try:
        status = json.loads((run_dir / "status.json").read_text())
    except (OSError, ValueError):
        return {}
    out: dict[int, float] = {}
    for record in status.get("resumes") or []:
        try:
            meta = json.loads((Path(record["resumed_from"]) / "checkpoint.json").read_text())
            saved = meta.get("saved_unix")
            out[int(record["segment"])] = (
                float(saved) if saved is not None else pd.Timestamp(meta["saved"]).timestamp()
            )
        except (OSError, ValueError, KeyError):
            continue
    return out


def read_monitor(run_dir: Path, include_abandoned: bool = False) -> pd.DataFrame:
    """Read every worker's monitor file into one frame with cumulative steps.

    A resumed run (:mod:`rld.rl.resume`) has one file per worker per segment; each file's
    ``t`` is relative to its own start, so rows are ordered by absolute time
    (``t_start`` from the file header + ``t``). Episodes of an earlier segment that ended
    after the checkpoint a later segment resumed from belong to the abandoned branch and
    are dropped unless ``include_abandoned``.

    Args:
        run_dir: The run directory.
        include_abandoned: Keep the abandoned branch's episodes (flagged ``abandoned``).

    Returns:
        Columns ``r`` (episode return), ``l`` (length, steps), ``t`` (host seconds since
        its file's start), ``t_abs`` (Unix seconds), ``segment``, ``abandoned``, plus the
        logged info keys, sorted by ``t_abs``, with ``steps`` = cumulative ``l`` across all
        workers (the approximate training step at each episode's end).
    """
    frames = []
    for path in sorted((run_dir / "monitor").glob("*.monitor.csv")):
        with path.open() as handle:
            header = handle.readline()
            t_start = 0.0
            if header.startswith("#"):
                try:
                    t_start = float(json.loads(header[1:]).get("t_start", 0.0))
                except ValueError:
                    t_start = 0.0
            else:
                handle.seek(0)
            frame = pd.read_csv(handle)
        frame["t_abs"] = t_start + frame["t"]
        frame["segment"] = _monitor_segment(path.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["r", "l", "t", "t_abs", "segment", "abandoned", "steps"])
    df = pd.concat(frames, ignore_index=True)
    df["abandoned"] = False
    for segment, saved in _resume_cutoffs(run_dir).items():
        df.loc[(df["segment"] < segment) & (df["t_abs"] > saved), "abandoned"] = True
    if not include_abandoned:
        df = df[~df["abandoned"]]
    df = df.sort_values("t_abs").reset_index(drop=True)
    df["steps"] = df["l"].cumsum()
    return df


def binned_return(monitor: pd.DataFrame, edges: np.ndarray) -> np.ndarray:
    """Return the mean episode return per step bin (NaN for an empty bin).

    Args:
        monitor: :func:`read_monitor` output.
        edges: Bin edges, env steps.

    Returns:
        ``len(edges) - 1`` means.
    """
    out = np.full(len(edges) - 1, np.nan)
    if monitor.empty:
        return out
    idx = np.digitize(monitor["steps"].to_numpy(), edges) - 1
    r = monitor["r"].to_numpy(dtype=np.float64)
    for b in range(len(out)):
        sel = idx == b
        if sel.any():
            out[b] = float(r[sel].mean())
    return out


def aggregate(
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put several ``(steps, values)`` curves on one grid and reduce across them.

    Args:
        curves: One ``(steps, values)`` pair per seed.

    Returns:
        ``(grid, mean, std)``; ``std`` is 0 for a single curve. The grid is the first
        curve's steps up to the shortest curve's last step.
    """
    end = min(float(s[-1]) for s, _ in curves)
    grid = curves[0][0][curves[0][0] <= end]
    stack = np.vstack([np.interp(grid, s, v) for s, v in curves])
    with warnings.catch_warnings():  # an all-NaN column (nothing touched down) stays NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
    return grid, mean, std


def _method(run_dir: Path) -> str:
    """Return a run's method label (config.yaml), else its group directory name."""
    try:
        return str(yaml.safe_load((run_dir / "config.yaml").read_text())["method"])
    except (OSError, KeyError, TypeError):
        return run_dir.parent.name


def plot_learning_curves(run_dirs: Sequence[Path], out_png: Path, title: str = "") -> Path:
    """Plot the learning curves of one run, or mean +- std over several seeds of one method.

    Args:
        run_dirs: Run directories, each with ``evals.csv``.
        out_png: Output image.
        title: Figure title (default: method and seed count).

    Returns:
        ``out_png``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [Path(d) for d in run_dirs]
    evals = [read_evals(d) for d in runs]
    method = _method(runs[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax_s, ax_r, ax_v = axes
    single = len(runs) == 1
    label = f"{method} (n={len(runs)} seed{'s' if len(runs) > 1 else ''})"

    def band(ax: object, col: str, colour: str, name: str) -> None:
        curves = [(e["steps"].to_numpy(dtype=float), e[col].to_numpy(dtype=float)) for e in evals]
        grid, mean, std = aggregate(curves)
        ax.plot(grid, mean, color=colour, label=name)  # type: ignore[attr-defined]
        if not single:
            ax.fill_between(grid, mean - std, mean + std, color=colour, alpha=0.25)  # type: ignore[attr-defined]

    band(ax_s, "success", "C0", f"{label}: SS3-SS5 mean")
    sea_states = [
        c[: -len("_frac_success")] for c in evals[0].columns if c.endswith("_frac_success")
    ]
    if single:
        for k, ss in enumerate(sea_states):
            ax_s.plot(
                evals[0]["steps"],
                evals[0][f"{ss}_frac_success"],
                ls="--",
                color=f"C{k + 1}",
                label=ss,
            )
        for _, promo in evals[0][evals[0]["promoted"].astype(str) == "True"].iterrows():
            ax_s.axvline(promo["steps"], color="grey", lw=0.8, ls=":")
            ax_s.text(promo["steps"], 0.02, f"-> {promo['stage_after']}", rotation=90, fontsize=7)
    ax_s.set_ylim(-0.02, 1.02)
    ax_s.set_ylabel("tune-pool success (deterministic)")

    monitors = [read_monitor(d) for d in runs]
    top = max(float(m["steps"].max()) if not m.empty else 1.0 for m in monitors)
    edges = np.linspace(0.0, top, 41)
    centres = 0.5 * (edges[:-1] + edges[1:])
    stack = np.vstack([binned_return(m, edges) for m in monitors])
    with warnings.catch_warnings():  # empty bins stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0, ddof=1) if not single else np.zeros_like(mean)
    ax_r.plot(centres, mean, color="C0", label=label)
    if not single:
        ax_r.fill_between(centres, mean - std, mean + std, color="C0", alpha=0.25)
    ax_r.set_ylabel("training episode return (binned)")

    band(ax_v, "pooled_td_closing_p95_m_s", "C3", f"{label}: p95 closing speed")
    ax_v.axhline(0.5, color="k", lw=0.8, ls="--", label="0.5 m/s success limit")
    ax_v.set_ylabel("p95 touchdown closing speed (m/s, model)")

    for ax in axes:
        ax.set_xlabel("training env steps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    status = runs[0] / "status.json"
    budget = json.loads(status.read_text()).get("budget") if status.exists() else None
    fig.suptitle(title or f"{label}; budget {budget} steps; simulation only")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
