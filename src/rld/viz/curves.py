"""Success rate against sea state, and touchdown closing speed, from committed Phase 7 CSVs.

Sources, all committed:

- learned methods: ``results/e07/matrix/aggregate.csv`` rows with ``metric = success_rate``
  and ``statistic = iqm``. The point is the IQM over 5 training seeds, and the interval is its
  stratified-bootstrap 95 % CI, which reflects seed variation only (P3-D1 §4);
- classical baselines: ``results/e07/matrix/carried_summary_e01.csv`` and
  ``carried_summary_e01_lowvz_cut.csv``. One deterministic run, so the interval is the Wilson
  95 % CI over the cell's 200 episodes.

The 12 methods are split into four panels of three, because identity is carried by colour and
only three categorical slots stay distinguishable when every pair of lines can cross. Every
panel but the always-printed-baseline panel repeats ``pid_feedforward`` as a dashed grey
reference. Success is never pooled across sea states. Units: success in percent; sea states are
the dmf JONSWAP classes (model scale λ = 1/25, aft pad, JONSWAP motion).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = [
    "PANELS",
    "REFERENCE_METHOD",
    "closing_speed_table",
    "plot_closing_speed_ecdf",
    "plot_success_vs_seastate",
    "success_table",
]

#: Sea states on the x axis, in order.
SEA_STATES: tuple[str, ...] = ("SS3", "SS4", "SS5", "SS6")

#: The three categorical slots that validate on every pair (dataviz reference palette).
SLOTS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")

#: Ink and surface (dataviz reference palette, light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE_GREY = "#8a8985"

#: What "outside the training distribution" covers (P3-D2): more than SS6.
TRAINING_POOL_NOTE = (
    "Training pool (P3-D2): frigate, SS3–SS5, headings 45/135/180°; so id's 90° episodes and every "
    "unseen_heading / unseen_vessel cell are also outside the training distribution."
)

#: The baseline repeated as a dashed reference in every other panel.
REFERENCE_METHOD = "pid_feedforward"


@dataclass(frozen=True)
class Panel:
    """One small multiple: a title and up to three methods.

    Attributes:
        title: Panel title.
        methods: Methods drawn in categorical slot order.
    """

    title: str
    methods: tuple[str, ...]


PANELS: tuple[Panel, ...] = (
    Panel("Pure RL", ("ppo", "sac", "ppo_sinusoid")),
    Panel("Residual and forecast RL", ("residual_ppo", "residual_ppo_forecast", "ppo_forecast")),
    Panel("Always-printed baselines", ("pid_track_descend", "pid_feedforward", "oracle_gated")),
    Panel(
        "Other classical",
        ("pid_feedforward_lowvz", "pid_feedforward_lowvz_cut", "gated"),
    ),
)

#: Display names. ``sac`` carries its budget confound; the forecast methods their ideal feed;
#: ``oracle_gated`` its privilege.
LABELS: dict[str, str] = {
    "sac": "sac (2 M steps)",
    "ppo_forecast": "ppo_forecast (ideal feed)",
    "residual_ppo_forecast": "residual_ppo_forecast (ideal feed)",
    "oracle_gated": "oracle_gated (privileged)",
}


def success_table(results_dir: Path) -> pd.DataFrame:
    """Collect success rate and its 95 % CI per (method, regime, sea state), in percent.

    Args:
        results_dir: The repository's ``results/`` directory.

    Returns:
        Columns ``method, regime, ss, point, lo, hi, kind``, where ``kind`` is ``learned``
        (IQM over seeds, seed-bootstrap CI) or ``classical`` (one run, Wilson CI). Success in
        percent.
    """
    matrix = results_dir / "e07" / "matrix"
    agg = pd.read_csv(matrix / "aggregate.csv")
    agg = agg[(agg["metric"] == "success_rate") & (agg["statistic"] == "iqm")]
    learned = pd.DataFrame(
        {
            "method": agg["method"],
            "regime": agg["regime"],
            "ss": agg["ss"],
            "point": 100.0 * agg["point"],
            "lo": 100.0 * agg["ci_lo"],
            "hi": 100.0 * agg["ci_hi"],
            "kind": "learned",
        }
    )
    carried = pd.concat(
        [
            pd.read_csv(matrix / "carried_summary_e01.csv"),
            pd.read_csv(matrix / "carried_summary_e01_lowvz_cut.csv"),
        ],
        ignore_index=True,
    )
    classical = pd.DataFrame(
        {
            "method": carried["method"],
            "regime": carried["regime"],
            "ss": carried["ss"],
            "point": 100.0 * carried["success_rate"],
            "lo": 100.0 * carried["success_wilson_lo"],
            "hi": 100.0 * carried["success_wilson_hi"],
            "kind": "classical",
        }
    )
    return pd.concat([learned, classical], ignore_index=True)


def _style_axes(ax: object, ylim: tuple[float, float]) -> None:
    """Recessive grid and spines, light surface."""
    ax.set_facecolor(SURFACE)  # type: ignore[attr-defined]
    ax.set_ylim(*ylim)  # type: ignore[attr-defined]
    ax.set_xlim(-0.35, len(SEA_STATES) - 0.65)  # type: ignore[attr-defined]
    ax.set_xticks(range(len(SEA_STATES)), SEA_STATES)  # type: ignore[attr-defined]
    ax.grid(axis="y", color=GRID, lw=0.8)  # type: ignore[attr-defined]
    ax.set_axisbelow(True)  # type: ignore[attr-defined]
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)  # type: ignore[attr-defined]
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)  # type: ignore[attr-defined]
    ax.tick_params(colors=INK_2, labelsize=8)  # type: ignore[attr-defined]


def _draw_panel(
    ax: object,
    table: pd.DataFrame,
    regime: str,
    panel: Panel,
    ylim: tuple[float, float],
) -> None:
    """Draw one panel: three methods with CI bars, plus the dashed reference."""
    in_regime = table["regime"] == regime
    sea_states = [ss for ss in SEA_STATES if (in_regime & (table["ss"] == ss)).any()]
    xs_all = {ss: SEA_STATES.index(ss) for ss in sea_states}
    _style_axes(ax, ylim)
    if REFERENCE_METHOD not in panel.methods:
        ref = table[(table["regime"] == regime) & (table["method"] == REFERENCE_METHOD)]
        ref = ref.set_index("ss").reindex(sea_states)
        ax.plot(  # type: ignore[attr-defined]
            [xs_all[s] for s in sea_states],
            ref["point"],
            color=REFERENCE_GREY,
            lw=1.4,
            ls=(0, (4, 3)),
            marker="_" if len(sea_states) == 1 else None,
            markersize=16,
            markeredgewidth=1.6,
            zorder=1,
            label=f"{REFERENCE_METHOD} (reference)",
        )
    n = len(panel.methods)
    for k, method in enumerate(panel.methods):
        rows = table[(table["regime"] == regime) & (table["method"] == method)]
        rows = rows.set_index("ss").reindex(sea_states)
        dodge = (k - (n - 1) / 2) * 0.09
        xs = [xs_all[s] + dodge for s in sea_states]
        colour = SLOTS[k]
        ax.errorbar(  # type: ignore[attr-defined]
            xs,
            rows["point"],
            yerr=[rows["point"] - rows["lo"], rows["hi"] - rows["point"]],
            color=colour,
            lw=2.0,
            elinewidth=1.4,
            capsize=0,
            marker="o",
            markersize=5.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            zorder=3 + k,
            label=LABELS.get(method, method),
        )
    ax.set_title(panel.title, fontsize=9.5, color=INK, loc="left")  # type: ignore[attr-defined]
    ax.legend(  # type: ignore[attr-defined]
        loc="lower left", fontsize=7.2, frameon=False, labelcolor=INK_2, handlelength=1.8
    )


def plot_success_vs_seastate(
    results_dir: Path,
    out_png: Path,
    regimes: Sequence[str] = ("id",),
    ylim: tuple[float, float] = (55.0, 101.0),
    title: str = "",
) -> Path:
    """Plot success rate against sea state, one row of four panels per regime.

    Args:
        results_dir: The repository's ``results/`` directory.
        out_png: Output image.
        regimes: Regimes, one row each, from ``id``, ``unseen_heading``, ``unseen_vessel``,
            ``unseen_seastate`` (SS6 only).
        ylim: Shared y range, percent. The axis does not start at 0; the marks are points with
            intervals, not bars.
        title: Figure title.

    Returns:
        ``out_png``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = success_table(results_dir)
    fig, axes = plt.subplots(
        len(regimes),
        len(PANELS),
        figsize=(13.0, 3.5 * len(regimes) + 0.9),
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)
    for r, regime in enumerate(regimes):
        for c, panel in enumerate(PANELS):
            ax = axes[r][c]
            _draw_panel(ax, table, regime, panel, ylim)
            if c == 0:
                ax.set_ylabel(f"{regime}\nsuccess (%)", fontsize=9, color=INK)
            ax.axvspan(2.5, 3.5, color="#f0efec", zorder=0, lw=0)
    fig.suptitle(
        title
        or (
            "Landing success by sea state. Simulation: Froude-scaled JONSWAP deck motion "
            "(λ = 1/25), aft pad"
        ),
        fontsize=11,
        color=INK,
        x=0.01,
        y=0.995,
        ha="left",
    )
    fig.text(
        0.01,
        0.005,
        "Learned: IQM over 5 seeds × 200 episodes, 95 % stratified-bootstrap CI over seeds. "
        "Classical: one deterministic run × 200 episodes, Wilson 95 % CI. "
        "Same frozen episode list for every method.\n"
        f"Shaded: SS6, a sea state no method trained on. {TRAINING_POOL_NOTE}\n"
        "Sources: results/e07/matrix/"
        "{aggregate,carried_summary_e01,carried_summary_e01_lowvz_cut}.csv",
        fontsize=7,
        color=INK_2,
    )
    height = 3.5 * len(regimes) + 0.9
    fig.tight_layout(rect=(0, 0.62 / height, 1, 1 - 0.12 / height), h_pad=1.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return out_png


#: Touchdown limit on closing speed along the deck normal, m/s (P3-D1 §1).
CLOSING_LIMIT_M_S = 0.5


def closing_speed_table(results_dir: Path, regime: str = "id") -> pd.DataFrame:
    """Per-episode touchdown closing speed of every method, from the committed episode rows.

    Args:
        results_dir: The repository's ``results/`` directory.
        regime: Regime.

    Returns:
        Columns ``method, run_seed, ss, closing_m_s``: |``rel_vz_normal_m_s``| of every
        touched-down episode at the aft pad (no touchdown: no row; never imputed, P3-D1 §4),
        metres per second along the deck normal, model scale.
    """
    cols = ["regime", "ss", "pad", "method", "run_seed", "rel_vz_normal_m_s"]
    cut = pd.read_csv(results_dir / "e01_lowvz_cut" / "episodes.csv", usecols=cols)
    frames = [
        pd.read_csv(results_dir / "e07" / "matrix" / "episodes.csv.gz", usecols=cols),
        pd.read_csv(results_dir / "e01" / "episodes.csv", usecols=cols),
        # e01_lowvz_cut re-carries the e01 baselines; take only the method it adds.
        cut[cut["method"] == "pid_feedforward_lowvz_cut"],
    ]
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[(rows["regime"] == regime) & (rows["pad"] == "aft")]
    rows = rows[rows["rel_vz_normal_m_s"].notna()].copy()
    rows["closing_m_s"] = rows["rel_vz_normal_m_s"].abs()
    return rows[["method", "run_seed", "ss", "closing_m_s"]].reset_index(drop=True)


def plot_closing_speed_ecdf(
    results_dir: Path,
    out_png: Path,
    sea_states: Sequence[str] = ("SS5", "SS6"),
    regime: str = "id",
) -> Path:
    """Plot the distribution (ECDF) of touchdown closing speed, one row per sea state.

    Learned methods pool their 5 seeds' touched-down episodes (up to 1 000) into one curve, which
    shows the distribution, not a seed-level statistic; baselines show their one run (up to 200).
    Episodes are never pooled across sea states.

    Args:
        results_dir: The repository's ``results/`` directory.
        out_png: Output image.
        sea_states: Sea states, one row each.
        regime: Regime.

    Returns:
        ``out_png``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    table = closing_speed_table(results_dir, regime)
    height = 3.3 * len(sea_states) + 1.0
    fig, axes = plt.subplots(
        len(sea_states),
        len(PANELS),
        figsize=(13.0, height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)

    def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.sort(values)
        return x, np.arange(1, len(x) + 1) / len(x)

    for r, ss in enumerate(sea_states):
        cell = table[table["ss"] == ss]
        for c, panel in enumerate(PANELS):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK_2)
            ax.tick_params(colors=INK_2, labelsize=8)
            ax.axvline(CLOSING_LIMIT_M_S, color=INK_2, lw=1.0, ls=":", zorder=1)
            if REFERENCE_METHOD not in panel.methods:
                x, y = ecdf(cell.loc[cell["method"] == REFERENCE_METHOD, "closing_m_s"].to_numpy())
                ax.step(
                    x,
                    y,
                    where="post",
                    color=REFERENCE_GREY,
                    lw=1.4,
                    ls=(0, (4, 3)),
                    label=f"{REFERENCE_METHOD} (reference)",
                )
            for k, method in enumerate(panel.methods):
                v = cell.loc[cell["method"] == method, "closing_m_s"].to_numpy()
                x, y = ecdf(v)
                ax.step(
                    x,
                    y,
                    where="post",
                    color=SLOTS[k],
                    lw=2.0,
                    label=f"{LABELS.get(method, method)} (n={len(v)})",
                )
            ax.set_xlim(0.0, 0.9)
            ax.set_ylim(0.0, 1.01)
            if r == 0:
                ax.set_title(panel.title, fontsize=9.5, color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(f"{regime} {ss}\nfraction of touchdowns ≤ x", fontsize=9, color=INK)
            ax.legend(
                loc="lower right", fontsize=6.8, frameon=False, labelcolor=INK_2, handlelength=1.8
            )
    fig.suptitle(
        "Touchdown closing speed distribution. Simulation: Froude-scaled JONSWAP deck motion "
        "(λ = 1/25), aft pad",
        fontsize=11,
        color=INK,
        x=0.01,
        y=0.995,
        ha="left",
    )
    fig.text(
        0.01,
        0.005,
        "Empirical CDF over touched-down episodes only (no touchdown: excluded, never imputed). "
        "Learned: 5 seeds' episodes pooled; classical: one run. Dotted: the 0.5 m/s limit.\n"
        "x: touchdown closing speed along the deck normal, m/s model scale; it is "
        "understated by about 7 % for every method alike (P5-D14); axis clipped at 0.9 m/s.\n"
        "Sources: results/e07/matrix/episodes.csv.gz, results/e01/episodes.csv, "
        "results/e01_lowvz_cut/episodes.csv (rel_vz_normal_m_s)",
        fontsize=7,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.62 / height, 1, 1 - 0.12 / height), h_pad=1.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return out_png
