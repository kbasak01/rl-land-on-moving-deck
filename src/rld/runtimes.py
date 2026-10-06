"""Per-stage wall clock for the README's Reproduce table (Phase 9).

Collects every recorded wall-clock time into one table, ``results/runtime_stages.csv``: one
row per measured item, with its source and whether that source is committed. Sources:

- committed ``run_info.json`` files of each flown stage (``results/e01``, ``e01_lowvz_cut``,
  ``e02``, ``e02_tcn_seeds``, ``e07``, ``latency``, ``latency/closed_loop_investigation``);
- ``results/e00_env_sanity.csv`` (``wall_s``), ``results/forecast/fit_manifest.json``
  (``wall_time_s``) and ``results/tune/<search>/trials.csv`` (``wall_s``);
- the gitignored training ``artifacts/runs/<method>/<seed>/status.json`` (``wall_s``) and the
  gitignored ``artifacts/dmf/*.log`` (shell ``time`` output). These rows carry
  ``committed = False``: the CSV is then their only committed record, as in dmf;
- optionally, stages timed by :func:`measure` on this machine (``source = measured``).

Units: seconds of wall clock, on the machine the stage ran on (i9-10980XE, 36 logical CPUs, RTX
A4000, WSL2). Nothing here is simulated time.
"""

import csv
import io
import json
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["COLUMNS", "RuntimeRow", "collect", "measure", "write_csv"]

COLUMNS: tuple[str, ...] = (
    "stage",
    "target",
    "item",
    "wall_s",
    "workers",
    "source",
    "committed",
)

#: Training run groups whose final runs are the evaluated checkpoints (5 seeds each).
TRAINING_GROUPS: tuple[str, ...] = (
    "ppo",
    "sac",
    "residual_ppo",
    "ppo_forecast",
    "residual_ppo_forecast",
    "ppo_sinusoid",
)


@dataclass(frozen=True)
class RuntimeRow:
    """One measured item.

    Attributes:
        stage: Stage name (``make`` target or training family).
        target: The ``make`` target that runs it (``train-bg`` / ``sweep`` / ``tune`` for
            training, launched by hand).
        item: What was timed.
        wall_s: Wall clock, seconds.
        workers: Worker processes, or ``""`` when not recorded.
        source: File (and field) the number is read from.
        committed: Whether ``source`` is a committed file.
    """

    stage: str
    target: str
    item: str
    wall_s: float
    workers: str
    source: str
    committed: bool


def _json(path: Path) -> dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _run_info_rows(root: Path, rel: str, stage: str, target: str, item: str) -> list[RuntimeRow]:
    path = root / rel / "run_info.json"
    if not path.exists():
        return []
    info = _json(path)
    return [
        RuntimeRow(
            stage,
            target,
            item,
            float(str(info["wall_s"])),
            str(info.get("workers", "")),
            f"{rel}/run_info.json wall_s",
            True,
        )
    ]


def _time_log(path: Path) -> float | None:
    """Return the ``real`` time of a shell ``time`` block in a log, seconds."""
    if not path.exists():
        return None
    match = re.search(r"^real\s+(\d+)m([\d.]+)s", path.read_text(encoding="utf-8"), re.M)
    return None if match is None else 60.0 * float(match.group(1)) + float(match.group(2))


def collect(repo: Path) -> list[RuntimeRow]:
    """Collect every recorded wall-clock item.

    Args:
        repo: Repository root.

    Returns:
        Rows in stage order of ``make all``, then training and tuning.
    """
    results = repo / "results"
    artifacts = repo / "artifacts"
    rows: list[RuntimeRow] = []

    sanity = results / "e00_env_sanity.csv"
    if sanity.exists():
        with sanity.open(encoding="utf-8") as fh:
            blocks: dict[tuple[str, str], list[str]] = {}
            for rec in csv.DictReader(fh):  # one timed block can cover several sea states
                blocks.setdefault((rec["policy"], rec["wall_s"]), []).append(rec["ss"])
        for (policy, wall), sea_states in blocks.items():
            rows.append(
                RuntimeRow(
                    "env-sanity",
                    "env-sanity",
                    f"{policy} policy, {' + '.join(sea_states)}",
                    float(wall),
                    "",
                    "results/e00_env_sanity.csv wall_s",
                    True,
                )
            )

    rows += _run_info_rows(results, "e01", "baselines", "baselines", "e01 flights")
    for rel, item in (
        ("e01_lowvz_cut", "e01_lowvz_cut flights (Phase 5)"),
        ("e02", "e02 forecast-gated flights (Phase 4)"),
        ("e02_tcn_seeds", "e02 TCN seed arms (Phase 4)"),
    ):
        rows += _run_info_rows(results, rel, "baselines (later phases)", "-", item)

    corpus = _time_log(artifacts / "dmf" / "corpus.log")
    if corpus is not None:
        rows.append(
            RuntimeRow(
                "dmf-forecasters",
                "dmf-forecasters",
                "dmf corpus (2304 realizations)",
                corpus,
                "24",
                "artifacts/dmf/corpus.log real",
                False,
            )
        )
    closed = _time_log(artifacts / "dmf" / "fit_closed_form.log")
    if closed is not None:
        rows.append(
            RuntimeRow(
                "dmf-forecasters",
                "dmf-forecasters",
                "closed-form fits",
                closed,
                "",
                "artifacts/dmf/fit_closed_form.log real",
                False,
            )
        )
    manifest = results / "forecast" / "fit_manifest.json"
    if manifest.exists():
        models = dict(_json(manifest)["models"])  # type: ignore[call-overload]
        for name in ("tcn", "tcn_quantile"):
            for k, seed in enumerate(models[name]["seeds"]):
                rows.append(
                    RuntimeRow(
                        "dmf-forecasters",
                        "dmf-forecasters",
                        f"{name} fit seed {k}",
                        float(seed["wall_time_s"]),
                        "",
                        f"results/forecast/fit_manifest.json models.{name}.seeds[{k}].wall_time_s",
                        True,
                    )
                )

    e07 = results / "e07" / "run_info.json"
    if e07.exists():
        for k, inv in enumerate(list(_json(e07)["invocations"])):  # type: ignore[call-overload]
            if "wall_s" not in inv:
                continue
            argv = " ".join(inv["argv"])
            superseded = k == 4  # the first noise arm, superseded by P7-D4 and re-flown
            rows.append(
                RuntimeRow(
                    "eval",
                    "eval",
                    f"eval_phase7.py {argv}" + (" (superseded, P7-D4)" if superseded else ""),
                    float(inv["wall_s"]),
                    str(inv.get("workers", "")),
                    f"results/e07/run_info.json invocations[{k}].wall_s",
                    True,
                )
            )

    latency = results / "latency" / "run_info.json"
    if latency.exists():
        info = _json(latency)
        for key in sorted(k for k in info if k.startswith("wall_s_")):
            rows.append(
                RuntimeRow(
                    "bench",
                    "bench",
                    key[len("wall_s_") :],
                    float(str(info[key])),
                    "",
                    f"results/latency/run_info.json {key}",
                    True,
                )
            )
    inv_path = results / "latency" / "closed_loop_investigation" / "run_info.json"
    if inv_path.exists():
        rows.append(
            RuntimeRow(
                "bench-investigate",
                "bench-investigate",
                "total",
                float(str(_json(inv_path)["wall_s_total"])),
                "",
                "results/latency/closed_loop_investigation/run_info.json wall_s_total",
                True,
            )
        )

    for group in TRAINING_GROUPS:
        for seed in range(5):
            status = artifacts / "runs" / group / str(seed) / "status.json"
            if not status.exists():
                continue
            info = _json(status)
            rows.append(
                RuntimeRow(
                    f"training: {group}",
                    "sweep",
                    f"seed {seed}, {info.get('steps')} env steps",
                    float(str(info["wall_s"])),
                    "",
                    f"artifacts/runs/{group}/{seed}/status.json wall_s",
                    False,
                )
            )

    for search in ("ppo", "sac_v2"):
        trials = results / "tune" / search / "trials.csv"
        if not trials.exists():
            continue
        with trials.open(encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                if not rec["wall_s"]:
                    continue
                rows.append(
                    RuntimeRow(
                        f"tuning: {search}",
                        "tune",
                        f"trial {rec['trial']} ({rec['state']})",
                        float(rec["wall_s"]),
                        "",
                        f"results/tune/{search}/trials.csv wall_s",
                        True,
                    )
                )
    v1 = artifacts / "runs" / "tune_sac"
    if v1.exists():
        for status in sorted(v1.glob("trial_*/*/status.json")):
            info = _json(status)
            if "wall_s" not in info:
                continue
            rows.append(
                RuntimeRow(
                    "tuning: sac v1 (superseded, P5-D10)",
                    "tune",
                    f"{status.parent.parent.name} ({info.get('state')})",
                    float(str(info["wall_s"])),
                    "",
                    f"artifacts/runs/tune_sac/{status.parent.parent.name}/"
                    f"{status.parent.name}/status.json wall_s",
                    False,
                )
            )
    return rows


def measure(repo: Path, stages: Sequence[str], workers: int = 24) -> list[RuntimeRow]:
    """Time stages that recorded no wall clock, writing their outputs to a temporary directory.

    Args:
        repo: Repository root.
        stages: Any of ``deck-stats``, ``report``.
        workers: Worker processes for ``deck-stats``.

    Returns:
        One row per stage, ``source = "measured"``. Nothing under ``results/`` is written.
    """
    rows: list[RuntimeRow] = []
    with tempfile.TemporaryDirectory() as tmp:
        commands = {
            "deck-stats": (
                [
                    sys.executable,
                    "scripts/deck_stats.py",
                    "--out",
                    f"{tmp}/deck_stats.csv",
                    "--seeds-out",
                    f"{tmp}/deck_stats_seeds.csv",
                    "--feasibility-out",
                    f"{tmp}/deck_feasibility.csv",
                    "--workers",
                    str(workers),
                    "--threshold-reference",
                    "all",
                ],
                str(workers),
            ),
            "report": ([sys.executable, "scripts/report.py", "--check"], ""),
        }
        for stage in stages:
            cmd, used = commands[stage]
            start = time.perf_counter()
            subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
            wall = time.perf_counter() - start
            rows.append(
                RuntimeRow(
                    stage,
                    stage,
                    "re-run to a temporary directory"
                    if stage == "deck-stats"
                    else "report.py --check",
                    wall,
                    used,
                    "measured (scripts/collect_runtimes.py --measure, Phase 9)",
                    True,
                )
            )
    return rows


def write_csv(path: Path, rows: Sequence[RuntimeRow]) -> str:
    r"""Write the rows (``\n`` endings, wall clock to 0.1 s) and return the text."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        rec = asdict(row)
        rec["wall_s"] = f"{row.wall_s:.1f}"
        writer.writerow(rec)
    text = buf.getvalue()
    path.write_text(text, encoding="utf-8")
    return text
