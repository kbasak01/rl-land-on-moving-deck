r"""Evaluate registered controllers on the frozen episode lists -> results/e01/, results/e02/.

Argparse wrapper only; the logic lives in :mod:`rld.eval`.

Usage::

    python scripts/eval_baselines.py --workers 24          # Phase 3 e01: the 5 baselines, aft
    python scripts/eval_baselines.py --render-only         # re-render the markdown from CSV
    python scripts/eval_baselines.py --out-dir results/e02 --pad both \\
        --controllers pid_track_descend pid_feedforward pid_feedforward_lowvz gated \\
        oracle_gated gated_forecast gated_forecast_tcn     # Phase 4 e02

``--controllers`` defaults to the **five Phase 3 controllers**, not the whole registry, so
``make baselines`` reproduces e01 whatever is registered later.

Before anything runs, every list in ``results/episodes/`` is checked against
``MANIFEST.csv`` (file and content SHA-256); a mismatch aborts, because a result on a list
other than the frozen one is not a result.

Pads (``--pad``): ``aft`` flies the listed (frozen, aft) pad exactly as e01 did;
``cg`` flies the same listed episodes with the pad moved to the ship's CG
(``pad_override="cg"``, the control arm for dmf's roll/pitch-heave phase defect); ``both``
does both, aft first. The ``pad`` column of ``episodes.csv`` and ``summary.csv`` is the
effective pad; a summary written before the column existed (e01) is all-aft.

Controllers that consume the ship-motion feed (``gated_forecast*``) cannot fly the
static-pad list (there is no ship); those cells are recorded in ``summary.csv`` with
``n_episodes = 0`` and a ``skip_reason``, and printed as "not run" -- never dropped.

Reproduction check (``--reference-dir``, default ``results/e01``): unless the output *is*
the reference, every aft-pad row of a method the reference also holds is compared, line for
line, with the reference ``episodes.csv``, and the per-method counts go into
``run_info.json["reference_check"]`` together with the per-method provenance columns whose
committed value differs from the live one (``oracle_gated.yaml``'s comment-only relabel).
A mismatch is reported and the exit status is 1, after everything is written.

Scratch options -- refused when ``--out-dir`` lies under ``results/``: ``--lists`` (a subset
of lists), ``--per-cell K`` (the first K listed episodes of each cell), and
``--config-override NAME=PATH`` (fly a controller from another YAML, e.g. one pointing at a
smoke forecaster). A forecaster whose ``meta.json`` says ``smoke`` is also refused under
``results/``.

Writes into ``--out-dir`` (the tuning CSVs already there are left alone):

* ``episodes.csv`` -- one row per (pad, controller, episode),
  :data:`rld.eval.runner.EPISODE_COLUMNS`;
* ``summary.csv`` -- one row per (pad, controller, regime, sea state): the six outcome
  fractions, success with Wilson 95 % CI and N, touchdown quality, disagreement and
  tunnelling, the quiescent-touchdown fraction, the privileged flag, the
  ``in_training_distribution`` fraction, ``n_listed`` and ``skip_reason``, and
  deterministic provenance (versions, submodule SHAs, config and list SHA-256s, each
  controller's YAML SHA-256 beside the SHA-256 of its fully resolved config, and every
  forecaster file's SHA-256);
* ``success_vs_seastate.md`` -- rendered from ``summary.csv``;
* ``run_info.json`` -- the non-deterministic facts (timestamp, host, workers, wall time) and
  the reproduction check, kept out of the CSVs so that they are byte-identical at any
  ``--workers``.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from rld.config import REPO_ROOT
from rld.control.registry import REGISTRY
from rld.envs.config import LANDING_CONFIG, NOISE_CONFIG, OBSERVATION_CONFIG, SUCCESS_CONFIG
from rld.eval.controller_config import (
    PROVENANCE_COLUMNS,
    controller_provenance_columns,
    resolved_config_json,
)
from rld.eval.envs import load_eval_configs
from rld.eval.episodes import (
    DEFAULT_GENERATOR_SEED,
    EPISODES_DIR,
    MANIFEST_NAME,
    ListedEpisode,
    content_sha256,
    file_sha256,
    list_names,
    read_list,
    read_manifest,
)
from rld.eval.report import (
    SkipRecord,
    deterministic_provenance,
    read_rows,
    render_success_vs_seastate,
    summarise,
    summary_columns,
    write_rows,
)
from rld.eval.reproduce import compare_to_reference, summary_provenance_differences
from rld.eval.runner import (
    DEFAULT_CHUNK,
    DEFAULT_WORKERS,
    EPISODE_COLUMNS,
    EvalArm,
    controller_spec,
    run_arms,
    split_runnable,
)
from rld.provenance import environment_provenance

#: The Phase 3 controllers: ``make baselines`` flies exactly these, so it reproduces e01.
PHASE3_CONTROLLERS: tuple[str, ...] = (
    "pid_track_descend",
    "pid_feedforward",
    "pid_feedforward_lowvz",
    "gated",
    "oracle_gated",
)

#: Document titles by output directory name; the committed e01 title is unchanged.
TITLES: dict[str, str] = {
    "e01": "e01 — classical baselines: success versus sea state (frozen episode lists)",
    "e02": (
        "e02 — baselines, gated and forecast-gated controllers, aft pad and pad at CG: "
        "success versus sea state (frozen episode lists)"
    ),
}

#: ``--pad`` choice -> pad overrides, in output order (``None`` = the listed, frozen pad).
PAD_ARMS: dict[str, tuple[str | None, ...]] = {
    "aft": (None,),
    "cg": ("cg",),
    "both": (None, "cg"),
}

#: Provenance keys that vary between runs of identical results; they go to run_info.json.
VOLATILE_KEYS = ("omp_num_threads", "cpu_count", "host", "timestamp_utc")

RESULTS_ROOT = (REPO_ROOT / "results").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "e01")
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--controllers", nargs="+", default=list(PHASE3_CONTROLLERS))
    parser.add_argument("--pad", choices=sorted(PAD_ARMS), default="aft")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--reference-dir", type=Path, default=REPO_ROOT / "results" / "e01")
    parser.add_argument("--title", default=None, help="document title (default: by out-dir)")
    parser.add_argument("--render-only", action="store_true")
    scratch = parser.add_argument_group("scratch only (refused under results/)")
    scratch.add_argument("--lists", nargs="+", default=None, help="subset of list names")
    scratch.add_argument("--per-cell", type=int, default=None, help="first K episodes per cell")
    scratch.add_argument(
        "--config-override", nargs="+", default=[], metavar="NAME=PATH", help="controller YAML"
    )
    return parser.parse_args()


def title_for(out_dir: Path, title: str | None) -> str:
    """Return the document title: explicit, by directory name, or generic."""
    if title:
        return title
    return TITLES.get(
        out_dir.name, f"{out_dir.name} — success versus sea state (frozen episode lists)"
    )


def verify_lists(episodes_dir: Path) -> None:
    """Abort unless every list matches the committed manifest."""
    for row in read_manifest(episodes_dir / MANIFEST_NAME):
        path = episodes_dir / row["file"]
        if file_sha256(path) != row["file_sha256"]:
            raise SystemExit(f"{path}: file SHA-256 differs from MANIFEST.csv")
        if content_sha256(read_list(path)) != row["content_sha256"]:
            raise SystemExit(f"{path}: content SHA-256 differs from MANIFEST.csv")


def render(out_dir: Path, title: str) -> None:
    summary = read_rows(out_dir / "summary.csv")
    text = render_success_vs_seastate(summary, title)
    (out_dir / "success_vs_seastate.md").write_text(text, encoding="utf-8")


def first_per_cell(episodes: Sequence[ListedEpisode], k: int) -> list[ListedEpisode]:
    """Return the first ``k`` listed episodes of every (regime, sea state) cell, in order."""
    seen: dict[tuple[str, str], int] = {}
    out: list[ListedEpisode] = []
    for row in episodes:
        cell = (row.regime, row.ss)
        if seen.get(cell, 0) < k:
            seen[cell] = seen.get(cell, 0) + 1
            out.append(row)
    return out


def parse_overrides(items: Sequence[str]) -> dict[str, Path]:
    """Parse ``NAME=PATH`` controller YAML overrides."""
    out: dict[str, Path] = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or name not in REGISTRY:
            raise SystemExit(f"--config-override {item!r}: expected NAME=PATH with NAME registered")
        out[name] = Path(path).resolve()
        if not out[name].is_file():
            raise SystemExit(f"--config-override {item!r}: no such file")
    return out


def main() -> int:
    args = parse_args()
    title = title_for(args.out_dir, args.title)
    if args.render_only:
        render(args.out_dir, title)
        return 0
    unknown = sorted(set(args.controllers) - set(REGISTRY))
    if unknown:
        raise SystemExit(f"unknown controllers {unknown}")
    order = [name for name in REGISTRY if name in args.controllers]
    overrides = parse_overrides(args.config_override)
    committed = args.out_dir.resolve().is_relative_to(RESULTS_ROOT)
    if committed and (args.lists or args.per_cell or overrides):
        raise SystemExit("--lists / --per-cell / --config-override are scratch-only, not results/")
    names = list(list_names()) if args.lists is None else list(args.lists)
    if not set(names) <= set(list_names()):
        raise SystemExit(f"unknown lists {sorted(set(names) - set(list_names()))}")

    # 1. The frozen lists, before anything else.
    verify_lists(args.episodes_dir)
    episodes = [row for name in names for row in read_list(args.episodes_dir / f"{name}.parquet")]
    if args.per_cell is not None:
        episodes = first_per_cell(episodes, args.per_cell)

    # 2. Provenance up front: fails fast on a missing forecaster, refuses a smoke one.
    per_method = controller_provenance_columns(order, config_paths=overrides)
    if committed:
        for name in order:
            doc = json.loads(resolved_config_json(name))
            if doc.get("forecaster", {}).get("smoke"):
                raise SystemExit(
                    f"{name}: smoke forecaster {doc['forecaster']['dir']} under results/"
                )

    cfgs = load_eval_configs()
    specs = {name: controller_spec(name, overrides.get(name)) for name in order}
    arms: list[EvalArm] = []
    skipped: list[SkipRecord] = []
    for pad_override in PAD_ARMS[args.pad]:
        for name in order:
            runnable, skips = split_runnable(episodes, specs[name], pad_override)
            arms.append(EvalArm(specs[name], pad_override, runnable))
            skipped += [
                SkipRecord(
                    name,
                    specs[name].privileged,
                    specs[name].run_seed,
                    s.pad,
                    s.episode.regime,
                    s.episode.ss,
                    s.reason,
                )
                for s in skips
            ]

    started = time.perf_counter()
    rows = run_arms(arms, cfgs, workers=args.workers, chunk=args.chunk)
    wall_s = time.perf_counter() - started

    after = controller_provenance_columns(order, config_paths=overrides)
    if after != per_method:
        raise SystemExit("controller or forecaster files changed during the run; not writing")

    env_prov = environment_provenance(REPO_ROOT)
    versions = {k: v for k, v in env_prov.items() if k not in VOLATILE_KEYS}
    provenance = deterministic_provenance(
        versions,
        {
            "success_yaml_sha256": SUCCESS_CONFIG,
            "landing_yaml_sha256": LANDING_CONFIG,
            "observation_yaml_sha256": OBSERVATION_CONFIG,
            "noise_yaml_sha256": NOISE_CONFIG,
            "episodes_manifest_sha256": args.episodes_dir / MANIFEST_NAME,
        },
        {
            "generator_seed": str(DEFAULT_GENERATOR_SEED),
            "v_max_m_s": repr(cfgs.landing.v_max_m_s),
            "driver": cfgs.landing.platform.driver,
        },
    )
    summary = summarise(rows, order, per_method, provenance, skipped)
    extra_cols = [*PROVENANCE_COLUMNS, *provenance]

    write_rows(args.out_dir / "episodes.csv", rows, EPISODE_COLUMNS)
    write_rows(args.out_dir / "summary.csv", summary, summary_columns(extra_cols))
    render(args.out_dir, title)

    reference_check: dict[str, object] | None = None
    ref_dir = args.reference_dir.resolve()
    if ref_dir != args.out_dir.resolve() and (ref_dir / "episodes.csv").is_file():
        reference_check = compare_to_reference(rows, ref_dir / "episodes.csv", pad="aft")
        reference_check["summary_provenance_differences"] = summary_provenance_differences(
            ref_dir / "summary.csv", per_method
        )
    run_info = {
        **{k: env_prov[k] for k in VOLATILE_KEYS},
        "workers": args.workers,
        "chunk": args.chunk,
        "wall_s": round(wall_s, 1),
        "episodes": len(rows),
        "controllers": order,
        "pads": ["aft" if p is None else p for p in PAD_ARMS[args.pad]],
        "lists": names,
        "per_cell": args.per_cell,
        "config_overrides": {k: str(v) for k, v in overrides.items()},
        "skipped_episodes": len(skipped),
        "reference_check": reference_check,
    }
    (args.out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )

    for rec in summary:
        if rec["n_episodes"] == 0:
            print(
                f"{rec['method']:>21s} {rec['pad']:>3s} {rec['regime']:>15s} {rec['ss']:>6s} "
                f"NOT RUN n_listed={rec['n_listed']}: {rec['skip_reason']}"
            )
            continue
        print(
            f"{rec['method']:>21s} {rec['pad']:>3s} {rec['regime']:>15s} {rec['ss']:>6s} "
            f"success={rec['success_rate']:.3f} [{rec['success_wilson_lo']:.3f}, "
            f"{rec['success_wilson_hi']:.3f}] n={rec['n_episodes']} "
            f"crash={rec['frac_crash']:.3f} off={rec['frac_off_pad']:.3f} "
            f"hard={rec['frac_hard_landing']:.3f} bounce={rec['frac_bounce']:.3f} "
            f"timeout={rec['frac_timeout']:.3f} dis={rec['disagreement_n']} "
            f"tun={rec['tunnelling_n']}"
        )
    print(f"{len(rows)} episode rows, {len(summary)} summary rows -> {args.out_dir}")
    print(f"{wall_s:.1f} s wall with {args.workers} worker(s)")
    if reference_check is not None:
        print(f"reference check vs {ref_dir}: all_identical={reference_check['all_identical']}")
        if not reference_check["all_identical"]:
            print("REFERENCE MISMATCH: see run_info.json['reference_check']", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
