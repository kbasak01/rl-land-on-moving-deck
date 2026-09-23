"""Evaluate the registered classical baselines on the frozen episode lists -> results/e01/.

Phase 3, Step B. Argparse wrapper only; the logic lives in :mod:`rld.eval`.

Usage::

    python scripts/eval_baselines.py --workers 24          # evaluate, write, render
    python scripts/eval_baselines.py --render-only         # re-render the markdown from CSV

Before anything runs, every list in ``results/episodes/`` is checked against
``MANIFEST.csv`` (file and content SHA-256); a mismatch aborts, because a result on a list
other than the frozen one is not a Phase 3 result.

Writes into ``--out-dir`` (the tuning CSVs already there are left alone):

* ``episodes.csv`` -- one row per (controller, episode): 5 x 2 800 rows,
  :data:`rld.eval.runner.EPISODE_COLUMNS`;
* ``summary.csv`` -- one row per (controller, regime, sea state): the six outcome
  fractions, success with Wilson 95 % CI and N, touchdown quality, disagreement and
  tunnelling, the quiescent-touchdown fraction, the privileged flag, the
  ``in_training_distribution`` fraction, and deterministic provenance (versions, submodule
  SHAs, config and list SHA-256s, and each controller's YAML SHA-256 beside the SHA-256 of
  its fully resolved config, ``resolved_gains_sha256``);
* ``success_vs_seastate.md`` -- rendered from ``summary.csv``;
* ``run_info.json`` -- the non-deterministic facts (timestamp, host, workers, wall time),
  kept out of the CSVs so that they are byte-identical at any ``--workers``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from rld.config import REPO_ROOT
from rld.control.registry import REGISTRY
from rld.envs.config import LANDING_CONFIG, NOISE_CONFIG, OBSERVATION_CONFIG, SUCCESS_CONFIG
from rld.eval.controller_config import controller_provenance_columns
from rld.eval.envs import load_eval_configs
from rld.eval.episodes import (
    DEFAULT_GENERATOR_SEED,
    EPISODES_DIR,
    MANIFEST_NAME,
    content_sha256,
    file_sha256,
    list_names,
    read_list,
    read_manifest,
)
from rld.eval.report import (
    deterministic_provenance,
    read_rows,
    render_success_vs_seastate,
    summarise,
    summary_columns,
    write_rows,
)
from rld.eval.runner import (
    DEFAULT_CHUNK,
    DEFAULT_WORKERS,
    EPISODE_COLUMNS,
    controller_spec,
    run_matrix,
)
from rld.provenance import environment_provenance

TITLE = "e01 — classical baselines: success versus sea state (frozen episode lists)"

#: Provenance keys that vary between runs of identical results; they go to run_info.json.
VOLATILE_KEYS = ("omp_num_threads", "cpu_count", "host", "timestamp_utc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "e01")
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--controllers", nargs="+", default=list(REGISTRY))
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--render-only", action="store_true")
    return parser.parse_args()


def verify_lists(episodes_dir: Path) -> None:
    """Abort unless every list matches the committed manifest."""
    for row in read_manifest(episodes_dir / MANIFEST_NAME):
        path = episodes_dir / row["file"]
        if file_sha256(path) != row["file_sha256"]:
            raise SystemExit(f"{path}: file SHA-256 differs from MANIFEST.csv")
        if content_sha256(read_list(path)) != row["content_sha256"]:
            raise SystemExit(f"{path}: content SHA-256 differs from MANIFEST.csv")


def render(out_dir: Path) -> None:
    summary = read_rows(out_dir / "summary.csv")
    text = render_success_vs_seastate(summary, TITLE)
    (out_dir / "success_vs_seastate.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.render_only:
        render(args.out_dir)
        return 0
    unknown = sorted(set(args.controllers) - set(REGISTRY))
    if unknown:
        raise SystemExit(f"unknown controllers {unknown}")
    order = [name for name in REGISTRY if name in args.controllers]

    verify_lists(args.episodes_dir)
    episodes = [
        row for name in list_names() for row in read_list(args.episodes_dir / f"{name}.parquet")
    ]
    cfgs = load_eval_configs()
    specs = [controller_spec(name) for name in order]

    started = time.perf_counter()
    rows = run_matrix(episodes, specs, cfgs, workers=args.workers, chunk=args.chunk)
    wall_s = time.perf_counter() - started

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
    per_method = controller_provenance_columns(order)
    summary = summarise(rows, order, per_method, provenance)
    extra_cols = ["controller_config_sha256", "resolved_gains_sha256", *provenance]

    write_rows(args.out_dir / "episodes.csv", rows, EPISODE_COLUMNS)
    write_rows(args.out_dir / "summary.csv", summary, summary_columns(extra_cols))
    render(args.out_dir)
    run_info = {
        **{k: env_prov[k] for k in VOLATILE_KEYS},
        "workers": args.workers,
        "chunk": args.chunk,
        "wall_s": round(wall_s, 1),
        "episodes": len(rows),
        "controllers": order,
    }
    (args.out_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )

    for rec in summary:
        print(
            f"{rec['method']:>21s} {rec['regime']:>15s} {rec['ss']:>6s} "
            f"success={rec['success_rate']:.3f} [{rec['success_wilson_lo']:.3f}, "
            f"{rec['success_wilson_hi']:.3f}] n={rec['n_episodes']} "
            f"crash={rec['frac_crash']:.3f} off={rec['frac_off_pad']:.3f} "
            f"hard={rec['frac_hard_landing']:.3f} bounce={rec['frac_bounce']:.3f} "
            f"timeout={rec['frac_timeout']:.3f} dis={rec['disagreement_n']} "
            f"tun={rec['tunnelling_n']}"
        )
    print(f"{len(rows)} episode rows, {len(summary)} summary rows -> {args.out_dir}")
    print(f"{wall_s:.1f} s wall with {args.workers} worker(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
