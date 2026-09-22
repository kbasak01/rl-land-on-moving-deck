"""Measure deck-point statistics over the dmf grid and write the Phase 1 artifacts.

Phase 1, Gate 1. Argparse wrapper only; the statistics live in :mod:`rld.deck.stats`.

Usage::

    python scripts/deck_stats.py --out results/deck_stats.csv \
        --seeds-out results/deck_stats_seeds.csv \
        --feasibility-out results/deck_feasibility.csv --workers 24

Three artifacts:

* ``results/deck_stats.csv`` -- one row per (cell, pad), 192 rows over the committed grid.
* ``results/deck_stats_seeds.csv`` -- one row per (realization, pad), 4608 rows: the audit
  trail behind the table and the committed per-realization sinusoid parameters.
* ``results/deck_feasibility.csv`` -- the pre-registered lambda rule, one row per candidate
  reference speed, with ``is_gate`` marking the row Gate 1 reads.

``--workers`` is deliberately **not** recorded in the provenance block: the output does not
depend on it (verified at 24 and 7 workers over the full grid, and asserted at 1 and 4 workers
in ``tests/test_deck_stats.py``), so re-running is byte-identical apart from the timestamp.

Sampling is at the **model** physics rate (240 Hz model = 48 Hz full at lambda = 1/25) over
the whole 600 s full-scale record. Work is parallelised over the 96 grid cells; the output is
sorted into grid order, so it does not depend on ``--workers``. Needs no GPU.
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from dmf.config import load_sim

from rld.deck.config import (
    MOTION_JONSWAP_CONFIG,
    PAD_CONFIG,
    SCALING_CONFIG,
    load_motion,
    load_pads,
    load_scaling,
)
from rld.deck.stats import DEFAULT_WORKERS, compute_grid, feasibility_rows, rows_as_dicts
from rld.provenance import environment_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "deck_stats.csv",
        help="per-cell CSV (default: results/deck_stats.csv)",
    )
    parser.add_argument(
        "--seeds-out",
        type=Path,
        default=REPO_ROOT / "results" / "deck_stats_seeds.csv",
        help="per-realization CSV (default: results/deck_stats_seeds.csv)",
    )
    parser.add_argument(
        "--feasibility-out",
        type=Path,
        default=REPO_ROOT / "results" / "deck_feasibility.csv",
        help="feasibility verdict CSV (default: results/deck_feasibility.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"processes, split over grid cells (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--threshold-reference",
        choices=["all", "gate"],
        default="all",
        help="emit the feasibility rule under both candidate denominators, or only the gate",
    )
    parser.add_argument(
        "--episode-seed",
        type=int,
        default=0,
        help="episode seed for the committed sinusoid phases (default: 0)",
    )
    parser.add_argument("--scaling-config", type=Path, default=SCALING_CONFIG)
    parser.add_argument("--pad-config", type=Path, default=PAD_CONFIG)
    parser.add_argument("--motion-config", type=Path, default=MOTION_JONSWAP_CONFIG)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], provenance: dict[str, str]) -> None:
    """Write rows plus the provenance block, at full float precision and unrounded.

    Raises:
        ValueError: If a provenance key shadows a measured column. Without this check the
            provenance string would silently replace the measured value.
    """
    shadowed = sorted(set(rows[0]) & set(provenance))
    if shadowed:
        raise ValueError(f"provenance keys shadow measured columns: {shadowed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{**row, **provenance} for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0]))
        writer.writeheader()
        writer.writerows(payload)


def main() -> int:
    args = parse_args()
    scaling = load_scaling(args.scaling_config)
    pads = load_pads(args.pad_config)
    motion = load_motion(args.motion_config)
    sim_cfg = load_sim(motion.sim_config_path)
    provenance = environment_provenance(
        REPO_ROOT,
        # `lam` and `sample_rate_full_hz` are deliberately absent: they are per-row columns
        # already, and a provenance key that shadows a measured column would silently win the
        # merge in write_csv.
        extra={
            "physics_freq_model_hz": repr(motion.physics_freq_hz),
            "corpus_fs_full_hz": repr(sim_cfg.fs_hz),
            "duration_full_s": repr(sim_cfg.duration_s),
            "spinup_full_s": repr(sim_cfg.spinup_s),
            "episode_seed": str(args.episode_seed),
        },
    )

    started = time.perf_counter()
    cell_rows, seed_rows = compute_grid(
        sim_cfg,
        scaling,
        pads,
        physics_freq_hz=motion.physics_freq_hz,
        workers=args.workers,
        episode_seed=args.episode_seed,
    )
    wall_s = time.perf_counter() - started
    verdicts = feasibility_rows(cell_rows, scaling, reference=args.threshold_reference)

    write_csv(args.out, rows_as_dicts(cell_rows), provenance)
    write_csv(args.seeds_out, rows_as_dicts(seed_rows), provenance)
    write_csv(args.feasibility_out, rows_as_dicts(verdicts), provenance)

    print(f"{len(cell_rows)} cell rows -> {args.out}")
    print(f"{len(seed_rows)} realization rows -> {args.seeds_out}")
    for row in verdicts:
        gate = " (GATE)" if row.is_gate else ""
        print(
            f"feasibility [{row.reference_name}]{gate}: "
            f"{row.vessel} {row.ss} {row.heading_deg:.0f} deg {row.speed_kn:.0f} kn "
            f"{row.pad} v_z p99 = {row.vz_p99_model_m_s:.4f} m/s vs threshold "
            f"{row.threshold_m_s:.4f} m/s -> {row.verdict}"
        )
    print(f"{wall_s:.1f} s wall with {args.workers} worker(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
