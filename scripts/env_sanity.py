"""Run the e00 environment sanity sweep and write the two Phase 2 artifacts.

Phase 2, Gate 2. Argparse wrapper only; the sweep lives in :mod:`rld.envs.sanity`.

Usage::

    python scripts/env_sanity.py --out results/e00_env_sanity.csv \
        --episodes-out results/e00_env_sanity_episodes.csv --workers 24

Two artifacts:

* ``results/e00_env_sanity.csv`` -- one row per (policy, sea state): 4 rows for ``hover``
  and ``random`` at ``id`` SS3 and SS5, with the six outcome-class fractions, the touchdown
  statistics and the analytic-versus-contact disagreement rate Gate 2 reads.
* ``results/e00_env_sanity_episodes.csv`` -- one row per episode, 800 rows: the audit trail
  behind the table, carrying the realization key, the start offset, the initial state and
  both touchdown records.

Episodes come from the ``id`` split's **val** partition with the draw seed recorded in every
row (P2-D4), so they are neither trained on nor part of the Phase 3 frozen evaluation lists.
``--workers`` is deliberately **not** in the provenance block: work splits over chunks of
episodes and the output is sorted into draw order, so the CSVs do not depend on it.

Runs headless (PyBullet DIRECT) and needs no GPU.
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
from rld.envs.config import (
    LANDING_CONFIG,
    NOISE_CONFIG,
    OBSERVATION_CONFIG,
    REWARD_CONFIG,
    SUCCESS_CONFIG,
    load_landing,
    load_noise,
    load_observation,
    load_reward,
    load_success,
)
from rld.envs.sanity import (
    DEFAULT_CHUNK,
    DEFAULT_DRAW_SEED,
    DEFAULT_EPISODES,
    DEFAULT_SEA_STATES,
    DEFAULT_WORKERS,
    SweepConfigs,
    rows_as_dicts,
    run_sweep,
)
from rld.provenance import environment_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "e00_env_sanity.csv",
        help="per-cell CSV (default: results/e00_env_sanity.csv)",
    )
    parser.add_argument(
        "--episodes-out",
        type=Path,
        default=REPO_ROOT / "results" / "e00_env_sanity_episodes.csv",
        help="per-episode CSV (default: results/e00_env_sanity_episodes.csv)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"processes, split over chunks of episodes (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"episodes per (policy, sea state) cell (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--sea-states",
        nargs="+",
        default=list(DEFAULT_SEA_STATES),
        help=f"sea states to sweep (default: {list(DEFAULT_SEA_STATES)})",
    )
    parser.add_argument(
        "--pad",
        default="aft",
        help="pad from configs/deck/pad.yaml: 'aft' (primary) or 'cg' (control arm)",
    )
    parser.add_argument(
        "--draw-seed",
        type=int,
        default=DEFAULT_DRAW_SEED,
        help=f"committed episode-draw seed, P2-D4 (default: {DEFAULT_DRAW_SEED})",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CHUNK,
        help=f"episodes per parallel task (default: {DEFAULT_CHUNK})",
    )
    parser.add_argument("--landing-config", type=Path, default=LANDING_CONFIG)
    parser.add_argument("--success-config", type=Path, default=SUCCESS_CONFIG)
    parser.add_argument("--observation-config", type=Path, default=OBSERVATION_CONFIG)
    parser.add_argument("--noise-config", type=Path, default=NOISE_CONFIG)
    parser.add_argument("--reward-config", type=Path, default=REWARD_CONFIG)
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
    motion = load_motion(args.motion_config)
    landing = load_landing(args.landing_config)
    cfgs = SweepConfigs(
        sim=load_sim(motion.sim_config_path),
        scaling=load_scaling(args.scaling_config),
        pads=load_pads(args.pad_config),
        motion=motion,
        landing=landing,
        success=load_success(args.success_config),
        observation=load_observation(args.observation_config),
        noise=load_noise(args.noise_config),
        reward=load_reward(args.reward_config),
    )
    provenance = environment_provenance(
        REPO_ROOT,
        # `v_max_m_s`, `driver` and `episode_draw_seed` are deliberately absent: they are
        # per-row columns already, and a provenance key that shadows a measured column
        # would silently win the merge in write_csv.
        extra={
            "drone_model": "cf2x",
            "pyb_freq_hz": str(landing.physics_freq_hz),
            "ctrl_freq_hz": str(landing.ctrl_freq_hz),
            "episode_len_s": repr(landing.episode_len_s),
            "dwell_grace_s": repr(landing.dwell_grace_s),
        },
    )

    started = time.perf_counter()
    cell_rows, episode_rows = run_sweep(
        cfgs,
        pad=args.pad,
        sea_states=tuple(args.sea_states),
        n_episodes=args.episodes,
        draw_seed=args.draw_seed,
        workers=args.workers,
        chunk=args.chunk,
    )
    wall_s = time.perf_counter() - started

    write_csv(args.out, rows_as_dicts(cell_rows), provenance)
    write_csv(args.episodes_out, rows_as_dicts(episode_rows), provenance)

    for row in cell_rows:
        print(
            f"{row.policy:>6s} {row.ss} n={row.n_episodes:<4d} "
            f"crash={row.frac_crash:.3f} off_pad={row.frac_off_pad:.3f} "
            f"hard={row.frac_hard_landing:.3f} bounce={row.frac_bounce:.3f} "
            f"success={row.frac_success:.3f} timeout={row.frac_timeout:.3f} | "
            f"disagree={row.touchdown_disagreement_n}/{row.n_episodes} "
            f"({row.touchdown_disagreement_rate:.4f}) "
            f"pen_max={row.max_penetration_m:.5f} m tunnel={row.tunnelling_n}"
        )
    print(f"{len(cell_rows)} cell rows -> {args.out}")
    print(f"{len(episode_rows)} episode rows -> {args.episodes_out}")
    print(f"{wall_s:.1f} s wall with {args.workers} worker(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
