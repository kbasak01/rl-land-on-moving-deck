"""Tune the classical baselines on the P3-D2 tune pool; or check the final configs on it.

Phase 3, Step A. Argparse wrapper only; the procedure lives in :mod:`rld.control.tuning`
and is recorded in ``docs/protocol.md`` P3-D3.

Usage::

    # tune both, same budget, same draw (writes results/e01/tuning_<controller>.csv)
    python scripts/tune_controller.py --controller pid_track_descend pid_feedforward

    # after the winners are in configs/control/*.yaml: the five Phase 3 controllers
    # (rld.control.tuning.FINAL_CONTROLLERS, pinned) at their committed configs on the same
    # tune draw (writes results/e01/tune_pool_final.csv)
    python scripts/tune_controller.py --final

Tune pool only: frigate, SS3-SS5, heading != 90 deg, seed ordinals 27-31. Never the frozen
evaluation lists. The winning gains are printed; they are copied into the controller YAMLs
by hand so the YAML comments survive, and ``tests/test_control_pid_*.py`` asserts that the
committed YAML equals the selected trial in the committed log.

``--workers`` is deliberately not in the provenance block: the output does not depend on it.
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Any

from dmf.config import load_sim

from rld.control.registry import entry
from rld.control.tuning import (
    DEFAULT_CHUNK,
    FINAL_CONTROLLERS,
    TUNED_CONTROLLERS,
    TUNING_CONFIG,
    TrialResult,
    draw_tuning_episodes,
    load_tuning,
    run_trials,
    select_trial,
    summarise_trial,
    trial_points,
)
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
from rld.envs.sanity import SweepConfigs
from rld.provenance import environment_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        nargs="+",
        default=list(TUNED_CONTROLLERS),
        choices=list(TUNED_CONTROLLERS),
        help="controllers to tune (default: both, with the identical budget)",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="instead of tuning, run the five Phase 3 controllers (FINAL_CONTROLLERS) at "
        "their committed configs on the tune draw and write tune_pool_final.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "e01")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--tuning-config", type=Path, default=TUNING_CONFIG)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], provenance: dict[str, str]) -> None:
    """Write rows plus the provenance block, unrounded.

    Raises:
        ValueError: If a provenance key shadows a measured column.
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


def print_result(result: TrialResult, sea_states: tuple[str, ...], mark: str = "") -> None:
    row = result.row
    parts = [f"{row['controller']:>17s} t{row['trial']:02d}{mark}"]
    parts += [f"{k[6:]}={v:.4g}" for k, v in row.items() if k.startswith("param_")]
    for ss in sea_states:
        parts.append(
            f"{ss}: S={row[f'{ss}_frac_success']:.3f} C={row[f'{ss}_frac_crash']:.3f} "
            f"O={row[f'{ss}_frac_off_pad']:.3f} H={row[f'{ss}_frac_hard_landing']:.3f} "
            f"B={row[f'{ss}_frac_bounce']:.3f} T={row[f'{ss}_frac_timeout']:.3f}"
        )
    parts.append(f"mean={result.objective:.4f} p95vz={result.tie_break_m_s:.3f}")
    print(" | ".join(parts), flush=True)


def main() -> int:
    args = parse_args()
    tuning = load_tuning(args.tuning_config)
    motion = load_motion(MOTION_JONSWAP_CONFIG)
    landing = load_landing(LANDING_CONFIG)
    cfgs = SweepConfigs(
        sim=load_sim(motion.sim_config_path),
        scaling=load_scaling(SCALING_CONFIG),
        pads=load_pads(PAD_CONFIG),
        motion=motion,
        landing=landing,
        success=load_success(SUCCESS_CONFIG),
        observation=load_observation(OBSERVATION_CONFIG),
        noise=load_noise(NOISE_CONFIG),
        reward=load_reward(REWARD_CONFIG),
    )
    episodes = draw_tuning_episodes(cfgs.sim, tuning)
    pool = "dev_pool_tune:frigate|SS3-SS5|heading!=90|seeds27-31"
    provenance = environment_provenance(
        REPO_ROOT,
        extra={
            "pool": pool,
            "tuning_seed": str(tuning.tuning_seed),
            "sobol_seed": str(tuning.sobol_seed),
            "episodes_per_sea_state": str(tuning.episodes_per_sea_state),
            "sea_states": "|".join(tuning.sea_states),
            "pad": tuning.pad,
            "trials_budget": str(tuning.trials),
            "v_max_m_s": repr(landing.v_max_m_s),
            "driver": landing.platform.driver,
        },
    )

    if args.final:
        results: list[TrialResult] = []
        for name in FINAL_CONTROLLERS:
            started = time.perf_counter()
            rows = run_trials(name, [{}], episodes, cfgs, workers=args.workers, chunk=args.chunk)
            result = summarise_trial(name, 0, {}, rows, tuning.sea_states)
            result.row["privileged"] = entry(name).privileged
            result.row["config"] = str(entry(name).config_path.relative_to(REPO_ROOT))
            result.row["wall_s"] = time.perf_counter() - started
            results.append(result)
            print_result(result, tuning.sea_states)
        write_csv(args.out_dir / "tune_pool_final.csv", [r.row for r in results], provenance)
        print(f"-> {args.out_dir / 'tune_pool_final.csv'}")
        return 0

    for name in args.controller:
        points = trial_points(name, tuning)
        started = time.perf_counter()
        rows = run_trials(name, points, episodes, cfgs, workers=args.workers, chunk=args.chunk)
        wall = time.perf_counter() - started
        results = [
            summarise_trial(
                name, trial, point, [r for r in rows if r["trial"] == trial], tuning.sea_states
            )
            for trial, point in enumerate(points)
        ]
        best = select_trial(results)
        for trial, result in enumerate(results):
            result.row["selected"] = trial == best
            print_result(result, tuning.sea_states, " *" if trial == best else "")
        out = args.out_dir / f"tuning_{name}.csv"
        write_csv(out, [r.row for r in results], provenance)
        print(f"{name}: {len(points)} trials x {len(episodes)} episodes in {wall:.0f} s -> {out}")
        print(f"{name}: selected trial {best}: {points[best]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
