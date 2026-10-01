"""Per-episode sinusoidal deck motion for ``ppo_sinusoid`` (the H4 motion-realism arm, plan D0.4).

``ppo_sinusoid`` trains on the P3-D2 train pool with the P3-D1 curriculum, but every
episode's deck moves as a :class:`~rld.deck.sinusoid.SinusoidDeckMotion` matched to the
episode's realization, never as the JONSWAP realization itself; its periodic tune-pool
evaluation (learning curves and curriculum promotion) flies sinusoids too, so no JONSWAP
signal reaches its training.

Matching (``rld.deck.sinusoid``, P1):

* **amplitudes**, per DOF: ``sqrt(2) x`` the realization's RMS over its whole committed
  record, full scale (degrees for roll and pitch, metres for heave). The RMS is read from
  the committed ``results/deck_stats_seeds.csv`` (Phase 1), not recomputed: that file's
  ``sin_*_amp_*`` columns are exactly ``sqrt(2) x RMS``, so :func:`committed_rms_full`
  returns ``amp / sqrt(2)`` and :func:`~rld.deck.sinusoid.sinusoid_params` multiplies it
  back (a round trip of at most one float64 ulp). ``tests/test_rl_motion.py`` checks the
  value against :func:`~rld.deck.sinusoid.matched_rms` on JONSWAP realizations.
* **period**: the realization's peak encounter period.
* **phases**: one per DOF, from the **episode seed** (``sinusoid_params(...,
  episode_seed=...)``), so the same episode always has the same motion and different
  episodes of one realization differ.

The source has the JONSWAP source's time window and full-scale lookback, so episode start
offsets are drawn from the same span.

``rld.rl`` never reads ``results/episodes/`` (``tests/test_rl_leakage.py``); the deck-stats
file is a Phase 1 statistic over every realization, not an episode list.

Units: amplitudes and RMS full scale (degrees, metres); periods seconds full scale; the
environment consumes the source at model scale.
"""

import csv
import hashlib
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from dmf.data.splits import RealizationKey, realization_key
from dmf.sim.generate import RealizationSpec

from rld.config import REPO_ROOT
from rld.deck.bridge import load_vessel_cached
from rld.deck.sinusoid import SinusoidDeckMotion, sinusoid_params
from rld.deck.splits import key_for_spec
from rld.eval.envs import EvalConfigs

__all__ = [
    "DECK_STATS_SEEDS",
    "RmsTable",
    "committed_rms_full",
    "file_sha256",
    "sinusoid_motion",
]

#: Phase 1's per-(realization, pad) statistics, with the matched sinusoid's amplitudes.
DECK_STATS_SEEDS: Path = REPO_ROOT / "results" / "deck_stats_seeds.csv"

#: ``{realization key: (roll_rms_deg, pitch_rms_deg, heave_rms_m)}``, full scale.
type RmsTable = Mapping[RealizationKey, tuple[float, float, float]]


def file_sha256(path: Path) -> str:
    """Return a file's SHA-256 hex digest (for run provenance)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_rms_full(
    keys: Iterable[RealizationKey], path: Path = DECK_STATS_SEEDS, pad: str = "aft"
) -> dict[RealizationKey, tuple[float, float, float]]:
    """Read the committed per-realization RMS of roll, pitch and heave, full scale.

    Args:
        keys: The realizations needed (a run's train and tune pools).
        path: ``results/deck_stats_seeds.csv``.
        pad: Which pad's rows to read; the ship-motion amplitudes are the same on every
            pad row of a realization, and the primary arm is ``"aft"``.

    Returns:
        ``{key: (roll_rms_deg, pitch_rms_deg, heave_rms_m)}`` for every requested key,
        ``sin_*_amp / sqrt(2)``.

    Raises:
        KeyError: If a requested realization has no row.
        ValueError: If a row's amplitude is not finite and non-negative.
    """
    wanted = set(keys)
    root_two = math.sqrt(2.0)
    out: dict[RealizationKey, tuple[float, float, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["pad"] != pad:
                continue
            key = realization_key(
                row["ss"],
                float(row["heading_deg"]),
                float(row["speed_kn"]),
                row["vessel"],
                int(row["seed"]),
            )
            if key not in wanted:
                continue
            amps = (
                float(row["sin_roll_amp_deg"]),
                float(row["sin_pitch_amp_deg"]),
                float(row["sin_heave_amp_m"]),
            )
            if not all(math.isfinite(a) and a >= 0.0 for a in amps):
                raise ValueError(f"{path}: bad sinusoid amplitude for {key}: {amps}")
            out[key] = (amps[0] / root_two, amps[1] / root_two, amps[2] / root_two)
    if missing := sorted(wanted - set(out), key=str):
        raise KeyError(f"{path} has no {pad!r} row for {len(missing)} realizations: {missing[:3]}")
    return out


def sinusoid_motion(
    cfgs: EvalConfigs,
    spec: RealizationSpec,
    episode_seed: int,
    rms_full: tuple[float, float, float],
) -> SinusoidDeckMotion:
    """Build one episode's sinusoidal deck motion, matched to its realization.

    Args:
        cfgs: The committed configs (dmf corpus config, Froude scale, pads, lookback).
        spec: The episode's realization.
        episode_seed: The episode seed; the three DOF phases are drawn from it.
        rms_full: The realization's ``(roll_deg, pitch_deg, heave_m)`` RMS, full scale
            (:func:`committed_rms_full`).

    Returns:
        The source, with the JONSWAP source's window and full-scale lookback.
    """
    scale = cfgs.scaling.froude_scale()
    params = sinusoid_params(
        spec, cfgs.sim, scale, cfgs.pads, episode_seed=int(episode_seed), rms_full=rms_full
    )
    if params.key != key_for_spec(spec):
        raise RuntimeError(f"sinusoid key {params.key} != realization {key_for_spec(spec)}")
    return SinusoidDeckMotion(
        params,
        cfgs.sim,
        scale,
        cfgs.pads,
        length_full_m=float(load_vessel_cached(spec.vessel).length_m),
        lookback_full_s=cfgs.motion.forecast_lookback_full_s,
    )
