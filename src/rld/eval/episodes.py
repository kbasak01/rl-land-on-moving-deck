"""The frozen evaluation episode lists: drawn once, read back from a real reset, hashed.

What a list is
--------------
One row per episode: a realization key ``(vessel, ss, heading, speed, seed)``, an episode
seed, and what a **real** :class:`~rld.envs.landing_env.DeckLandingAviary` reset produced from
them -- the start offset ``t0_model_s`` and the drone's initial position. Every method and
every training seed is evaluated on the identical rows, so every method contrast is paired.

Cells (P3-D1, P3-D2)
--------------------
Thirteen (regime, sea state) cells, N = 200 each, drawn from each regime's dmf **test**
partition (``part_specs(build_all_splits(cfg)[regime], "test")``):

=====================  ======================  =============================================
regime                 sea states              realizations in the test partition per SS
=====================  ======================  =============================================
``id``                 SS3, SS4, SS5, SS6      96 (frigate, 4 headings, 3 speeds, seeds 32-39)
``unseen_seastate``    SS6                     480 (frigate, 4 headings, 3 speeds, seeds 0-39)
``unseen_heading``     SS3, SS4, SS5, SS6      120 (frigate, 90 deg, 3 speeds, seeds 0-39)
``unseen_vessel``      SS3, SS4, SS5, SS6      96 (s175, 4 headings, 3 speeds, seeds 0-7)
=====================  ======================  =============================================

plus ``static`` (200 episodes on :class:`~rld.envs.platform.StaticDeckMotion`), which exists
for Gate 3's static-pad criterion and is not a dmf realization.

``in_training_distribution`` is true iff the realization's grid cell ``(vessel, ss,
heading, speed)`` is a cell of the development pool :func:`rld.deck.splits.dev_pool` (P3-D2).
The pool covers every speed of its (vessel, ss, heading) cells, so this is identical to the
(vessel, ss, heading) definition for the committed grid -- and ``tests/test_episodes.py``
asserts that. It is false for ``id`` SS6, for ``id`` at 90 deg, for all of
``unseen_seastate``, ``unseen_heading`` and ``unseen_vessel``, and for ``static``.

The draw (balanced round-robin, user decision 2026-09-22)
--------------------------------------------------------
Per cell, the regime's test realizations at that sea state -- ``R`` of them, in
:func:`rld.deck.splits.sorted_specs` order -- are shuffled once by a generator seeded from
``(generator_seed, sha256(regime), sha256(ss))``. Episode ``i`` flies realization
``shuffled[i mod R]``:

* ``R <= N``: round-robin, so every realization gets ``floor(N/R)`` or ``ceil(N/R)``
  episodes, the extra ones going to the first ``N mod R`` realizations in shuffled order;
* ``R > N`` (``unseen_seastate`` SS6, ``R`` = 480): the first ``N`` of the shuffled list, one
  episode each, without replacement.

Coverage is therefore ``min(R, N)`` and the per-realization count spread within a cell is at
most 1. The episode seed is drawn from a generator seeded by ``(generator_seed,
sha256(regime), sha256(ss), index)``, so episode ``i`` is a pure function of its own index
and of the cell: regenerating the first ``k`` episodes of a cell reproduces the committed
first ``k`` rows exactly, which is what makes a cheap regeneration test possible. Labels are
mixed in through SHA-256, never ``hash()``, which is salted per interpreter. The rule is
recorded in the manifest's ``draw`` field as :data:`DRAW_RULE`.

This replaced the with-replacement pick of :func:`rld.envs.sanity.draw_episodes` before the
lists were first committed: with replacement an ``id`` cell covered only 84-90 of its 96
test realizations.

Hashes
------
``results/episodes/MANIFEST.csv`` records, per file, the SHA-256 of the parquet **bytes** and
a **content** SHA-256 over :func:`canonical_csv` -- fixed column order, ``%.17g`` floats
(round-trip exact for float64), ``true``/``false`` booleans, newline line endings. The file
hash depends on the pyarrow version; the content hash does not, and is the one to compare
across machines.

The MSS transfer lists (P7-D1 §6)
---------------------------------
A **separate** list directory, ``results/episodes_mss/``, with its own ``MANIFEST.csv``; the
frozen ``results/episodes/`` and its manifest are never touched (:func:`generate_mss_lists`,
``scripts/make_episodes.py --mss``). Two regimes, one cell each, N = 200:

* ``mss_transfer``: ``ss = "SS5/mss:mss"``, motion kind ``mss`` -- MSS's spectrum and RAOs
  (the primary arm);
* ``mss_transfer_corpus``: ``ss = "SS5/mss:corpus"``, motion kind ``mss_corpus`` -- dmf's own
  wave field through MSS's transfer function (the attribution control).

The candidates are the arm config's 18 realizations per grid kind (headings {180, 135} deg x
speeds {0, 6, 12} kn x seeds {0, 1, 2}, ``configs/deck/mss_s175_ss5.yaml``), sorted by
``(heading_deg, speed_kn, seed)``, and they are dealt by the same ``balanced_round_robin`` rule
with generator seed :data:`MSS_GENERATOR_SEED` (20261001): 200 = 11 x 18 + 2, so two
realizations get 12 episodes and sixteen get 11. Each row carries
``ss = rld.deck.mss.mss_ss_label(grid_kind)`` (so its key is :func:`rld.deck.mss.mss_key`,
never a dmf corpus key), vessel ``s175`` (the lever arm), pad ``aft``, and
``in_training_distribution = false`` (S175 is never trained on). ``t0`` and the initial state
are read back from a real reset of the MSS source, exactly as for the frozen lists, and the
runner's strict start check holds on them unchanged.

Units: ``t0_model_s`` seconds **model** scale (one model second is five full-scale seconds at
lambda = 1/25), ``init_*_m`` metres model scale in the world frame, ``heading_deg``
degrees, ``speed_kn`` knots full scale.
"""

import csv
import hashlib
import io
import multiprocessing as mp
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import astuple, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dmf.config import SimConfig
from dmf.data.splits import REGIMES, realization_key

from rld.config import REPO_ROOT
from rld.deck.splits import build_all_splits, dev_pool, part_specs
from rld.eval.arms import EPISODES_MSS_DIR, MSS_REGIME_MOTION
from rld.eval.envs import (
    STATIC_LABEL,
    EvalConfigs,
    MotionKind,
    load_eval_configs,
    make_env,
    motion_for,
    pad_offset_for,
)

__all__ = [
    "DEFAULT_CHUNK",
    "DEFAULT_GENERATOR_SEED",
    "DEFAULT_PAD",
    "DEFAULT_WORKERS",
    "DRAW_RULE",
    "EPISODES_DIR",
    "EPISODES_MSS_DIR",
    "EPISODES_PER_CELL",
    "LIST_COLUMNS",
    "MANIFEST_COLUMNS",
    "MANIFEST_NAME",
    "MSS_GENERATOR_SEED",
    "MSS_REGIME_GRID",
    "MSS_VESSEL",
    "REGIME_CELLS",
    "EpisodeDraw",
    "ListedEpisode",
    "ManifestRow",
    "canonical_csv",
    "cell_order",
    "content_sha256",
    "draw_cell",
    "draw_mss_cell",
    "file_sha256",
    "generate_lists",
    "generate_mss_lists",
    "label_entropy",
    "list_names",
    "manifest_row",
    "motion_kind_for",
    "mss_candidates",
    "mss_list_names",
    "mss_ss",
    "read_list",
    "read_manifest",
    "resolve_draws",
    "training_cells",
    "write_lists",
]

#: The committed generator seed. Distinct from the e00 sanity draw seed (20260922, P2-D4)
#: and the Phase 3 tuning draw seed (20260923).
DEFAULT_GENERATOR_SEED: int = 20260924

#: The draw rule, recorded in every manifest row (see the module docstring).
DRAW_RULE: str = "balanced_round_robin"

#: Episodes per (regime, sea state) cell, and in the static list.
EPISODES_PER_CELL: int = 200

#: The primary arm's pad. The pad-at-CG control arm reuses the same lists with the pad
#: overridden at run time: neither the start offset nor the initial state depends on the pad.
DEFAULT_PAD: str = "aft"

#: The evaluated cells, in committed file and row order.
REGIME_CELLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("id", ("SS3", "SS4", "SS5", "SS6")),
    ("unseen_seastate", ("SS6",)),
    ("unseen_heading", ("SS3", "SS4", "SS5", "SS6")),
    ("unseen_vessel", ("SS3", "SS4", "SS5", "SS6")),
    (STATIC_LABEL, (STATIC_LABEL,)),
)

#: Where the committed lists live.
EPISODES_DIR: Path = REPO_ROOT / "results" / "episodes"

#: The manifest's file name inside :data:`EPISODES_DIR`.
MANIFEST_NAME: str = "MANIFEST.csv"

#: The MSS transfer lists' generator seed (P7-D1 §6). Distinct from every other draw seed.
MSS_GENERATOR_SEED: int = 20261001

#: MSS list regime -> dmf.mss grid kind (P7-D1 §6), in manifest order.
MSS_REGIME_GRID: dict[str, str] = {"mss_transfer": "mss", "mss_transfer_corpus": "corpus"}

#: The vessel field of every MSS row: dmf's ``s175`` hull config sets the lever arm.
MSS_VESSEL: str = "s175"

#: Episodes per parallel reset task.
DEFAULT_CHUNK: int = 25

#: Default worker processes.
DEFAULT_WORKERS: int = 24


@dataclass(frozen=True)
class EpisodeDraw:
    """One episode as drawn, before the environment has been asked where it starts.

    Attributes:
        regime: dmf regime name, or ``"static"``.
        ss: Sea state label, or ``"static"``.
        index: Position in the cell's draw order, ``0 .. n - 1``.
        pad: Pad name.
        vessel: dmf vessel config stem, or ``"static"``.
        heading_deg: Encounter heading, degrees.
        speed_kn: Forward speed, knots full scale.
        realization_seed: dmf realization seed ordinal within its cell.
        episode_seed: The environment's episode seed; fixes start offset, initial state and
            noise stream.
        in_training_distribution: Whether the realization's grid cell is a dev-pool cell.
    """

    regime: str
    ss: str
    index: int
    pad: str
    vessel: str
    heading_deg: float
    speed_kn: float
    realization_seed: int
    episode_seed: int
    in_training_distribution: bool


@dataclass(frozen=True)
class ListedEpisode:
    """One committed row of an episode list. Field order is the file's column order.

    Attributes:
        regime: dmf regime name, or ``"static"``.
        ss: Sea state label, or ``"static"``.
        index: Position in the cell's draw order.
        pad: Pad name.
        vessel: dmf vessel config stem, or ``"static"``.
        heading_deg: Encounter heading, degrees.
        speed_kn: Forward speed, knots full scale.
        realization_seed: dmf realization seed ordinal within its cell.
        episode_seed: The environment's episode seed.
        t0_model_s: Start offset inside the dmf record, seconds model scale, as read back
            from a real reset.
        init_x_m: Initial drone x, metres model scale, world frame, read back from the
            physics state after reset.
        init_y_m: Initial drone y, metres model scale, world frame.
        init_z_m: Initial drone z, metres model scale, world frame.
        in_training_distribution: Whether the realization's grid cell is a dev-pool cell.
    """

    regime: str
    ss: str
    index: int
    pad: str
    vessel: str
    heading_deg: float
    speed_kn: float
    realization_seed: int
    episode_seed: int
    t0_model_s: float
    init_x_m: float
    init_y_m: float
    init_z_m: float
    in_training_distribution: bool

    @property
    def key(self) -> tuple[str, float, float, str, int]:
        """Return dmf's realization key ``(ss, heading_deg, speed_kn, vessel, seed)``."""
        return realization_key(
            self.ss, self.heading_deg, self.speed_kn, self.vessel, self.realization_seed
        )

    @property
    def init_xyz_m(self) -> tuple[float, float, float]:
        """Return the initial drone position, metres model scale, world frame."""
        return (self.init_x_m, self.init_y_m, self.init_z_m)


#: Committed column order of every list file.
LIST_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(ListedEpisode))

#: Arrow schema of every list file.
_LIST_SCHEMA = pa.schema(
    [
        ("regime", pa.string()),
        ("ss", pa.string()),
        ("index", pa.int64()),
        ("pad", pa.string()),
        ("vessel", pa.string()),
        ("heading_deg", pa.float64()),
        ("speed_kn", pa.float64()),
        ("realization_seed", pa.int64()),
        ("episode_seed", pa.int64()),
        ("t0_model_s", pa.float64()),
        ("init_x_m", pa.float64()),
        ("init_y_m", pa.float64()),
        ("init_z_m", pa.float64()),
        ("in_training_distribution", pa.bool_()),
    ]
)


@dataclass(frozen=True)
class ManifestRow:
    """One row of ``results/episodes/MANIFEST.csv``.

    Attributes:
        file: File name inside ``results/episodes/``.
        regime: The file's regime.
        rows: Episodes in the file.
        n_cells: Sea-state cells in the file.
        cells: The cells' sea states, ``|``-joined in file order.
        n_per_cell: Episodes per cell.
        generator_seed: The draw seed.
        draw: The draw rule, :data:`DRAW_RULE`.
        pad: The pad every row carries.
        frac_in_training_distribution: Fraction of rows with the flag set, dimensionless.
        file_sha256: SHA-256 of the parquet bytes.
        content_sha256: SHA-256 of :func:`canonical_csv` of the rows.
    """

    file: str
    regime: str
    rows: int
    n_cells: int
    cells: str
    n_per_cell: int
    generator_seed: int
    draw: str
    pad: str
    frac_in_training_distribution: float
    file_sha256: str
    content_sha256: str


#: Committed column order of the manifest.
MANIFEST_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(ManifestRow))


def label_entropy(label: str) -> int:
    """Return a stable 64-bit entropy value for a string label.

    Args:
        label: Any label, e.g. a regime or sea-state name.

    Returns:
        The first eight bytes of the label's SHA-256, big-endian, in ``[0, 2**64)``.
        ``hash()`` is salted per process and would make a spawned worker disagree.
    """
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")


def list_names() -> tuple[str, ...]:
    """Return the committed list names (one parquet file each), in manifest order."""
    return tuple(regime for regime, _ in REGIME_CELLS)


def training_cells(sim_cfg: SimConfig) -> frozenset[tuple[str, str, float, float]]:
    """Return the grid cells of the development pool (P3-D2).

    Args:
        sim_cfg: dmf's corpus config.

    Returns:
        ``(vessel, ss, heading_deg, speed_kn)`` for every cell that has a realization in
        :func:`rld.deck.splits.dev_pool`'s train or tune pool.
    """
    train, tune = dev_pool(sim_cfg)
    return frozenset(
        (s.vessel, s.sea_state, float(s.heading_deg), float(s.speed_kn)) for s in (*train, *tune)
    )


def _episode_rng(generator_seed: int, regime: str, ss: str, index: int) -> np.random.Generator:
    """Return the generator of one episode's draw -- a pure function of its coordinates.

    Args:
        generator_seed: The committed draw seed.
        regime: The list's regime.
        ss: The cell's sea state.
        index: The episode's index in its cell.

    Returns:
        A seeded ``numpy.random.Generator``.
    """
    entropy = [int(generator_seed), label_entropy(regime), label_entropy(ss), int(index)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def cell_order(generator_seed: int, regime: str, ss: str, n_realizations: int) -> list[int]:
    """Return the cell's shuffled realization order -- a pure function of the cell.

    Args:
        generator_seed: The committed draw seed.
        regime: The list's regime.
        ss: The cell's sea state.
        n_realizations: ``R``, the number of candidate realizations.

    Returns:
        A permutation of ``range(R)``: positions into the cell's
        :func:`rld.deck.splits.sorted_specs`-ordered candidates. Episode ``i`` flies
        candidate ``order[i % R]``.
    """
    entropy = [int(generator_seed), label_entropy(regime), label_entropy(ss)]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return [int(v) for v in rng.permutation(n_realizations)]


def draw_cell(
    sim_cfg: SimConfig,
    regime: str,
    ss: str,
    generator_seed: int,
    indices: Iterable[int],
    pad: str = DEFAULT_PAD,
) -> list[EpisodeDraw]:
    """Draw some episodes of one cell from that regime's dmf test partition.

    Args:
        sim_cfg: dmf's corpus config.
        regime: dmf regime name, or ``"static"``.
        ss: Sea state label (``"static"`` for the static list).
        generator_seed: The committed draw seed.
        indices: Which episode indices to draw. Episode ``i`` flies realization
            ``cell_order(...)[i % R]`` (balanced round-robin; see the module docstring)
            with an episode seed depending only on ``i`` and the cell.
        pad: Pad name.

    Returns:
        The draws, in the order of ``indices``.

    Raises:
        ValueError: If the regime is unknown or its test partition has no realization at
            ``ss``.
    """
    in_pool = training_cells(sim_cfg)
    draws: list[EpisodeDraw] = []
    if regime == STATIC_LABEL:
        for index in indices:
            rng = _episode_rng(generator_seed, regime, ss, index)
            draws.append(
                EpisodeDraw(
                    regime=regime,
                    ss=STATIC_LABEL,
                    index=int(index),
                    pad=pad,
                    vessel=STATIC_LABEL,
                    heading_deg=180.0,
                    speed_kn=0.0,
                    realization_seed=0,
                    episode_seed=int(rng.integers(0, 2**31 - 1)),
                    in_training_distribution=False,
                )
            )
        return draws
    splits = build_all_splits(sim_cfg)
    matches = [name for name in REGIMES if name == regime]
    if not matches:
        raise ValueError(f"unknown regime {regime!r}, expected one of {list(REGIMES)}")
    split = splits[matches[0]]
    candidates = [spec for spec in part_specs(split, "test") if spec.sea_state == ss]
    if not candidates:
        raise ValueError(f"no {regime!r} test realization at {ss}")
    order = cell_order(generator_seed, regime, ss, len(candidates))
    for index in indices:
        rng = _episode_rng(generator_seed, regime, ss, index)
        spec = candidates[order[int(index) % len(order)]]
        cell = (spec.vessel, spec.sea_state, float(spec.heading_deg), float(spec.speed_kn))
        draws.append(
            EpisodeDraw(
                regime=regime,
                ss=ss,
                index=int(index),
                pad=pad,
                vessel=spec.vessel,
                heading_deg=float(spec.heading_deg),
                speed_kn=float(spec.speed_kn),
                realization_seed=int(spec.seed),
                episode_seed=int(rng.integers(0, 2**31 - 1)),
                in_training_distribution=cell in in_pool,
            )
        )
    return draws


def mss_list_names() -> tuple[str, ...]:
    """Return the MSS list names (one parquet file each), in manifest order."""
    return tuple(MSS_REGIME_GRID)


def mss_ss(regime: str) -> str:
    """Return an MSS list's ``ss`` label, e.g. ``"SS5/mss:mss"`` for ``mss_transfer``.

    Raises:
        ValueError: If ``regime`` is not an MSS list.
    """
    from rld.deck.mss import load_mss_config, mss_ss_label  # dmf.mss; MSS lists only

    if regime not in MSS_REGIME_GRID:
        raise ValueError(f"unknown MSS list {regime!r}, expected one of {mss_list_names()}")
    return mss_ss_label(MSS_REGIME_GRID[regime], str(load_mss_config()["sea_state"]["name"]))


def mss_candidates() -> list[tuple[float, float, int]]:
    """Return the MSS arm's realizations per grid kind as ``(heading_deg, speed_kn, seed)``.

    Returns:
        The config's headings x speeds x seeds (``configs/deck/mss_s175_ss5.yaml``: 18),
        sorted by ``(heading_deg, speed_kn, seed)``: the candidates' order before the cell's
        shuffle. Headings degrees, speeds knots full scale.
    """
    from rld.deck.mss import load_mss_config  # dmf.mss; MSS lists only

    cfg = load_mss_config()
    return sorted(
        (float(h), float(u), int(s))
        for h in cfg["headings_deg"]
        for u in cfg["speeds_kn"]
        for s in cfg["seeds"]
    )


def draw_mss_cell(
    regime: str,
    generator_seed: int,
    indices: Iterable[int],
    pad: str = DEFAULT_PAD,
) -> list[EpisodeDraw]:
    """Draw some episodes of one MSS list by the same balanced round-robin rule.

    Args:
        regime: ``"mss_transfer"`` or ``"mss_transfer_corpus"``.
        generator_seed: The draw seed (:data:`MSS_GENERATOR_SEED`).
        indices: Which episode indices to draw; episode ``i`` flies candidate
            ``cell_order(...)[i % 18]`` with an episode seed depending only on ``i`` and the
            cell, exactly as :func:`draw_cell`.
        pad: Pad name.

    Returns:
        The draws, in the order of ``indices``; ``in_training_distribution`` is false.
    """
    ss = mss_ss(regime)
    candidates = mss_candidates()
    order = cell_order(generator_seed, regime, ss, len(candidates))
    draws: list[EpisodeDraw] = []
    for index in indices:
        rng = _episode_rng(generator_seed, regime, ss, index)
        heading, speed, seed = candidates[order[int(index) % len(order)]]
        draws.append(
            EpisodeDraw(
                regime=regime,
                ss=ss,
                index=int(index),
                pad=pad,
                vessel=MSS_VESSEL,
                heading_deg=heading,
                speed_kn=speed,
                realization_seed=seed,
                episode_seed=int(rng.integers(0, 2**31 - 1)),
                in_training_distribution=False,
            )
        )
    return draws


def motion_kind_for(regime: str) -> MotionKind:
    """Return the motion kind a listed regime is reset under: its MSS kind, else JONSWAP."""
    kind = MSS_REGIME_MOTION.get(regime, "jonswap")
    if kind == "mss":
        return "mss"
    if kind == "mss_corpus":
        return "mss_corpus"
    return "jonswap"


def _resolve_chunk(task: tuple[Sequence[EpisodeDraw], EvalConfigs]) -> list[ListedEpisode]:
    """Reset a real environment once per draw and read back where the episode starts.

    Module-level with one packed argument so ``multiprocessing`` can pickle it.

    Args:
        task: ``(draws, configs)``.

    Returns:
        One :class:`ListedEpisode` per draw, in the order given.

    Raises:
        AssertionError: If the reset's realization key disagrees with the draw, or the
            physics state disagrees with the environment's own initial position.
    """
    draws, cfgs = task
    if not draws:
        return []
    first = draws[0]
    env = make_env(
        cfgs,
        motion_for(
            cfgs,
            first.vessel,
            first.ss,
            first.heading_deg,
            first.speed_kn,
            first.realization_seed,
            kind=motion_kind_for(first.regime),
        ),
        first.vessel,
        first.pad,
        first.episode_seed,
    )
    rows: list[ListedEpisode] = []
    try:
        for draw in draws:
            motion = motion_for(
                cfgs,
                draw.vessel,
                draw.ss,
                draw.heading_deg,
                draw.speed_kn,
                draw.realization_seed,
                kind=motion_kind_for(draw.regime),
            )
            env.set_motion(motion, draw.pad, pad_offset_for(cfgs, draw.vessel, draw.pad))
            _, info = env.reset(seed=draw.episode_seed)
            expected_key = realization_key(
                draw.ss, draw.heading_deg, draw.speed_kn, draw.vessel, draw.realization_seed
            )
            assert info["realization_key"] == expected_key, (info["realization_key"], draw)
            position = np.asarray(env.pos[0], dtype=np.float64).reshape(3)
            configured = np.asarray(env.INIT_XYZS, dtype=np.float64).reshape(3)
            assert np.array_equal(position, configured), (position, configured)
            assert float(info["t0_model_s"]) == env.record.t0_model_s
            rows.append(
                ListedEpisode(
                    regime=draw.regime,
                    ss=draw.ss,
                    index=draw.index,
                    pad=draw.pad,
                    vessel=draw.vessel,
                    heading_deg=draw.heading_deg,
                    speed_kn=draw.speed_kn,
                    realization_seed=draw.realization_seed,
                    episode_seed=draw.episode_seed,
                    t0_model_s=float(env.record.t0_model_s),
                    init_x_m=float(position[0]),
                    init_y_m=float(position[1]),
                    init_z_m=float(position[2]),
                    in_training_distribution=draw.in_training_distribution,
                )
            )
    finally:
        env.close()
    return rows


def resolve_draws(
    draws: Sequence[EpisodeDraw],
    cfgs: EvalConfigs,
    *,
    workers: int = DEFAULT_WORKERS,
    chunk: int = DEFAULT_CHUNK,
) -> list[ListedEpisode]:
    """Read every draw's start offset and initial state back from a real environment reset.

    Nothing here re-derives ``_prepare_episode``: the environment is the only source of
    ``t0`` and the initial state, so a list can never disagree with the environment that
    will fly it.

    Args:
        draws: The draws, in committed order.
        cfgs: The committed configs.
        workers: Processes. Work splits over contiguous chunks and is reassembled in input
            order, so the result does not depend on this value.
        chunk: Draws per parallel task.

    Returns:
        One :class:`ListedEpisode` per draw, in input order.

    Raises:
        ValueError: If ``workers`` or ``chunk`` is not positive.
    """
    if workers < 1 or chunk < 1:
        raise ValueError(f"workers and chunk must be positive, got {workers}, {chunk}")
    tasks = [(draws[i : i + chunk], cfgs) for i in range(0, len(draws), chunk)]
    if workers == 1 or len(tasks) <= 1:
        results = [_resolve_chunk(task) for task in tasks]
    else:
        with mp.get_context("spawn").Pool(processes=min(workers, len(tasks))) as pool:
            results = list(pool.imap(_resolve_chunk, tasks, chunksize=1))
    return [row for rows in results for row in rows]


def generate_lists(
    sim_cfg: SimConfig,
    generator_seed: int = DEFAULT_GENERATOR_SEED,
    n: int = EPISODES_PER_CELL,
    *,
    cfgs: EvalConfigs | None = None,
    names: Sequence[str] | None = None,
    pad: str = DEFAULT_PAD,
    workers: int = DEFAULT_WORKERS,
    chunk: int = DEFAULT_CHUNK,
) -> dict[str, list[ListedEpisode]]:
    """Draw and resolve the evaluation lists.

    Args:
        sim_cfg: dmf's corpus config; the splits and the bridge are built from it.
        generator_seed: The draw seed.
        n: Episodes per cell. Episode ``i`` does not depend on ``n``, so a smaller ``n``
            reproduces a prefix of every cell.
        cfgs: Environment configs; the committed ones when omitted. Its ``sim`` is replaced
            by ``sim_cfg`` so the splits and the bridge cannot come from two corpora.
        names: Which lists to generate (default: all of :func:`list_names`).
        pad: Pad name carried by every row.
        workers: Reset processes; the result does not depend on it.
        chunk: Draws per reset task.

    Returns:
        ``{list name: rows}`` with rows ordered by sea state (in :data:`REGIME_CELLS`
        order) then index.

    Raises:
        ValueError: If ``n`` is not positive or a name is unknown.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    wanted = list_names() if names is None else tuple(names)
    unknown = sorted(set(wanted) - set(list_names()))
    if unknown:
        raise ValueError(f"unknown lists {unknown}, expected a subset of {list_names()}")
    env_cfgs = replace(cfgs if cfgs is not None else load_eval_configs(), sim=sim_cfg)
    draws: list[EpisodeDraw] = []
    for regime, sea_states in REGIME_CELLS:
        if regime not in wanted:
            continue
        for ss in sea_states:
            draws += draw_cell(sim_cfg, regime, ss, generator_seed, range(n), pad)
    rows = resolve_draws(draws, env_cfgs, workers=workers, chunk=chunk)
    return {name: [row for row in rows if row.regime == name] for name in wanted}


def generate_mss_lists(
    generator_seed: int = MSS_GENERATOR_SEED,
    n: int = EPISODES_PER_CELL,
    *,
    cfgs: EvalConfigs | None = None,
    names: Sequence[str] | None = None,
    pad: str = DEFAULT_PAD,
    workers: int = DEFAULT_WORKERS,
    chunk: int = DEFAULT_CHUNK,
) -> dict[str, list[ListedEpisode]]:
    """Draw and resolve the MSS transfer lists (P7-D1 §6; module docstring).

    Args:
        generator_seed: The draw seed (:data:`MSS_GENERATOR_SEED`).
        n: Episodes per list; episode ``i`` does not depend on ``n``, so a smaller ``n``
            reproduces a prefix.
        cfgs: Environment configs; the committed ones when omitted. The MSS source reads
            their ``scaling`` (lambda 1/25), ``pads`` and full-scale lookback.
        names: Which MSS lists (default: :func:`mss_list_names`).
        pad: Pad name carried by every row (``aft``).
        workers: Reset processes; the result does not depend on it.
        chunk: Draws per reset task.

    Returns:
        ``{list name: rows}`` in index order, ``t0`` and the initial state read back from a
        real reset of each row's MSS source.

    Raises:
        ValueError: If ``n`` is not positive or a name is unknown.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    wanted = mss_list_names() if names is None else tuple(names)
    unknown = sorted(set(wanted) - set(mss_list_names()))
    if unknown:
        raise ValueError(f"unknown MSS lists {unknown}, expected a subset of {mss_list_names()}")
    env_cfgs = cfgs if cfgs is not None else load_eval_configs()
    draws: list[EpisodeDraw] = []
    for regime in mss_list_names():
        if regime in wanted:
            draws += draw_mss_cell(regime, generator_seed, range(n), pad)
    rows = resolve_draws(draws, env_cfgs, workers=workers, chunk=chunk)
    return {name: [row for row in rows if row.regime == name] for name in wanted}


def canonical_csv(rows: Sequence[ListedEpisode]) -> str:
    """Serialise rows canonically, for the content hash.

    Fixed column order (:data:`LIST_COLUMNS`), floats as ``%.17g`` (exact round trip for
    float64), booleans as ``true``/``false``, integers in decimal, newline line endings, a
    header line, no index. Independent of pandas, pyarrow and the platform.

    Args:
        rows: The rows, in committed order.

    Returns:
        The CSV text.

    Raises:
        ValueError: If a string field contains a comma, quote or newline, which this
            format does not escape.
    """
    lines = [",".join(LIST_COLUMNS)]
    for row in rows:
        cells: list[str] = []
        for value in astuple(row):
            if isinstance(value, bool):
                cells.append("true" if value else "false")
            elif isinstance(value, int):
                cells.append(str(value))
            elif isinstance(value, float):
                cells.append("%.17g" % value)  # noqa: UP031 -- the format is the contract
            else:
                text = str(value)
                if any(ch in text for ch in ',"\n\r'):
                    raise ValueError(f"unescapable string in canonical CSV: {text!r}")
                cells.append(text)
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


def content_sha256(rows: Sequence[ListedEpisode]) -> str:
    """Return the SHA-256 hex digest of :func:`canonical_csv` of ``rows``."""
    return hashlib.sha256(canonical_csv(rows).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_table(rows: Sequence[ListedEpisode]) -> pa.Table:
    """Return the rows as an Arrow table with the committed schema and no pandas metadata."""
    columns: dict[str, list[Any]] = {name: [] for name in LIST_COLUMNS}
    for row in rows:
        for name, value in zip(LIST_COLUMNS, astuple(row), strict=True):
            columns[name].append(value)
    return pa.Table.from_pydict(columns, schema=_LIST_SCHEMA)


def read_list(path: Path) -> list[ListedEpisode]:
    """Read one committed list back into rows.

    Args:
        path: A ``results/episodes/<name>.parquet`` file.

    Returns:
        The rows, in file order, with Python ``int``/``float``/``bool``/``str`` values.

    Raises:
        ValueError: If the file's columns are not :data:`LIST_COLUMNS` in order.
    """
    table = pq.read_table(path)
    if tuple(table.column_names) != LIST_COLUMNS:
        raise ValueError(f"{path}: columns {table.column_names} != {list(LIST_COLUMNS)}")
    return [ListedEpisode(**record) for record in table.to_pylist()]


def manifest_row(
    name: str, path: Path, rows: Sequence[ListedEpisode], generator_seed: int
) -> ManifestRow:
    """Summarise one written list file for the manifest.

    Args:
        name: List name (the regime).
        path: The written parquet file.
        rows: The rows that were written.
        generator_seed: The draw seed.

    Returns:
        The :class:`ManifestRow`.

    Raises:
        ValueError: If the cells do not all hold the same number of rows, or the pad is
            not uniform.
    """
    ss_order = list(dict.fromkeys(row.ss for row in rows))
    per_cell = {ss: sum(row.ss == ss for row in rows) for ss in ss_order}
    if len(set(per_cell.values())) != 1:
        raise ValueError(f"{name}: unequal cells {per_cell}")
    pads = {row.pad for row in rows}
    if len(pads) != 1:
        raise ValueError(f"{name}: mixed pads {sorted(pads)}")
    return ManifestRow(
        file=path.name,
        regime=name,
        rows=len(rows),
        n_cells=len(ss_order),
        cells="|".join(ss_order),
        n_per_cell=next(iter(per_cell.values())),
        generator_seed=generator_seed,
        draw=DRAW_RULE,
        pad=next(iter(pads)),
        frac_in_training_distribution=sum(r.in_training_distribution for r in rows) / len(rows),
        file_sha256=file_sha256(path),
        content_sha256=content_sha256(rows),
    )


def write_lists(
    lists: Mapping[str, Sequence[ListedEpisode]],
    out_dir: Path,
    generator_seed: int,
) -> list[ManifestRow]:
    """Write every list as parquet plus ``MANIFEST.csv``.

    Args:
        lists: ``{list name: rows}``.
        out_dir: Target directory, created if missing.
        generator_seed: The draw seed, recorded in the manifest.

    Returns:
        The manifest rows, in :func:`list_names` order, then any other list (the MSS lists)
        in the order given.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[ManifestRow] = []
    order = [name for name in list_names() if name in lists]
    order += [name for name in lists if name not in order]
    for name in order:
        path = out_dir / f"{name}.parquet"
        pq.write_table(_to_table(lists[name]), path, compression="snappy")
        manifest.append(manifest_row(name, path, lists[name], generator_seed))
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    for row in manifest:
        writer.writerow([repr(v) if isinstance(v, float) else v for v in astuple(row)])
    (out_dir / MANIFEST_NAME).write_text(buffer.getvalue(), encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read ``MANIFEST.csv`` as raw string records.

    Args:
        path: The manifest file.

    Returns:
        One ``{column: text}`` per list file, in file order.

    Raises:
        ValueError: If the header is not :data:`MANIFEST_COLUMNS`.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(f"{path}: header {reader.fieldnames} != {list(MANIFEST_COLUMNS)}")
        return list(reader)
