"""MSS strip-theory deck motion: the Phase 7 transfer arm (protocol P7-D1 section 6).

Project 4 (``dmf``) exports ITTC S-175 motion synthesised from the Marine Systems Simulator's
ShipX strip-theory RAOs (``dmf.mss``). This module turns the *same* realizations into a
:class:`~rld.deck.bridge.DeckMotionSource`, so every method can fly them unchanged. These
are another simulator's trajectories, **not measurements of a real ship**.

What this module guarantees
---------------------------
1. **dmf's export path, exactly.** One realization is
   ``dmf.mss.export.MSSRealizationSpec(grid_kind, heading_deg, speed_kn, seed)``. Its wave
   grid is ``_build_grid(spec, cfg, realization_seed(spec))``, the generator and grid builder
   that ``dmf.mss.export.generate_realization`` uses. Motion is
   ``dmf.mss.synth.synthesize_mss_motion(..., speed_index=cfg["vessel"]["speed_index"])``
   at ``speed_m_s = speed_kn * 0.514444`` (the export's literal knot factor). Units and signs
   go through ``dmf.mss.convert.mss_motion_to_frame``. ``tests/test_deck_mss.py`` asserts
   that the 10 Hz samples equal the exported CSV rows.
2. **Analytic, on any time grid.** ``synthesize_mss_motion`` is a memoryless sum of harmonics
   in ``t``. It is evaluated directly on the full-scale times that map to the model-scale
   physics grid. The 10 Hz CSV is never read here, never interpolated, and rates are never
   differenced. dmf's ``eta_dot`` is the analytic derivative. The accelerations, which dmf
   does not export, come from one more ``synthesize_mss_motion`` call on the *same*
   frequencies with amplitudes ``a * w_e**2`` and phases ``phi + pi``. Its ``eta`` is then
   ``sum |H| a w_e**2 cos(w_e t + arg H + phi + pi)``, the exact second derivative. ``w_e``
   is MSS's own ``|w - w**2 U cos(beta) / g|`` with MSS's ``g`` (9.8100004 m/s^2).
3. **Row-invariant bits.** A one-sample ``synthesize_mss_motion`` call reduces its
   ``(n_components, 1)`` array with numpy's pairwise sum, while two or more samples reduce
   component by component. The last bits of a sample would then depend on how many
   neighbours it was evaluated with. A one-sample request is therefore evaluated as the
   duplicated pair ``[t, t]``, so ``channels(t[a:b])`` equals ``channels(t)[a:b]`` bit for
   bit for every slice (asserted). This makes the source elementwise in time, like the
   sinusoid, so a lazily chunked episode grid is exact.

Sign conventions
----------------
MSS works in SNAME axes (x forward, y starboard, z **down**). dmf's
``MSS_TO_CORPUS_SIGN`` maps them to the corpus convention (``docs/corpus_card.md`` of dmf):
roll **positive to starboard** (+1), pitch **positive bow-up** (+1), heave positive **up**
(-1, the only flip). Angles go radians -> degrees inside ``mss_motion_to_frame``, and nowhere
else. :func:`rld.deck.kinematics.deck_point_state` then maps those signs into this project's
world frame (x bow, y port, z up, ZYX), exactly as for JONSWAP: for a centreline pad at
signed ``x`` (``-0.4 L = -70.0`` m full scale aft on the S175),
``z_pad = heave + x * sin(pitch)``. So a bow-up pitch lowers the aft pad.

Units and scales
----------------
``channels`` is **full scale**: degrees, degrees per second, degrees per second squared,
metres, metres per second, metres per second squared, and absolute full-scale seconds.
``deck_point`` is **model scale** in the world frame. Kinematics are evaluated at full scale
with the full-scale lever arm, then Froude-scaled once by the project ``lam``
(``to_model_state``). The lever arm uses dmf's ``s175`` hull config (``L = 175.0`` m, equal
to MSS's ``Lpp``; checked at construction). That is the length
:func:`rld.envs.platform.pad_offset_model_m` and the forecast feed use, so the plate anchor,
the deck point and the forecast pad band agree.

Time and window
---------------
The MSS record starts at ``record.t_start_s = 120.0`` s full scale (matching the corpus's
120 s spin-up) and lasts ``record.duration_s = 600.0`` s at ``fs_hz = 10`` Hz: 6000 rows,
``t = 120.0 ... 719.9``. Model time is measured from the record start:

    t_full_s = t_start_s + t_model_s / sqrt(lam)

The usable model window is ``(lookback, duration)`` converted to model time. With the
forecaster's 200-sample, 10 Hz full-scale lookback (20 s full = 4.0 s model at
``lam = 1/25``), that is ``(4.0, 120.0)`` s model, the JONSWAP source's window. A 12.5 s
episode therefore starts in ``(4.0, 107.5)``. History before an episode starts is
legitimate: the ship was already moving.

Identity
--------
dmf's ``RealizationKey`` is ``(ss, heading_deg, speed_kn, vessel, seed)``. The vessel field
stays ``"s175"``, because every lever-arm lookup in the project resolves it as a dmf hull
config. The sea-state field carries the source and the grid kind:
``mss_ss_label("mss") == "SS5/mss:mss"`` and ``mss_ss_label("corpus") == "SS5/mss:corpus"``.
No dmf corpus key has a ``/`` in its sea state, so an MSS key never equals one, and the two
grid kinds never equal each other.

The dmf roll/pitch-heave phase defect
-------------------------------------
That defect belongs to dmf's own reduced-order response model. These records use MSS's
strip-theory RAO phases, so the defect is not expected here. The aft pad and the pad-at-CG
control are still flown and reported side by side (P7-D1 section 6, "aft and CG").
"""

import math
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
from dmf.data.splits import RealizationKey, realization_key
from dmf.mss.convert import mss_motion_to_frame
from dmf.mss.export import (
    GRID_KINDS,
    MSSRealizationSpec,
    _build_grid,
    _time_axis,
    realization_seed,
)
from dmf.mss.synth import MSSMotion, WaveGrid, mss_encounter_frequency, synthesize_mss_motion
from dmf.mss.vessel import MSSVessel, load_mss_vessel
from dmf.typedefs import FloatArray

from rld.config import CONFIG_DIR, REPO_ROOT, load_yaml
from rld.deck.bridge import (
    TIME_BOUND_SLACK_S,
    DeckPointTrajectory,
    MotionChannels,
    load_vessel_cached,
)
from rld.deck.config import MotionConfig, PadConfig, ScalingConfig
from rld.deck.kinematics import deck_point_state, to_model_state
from rld.deck.scaling import FroudeScale

__all__ = [
    "GRID_KINDS",
    "KNOT_M_S",
    "MSS_CONFIG",
    "MSS_RECORDS_DIR",
    "MSS_ROOT",
    "MSS_UPSTREAM_DIR",
    "MSS_UPSTREAM_SHA",
    "MSS_UPSTREAM_URL",
    "MSS_VESSEL",
    "MssDeckMotion",
    "MssEvalConfigs",
    "load_mss_config",
    "load_mss_vessel_cached",
    "mss_key",
    "mss_motion",
    "mss_record_path",
    "mss_ss_label",
    "parse_mss_ss_label",
    "record_time_full_s",
    "resolve_mat_path",
]

#: The arm's config: dmf's ``configs/mss/s175_ss5.yaml`` with only ``vessel.mat_path``
#: changed (to the clone below).
MSS_CONFIG: Final[Path] = CONFIG_DIR / "deck" / "mss_s175_ss5.yaml"

#: Everything the arm builds lives here. It is gitignored and never inside ``third_party/``.
MSS_ROOT: Final[Path] = REPO_ROOT / "artifacts" / "mss"

#: The MSS toolbox clone, at :data:`MSS_UPSTREAM_SHA`.
MSS_UPSTREAM_DIR: Final[Path] = MSS_ROOT / "upstream"

#: dmf's exported records: ``<grid_kind>/<stem>.csv`` plus ``manifest.csv``.
MSS_RECORDS_DIR: Final[Path] = MSS_ROOT / "records"

#: Upstream repository, as named in dmf's ``THIRD_PARTY_NOTICES.md``.
MSS_UPSTREAM_URL: Final[str] = "https://github.com/cybergalactic/MSS.git"

#: dmf's pinned MSS commit (``mss/PATCHES.md``), 2026-09-07.
MSS_UPSTREAM_SHA: Final[str] = "98970f71a21cfe81e7e29abdcc1bb6741789cddc"

#: dmf hull config stem that sets the lever arm: ``-0.4 L = -70.0`` m full scale aft.
MSS_VESSEL: Final[str] = "s175"

#: Metres per second per knot, the literal ``dmf.mss.export.generate_realization`` uses.
#: (Equal to ``dmf.sim.encounter.knots_to_m_s(1.0)``. The literal is kept so the speed is
#: the export's by construction, not by coincidence.)
KNOT_M_S: Final[float] = 0.514444

#: Separator between the sea-state name and the source tag in an MSS key's ``ss`` field.
_SS_SEPARATOR: Final[str] = "/mss:"


class MssEvalConfigs(Protocol):
    """What :func:`mss_motion` reads from the eval side's ``EvalConfigs``.

    A structural type rather than an import of :class:`rld.eval.envs.EvalConfigs`.
    ``rld.eval`` imports ``rld.deck``, so importing back would be a cycle.
    """

    @property
    def scaling(self) -> ScalingConfig:
        """Froude scale config (``lam``, dimensionless)."""
        ...

    @property
    def pads(self) -> PadConfig:
        """Pad geometry; offsets are fractions of the full-scale vessel length."""
        ...

    @property
    def motion(self) -> MotionConfig:
        """Rates; ``forecast_lookback_full_s`` is seconds full scale."""
        ...


def load_mss_config(path: Path = MSS_CONFIG) -> dict[str, Any]:
    """Load the arm's config, cached per process and path.

    The dict is shared between callers and passed to dmf's functions unchanged. Do not
    mutate it.

    Args:
        path: ``configs/deck/mss_s175_ss5.yaml``.

    Returns:
        The parsed mapping. Sea state in metres and seconds full scale, headings in degrees,
        speeds in knots, record geometry in seconds and hertz full scale.
    """
    return _load_mss_config_cached(Path(path).resolve())


@cache
def _load_mss_config_cached(path: Path) -> dict[str, Any]:
    """Cached body of :func:`load_mss_config`, keyed on the resolved path."""
    return load_yaml(path)


def resolve_mat_path(cfg: dict[str, Any]) -> Path:
    """Return the S175 ``.mat`` path from the config, resolved against the repository root.

    Args:
        cfg: The arm's config.

    Returns:
        Absolute path to ``s175.mat`` inside the MSS clone.
    """
    raw = Path(str(cfg["vessel"]["mat_path"]))
    return raw if raw.is_absolute() else REPO_ROOT / raw


@cache
def load_mss_vessel_cached(path: Path) -> MSSVessel:
    """Load the MSS S175 vessel (RAO tables) once per process.

    Args:
        path: Absolute path to ``s175.mat``.

    Returns:
        dmf's :class:`~dmf.mss.vessel.MSSVessel`. RAOs are m/m and rad/m, frequencies rad/s,
        lengths metres full scale.

    Raises:
        FileNotFoundError: If the clone is absent (run ``make mss-export``).
    """
    return load_mss_vessel(path)


def record_time_full_s(cfg: dict[str, Any]) -> FloatArray:
    """Return the exported record's time axis, seconds full scale, absolute.

    Args:
        cfg: The arm's config.

    Returns:
        dmf's ``_time_axis(cfg)``: ``t_start_s + arange(n) / fs_hz``, i.e. 120.0 ... 719.9 s,
        6000 samples. These are the exact floats the CSV ``t`` column was written from.
    """
    return np.asarray(_time_axis(cfg), dtype=np.float64)


def mss_ss_label(grid_kind: str, sea_state: str = "SS5") -> str:
    """Return the ``ss`` field of an MSS realization key.

    Args:
        grid_kind: ``"mss"`` (primary) or ``"corpus"`` (attribution control).
        sea_state: Sea-state name from the config.

    Returns:
        ``"<sea_state>/mss:<grid_kind>"``, e.g. ``"SS5/mss:mss"``.

    Raises:
        ValueError: If ``grid_kind`` is not one of dmf's :data:`GRID_KINDS`.
    """
    if grid_kind not in GRID_KINDS:
        raise ValueError(f"unknown grid_kind {grid_kind!r}, expected one of {GRID_KINDS}")
    return f"{sea_state}{_SS_SEPARATOR}{grid_kind}"


def parse_mss_ss_label(ss: str) -> tuple[str, str]:
    """Invert :func:`mss_ss_label`.

    Args:
        ss: An MSS key's ``ss`` field.

    Returns:
        ``(sea_state, grid_kind)``.

    Raises:
        ValueError: If ``ss`` is not an MSS label (e.g. a dmf corpus ``"SS5"``).
    """
    sea_state, sep, grid_kind = ss.partition(_SS_SEPARATOR)
    if not sep or not sea_state or grid_kind not in GRID_KINDS:
        raise ValueError(f"{ss!r} is not an MSS sea-state label like 'SS5/mss:mss'")
    return sea_state, grid_kind


def mss_key(
    grid_kind: str, heading_deg: float, speed_kn: float, seed: int, sea_state: str = "SS5"
) -> RealizationKey:
    """Build an MSS realization's key, never equal to a dmf corpus key.

    Args:
        grid_kind: ``"mss"`` or ``"corpus"``.
        heading_deg: Encounter angle, degrees (180 head, 135 bow-quartering).
        speed_kn: Forward speed, knots full scale.
        seed: Seed ordinal, dimensionless.
        sea_state: Sea-state name.

    Returns:
        ``(mss_ss_label(grid_kind, sea_state), heading_deg, speed_kn, "s175", seed)``,
        through dmf's single cast site :func:`dmf.data.splits.realization_key`.
    """
    return realization_key(
        mss_ss_label(grid_kind, sea_state), heading_deg, speed_kn, MSS_VESSEL, seed
    )


def mss_record_path(spec: MSSRealizationSpec, records_dir: Path = MSS_RECORDS_DIR) -> Path:
    """Return where dmf's export wrote a realization's 10 Hz CSV.

    Args:
        spec: The realization.
        records_dir: The export's ``--out-dir``.

    Returns:
        ``records_dir / grid_kind / f"{spec.stem}.csv"``. Read by tests only.
    """
    return records_dir / spec.grid_kind / f"{spec.stem}.csv"


class MssDeckMotion:
    """One MSS realization as model-scale deck-point trajectories.

    Construction draws the realization's wave grid through dmf's export seed path. Every
    later call is a pure function of time.

    Attributes:
        spec: dmf's MSS realization coordinates.
        cfg: The arm's config (shared, do not mutate).
        scale: Froude scale, full to model.
        pads: Pad geometry; offsets are fractions of the full-scale vessel length.
        lookback_full_s: History reserved before the earliest usable model time, seconds
            full scale.
        vessel: MSS S175 RAO tables.
        length_full_m: Lever-arm length, metres full scale (dmf's ``s175`` config, 175.0).
        speed_m_s: Forward speed, metres per second full scale.
        grid: The realization's wave grid (frequencies rad/s, amplitudes m, phases rad).
        t_start_s: Record start, seconds full scale, absolute (120.0).
        duration_s: Record length, seconds full scale (600.0).
    """

    def __init__(
        self,
        spec: MSSRealizationSpec,
        cfg: dict[str, Any],
        scale: FroudeScale,
        pads: PadConfig,
        lookback_full_s: float = 0.0,
        *,
        vessel: MSSVessel | None = None,
    ) -> None:
        """Draw the realization's wave grid exactly as dmf's export does.

        Args:
            spec: Grid kind, heading (degrees), speed (knots) and seed ordinal.
            cfg: The arm's config (:func:`load_mss_config`).
            scale: Froude scale.
            pads: Pad geometry.
            lookback_full_s: Reserved history, seconds full scale, non-negative.
            vessel: Pre-loaded MSS vessel; loaded from ``cfg``'s ``mat_path`` if None.

        Raises:
            ValueError: If the realization is not one of the config's 18 per grid kind, the
                lookback is negative or leaves no window, or MSS's ``Lpp`` disagrees with
                dmf's ``s175`` length.
            FileNotFoundError: If the MSS clone is absent.
        """
        if spec.grid_kind not in GRID_KINDS:
            raise ValueError(f"unknown grid_kind {spec.grid_kind!r}, expected {GRID_KINDS}")
        for name, value, allowed in (
            ("heading_deg", spec.heading_deg, cfg["headings_deg"]),
            ("speed_kn", spec.speed_kn, cfg["speeds_kn"]),
            ("seed", spec.seed, cfg["seeds"]),
        ):
            if float(value) not in {float(a) for a in allowed}:
                raise ValueError(f"{name} {value} is not in the MSS arm's {list(allowed)}")
        rec = cfg["record"]
        self.t_start_s = float(rec["t_start_s"])
        self.duration_s = float(rec["duration_s"])
        if lookback_full_s < 0.0:
            raise ValueError(f"lookback_full_s must be non-negative, got {lookback_full_s}")
        if lookback_full_s >= self.duration_s:
            raise ValueError(
                f"lookback_full_s {lookback_full_s} leaves no usable window inside a "
                f"{self.duration_s} s record"
            )
        self.spec = spec
        self.cfg = cfg
        self.scale = scale
        self.pads = pads
        self.lookback_full_s = float(lookback_full_s)
        self.vessel = (
            vessel if vessel is not None else load_mss_vessel_cached(resolve_mat_path(cfg))
        )
        self.length_full_m = float(load_vessel_cached(MSS_VESSEL).length_m)
        if not math.isclose(self.vessel.length_m, self.length_full_m, rel_tol=1e-12):
            raise ValueError(
                f"MSS Lpp {self.vessel.length_m} m != dmf {MSS_VESSEL} length "
                f"{self.length_full_m} m: the lever arm would disagree with the plate anchor"
            )
        self.speed_m_s = float(spec.speed_kn) * KNOT_M_S
        self._speed_index = int(cfg["vessel"]["speed_index"])
        # dmf.mss.export.generate_realization, verbatim: one generator, then the grid.
        self.grid: WaveGrid = _build_grid(spec, cfg, realization_seed(spec))
        w_e = mss_encounter_frequency(
            self.grid.w_rad_s,
            self.speed_m_s,
            spec.heading_deg,
            gravity_m_s2=self.vessel.gravity_m_s2,
        )
        # Second derivative of every component: amplitude x w_e**2, phase + pi. Same
        # frequencies, so the RAO interpolation and w_e inside synthesize_mss_motion match.
        self._acc_grid = WaveGrid(
            w_rad_s=self.grid.w_rad_s,
            amplitude_m=self.grid.amplitude_m * w_e**2,
            phase_rad=self.grid.phase_rad + np.pi,
            source=self.grid.source,
        )

    @property
    def key(self) -> RealizationKey:
        """Return ``(ss/mss:<grid_kind>, heading_deg, speed_kn, "s175", seed)``; see module."""
        return mss_key(
            self.spec.grid_kind,
            self.spec.heading_deg,
            self.spec.speed_kn,
            self.spec.seed,
            str(self.cfg["sea_state"]["name"]),
        )

    @property
    def x_pad_full_m(self) -> float:
        """Vessel length the pad fractions multiply, metres full scale (175.0)."""
        return self.length_full_m

    @property
    def t_model_window_s(self) -> tuple[float, float]:
        """Usable model-scale span, seconds: ``(lookback, duration)`` in model time.

        ``(4.0, 120.0)`` at ``lam = 1/25`` with the 20 s full-scale lookback, identical to
        the JONSWAP source's window.
        """
        return (
            float(self.scale.to_model_time(self.lookback_full_s)),
            float(self.scale.to_model_time(self.duration_s)),
        )

    def episode_start_window_s(self, episode_len_model_s: float) -> tuple[float, float]:
        """Return the model-scale window of legal episode start offsets, seconds.

        Args:
            episode_len_model_s: Episode length, seconds model scale.

        Returns:
            ``(lo, hi - episode_len_model_s)`` model-scale seconds, as on the JONSWAP source.

        Raises:
            ValueError: If the episode does not fit in the usable window.
        """
        lo, hi = self.t_model_window_s
        if not 0.0 < episode_len_model_s <= hi - lo:
            raise ValueError(
                f"episode of {episode_len_model_s} s (model) does not fit in the usable "
                f"window {(lo, hi)}"
            )
        return (lo, hi - episode_len_model_s)

    def full_time_s(self, t_model_s: FloatArray) -> FloatArray:
        """Map model-scale seconds to absolute full-scale seconds.

        Args:
            t_model_s: Times, seconds model scale, from the start of the record.

        Returns:
            ``t_start_s + t_model_s / sqrt(lam)``, seconds full scale, absolute.
        """
        t_model = np.asarray(t_model_s, dtype=np.float64)
        return np.asarray(self.t_start_s + self.scale.to_full_time(t_model))

    def _check_record(self, t_full: FloatArray) -> None:
        """Reject full-scale times outside ``[t_start_s, t_start_s + duration_s]``.

        Args:
            t_full: Times, seconds full scale, absolute.

        Raises:
            ValueError: If any time leaves the record (beyond :data:`TIME_BOUND_SLACK_S`).
        """
        lo = self.t_start_s - TIME_BOUND_SLACK_S
        hi = self.t_start_s + self.duration_s + TIME_BOUND_SLACK_S
        if t_full.size and (float(t_full.min()) < lo or float(t_full.max()) > hi):
            raise ValueError(
                f"full-scale times [{float(t_full.min())}, {float(t_full.max())}] leave the "
                f"MSS record [{self.t_start_s}, {self.t_start_s + self.duration_s}]"
            )

    def _synthesize(self, grid: WaveGrid, t_full: FloatArray) -> MSSMotion:
        """Call ``synthesize_mss_motion``, widening a one-sample request to ``[t, t]``.

        Args:
            grid: The realization's grid, or its acceleration twin.
            t_full: Times, seconds full scale, 1-D, at least one sample.

        Returns:
            dmf's :class:`~dmf.mss.synth.MSSMotion` in MSS units and signs, possibly on the
            widened two-sample axis. The caller keeps the first ``t_full.size`` samples.
        """
        t_eval = np.repeat(t_full, 2) if t_full.size == 1 else t_full
        return synthesize_mss_motion(
            self.vessel,
            grid,
            self.spec.heading_deg,
            self.speed_m_s,
            t_eval,
            speed_index=self._speed_index,
        )

    def channels(self, t_full_s: FloatArray) -> MotionChannels:
        """Evaluate the motion channels at absolute full-scale times.

        Args:
            t_full_s: 1-D times, seconds full scale, absolute, inside
                ``[t_start_s, t_start_s + duration_s]``.

        Returns:
            :class:`~rld.deck.bridge.MotionChannels`, full scale, corpus signs. The six
            forecast channels are ``mss_motion_to_frame(synthesize_mss_motion(...))``, the
            export's frame. ``heave_acc``, ``roll_acc`` and ``pitch_acc`` are the analytic
            second derivatives, through the same conversion.

        Raises:
            ValueError: If ``t_full_s`` is not 1-D or a time leaves the record.
        """
        t_full = np.asarray(t_full_s, dtype=np.float64)
        if t_full.ndim != 1:
            raise ValueError(f"t_full_s must be one-dimensional, got shape {t_full.shape}")
        self._check_record(t_full)
        n = t_full.size
        if n == 0:
            empty = np.zeros(0, dtype=np.float64)
            return MotionChannels(t_full, *(empty,) * 9)
        motion = mss_motion_to_frame(self._synthesize(self.grid, t_full))
        # Same sign and rad->deg conversion as the motion itself. Validation is off: its
        # limits are plausibility bounds for angles and rates, not for accelerations.
        acc = mss_motion_to_frame(self._synthesize(self._acc_grid, t_full), validate=False)

        def col(frame: Any, name: str) -> FloatArray:
            return np.asarray(frame[name].to_numpy(dtype=np.float64)[:n])

        return MotionChannels(
            t_full_s=t_full,
            roll_deg=col(motion, "roll"),
            pitch_deg=col(motion, "pitch"),
            heave_m=col(motion, "heave"),
            roll_rate_dps=col(motion, "roll_rate"),
            pitch_rate_dps=col(motion, "pitch_rate"),
            heave_rate_m_s=col(motion, "heave_rate"),
            heave_acc_m_s2=col(acc, "heave"),
            roll_acc_dps2=col(acc, "roll"),
            pitch_acc_dps2=col(acc, "pitch"),
        )

    def deck_points(
        self, t_model_s: FloatArray, pads: Sequence[str]
    ) -> dict[str, DeckPointTrajectory]:
        """Evaluate several pads from one channel evaluation.

        Args:
            t_model_s: Times, seconds model scale, from the start of the record.
            pads: Pad names from ``configs/deck/pad.yaml``.

        Returns:
            One model-scale world-frame :class:`~rld.deck.bridge.DeckPointTrajectory` per
            pad, keyed by name.

        Raises:
            ValueError: If a pad is unknown or a time leaves the record.
        """
        t_model = np.asarray(t_model_s, dtype=np.float64)
        t_full = self.full_time_s(t_model)
        ch = self.channels(t_full)
        out: dict[str, DeckPointTrajectory] = {}
        for pad in pads:
            state_full = deck_point_state(
                roll_deg=ch.roll_deg,
                pitch_deg=ch.pitch_deg,
                heave_m=ch.heave_m,
                roll_rate_dps=ch.roll_rate_dps,
                pitch_rate_dps=ch.pitch_rate_dps,
                heave_rate_m_s=ch.heave_rate_m_s,
                roll_acc_dps2=ch.roll_acc_dps2,
                pitch_acc_dps2=ch.pitch_acc_dps2,
                heave_acc_m_s2=ch.heave_acc_m_s2,
                r_pad_m=self.pads.spec(pad).r_pad_full_m(self.length_full_m),
            )
            out[pad] = DeckPointTrajectory(
                pad=pad,
                t_model_s=t_model,
                t_full_s=t_full,
                state=to_model_state(state_full, self.scale),
            )
        return out

    def deck_point(self, t_model_s: FloatArray, pad: str) -> DeckPointTrajectory:
        """Evaluate one pad's world-frame trajectory at model-scale times.

        Kinematics at full scale with the full-scale lever arm (``-70.0`` m aft), then one
        Froude scaling: lengths x ``lam``, velocities x ``sqrt(lam)``, accelerations x 1,
        angles x 1, angular rates x ``1/sqrt(lam)``.

        Args:
            t_model_s: Times, seconds model scale, from the start of the record.
            pad: Pad name, e.g. ``"aft"`` or ``"cg"``.

        Returns:
            The trajectory at **model** scale, world frame (x bow, y port, z up).

        Raises:
            ValueError: If the pad is unknown or a time leaves the record.
        """
        return self.deck_points(t_model_s, (pad,))[pad]

    def deck_point_rows(
        self, t_model_s: FloatArray, pad: str, start: int, stop: int
    ) -> DeckPointTrajectory:
        """Evaluate rows ``[start, stop)`` of :meth:`deck_point` on a grid, bit for bit.

        The source is elementwise in time (module guarantee 3), so this is
        ``deck_point(t[start:stop])`` after checking the whole grid against the record. It
        has the same contract as :meth:`rld.deck.bridge.JonswapDeckMotion.deck_point_rows`.

        Args:
            t_model_s: The whole grid, seconds model scale.
            pad: Pad name.
            start: First row, inclusive.
            stop: Last row, exclusive; ``start < stop <= len(t_model_s)``.

        Returns:
            The trajectory for the ``stop - start`` rows, model scale, world frame.

        Raises:
            ValueError: If the row range is empty or out of bounds, the pad is unknown, or a
                grid time leaves the record.
        """
        grid = np.asarray(t_model_s, dtype=np.float64).reshape(-1)
        if not 0 <= start < stop <= grid.size:
            raise ValueError(f"rows [{start}, {stop}) outside a grid of {grid.size} samples")
        self._check_record(self.full_time_s(grid))
        return self.deck_point(grid[start:stop], pad)


def mss_motion(
    cfgs: MssEvalConfigs,
    grid_kind: str,
    heading_deg: float,
    speed_kn: float,
    seed: int,
    *,
    config_path: Path = MSS_CONFIG,
) -> MssDeckMotion:
    """Build one MSS realization's deck-motion source for the evaluation runner.

    Args:
        cfgs: The committed eval configs (``rld.eval.envs.EvalConfigs``). The arm reads
            ``scaling`` (lam), ``pads`` and ``motion.forecast_lookback_full_s``.
        grid_kind: ``"mss"`` (regime ``mss_transfer``) or ``"corpus"``
            (``mss_transfer_corpus``).
        heading_deg: 180.0 or 135.0, degrees.
        speed_kn: 0.0, 6.0 or 12.0, knots full scale.
        seed: 0, 1 or 2.
        config_path: The arm's config.

    Returns:
        The source, with the JONSWAP source's window and the 20 s full-scale lookback. Its
        ``key`` is :func:`mss_key`, and the vessel for ``pad_offset_for`` is ``"s175"``.

    Raises:
        ValueError: If the realization is not one of the arm's.
        FileNotFoundError: If the MSS clone is absent (``make mss-export``).
    """
    cfg = load_mss_config(config_path)
    return MssDeckMotion(
        MSSRealizationSpec(
            grid_kind=str(grid_kind),
            heading_deg=float(heading_deg),
            speed_kn=float(speed_kn),
            seed=int(seed),
        ),
        cfg,
        cfgs.scaling.froude_scale(),
        cfgs.pads,
        cfgs.motion.forecast_lookback_full_s,
    )
