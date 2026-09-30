"""Bridge from the read-only ``dmf`` seakeeping simulator to model-scale deck points.

What this module guarantees
---------------------------
1. **The seed path is dmf's, exactly.** ``realization_seed_sequence(spec).spawn(2)``, the
   **first** child into ``sample_components``, the second spawned and discarded (spawning is
   what advances the sequence, so skipping it changes the wave draw), then
   ``dmf.sim.response.synthesize_motion`` on the requested time grid. Reproduced from
   ``dmf/sim/generate.py::simulate_realization`` and asserted against it by
   ``tests/test_bridge.py``.
2. **Absolute full-scale time.** dmf's corpus ``t`` column starts at ``spinup_s = 120.0`` s
   and ``synthesize_motion`` is evaluated on that same absolute grid, so corpus time *is*
   synthesis time. Model time is measured from the start of the committed record:

       t_full_s = spinup_s + t_model_s / sqrt(lam)

   Using a relative grid instead shifts every channel by 120 s of full-scale time and still
   produces plausible statistics -- it costs ~12 deg of roll error against a 9e-4 tolerance.
3. **No interpolation and no finite differences.** ``synthesize_motion`` is a memoryless sum
   of harmonics in ``t``, so it may be evaluated directly on the sparse model-scale physics
   grid: the result is bit-identical to evaluating the dense record and slicing. Rates come
   from dmf's analytic frequency-domain derivatives. The two **angular accelerations** dmf
   does not store are rebuilt in the same phasor form, ``-w_e**2 * z_dof``, from dmf's
   public :func:`dmf.sim.response.dof_transfer` and
   :func:`dmf.sim.encounter.encounter_frequency`; that inherits dmf's signed-``w_e`` /
   ``abs(w_e)`` handling of the encounter-non-monotonic 45 deg cells for free.

Units and scales
----------------
``channels`` is **full scale**: degrees, degrees per second, degrees per second squared,
metres, metres per second, metres per second squared, and full-scale seconds for ``t``.
``deck_point`` is **model scale** in the world frame of :mod:`rld.deck.kinematics`
(x = bow, y = port, z = up). Nothing in between is ever half-scaled: kinematics are
evaluated at full scale with the full-scale lever arm, then Froude-scaled once.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Protocol

import numpy as np
from dmf.config import SeaState, SimConfig
from dmf.data.splits import RealizationKey, realization_key
from dmf.sim.encounter import encounter_frequency, knots_to_m_s
from dmf.sim.generate import VESSEL_CONFIG_DIR, RealizationSpec, realization_seed_sequence
from dmf.sim.response import MotionRecord, dof_transfer, synthesize_motion
from dmf.sim.spectra import WaveComponents, sample_components
from dmf.sim.vessel import Vessel, load_vessel
from dmf.typedefs import ComplexArray, FloatArray

from rld.deck.config import PadConfig
from rld.deck.kinematics import DeckPointState, deck_point_state, to_model_state
from rld.deck.scaling import FroudeScale

__all__ = [
    "CLEAN_CHANNELS",
    "TIME_CHUNK",
    "FORECAST_CHANNELS",
    "DeckMotionSource",
    "DeckPointTrajectory",
    "DofPhasors",
    "JonswapDeckMotion",
    "MotionChannels",
    "dof_phasors",
    "harmonic_sum",
    "harmonic_sum_rows",
    "MIN_SYNTHESIS_ROWS",
    "load_vessel_cached",
]

#: Corpus column name -> :class:`MotionChannels` field, for dmf's seven clean channels.
#: These are the columns ``tests/test_bridge.py`` checks parity on, in corpus order.
CLEAN_CHANNELS: dict[str, str] = {
    "roll": "roll_deg",
    "pitch": "pitch_deg",
    "heave": "heave_m",
    "roll_rate": "roll_rate_dps",
    "pitch_rate": "pitch_rate_dps",
    "heave_rate": "heave_rate_m_s",
    "heave_acc": "heave_acc_m_s2",
}

#: The six channels the Phase 4 forecaster adapter feeds the dmf models, in dmf's
#: ``DataConfig.input_channels`` order. Degrees, degrees per second, metres, metres per
#: second -- all **full scale**, sampled at 10 Hz full scale.
FORECAST_CHANNELS: tuple[str, ...] = (
    "roll",
    "pitch",
    "heave",
    "roll_rate",
    "pitch_rate",
    "heave_rate",
)

#: Time-sample block size of the superposition, matching ``dmf.sim.response._TIME_CHUNK``.
#: Keeps the ``(n_samples, n_components)`` trig temporaries in cache instead of allocating a
#: 28 800 x 299 array per channel for a whole model-scale record.
TIME_CHUNK: int = 8192

#: Slack, seconds full scale, allowed when checking a requested time against the committed
#: record. Covers float round-off in ``spinup_s + t_model/sqrt(lam)``, nothing more.
TIME_BOUND_SLACK_S: float = 1e-9

#: Fewest time rows :meth:`JonswapDeckMotion.deck_point_rows` hands dmf's
#: ``synthesize_motion`` (P5-D5). A one-row ``(1, 299) @ (299, 7)`` product is dispatched by
#: numpy to a matrix-vector BLAS routine whose summation order differs from the matrix-matrix
#: one the full grid uses (P2-D8 measured exactly that); a one-row request is therefore
#: widened by a neighbouring row and the extra row discarded.
MIN_SYNTHESIS_ROWS: int = 2


@dataclass(frozen=True)
class MotionChannels:
    """dmf's clean motion channels plus the two angular accelerations, **full scale**.

    Attributes:
        t_full_s: Sample times, seconds, full scale, absolute (starting at ``spinup_s``).
        roll_deg: Roll, degrees, positive to starboard.
        pitch_deg: Pitch, degrees, positive bow-up.
        heave_m: Heave, metres, positive up.
        roll_rate_dps: Roll rate, degrees per second.
        pitch_rate_dps: Pitch rate, degrees per second.
        heave_rate_m_s: Heave rate, metres per second.
        heave_acc_m_s2: Heave acceleration, metres per second squared.
        roll_acc_dps2: Roll acceleration, degrees per second squared. Not a dmf corpus
            column; reconstructed analytically (see the module docstring).
        pitch_acc_dps2: Pitch acceleration, degrees per second squared. Likewise.
    """

    t_full_s: FloatArray
    roll_deg: FloatArray
    pitch_deg: FloatArray
    heave_m: FloatArray
    roll_rate_dps: FloatArray
    pitch_rate_dps: FloatArray
    heave_rate_m_s: FloatArray
    heave_acc_m_s2: FloatArray
    roll_acc_dps2: FloatArray
    pitch_acc_dps2: FloatArray


@dataclass(frozen=True)
class DeckPointTrajectory:
    """One pad's trajectory over a model-scale time grid.

    Attributes:
        pad: Pad name, e.g. ``"aft"`` or ``"cg"``.
        t_model_s: Sample times, seconds, **model** scale, measured from the start of the
            committed dmf record.
        t_full_s: The same samples in full-scale absolute seconds, kept so that any number
            downstream can be traced back to a corpus row.
        state: Pose, velocity and acceleration at **model** scale in the world frame.
    """

    pad: str
    t_model_s: FloatArray
    t_full_s: FloatArray
    state: DeckPointState


@dataclass(frozen=True)
class DofPhasors:
    """Per-component complex response phasors for the three DOFs.

    ``x(t) = Re(sum_i z_i exp(1j * w_e_i * t))`` reproduces dmf's superposition, with the
    **signed** encounter frequency in the exponent and the transfer function evaluated at
    ``abs(w_e)`` inside :func:`dmf.sim.response.dof_transfer`.

    Attributes:
        w_e_rad_s: Signed encounter frequencies, radians per second, shape ``(n,)``.
        roll: Roll phasors, **radians** per component, shape ``(n,)``.
        pitch: Pitch phasors, radians per component, shape ``(n,)``.
        heave: Heave phasors, metres per component, shape ``(n,)``.
    """

    w_e_rad_s: FloatArray
    roll: ComplexArray
    pitch: ComplexArray
    heave: ComplexArray


class DeckMotionSource(Protocol):
    """The interface every deck-motion model implements (JONSWAP and sinusoid alike).

    Phase 2's platform body and Phase 4's forecaster adapter are written against this
    protocol, not against :class:`JonswapDeckMotion`, which is what makes the H4
    motion-realism arm a config change rather than a code change. Not
    ``runtime_checkable``: it carries properties, so ``issubclass`` would raise; conformance
    is checked statically and by attribute in the tests.
    """

    @property
    def key(self) -> RealizationKey:
        """Return dmf's realization key ``(ss, heading_deg, speed_kn, vessel, seed)``."""
        ...

    @property
    def t_model_window_s(self) -> tuple[float, float]:
        """Usable model-scale span, seconds: ``(lookback, duration)`` (see the class)."""
        ...

    def full_time_s(self, t_model_s: FloatArray) -> FloatArray:
        """Map model-scale seconds to absolute full-scale seconds."""
        ...

    def channels(self, t_full_s: FloatArray) -> MotionChannels:
        """Evaluate the motion channels at absolute full-scale times."""
        ...

    def deck_point(self, t_model_s: FloatArray, pad: str) -> DeckPointTrajectory:
        """Evaluate one pad's model-scale world-frame trajectory."""
        ...


@cache
def load_vessel_cached(name: str) -> Vessel:
    """Load one dmf vessel definition, cached per process.

    Uses dmf's public :func:`dmf.sim.vessel.load_vessel` and public
    :data:`dmf.sim.generate.VESSEL_CONFIG_DIR` rather than dmf's private ``_vessel`` cache.

    Args:
        name: Vessel config stem, ``"frigate"`` or ``"s175"``.

    Returns:
        The parsed vessel. ``length_m`` and ``draft_m`` are metres, **full scale**.

    Raises:
        FileNotFoundError: If no config exists for ``name``.
    """
    return load_vessel(VESSEL_CONFIG_DIR / f"{name}.yaml")


def dof_phasors(
    components: WaveComponents, vessel: Vessel, heading_deg: float, speed_m_s: float
) -> DofPhasors:
    """Build the per-component response phasors for one realization.

    Mirrors the phasor construction inside :func:`dmf.sim.response.synthesize_motion`: one
    complex number per wave component per DOF, ``z = a * H * exp(1j*phi)``, with the wave
    phase shared across DOFs (which is what preserves their phase relationships, defect
    included).

    Args:
        components: The realization's wave component set.
        vessel: Hull parameters, full scale.
        heading_deg: Encounter angle, degrees (180 head, 90 beam).
        speed_m_s: Forward speed, metres per second, full scale.

    Returns:
        The :class:`DofPhasors`. Roll and pitch are in radians, heave in metres.
    """
    w = components.w_rad_s
    w_e = encounter_frequency(w, speed_m_s, heading_deg)
    wave = components.amplitude_m * np.exp(1j * components.phase_rad)
    return DofPhasors(
        w_e_rad_s=w_e,
        roll=dof_transfer("roll", w, w_e, vessel, heading_deg) * wave,
        pitch=dof_transfer("pitch", w, w_e, vessel, heading_deg) * wave,
        heave=dof_transfer("heave", w, w_e, vessel, heading_deg) * wave,
    )


def harmonic_sum(phasors: ComplexArray, w_e_rad_s: FloatArray, t_s: FloatArray) -> FloatArray:
    """Evaluate ``Re(sum_i z_i exp(1j*w_e_i*t))`` for one or more phasor sets.

    The same superposition dmf uses, in the same ``cos @ Re - sin @ Im`` form, so a channel
    rebuilt here agrees with ``synthesize_motion`` to floating-point round-off
    (``tests/test_bridge.py`` asserts ``rtol = 1e-12`` on all seven dmf channels).

    Args:
        phasors: Complex phasors, shape ``(n_channels, n_components)``.
        w_e_rad_s: Signed encounter frequencies, radians per second, shape
            ``(n_components,)``.
        t_s: Sample times, seconds, shape ``(n_samples,)``. Full scale here; the function
            itself is scale-agnostic.

    Returns:
        Real values, shape ``(n_samples, n_channels)``, in the units of ``phasors``.
    """
    flat_t = np.asarray(t_s, dtype=np.float64).reshape(-1)
    real = np.ascontiguousarray(phasors.real.T)
    imag = np.ascontiguousarray(phasors.imag.T)
    out = np.empty((flat_t.size, real.shape[1]), dtype=np.float64)
    # Chunked over time exactly as dmf's own superposition is, so a 28 800-sample model-scale
    # record does not materialise an (n_samples, n_components) temporary. Each output row
    # depends only on its own time, so chunking is bit-identical to one large matmul.
    for start in range(0, flat_t.size, TIME_CHUNK):
        stop = min(start + TIME_CHUNK, flat_t.size)
        theta = np.outer(flat_t[start:stop], w_e_rad_s)
        out[start:stop] = np.cos(theta) @ real - np.sin(theta) @ imag
    return out


def harmonic_sum_rows(
    phasors: ComplexArray, w_e_rad_s: FloatArray, t_s: FloatArray, start: int, stop: int
) -> FloatArray:
    """Return rows ``[start, stop)`` of :func:`harmonic_sum`, bit for bit, evaluating only them.

    Why this is not ``harmonic_sum(phasors, w_e_rad_s, t_s[start:stop])`` (P5-D5): the
    trig is elementwise, but the reduction over wave components is a BLAS matrix product,
    and OpenBLAS picks its kernel -- hence its summation order -- from the operand
    **shape**. On this project's pinned stack (OpenBLAS 0.3.34, SkylakeX) a
    ``(m, 299) @ (299, 2)`` product takes the small-matrix kernel while ``m * 2 * 299 <= 1e6``
    and the blocked kernel above it, and the two differ in the last bits (<= 8e-14 of a
    quantity of order 1). A slice of the 3 001-sample physics grid therefore does not
    reproduce the full-grid rows.

    So the cos/sin are evaluated for the requested rows only -- that is the whole cost --
    and written into a block with **exactly the shape, and at exactly the row positions**,
    that :func:`harmonic_sum` hands BLAS for the same rows of the same grid. The product
    then runs the same kernel on the same shape; each output row depends only on its own
    input row, so the block's other rows (zeros, or finite leftovers from an earlier call:
    the block is a reused workspace) cannot change it. The equality is
    asserted with ``np.array_equal`` by ``tests/test_lazy_trajectory.py``.

    Args:
        phasors: Complex phasors, shape ``(n_channels, n_components)``.
        w_e_rad_s: Signed encounter frequencies, radians per second, shape
            ``(n_components,)``.
        t_s: The **whole** sample grid :func:`harmonic_sum` would be called on, seconds,
            shape ``(n_samples,)``. Full scale here; the function is scale-agnostic.
        start: First row, inclusive, in ``[0, n_samples]``.
        stop: Last row, exclusive, in ``[start, n_samples]``.

    Returns:
        Real values, shape ``(stop - start, n_channels)``, in the units of ``phasors``.

    Raises:
        ValueError: If ``[start, stop)`` is not a valid row range of ``t_s``.
    """
    flat_t = np.asarray(t_s, dtype=np.float64).reshape(-1)
    if not 0 <= start <= stop <= flat_t.size:
        raise ValueError(f"rows [{start}, {stop}) outside a grid of {flat_t.size} samples")
    real = np.ascontiguousarray(phasors.real.T)
    imag = np.ascontiguousarray(phasors.imag.T)
    out = np.empty((stop - start, real.shape[1]), dtype=np.float64)
    # The same TIME_CHUNK blocks as harmonic_sum, visited only where they overlap the rows.
    first_block = (start // TIME_CHUNK) * TIME_CHUNK
    for block in range(first_block, stop, TIME_CHUNK):
        block_stop = min(block + TIME_CHUNK, flat_t.size)
        lo, hi = max(start, block), min(stop, block_stop)
        theta = np.outer(flat_t[lo:hi], w_e_rad_s)
        cos_block, sin_block = _rows_workspace(block_stop - block, w_e_rad_s.size)
        cos_block[lo - block : hi - block] = np.cos(theta)
        sin_block[lo - block : hi - block] = np.sin(theta)
        product = cos_block @ real - sin_block @ imag
        out[lo - start : hi - start] = product[lo - block : hi - block]
    return out


#: Reused ``(n_rows, n_components)`` cos/sin operands for :func:`harmonic_sum_rows`, keyed by
#: shape. Allocating and zeroing two 7 MB blocks per chunk cost ~1.2 ms, a fifth of a
#: chunk. Reuse is exact: rows outside the requested range hold whatever an earlier call
#: left there (always finite cos/sin values or the initial zeros), and a matrix product's
#: output row depends on its own input row only, so they cannot reach the rows returned.
_ROWS_WORKSPACE: dict[tuple[int, int], tuple[FloatArray, FloatArray]] = {}


def _rows_workspace(n_rows: int, n_components: int) -> tuple[FloatArray, FloatArray]:
    """Return the cached cos and sin operands for one block shape, zero-filled on creation.

    Args:
        n_rows: Rows of the block, i.e. the time samples of the ``harmonic_sum`` block.
        n_components: Wave components, dimensionless.

    Returns:
        ``(cos_block, sin_block)``, each ``(n_rows, n_components)`` float64, C-contiguous.
    """
    key = (int(n_rows), int(n_components))
    cached = _ROWS_WORKSPACE.get(key)
    if cached is None:
        cached = (np.zeros(key, dtype=np.float64), np.zeros(key, dtype=np.float64))
        _ROWS_WORKSPACE[key] = cached
    return cached


class JonswapDeckMotion:
    """JONSWAP deck motion for one dmf realization, as model-scale deck-point trajectories.

    Construction draws the realization's wave components through dmf's seed path; every
    later call is a pure function of time, so one instance can serve an entire episode or
    the whole 600 s record.

    Attributes:
        spec: The dmf realization coordinates.
        sim_cfg: dmf's corpus config -- the source of ``spinup_s``, ``duration_s`` and the
            synthesis band, all full scale.
        scale: The Froude scale between full and model scale.
        pads: Pad geometry; offsets are fractions of the vessel's full-scale length.
        lookback_full_s: Full-scale seconds of history reserved before the earliest usable
            model time, so that Phase 4's 200-sample, 10 Hz full-scale lookback always lands
            inside the committed record. History before an episode starts is legitimate --
            the ship was already moving.
    """

    def __init__(
        self,
        spec: RealizationSpec,
        sim_cfg: SimConfig,
        scale: FroudeScale,
        pads: PadConfig,
        lookback_full_s: float = 0.0,
    ) -> None:
        """Draw the realization's wave components through dmf's exact seed path.

        Args:
            spec: Grid coordinates and seed ordinal of the realization.
            sim_cfg: dmf's corpus config, normally loaded from the path in
                ``configs/deck/motion_jonswap.yaml``.
            scale: Froude scale, full scale to model scale.
            pads: Pad geometry.
            lookback_full_s: Reserved history, seconds full scale, non-negative.

        Raises:
            ValueError: If ``spec.sea_state`` is not in ``sim_cfg``, or if
                ``lookback_full_s`` is negative or leaves no usable window.
        """
        if lookback_full_s < 0.0:
            raise ValueError(f"lookback_full_s must be non-negative, got {lookback_full_s}")
        if lookback_full_s >= sim_cfg.duration_s:
            raise ValueError(
                f"lookback_full_s {lookback_full_s} leaves no usable window inside a "
                f"{sim_cfg.duration_s} s record"
            )
        self.spec = spec
        self.sim_cfg = sim_cfg
        self.scale = scale
        self.pads = pads
        self.lookback_full_s = float(lookback_full_s)
        self.vessel = load_vessel_cached(spec.vessel)
        self.speed_m_s = knots_to_m_s(spec.speed_kn)
        sea = self._sea_state(spec.sea_state, sim_cfg)
        # dmf's seed path, verbatim: the second child is the IMU stream. It is unused here,
        # but spawning it is what advances the sequence, so it cannot be skipped.
        wave_seq, _imu_seq = realization_seed_sequence(spec).spawn(2)
        self.components = sample_components(
            hs_m=sea.hs_m,
            tp_s=sea.tp_s,
            gamma=sea.gamma,
            n_components=sim_cfg.n_components,
            w_min_rad_s=sim_cfg.w_min_rad_s,
            w_max_rad_s=sim_cfg.w_max_rad_s,
            rng=np.random.default_rng(wave_seq),
            jitter=sim_cfg.jitter_frequencies,
        )
        self._phasors = dof_phasors(self.components, self.vessel, spec.heading_deg, self.speed_m_s)

    @staticmethod
    def _sea_state(name: str, sim_cfg: SimConfig) -> SeaState:
        """Return the named sea state from ``sim_cfg``.

        Args:
            name: Sea state label, e.g. ``"SS5"``.
            sim_cfg: dmf's corpus config.

        Returns:
            The matching :class:`dmf.config.SeaState`; ``hs_m`` metres, ``tp_s`` seconds,
            both full scale.

        Raises:
            ValueError: If the sea state is not configured.
        """
        for sea in sim_cfg.sea_states:
            if sea.name == name:
                return sea
        known = [sea.name for sea in sim_cfg.sea_states]
        raise ValueError(f"unknown sea state {name!r}, expected one of {known}")

    @property
    def key(self) -> RealizationKey:
        """Return dmf's realization key ``(ss, heading_deg, speed_kn, vessel, seed)``.

        Built through :func:`dmf.data.splits.realization_key` so that it compares equal to
        keys built from a manifest row, which is what the Phase 1 splits are keyed on.
        """
        return realization_key(
            self.spec.sea_state,
            self.spec.heading_deg,
            self.spec.speed_kn,
            self.spec.vessel,
            self.spec.seed,
        )

    @property
    def x_pad_full_m(self) -> float:
        """Longitudinal lever arm of the vessel's length unit, metres full scale (= L)."""
        return float(self.vessel.length_m)

    @property
    def t_model_window_s(self) -> tuple[float, float]:
        """Usable model-scale time span, seconds, as ``(lo, hi)``.

        ``lo`` is ``lookback_full_s`` converted to model time: at or after it, a full
        forecaster lookback still lands inside the committed record. ``hi`` is the record
        length in model time (120.0 s for dmf's 600 s record at ``lam = 1/25``). Times
        outside ``[0, hi]`` are rejected by :meth:`channels`: they are physically identical
        to times inside it, but they were never committed, so they stay out of reach. An
        episode must additionally leave its own length before ``hi`` -- see
        :meth:`episode_start_window_s`.
        """
        return (
            float(self.scale.to_model_time(self.lookback_full_s)),
            float(self.scale.to_model_time(self.sim_cfg.duration_s)),
        )

    def episode_start_window_s(self, episode_len_model_s: float) -> tuple[float, float]:
        """Return the model-scale window of legal episode start offsets, seconds.

        Args:
            episode_len_model_s: Episode length, seconds, model scale (Phase 2 uses
                <= 12 s).

        Returns:
            ``(lo, hi)`` model-scale seconds, such that an episode starting anywhere in the
            closed interval evaluates only committed times and has a full lookback available
            at its start.

        Raises:
            ValueError: If the episode does not fit in the record.
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
            t_model_s: Times, seconds, model scale, measured from the start of the committed
                record.

        Returns:
            ``spinup_s + t_model_s / sqrt(lam)``, seconds, full scale, absolute -- the
            convention of dmf's corpus ``t`` column.
        """
        t_model = np.asarray(t_model_s, dtype=np.float64)
        return np.asarray(self.sim_cfg.spinup_s + self.scale.to_full_time(t_model))

    def _check_record(self, t_full: FloatArray) -> None:
        """Reject full-scale times outside the committed record.

        Args:
            t_full: Times, seconds, full scale, absolute.

        Raises:
            ValueError: If any time lies outside ``[spinup_s, spinup_s + duration_s]``
                (plus :data:`TIME_BOUND_SLACK_S`).
        """
        lo = self.sim_cfg.spinup_s - TIME_BOUND_SLACK_S
        hi = self.sim_cfg.spinup_s + self.sim_cfg.duration_s + TIME_BOUND_SLACK_S
        if t_full.size and (float(t_full.min()) < lo or float(t_full.max()) > hi):
            raise ValueError(
                f"full-scale times [{float(t_full.min())}, {float(t_full.max())}] leave the "
                f"committed record [{self.sim_cfg.spinup_s}, "
                f"{self.sim_cfg.spinup_s + self.sim_cfg.duration_s}]"
            )

    def _angular_acc_phasors(self) -> ComplexArray:
        """Return the roll and pitch **acceleration** phasors, ``-w_e**2 * z``, radians.

        Returns:
            Shape ``(2, n_components)``: roll, then pitch, radians per second squared per
            component.
        """
        return np.stack(
            [
                -(self._phasors.w_e_rad_s**2) * self._phasors.roll,
                -(self._phasors.w_e_rad_s**2) * self._phasors.pitch,
            ]
        )

    @staticmethod
    def _assemble_channels(
        t_full: FloatArray, motion: MotionRecord, angular: FloatArray
    ) -> MotionChannels:
        """Pack dmf's record and the rebuilt angular accelerations into full-scale channels.

        Args:
            t_full: Times, seconds, full scale, absolute.
            motion: dmf's :class:`~dmf.sim.response.MotionRecord` on ``t_full``.
            angular: ``(n, 2)`` roll and pitch accelerations on ``t_full``, radians per
                second squared.

        Returns:
            The :class:`MotionChannels`, full scale, angles in degrees.
        """
        to_deg = float(np.degrees(1.0))
        return MotionChannels(
            t_full_s=t_full,
            roll_deg=motion.roll_deg,
            pitch_deg=motion.pitch_deg,
            heave_m=motion.heave_m,
            roll_rate_dps=motion.roll_rate_dps,
            pitch_rate_dps=motion.pitch_rate_dps,
            heave_rate_m_s=motion.heave_rate_m_s,
            heave_acc_m_s2=motion.heave_acc_m_s2,
            roll_acc_dps2=angular[:, 0] * to_deg,
            pitch_acc_dps2=angular[:, 1] * to_deg,
        )

    def channels(self, t_full_s: FloatArray) -> MotionChannels:
        """Evaluate the motion channels at absolute full-scale times.

        Args:
            t_full_s: Times, seconds, full scale, absolute -- inside the committed record
                ``[spinup_s, spinup_s + duration_s]``.

        Returns:
            The :class:`MotionChannels`, full scale. The seven dmf channels come from
            :func:`dmf.sim.response.synthesize_motion` itself; the two angular
            accelerations are rebuilt as ``-w_e**2 * z_dof`` in the same phasor form.

        Raises:
            ValueError: If any requested time lies outside the committed record.
        """
        t_full = np.asarray(t_full_s, dtype=np.float64)
        self._check_record(t_full)
        motion = synthesize_motion(
            self.components, self.vessel, self.spec.heading_deg, self.speed_m_s, t_full
        )
        angular = harmonic_sum(self._angular_acc_phasors(), self._phasors.w_e_rad_s, t_full)
        return self._assemble_channels(t_full, motion, angular)

    def _pad_trajectory(
        self, ch: MotionChannels, t_model: FloatArray, t_full: FloatArray, pad: str
    ) -> DeckPointTrajectory:
        """Turn full-scale channels into one pad's model-scale trajectory.

        Args:
            ch: Full-scale channels on ``t_full``.
            t_model: The same samples, seconds model scale.
            t_full: The same samples, seconds full scale, absolute.
            pad: Pad name from ``configs/deck/pad.yaml``.

        Returns:
            The :class:`DeckPointTrajectory`, model scale, world frame.

        Raises:
            ValueError: If ``pad`` is unknown.
        """
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
            r_pad_m=self.pads.spec(pad).r_pad_full_m(self.vessel.length_m),
        )
        return DeckPointTrajectory(
            pad=pad,
            t_model_s=t_model,
            t_full_s=t_full,
            state=to_model_state(state_full, self.scale),
        )

    def deck_points(
        self, t_model_s: FloatArray, pads: Sequence[str]
    ) -> dict[str, DeckPointTrajectory]:
        """Evaluate several pads' trajectories from a single channel evaluation.

        The statistics driver needs the aft pad and the CG pad of the same realization side
        by side; evaluating the channels once halves the cost and guarantees the two rows
        come from bit-identical motion.

        Args:
            t_model_s: Times, seconds, model scale, from the start of the committed record.
            pads: Pad names from ``configs/deck/pad.yaml``, e.g. ``("aft", "cg")``.

        Returns:
            One :class:`DeckPointTrajectory` per requested pad, at **model** scale in the
            world frame, keyed by pad name.

        Raises:
            ValueError: If a pad is unknown or a time leaves the committed record.
        """
        t_model = np.asarray(t_model_s, dtype=np.float64)
        t_full = self.full_time_s(t_model)
        ch = self.channels(t_full)
        return {pad: self._pad_trajectory(ch, t_model, t_full, pad) for pad in pads}

    def deck_point(self, t_model_s: FloatArray, pad: str) -> DeckPointTrajectory:
        """Evaluate one pad's world-frame trajectory at model-scale times.

        Kinematics are evaluated at **full** scale with the vessel's full-scale lever arm
        (``-0.4 L`` = -49.6 m on the frigate, -70.0 m on the s175) and Froude-scaled once
        afterwards: lengths by ``lam``, velocities by ``sqrt(lam)``, accelerations by 1,
        angles unchanged, angular rates by ``1/sqrt(lam)``.

        Args:
            t_model_s: Times, seconds, model scale, from the start of the committed record.
            pad: Pad name from ``configs/deck/pad.yaml``, e.g. ``"aft"`` or ``"cg"``.

        Returns:
            The :class:`DeckPointTrajectory` at **model** scale in the world frame
            (x = bow, y = port, z = up).

        Raises:
            ValueError: If ``pad`` is unknown or a time leaves the committed record.
        """
        return self.deck_points(t_model_s, (pad,))[pad]

    def deck_point_rows(
        self, t_model_s: FloatArray, pad: str, start: int, stop: int
    ) -> DeckPointTrajectory:
        """Evaluate rows ``[start, stop)`` of :meth:`deck_point` on a grid, bit for bit.

        ``deck_point_rows(t, pad, a, b)`` equals ``deck_point(t, pad)`` restricted to rows
        ``a:b`` in every array, exactly (``np.array_equal``, not a tolerance), while paying
        the synthesis cost of ``b - a`` rows instead of ``len(t)``. This is what lets the
        landing environment evaluate its 3 001-sample physics grid lazily, in chunks, as
        the episode advances (P5-D5), without changing a bit of any episode.

        How each stage stays exact:

        * **Time mapping, kinematics, Froude scaling**: elementwise in the sample, so a
          row's value does not depend on which other rows are evaluated with it.
        * **dmf's seven channels**: :func:`dmf.sim.response.synthesize_motion`, called on
          the row slice itself (dmf is read-only). Its ``(m, 299) @ (299, 7)`` reduction
          is row-invariant on the pinned OpenBLAS for every ``m >= 2``; a one-row request
          is widened to :data:`MIN_SYNTHESIS_ROWS` because numpy sends a one-row product to
          a matrix-vector routine with a different summation order.
        * **The two angular accelerations**: :func:`harmonic_sum_rows`, which feeds BLAS
          the same-shaped operand as the full grid, because the ``(m, 299) @ (299, 2)``
          product is **not** row-invariant (it switches kernel with ``m``).

        The whole grid is checked against the committed record up front, so a grid that
        :meth:`deck_point` would reject is rejected here on the first chunk, not
        mid-episode.

        Args:
            t_model_s: The **whole** grid, seconds model scale, from the start of the
                committed record.
            pad: Pad name from ``configs/deck/pad.yaml``.
            start: First row, inclusive.
            stop: Last row, exclusive; ``start < stop <= len(t_model_s)``.

        Returns:
            The :class:`DeckPointTrajectory` for the ``stop - start`` rows, model scale,
            world frame.

        Raises:
            ValueError: If the row range is empty or out of bounds, the pad is unknown, or
                any time in the grid leaves the committed record.
        """
        t_model_grid = np.asarray(t_model_s, dtype=np.float64).reshape(-1)
        n = t_model_grid.size
        if not 0 <= start < stop <= n:
            raise ValueError(f"rows [{start}, {stop}) outside a grid of {n} samples")
        t_full_grid = self.full_time_s(t_model_grid)
        self._check_record(t_full_grid)
        lo, hi = start, stop
        if hi - lo < MIN_SYNTHESIS_ROWS:
            hi = min(n, lo + MIN_SYNTHESIS_ROWS)
            lo = max(0, hi - MIN_SYNTHESIS_ROWS)
        widened = synthesize_motion(
            self.components,
            self.vessel,
            self.spec.heading_deg,
            self.speed_m_s,
            t_full_grid[lo:hi],
        )
        keep = slice(start - lo, stop - lo)
        motion = MotionRecord(
            t_s=widened.t_s[keep],
            roll_deg=widened.roll_deg[keep],
            pitch_deg=widened.pitch_deg[keep],
            heave_m=widened.heave_m[keep],
            roll_rate_dps=widened.roll_rate_dps[keep],
            pitch_rate_dps=widened.pitch_rate_dps[keep],
            heave_rate_m_s=widened.heave_rate_m_s[keep],
            heave_acc_m_s2=widened.heave_acc_m_s2[keep],
        )
        angular = harmonic_sum_rows(
            self._angular_acc_phasors(), self._phasors.w_e_rad_s, t_full_grid, start, stop
        )
        t_full = t_full_grid[start:stop]
        ch = self._assemble_channels(t_full, motion, angular)
        return self._pad_trajectory(ch, t_model_grid[start:stop], t_full, pad)
