"""Config dataclasses for ``rld.envs``: the landing environment and its frozen criteria.

One frozen dataclass per YAML file under ``configs/env/``, loaded through
:func:`rld.config.load_yaml`, exactly the pattern :mod:`rld.deck.config` established in
Phase 1. Nothing in :mod:`rld.envs` hard-codes a success threshold, a rate, a weight or a
geometry: they all arrive here.

Units and scales
----------------
Every length is metres **model** scale, every speed metres per second model scale, every
time seconds model scale, every angle degrees at the boundary. The drone is a Crazyflie
2.X and is *not* Froude-scaled; the deck is scaled down to it at ``lam = 1/25`` (P1-D1), so
one model second is five full-scale seconds. The only full-scale quantity that reaches this
module is the dmf record geometry, and that arrives through :mod:`rld.deck.config`, not
here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rld.config import CONFIG_DIR, load_yaml, require

__all__ = [
    "CRASH_TILT_COMPONENTS",
    "DRIVERS",
    "LANDING_CONFIG",
    "NOISE_CONFIG",
    "OBSERVATION_CONFIG",
    "REWARD_CONFIG",
    "SUCCESS_CONFIG",
    "VZ_COMPONENTS",
    "BoundsConfig",
    "CrashTiltComponent",
    "Driver",
    "InitConfig",
    "LandingConfig",
    "NoiseConfig",
    "ObservationConfig",
    "PlatformConfig",
    "RewardConfig",
    "SuccessConfig",
    "VzComponent",
    "load_landing",
    "load_noise",
    "load_observation",
    "load_reward",
    "load_success",
]

#: How the deck plate is driven along the deck-motion trajectory. Measured, not assumed:
#: see ``docs/protocol.md`` P2-D1 for both drivers' tracking and velocity numbers.
type Driver = Literal["constraint", "kinematic"]

#: Tuple form of :data:`Driver`, for validation.
DRIVERS: tuple[Driver, ...] = ("constraint", "kinematic")

#: Which component of the relative velocity the touchdown criterion is on. ``deck_normal``
#: is the frozen choice (P2-D5); ``world_z`` is committed beside it per episode.
type VzComponent = Literal["deck_normal", "world_z"]

#: Tuple form of :data:`VzComponent`.
VZ_COMPONENTS: tuple[VzComponent, ...] = ("deck_normal", "world_z")

#: Which tilt the ``crash`` threshold is read on. ``absolute`` is the frozen choice: it is
#: a loss-of-control detector, unlike the 15 deg *relative* landing-quality criterion.
type CrashTiltComponent = Literal["absolute", "relative"]

#: Tuple form of :data:`CrashTiltComponent`.
CRASH_TILT_COMPONENTS: tuple[CrashTiltComponent, ...] = ("absolute", "relative")

#: Committed config paths. Callers pass these explicitly, so that a test or a script never
#: silently reads a different file than the one it names.
LANDING_CONFIG: Path = CONFIG_DIR / "env" / "landing.yaml"
SUCCESS_CONFIG: Path = CONFIG_DIR / "env" / "success.yaml"
OBSERVATION_CONFIG: Path = CONFIG_DIR / "env" / "observation.yaml"
NOISE_CONFIG: Path = CONFIG_DIR / "env" / "noise.yaml"
REWARD_CONFIG: Path = CONFIG_DIR / "env" / "reward.yaml"


@dataclass(frozen=True)
class PlatformConfig:
    """The deck plate and how it is driven (P2-D1, P2-D3).

    Attributes:
        driver: ``"kinematic"`` or ``"constraint"``; see :data:`Driver`.
        half_extents_m: Box half-extents ``(hx, hy, hz)``, metres model scale. The
            committed plate is ``(0.40, 0.26, 0.01)`` = 0.80 x 0.52 x 0.02 m, the same for
            both hulls.
        deck_origin_m: Where the deck point sits on flat water, metres model scale, world
            frame (x = bow, y = port, z = up).
        mass_kg: Plate mass, kilograms. Large enough that the 27 g drone cannot move the
            ship within one physics step.
        constraint_max_force_n: ``maxForce`` passed to ``changeConstraint``, newtons.
            ``constraint`` driver only.
        lateral_friction: Dimensionless Coulomb coefficient of the plate. Bullet multiplies
            the two bodies' coefficients.
        restitution: Dimensionless coefficient of restitution of the plate.
    """

    driver: Driver
    half_extents_m: tuple[float, float, float]
    deck_origin_m: tuple[float, float, float]
    mass_kg: float
    constraint_max_force_n: float
    lateral_friction: float
    restitution: float

    @property
    def deck_origin_z_m(self) -> float:
        """Mean deck height above PyBullet's ground plane, metres model scale."""
        return self.deck_origin_m[2]


@dataclass(frozen=True)
class InitConfig:
    """The per-episode initial drone state distribution.

    Attributes:
        height_above_deck_m: ``(lo, hi)`` metres model scale above the **mean** deck, drawn
            uniformly.
        lateral_m: Half-width of the uniform lateral draw, metres model scale, applied
            independently in world x and y about the mean deck point.
    """

    height_above_deck_m: tuple[float, float]
    lateral_m: float


@dataclass(frozen=True)
class BoundsConfig:
    """Divergence bounds. Leaving them is a ``crash`` (P2-D5).

    Attributes:
        rel_xy_max_m: Maximum |x| and |y| of the drone relative to the pad centre, metres
            model scale.
        rel_z_max_m: Maximum |z| of the drone relative to the pad centre, metres model
            scale.
        below_deck_m: Metres model scale below the **mean** deck at which the drone is
            declared lost, reached well before PyBullet's ground plane at z = 0.
    """

    rel_xy_max_m: float
    rel_z_max_m: float
    below_deck_m: float


@dataclass(frozen=True)
class LandingConfig:
    """``configs/env/landing.yaml``: rates, episode geometry and the shared action space.

    Attributes:
        physics_freq_hz: PyBullet step rate, hertz model scale. An integer multiple of
            ``ctrl_freq_hz``.
        ctrl_freq_hz: Control and policy rate, hertz model scale.
        episode_len_s: Flight budget, seconds model scale. A ``timeout`` is declared here.
        dwell_grace_s: Extra simulated time after the budget, seconds model scale, granted
            only so that a late touchdown can finish its dwell. Not flight time.
        v_max_m_s: Action scale, metres per second model scale (P2-D2).
        v_max_is_norm_cap: When true, ``v_max_m_s`` caps the commanded **speed**, not each
            axis: a longer vector is rescaled rather than clipped per component.
        yaw_setpoint_deg: Yaw held for the whole episode, degrees.
        platform: Deck plate geometry and driver.
        init: Initial-state distribution.
        bounds: Divergence bounds.
    """

    physics_freq_hz: int
    ctrl_freq_hz: int
    episode_len_s: float
    dwell_grace_s: float
    v_max_m_s: float
    v_max_is_norm_cap: bool
    yaw_setpoint_deg: float
    platform: PlatformConfig
    init: InitConfig
    bounds: BoundsConfig

    @property
    def total_len_s(self) -> float:
        """Simulated span of one episode, seconds model scale (flight plus dwell grace)."""
        return self.episode_len_s + self.dwell_grace_s

    @property
    def physics_dt_s(self) -> float:
        """Physics timestep, seconds model scale."""
        return 1.0 / self.physics_freq_hz

    @property
    def ctrl_dt_s(self) -> float:
        """Control timestep, seconds model scale."""
        return 1.0 / self.ctrl_freq_hz

    @property
    def pyb_steps_per_ctrl(self) -> int:
        """Physics substeps per control step, dimensionless."""
        return self.physics_freq_hz // self.ctrl_freq_hz

    @property
    def n_physics_samples(self) -> int:
        """Deck samples one episode needs, dimensionless.

        ``total_len_s * physics_freq_hz + 1`` -- one per substep boundary, inclusive of
        both ends, so that the last sample is exactly ``t0 + total_len_s``. Making it any
        larger would push the last requested time past the committed dmf record when the
        start offset sits at the top of ``episode_start_window_s``.
        """
        return int(round(self.total_len_s * self.physics_freq_hz)) + 1

    @property
    def max_substeps(self) -> int:
        """Physics substeps one episode may execute, dimensionless."""
        return self.n_physics_samples - 1


@dataclass(frozen=True)
class SuccessConfig:
    """``configs/env/success.yaml``: the four frozen criteria and the outcome thresholds.

    Frozen at Gate 3; changing any field after that needs a dated deviation entry in
    ``docs/protocol.md``. See the YAML for the reasoning behind each number.

    Attributes:
        rel_vertical_velocity_max_m_s: Criterion 1, metres per second model scale, on the
            closing speed named by ``vz_component``.
        vz_component: Which component criterion 1 is on (P2-D5). ``deck_normal``.
        lateral_offset_max_m: Criterion 2, metres model scale, drone centre to pad centre
            measured in the deck plane.
        relative_tilt_max_deg: Criterion 3, degrees, drone body z versus the **deck
            normal**.
        contact_dwell_min_s: Criterion 4, seconds model scale.
        contact_loss_grace_s: Seconds model scale of contact flicker tolerated before a
            release is declared.
        crash_tilt_deg: Loss-of-control threshold, degrees.
        crash_tilt_component: Which tilt ``crash_tilt_deg`` is read on. ``absolute``.
        hard_landing_includes_relative_tilt: Whether a relative-tilt violation on the pad
            is a ``hard_landing`` (P2-D5). True.
        tunnelling_penetration_m: Penetration depth at first contact, metres model scale,
            beyond which the event is flagged as tunnelling.
        contact_normal_force_min_n: Newtons. Contacts carrying less are discarded, because
            Bullet reports candidate points up to ~2 cm early.
        analytic_contact_margin_m: Metres model scale. Where PyBullet places the surface:
            resting bodies separate by Bullet's collision margin, so the analytic detector
            triggers at ``clearance <= this`` rather than at exactly zero.
        detector_disagreement_window_s: Seconds model scale. The two detectors disagree
            when they fire further apart than this, or when exactly one fires.
    """

    rel_vertical_velocity_max_m_s: float
    vz_component: VzComponent
    lateral_offset_max_m: float
    relative_tilt_max_deg: float
    contact_dwell_min_s: float
    contact_loss_grace_s: float
    crash_tilt_deg: float
    crash_tilt_component: CrashTiltComponent
    hard_landing_includes_relative_tilt: bool
    tunnelling_penetration_m: float
    contact_normal_force_min_n: float
    analytic_contact_margin_m: float
    detector_disagreement_window_s: float


@dataclass(frozen=True)
class ObservationConfig:
    """``configs/env/observation.yaml``: which blocks are on and their clipping bounds.

    Attributes:
        include_last_action: Append ``a_{t-1}``, dimensionless, in ``[-1, 1]^3``. The
            smoothness penalty makes the MDP non-Markov without it.
        include_contact: Append a 0/1 contact flag. The dwell criterion is not learnable
            without it.
        max_body_rate_rad_s: Clip bound on the body-rate block, radians per second.
        max_velocity_m_s: Clip bound on the drone velocity block, metres per second model
            scale.
        max_rel_position_m: Clip bound on the relative pad position block, metres model
            scale.
        max_rel_velocity_m_s: Clip bound on the relative pad velocity block, metres per
            second model scale.
        max_height_m: Clip bound on the signed clearance, metres model scale.
    """

    include_last_action: bool
    include_contact: bool
    max_body_rate_rad_s: float
    max_velocity_m_s: float
    max_rel_position_m: float
    max_rel_velocity_m_s: float
    max_height_m: float


@dataclass(frozen=True)
class NoiseConfig:
    """``configs/env/noise.yaml``: the perception stand-in, applied to the relative block.

    Attributes:
        enabled: Off by default; Phase 7 ablates it.
        position_sigma_m: Gaussian standard deviation on relative position, metres model
            scale, independent per axis.
        velocity_sigma_m_s: Gaussian standard deviation on relative velocity, metres per
            second model scale.
        latency_ms: Fixed transport delay, milliseconds model scale, quantised down to a
            whole number of control steps.
        hold_freq_hz: Sample-and-hold refresh rate, hertz model scale. Equal to the control
            rate means no hold.
    """

    enabled: bool
    position_sigma_m: float
    velocity_sigma_m_s: float
    latency_ms: float
    hold_freq_hz: float


@dataclass(frozen=True)
class RewardConfig:
    """``configs/env/reward.yaml``: the weights of the pre-registered reward structure.

    The structure is fixed; only these weights are tuned, and only on ``id`` validation
    realizations. ``timeout`` carries no terminal term at all -- see the YAML.

    Attributes:
        w_progress: Weight on ``d_{t-1} - d_t``, per metre model scale.
        w_vz: Weight on the excess closing speed near the deck, per metre per second.
        v_safe_m_s: Closing speed below which no penalty applies, metres per second model
            scale.
        gate_height_m: Clearance below which the closing-speed penalty is active, metres
            model scale.
        w_smooth: Weight on ``||a_t - a_{t-1}||^2``, dimensionless.
        w_time: Per-control-step cost, dimensionless.
        r_success: Terminal bonus on ``success``, dimensionless.
        r_hard_landing: Terminal penalty magnitude on ``hard_landing``.
        r_off_pad: Terminal penalty magnitude on ``off_pad``.
        r_bounce: Terminal penalty magnitude on ``bounce``.
        r_crash: Terminal penalty magnitude on ``crash``.
    """

    w_progress: float
    w_vz: float
    v_safe_m_s: float
    gate_height_m: float
    w_smooth: float
    w_time: float
    r_success: float
    r_hard_landing: float
    r_off_pad: float
    r_bounce: float
    r_crash: float


def _triple(raw: Any, path: Path, key: str) -> tuple[float, float, float]:
    """Coerce a YAML sequence to a 3-tuple of floats.

    Args:
        raw: The parsed value.
        path: File it came from, for the error message.
        key: Key it came from, for the error message.

    Returns:
        ``(a, b, c)`` floats, in the units the schema documents.

    Raises:
        ValueError: If the value is not a sequence of exactly three numbers.
    """
    values = [float(v) for v in raw]
    if len(values) != 3:
        raise ValueError(f"{path}: {key} must have 3 entries, got {len(values)}")
    return (values[0], values[1], values[2])


def _pair(raw: Any, path: Path, key: str) -> tuple[float, float]:
    """Coerce a YAML sequence to an ordered 2-tuple of floats.

    Args:
        raw: The parsed value.
        path: File it came from, for the error message.
        key: Key it came from, for the error message.

    Returns:
        ``(lo, hi)`` floats with ``lo <= hi``, in the units the schema documents.

    Raises:
        ValueError: If the value is not a sequence of two numbers, or is not ordered.
    """
    values = [float(v) for v in raw]
    if len(values) != 2:
        raise ValueError(f"{path}: {key} must have 2 entries, got {len(values)}")
    if values[0] > values[1]:
        raise ValueError(f"{path}: {key} must be ordered (lo, hi), got {values}")
    return (values[0], values[1])


def load_landing(path: Path) -> LandingConfig:
    """Load ``configs/env/landing.yaml``.

    Args:
        path: Path to the YAML file, normally :data:`LANDING_CONFIG`.

    Returns:
        The parsed config. Rates hertz model scale, lengths metres model scale, times
        seconds model scale, ``v_max_m_s`` metres per second model scale.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required key is missing, if the driver is unknown, if the physics
            rate is not an integer multiple of the control rate, or if a rate, length or
            time that must be positive is not.
    """
    raw = load_yaml(path)
    pyb = int(require(raw, "physics_freq_hz", path))
    ctrl = int(require(raw, "ctrl_freq_hz", path))
    if pyb <= 0 or ctrl <= 0:
        raise ValueError(f"{path}: physics_freq_hz and ctrl_freq_hz must be positive")
    if pyb % ctrl != 0:
        raise ValueError(f"{path}: physics_freq_hz {pyb} is not a multiple of ctrl_freq_hz {ctrl}")

    plat_raw = require(raw, "platform", path)
    if not isinstance(plat_raw, dict):
        raise ValueError(f"{path}: 'platform' must be a mapping")
    driver = str(plat_raw["driver"])
    if driver not in DRIVERS:
        raise ValueError(f"{path}: unknown driver {driver!r}, expected one of {list(DRIVERS)}")
    platform = PlatformConfig(
        driver=driver,
        half_extents_m=_triple(plat_raw["half_extents_m"], path, "platform.half_extents_m"),
        deck_origin_m=_triple(plat_raw["deck_origin_m"], path, "platform.deck_origin_m"),
        mass_kg=float(plat_raw["mass_kg"]),
        constraint_max_force_n=float(plat_raw["constraint_max_force_n"]),
        lateral_friction=float(plat_raw["lateral_friction"]),
        restitution=float(plat_raw["restitution"]),
    )
    if min(platform.half_extents_m) <= 0.0 or platform.mass_kg <= 0.0:
        raise ValueError(f"{path}: plate half-extents and mass must be positive")

    init_raw = require(raw, "init", path)
    bounds_raw = require(raw, "bounds", path)
    if not isinstance(init_raw, dict) or not isinstance(bounds_raw, dict):
        raise ValueError(f"{path}: 'init' and 'bounds' must be mappings")

    episode_len_s = float(require(raw, "episode_len_s", path))
    dwell_grace_s = float(require(raw, "dwell_grace_s", path))
    if episode_len_s <= 0.0 or dwell_grace_s < 0.0:
        raise ValueError(f"{path}: episode_len_s must be positive and dwell_grace_s non-negative")
    v_max = float(require(raw, "v_max_m_s", path))
    if v_max <= 0.0:
        raise ValueError(f"{path}: v_max_m_s must be positive, got {v_max}")

    return LandingConfig(
        physics_freq_hz=pyb,
        ctrl_freq_hz=ctrl,
        episode_len_s=episode_len_s,
        dwell_grace_s=dwell_grace_s,
        v_max_m_s=v_max,
        v_max_is_norm_cap=bool(require(raw, "v_max_is_norm_cap", path)),
        yaw_setpoint_deg=float(require(raw, "yaw_setpoint_deg", path)),
        platform=platform,
        init=InitConfig(
            height_above_deck_m=_pair(
                init_raw["height_above_deck_m"], path, "init.height_above_deck_m"
            ),
            lateral_m=float(init_raw["lateral_m"]),
        ),
        bounds=BoundsConfig(
            rel_xy_max_m=float(bounds_raw["rel_xy_max_m"]),
            rel_z_max_m=float(bounds_raw["rel_z_max_m"]),
            below_deck_m=float(bounds_raw["below_deck_m"]),
        ),
    )


def load_success(path: Path) -> SuccessConfig:
    """Load ``configs/env/success.yaml``, the frozen criteria.

    Args:
        path: Path to the YAML file, normally :data:`SUCCESS_CONFIG`.

    Returns:
        The parsed criteria. Speeds metres per second model scale, lengths metres model
        scale, times seconds model scale, angles degrees, forces newtons.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required key is missing, if ``vz_component`` or
            ``crash_tilt_component`` is unknown, or if a threshold that must be positive is
            not.
    """
    raw = load_yaml(path)
    vz = str(require(raw, "vz_component", path))
    if vz not in VZ_COMPONENTS:
        raise ValueError(f"{path}: vz_component must be one of {list(VZ_COMPONENTS)}, got {vz!r}")
    crash_component = str(require(raw, "crash_tilt_component", path))
    if crash_component not in CRASH_TILT_COMPONENTS:
        raise ValueError(f"{path}: crash_tilt_component must be 'absolute' or 'relative'")
    cfg = SuccessConfig(
        rel_vertical_velocity_max_m_s=float(require(raw, "rel_vertical_velocity_max_m_s", path)),
        vz_component=vz,
        lateral_offset_max_m=float(require(raw, "lateral_offset_max_m", path)),
        relative_tilt_max_deg=float(require(raw, "relative_tilt_max_deg", path)),
        contact_dwell_min_s=float(require(raw, "contact_dwell_min_s", path)),
        contact_loss_grace_s=float(require(raw, "contact_loss_grace_s", path)),
        crash_tilt_deg=float(require(raw, "crash_tilt_deg", path)),
        crash_tilt_component=crash_component,
        hard_landing_includes_relative_tilt=bool(
            require(raw, "hard_landing_includes_relative_tilt", path)
        ),
        tunnelling_penetration_m=float(require(raw, "tunnelling_penetration_m", path)),
        contact_normal_force_min_n=float(require(raw, "contact_normal_force_min_n", path)),
        analytic_contact_margin_m=float(require(raw, "analytic_contact_margin_m", path)),
        detector_disagreement_window_s=float(require(raw, "detector_disagreement_window_s", path)),
    )
    positive = {
        "rel_vertical_velocity_max_m_s": cfg.rel_vertical_velocity_max_m_s,
        "lateral_offset_max_m": cfg.lateral_offset_max_m,
        "relative_tilt_max_deg": cfg.relative_tilt_max_deg,
        "contact_dwell_min_s": cfg.contact_dwell_min_s,
        "crash_tilt_deg": cfg.crash_tilt_deg,
        "tunnelling_penetration_m": cfg.tunnelling_penetration_m,
        "contact_normal_force_min_n": cfg.contact_normal_force_min_n,
        "analytic_contact_margin_m": cfg.analytic_contact_margin_m,
        "detector_disagreement_window_s": cfg.detector_disagreement_window_s,
    }
    bad = sorted(name for name, value in positive.items() if value <= 0.0)
    if bad:
        raise ValueError(f"{path}: these must be positive: {bad}")
    if cfg.relative_tilt_max_deg >= cfg.crash_tilt_deg:
        raise ValueError(
            f"{path}: relative_tilt_max_deg {cfg.relative_tilt_max_deg} must be below "
            f"crash_tilt_deg {cfg.crash_tilt_deg}, or no landing can be a hard_landing"
        )
    return cfg


def load_observation(path: Path) -> ObservationConfig:
    """Load ``configs/env/observation.yaml``.

    Args:
        path: Path to the YAML file, normally :data:`OBSERVATION_CONFIG`.

    Returns:
        The parsed config; bounds in metres, metres per second (model scale) and radians
        per second.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required key is missing or a bound is not positive.
    """
    raw = load_yaml(path)
    cfg = ObservationConfig(
        include_last_action=bool(require(raw, "include_last_action", path)),
        include_contact=bool(require(raw, "include_contact", path)),
        max_body_rate_rad_s=float(require(raw, "max_body_rate_rad_s", path)),
        max_velocity_m_s=float(require(raw, "max_velocity_m_s", path)),
        max_rel_position_m=float(require(raw, "max_rel_position_m", path)),
        max_rel_velocity_m_s=float(require(raw, "max_rel_velocity_m_s", path)),
        max_height_m=float(require(raw, "max_height_m", path)),
    )
    bounds = (
        cfg.max_body_rate_rad_s,
        cfg.max_velocity_m_s,
        cfg.max_rel_position_m,
        cfg.max_rel_velocity_m_s,
        cfg.max_height_m,
    )
    if min(bounds) <= 0.0:
        raise ValueError(f"{path}: every clipping bound must be positive, got {bounds}")
    return cfg


def load_noise(path: Path) -> NoiseConfig:
    """Load ``configs/env/noise.yaml``.

    Args:
        path: Path to the YAML file, normally :data:`NOISE_CONFIG`.

    Returns:
        The parsed config. Sigmas in metres and metres per second model scale, latency in
        milliseconds model scale, hold rate in hertz model scale.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required key is missing, a sigma or the latency is negative, or
            the hold rate is not positive.
    """
    raw = load_yaml(path)
    cfg = NoiseConfig(
        enabled=bool(require(raw, "enabled", path)),
        position_sigma_m=float(require(raw, "position_sigma_m", path)),
        velocity_sigma_m_s=float(require(raw, "velocity_sigma_m_s", path)),
        latency_ms=float(require(raw, "latency_ms", path)),
        hold_freq_hz=float(require(raw, "hold_freq_hz", path)),
    )
    if min(cfg.position_sigma_m, cfg.velocity_sigma_m_s, cfg.latency_ms) < 0.0:
        raise ValueError(f"{path}: sigmas and latency must be non-negative")
    if cfg.hold_freq_hz <= 0.0:
        raise ValueError(f"{path}: hold_freq_hz must be positive, got {cfg.hold_freq_hz}")
    return cfg


def load_reward(path: Path) -> RewardConfig:
    """Load ``configs/env/reward.yaml``.

    Args:
        path: Path to the YAML file, normally :data:`REWARD_CONFIG`.

    Returns:
        The parsed weights. ``v_safe_m_s`` is metres per second model scale and
        ``gate_height_m`` metres model scale; the weights themselves are dimensionless.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required key is missing or any weight is negative. Penalties are
            stored as positive magnitudes and subtracted at use.
    """
    raw = load_yaml(path)
    cfg = RewardConfig(
        w_progress=float(require(raw, "w_progress", path)),
        w_vz=float(require(raw, "w_vz", path)),
        v_safe_m_s=float(require(raw, "v_safe_m_s", path)),
        gate_height_m=float(require(raw, "gate_height_m", path)),
        w_smooth=float(require(raw, "w_smooth", path)),
        w_time=float(require(raw, "w_time", path)),
        r_success=float(require(raw, "r_success", path)),
        r_hard_landing=float(require(raw, "r_hard_landing", path)),
        r_off_pad=float(require(raw, "r_off_pad", path)),
        r_bounce=float(require(raw, "r_bounce", path)),
        r_crash=float(require(raw, "r_crash", path)),
    )
    negative = sorted(
        name for name, value in vars(cfg).items() if isinstance(value, float) and value < 0.0
    )
    if negative:
        raise ValueError(
            f"{path}: these must be non-negative (penalties are magnitudes): {negative}"
        )
    return cfg
