"""Config dataclasses for ``rld.deck``: Froude scale, pad geometry, motion sources.

One dataclass per YAML file under ``configs/deck/``, loaded through
:func:`rld.config.load_yaml`. Per-package config modules, rather than one monolithic
``rld.config``, because every later phase adds a family of its own.

Units, and model scale versus full scale, are stated on every field. In short: the sea
state, the corpus grid and dmf's record geometry are **full scale**; the pad radius and the
physics/control rates are **model scale**; pad offsets are dimensionless fractions of the
vessel's full-scale length.
"""

from dataclasses import dataclass
from pathlib import Path

from rld.config import CONFIG_DIR, load_yaml, require
from rld.deck.scaling import FroudeScale

__all__ = [
    "AMPLITUDE_RULE",
    "MOTION_JONSWAP_CONFIG",
    "MOTION_SINUSOID_CONFIG",
    "PAD_CONFIG",
    "SCALING_CONFIG",
    "FeasibilityReference",
    "FeasibilityRule",
    "PERIOD_RULE",
    "PHASE_RULE",
    "MotionConfig",
    "PadConfig",
    "PadSpec",
    "ScalingConfig",
    "SinusoidConfig",
    "load_motion",
    "load_pads",
    "load_scaling",
    "load_sinusoid",
]

#: The only per-DOF amplitude rule :mod:`rld.deck.sinusoid` implements: amplitude =
#: ``sqrt(2) * RMS`` of the matched JONSWAP realization, so a sinusoid and its twin share a
#: variance. A config naming anything else is rejected rather than silently ignored.
AMPLITUDE_RULE: str = "sqrt2_times_matched_jonswap_rms"

#: The only period rule implemented: the realization's peak encounter period.
PERIOD_RULE: str = "peak_encounter_period"

#: The only phase rule implemented: one phase per DOF, ``U(0, 2*pi)`` from the episode seed.
PHASE_RULE: str = "uniform_per_dof_from_episode_seed"

#: Committed config paths. Callers pass these explicitly rather than relying on defaults,
#: so that a test or script never silently reads a different file than the one it names.
SCALING_CONFIG: Path = CONFIG_DIR / "deck" / "scaling.yaml"
PAD_CONFIG: Path = CONFIG_DIR / "deck" / "pad.yaml"
MOTION_JONSWAP_CONFIG: Path = CONFIG_DIR / "deck" / "motion_jonswap.yaml"
MOTION_SINUSOID_CONFIG: Path = CONFIG_DIR / "deck" / "motion_sinusoid.yaml"


@dataclass(frozen=True)
class FeasibilityReference:
    """One reading of "the CF2X's commanded maximum speed", with its source.

    Attributes:
        name: Short identifier, written into ``results/deck_feasibility.csv``.
        v_max_m_s: The reference speed, metres per second, **model** scale (the drone is
            not Froude-scaled).
        source: Where the number comes from, file and line, as committed evidence.
        is_gate: True for the single reference Gate 1 reads. Exactly one reference in a
            config must set it.
    """

    name: str
    v_max_m_s: float
    source: str
    is_gate: bool


@dataclass(frozen=True)
class FeasibilityRule:
    """The pre-registered lambda feasibility rule.

    Attributes:
        rule: The rule text, committed verbatim so the CSV carries its own definition.
        fraction: Dimensionless fraction of the reference speed that the SS6 head-seas
            aft-pad model-scale vertical velocity p99 must not exceed.
        references: Candidate denominators, in config order.
    """

    rule: str
    fraction: float
    references: tuple[FeasibilityReference, ...]

    @property
    def gate_reference(self) -> FeasibilityReference:
        """Return the reference Gate 1 reads.

        Returns:
            The single :class:`FeasibilityReference` with ``is_gate`` set.

        Raises:
            ValueError: If zero or more than one reference is marked.
        """
        marked = [r for r in self.references if r.is_gate]
        if len(marked) != 1:
            raise ValueError(f"exactly one feasibility reference must be the gate, got {marked}")
        return marked[0]

    def threshold_m_s(self, reference: FeasibilityReference) -> float:
        """Return the threshold in metres per second, model scale, for one reference.

        Args:
            reference: The denominator to apply ``fraction`` to.

        Returns:
            ``fraction * reference.v_max_m_s``, metres per second, model scale.
        """
        return self.fraction * reference.v_max_m_s


@dataclass(frozen=True)
class ScalingConfig:
    """Froude scale, the declared fallback ladder, and the feasibility rule.

    Attributes:
        lam_inverse: ``1 / lam``, dimensionless. Stored inverted so that ladder entries
            such as 1/60 are exact in the YAML rather than rounded.
        lam_ladder_inverse: The declared ladder of fallbacks, as inverses, in the order the
            rule walks them.
        gravity_m_s2: Gravitational acceleration, metres per second squared, assumed equal
            at both scales.
        feasibility: The pre-registered rule.
    """

    lam_inverse: float
    lam_ladder_inverse: tuple[float, ...]
    gravity_m_s2: float
    feasibility: FeasibilityRule

    @property
    def lam(self) -> float:
        """Length scale ``L_model / L_full``, dimensionless."""
        return 1.0 / self.lam_inverse

    @property
    def lam_ladder(self) -> tuple[float, ...]:
        """The declared fallback ladder as length scales, dimensionless, largest first."""
        return tuple(1.0 / value for value in self.lam_ladder_inverse)

    def froude_scale(self) -> FroudeScale:
        """Return the :class:`~rld.deck.scaling.FroudeScale` this config describes."""
        return FroudeScale(lam=self.lam, gravity_m_s2=self.gravity_m_s2)


@dataclass(frozen=True)
class PadSpec:
    """One landing pad, as an offset from the vessel's centre of gravity.

    Attributes:
        name: Pad identifier, e.g. ``"aft"`` or ``"cg"``.
        r_pad_frac: ``(x, y, z)`` offsets as dimensionless fractions of the vessel's
            full-scale length between perpendiculars, in the ship body frame with
            **x = bow, y = port, z = up**. Negative x is aft.
    """

    name: str
    r_pad_frac: tuple[float, float, float]

    def r_pad_full_m(self, length_full_m: float) -> tuple[float, float, float]:
        """Return the pad offset in metres, full scale, in the body frame.

        Args:
            length_full_m: Vessel length between perpendiculars, metres, full scale
                (``dmf.sim.vessel.Vessel.length_m``).

        Returns:
            ``(x, y, z)`` metres, full scale, x = bow, y = port, z = up. For the default
            aft pad this is ``(-49.6, 0, 0)`` on the frigate and ``(-70.0, 0, 0)`` on the
            s175 -- the lever arm differs across hulls, which is a named confound of the
            ``unseen_vessel`` regime.
        """
        x_frac, y_frac, z_frac = self.r_pad_frac
        return (x_frac * length_full_m, y_frac * length_full_m, z_frac * length_full_m)


@dataclass(frozen=True)
class PadConfig:
    """The set of landing pads and their shared geometry.

    Attributes:
        radius_model_m: Pad radius, metres, **model** scale.
        pads: The pads, in config order. ``"aft"`` is the primary arm and ``"cg"`` the
            control arm that removes the lever-arm coupling (plan D0.5).
    """

    radius_model_m: float
    pads: tuple[PadSpec, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Pad names in config order, dimensionless labels."""
        return tuple(pad.name for pad in self.pads)

    def spec(self, name: str) -> PadSpec:
        """Return one pad by name.

        Args:
            name: Pad identifier.

        Returns:
            The matching :class:`PadSpec`.

        Raises:
            ValueError: If no pad has that name.
        """
        for pad in self.pads:
            if pad.name == name:
                return pad
        raise ValueError(f"unknown pad {name!r}, expected one of {list(self.names)}")


@dataclass(frozen=True)
class MotionConfig:
    """Sampling rates and the dmf corpus config behind one deck-motion source.

    Attributes:
        sim_config_path: Absolute path to dmf's corpus config, already resolved. It fixes
            the full-scale record geometry the bridge reads: ``spinup_s``, ``duration_s``
            and ``fs_hz``.
        physics_freq_hz: PyBullet step rate, hertz, **model** scale.
        ctrl_freq_hz: Controller and policy rate, hertz, **model** scale.
        forecast_fs_hz: The dmf forecaster's input rate, hertz, **full** scale.
        forecast_lookback_samples: Lookback length in samples at ``forecast_fs_hz``.

    Keys beyond these are ignored here -- ``motion_sinusoid.yaml``'s matching rules are read
    by :func:`load_sinusoid` into a :class:`SinusoidConfig`. Every key this dataclass needs is
    required explicitly, so an ignored key can only ever be an addition, never a typo of a
    field that matters.
    """

    sim_config_path: Path
    physics_freq_hz: float
    ctrl_freq_hz: float
    forecast_fs_hz: float
    forecast_lookback_samples: int

    @property
    def forecast_lookback_full_s(self) -> float:
        """Forecaster lookback span, seconds, **full** scale (200 / 10 Hz = 20.0 s)."""
        return self.forecast_lookback_samples / self.forecast_fs_hz

    @property
    def physics_dt_model_s(self) -> float:
        """Physics timestep, seconds, **model** scale."""
        return 1.0 / self.physics_freq_hz


def load_scaling(path: Path) -> ScalingConfig:
    """Load ``configs/deck/scaling.yaml``.

    Args:
        path: Path to the YAML file, normally :data:`SCALING_CONFIG`.

    Returns:
        The parsed config. ``lam`` is dimensionless, ``gravity_m_s2`` metres per second
        squared, and the feasibility threshold speeds are metres per second model scale.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required key is missing, if ``lam_inverse`` is not positive, or if
            the feasibility rule does not mark exactly one gate reference.
    """
    raw = load_yaml(path)
    lam_inverse = float(require(raw, "lam_inverse", path))
    if lam_inverse < 1.0:
        raise ValueError(f"{path}: lam_inverse must be >= 1 (lam <= 1), got {lam_inverse}")
    rule_raw = require(raw, "feasibility", path)
    if not isinstance(rule_raw, dict):
        raise ValueError(f"{path}: 'feasibility' must be a mapping")
    references = tuple(
        FeasibilityReference(
            name=str(entry["name"]),
            v_max_m_s=float(entry["v_max_m_s"]),
            source=str(entry["source"]),
            is_gate=bool(entry.get("is_gate", False)),
        )
        for entry in rule_raw["references"]
    )
    rule = FeasibilityRule(
        rule=str(rule_raw["rule"]).strip(),
        fraction=float(rule_raw["fraction"]),
        references=references,
    )
    _ = rule.gate_reference  # validate at load time, not at gate time
    return ScalingConfig(
        lam_inverse=lam_inverse,
        lam_ladder_inverse=tuple(float(v) for v in require(raw, "lam_ladder_inverse", path)),
        gravity_m_s2=float(require(raw, "gravity_m_s2", path)),
        feasibility=rule,
    )


def load_pads(path: Path) -> PadConfig:
    """Load ``configs/deck/pad.yaml``.

    Args:
        path: Path to the YAML file, normally :data:`PAD_CONFIG`.

    Returns:
        The parsed config: ``radius_model_m`` metres model scale, pad offsets
        dimensionless fractions of the vessel's full-scale length.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required key is missing, if a pad offset is not a 3-vector, or if
            two pads share a name.
    """
    raw = load_yaml(path)
    pads: list[PadSpec] = []
    for entry in require(raw, "pads", path):
        frac = [float(v) for v in entry["r_pad_frac"]]
        if len(frac) != 3:
            raise ValueError(f"{path}: pad {entry.get('name')!r} r_pad_frac must have 3 entries")
        pads.append(PadSpec(name=str(entry["name"]), r_pad_frac=(frac[0], frac[1], frac[2])))
    names = [pad.name for pad in pads]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate pad names {names}")
    return PadConfig(
        radius_model_m=float(require(raw, "radius_model_m", path)),
        pads=tuple(pads),
    )


def load_motion(path: Path) -> MotionConfig:
    """Load ``configs/deck/motion_jonswap.yaml`` or ``motion_sinusoid.yaml``.

    ``sim_config`` is resolved relative to ``path``'s directory, dmf's convention, so the
    working directory never matters.

    Args:
        path: Path to the YAML file, normally :data:`MOTION_JONSWAP_CONFIG` or
            :data:`MOTION_SINUSOID_CONFIG`.

    Returns:
        The parsed config. Rates are hertz -- ``physics_freq_hz`` and ``ctrl_freq_hz`` model
        scale, ``forecast_fs_hz`` full scale.

    Raises:
        FileNotFoundError: If ``path`` or the referenced ``sim_config`` does not exist.
        ValueError: If a required key is missing or a rate is not positive.
    """
    raw = load_yaml(path)
    sim_config_path = (path.parent / str(require(raw, "sim_config", path))).resolve()
    if not sim_config_path.exists():
        raise FileNotFoundError(f"{path}: sim_config does not exist: {sim_config_path}")
    rates = {
        key: float(require(raw, key, path))
        for key in ("physics_freq_hz", "ctrl_freq_hz", "forecast_fs_hz")
    }
    if any(value <= 0.0 for value in rates.values()):
        raise ValueError(f"{path}: rates must be positive, got {rates}")
    lookback = int(require(raw, "forecast_lookback_samples", path))
    if lookback <= 0:
        raise ValueError(f"{path}: forecast_lookback_samples must be positive, got {lookback}")
    return MotionConfig(
        sim_config_path=sim_config_path,
        physics_freq_hz=rates["physics_freq_hz"],
        ctrl_freq_hz=rates["ctrl_freq_hz"],
        forecast_fs_hz=rates["forecast_fs_hz"],
        forecast_lookback_samples=lookback,
    )


@dataclass(frozen=True)
class SinusoidConfig:
    """The Angelis-style sinusoidal motion model's matching rule (plan D0.4, H4).

    Attributes:
        motion: Rates and corpus config, shared with the JONSWAP source so that the two
            motion models are interchangeable at the same physics and control rates.
        amplitude_rule: Must equal :data:`AMPLITUDE_RULE`. Per DOF, degrees for roll and
            pitch and metres for heave, matched at **full** scale before Froude scaling.
        period_rule: Must equal :data:`PERIOD_RULE`. Seconds, **full** scale.
        phase_rule: Must equal :data:`PHASE_RULE`. Radians.
    """

    motion: MotionConfig
    amplitude_rule: str
    period_rule: str
    phase_rule: str


def load_sinusoid(path: Path) -> SinusoidConfig:
    """Load ``configs/deck/motion_sinusoid.yaml``.

    Args:
        path: Path to the YAML file, normally :data:`MOTION_SINUSOID_CONFIG`.

    Returns:
        The parsed config. Rates are hertz (model scale for physics and control, full scale
        for the forecaster); the three rule fields are the committed statement of the
        matching rule, not free text.

    Raises:
        FileNotFoundError: If ``path`` or the referenced ``sim_config`` does not exist.
        ValueError: If a required key is missing, or if a rule names a variant this package
            does not implement -- a silently ignored rule would change the H4 arm without a
            recorded decision.
    """
    raw = load_yaml(path)
    rules = {
        "amplitude_rule": AMPLITUDE_RULE,
        "period_rule": PERIOD_RULE,
        "phase_rule": PHASE_RULE,
    }
    parsed = {key: str(require(raw, key, path)) for key in rules}
    for key, implemented in rules.items():
        if parsed[key] != implemented:
            raise ValueError(
                f"{path}: {key} is {parsed[key]!r}, but rld.deck.sinusoid implements only "
                f"{implemented!r}"
            )
    return SinusoidConfig(
        motion=load_motion(path),
        amplitude_rule=parsed["amplitude_rule"],
        period_rule=parsed["period_rule"],
        phase_rule=parsed["phase_rule"],
    )
