"""Config dataclasses for ``configs/control/*.yaml``: every gain and threshold, none in code.

One frozen dataclass per controller family, loaded through :func:`rld.config.load_yaml`,
following :mod:`rld.envs.config`. The tuned gains of ``pid_feedforward`` are **referenced**,
not copied, by ``gated`` and ``oracle_gated`` (their YAMLs carry
``gains_from: pid_feedforward.yaml``), so re-tuning the base cannot leave the gated
controllers flying stale gains. ``gated_forecast`` and ``gated_forecast_tcn`` carry the same
keys plus the forecaster they consult (:class:`GatedForecastConfig`). ``pid_feedforward_lowvz_cut``
loads ``pid_feedforward_lowvz.yaml``'s gains the same way and adds its post-contact cut
(:class:`LowvzCutConfig`).

Units and scales
----------------
Every length is metres **model** scale, every speed metres per second model scale, every
time seconds model scale. A gain's unit is in its name: ``_per_s`` is (m/s)/m,
``_per_s2`` is (m/s)/(m s). Quiescence thresholds are **not** here: they are dmf's,
named by set and converted by :mod:`rld.control.quiescence`.
"""

import math
from dataclasses import dataclass
from pathlib import Path

from rld.config import CONFIG_DIR, REPO_ROOT, load_yaml, require
from rld.control.quiescence import THRESHOLD_SETS

__all__ = [
    "CONTROL_CONFIG_DIR",
    "FeedforwardConfig",
    "GatedConfig",
    "GatedForecastConfig",
    "LowvzCutConfig",
    "PidConfig",
    "load_feedforward",
    "load_gated",
    "load_gated_forecast",
    "load_lowvz_cut",
    "load_pid",
]

#: Directory of the committed controller configs.
CONTROL_CONFIG_DIR: Path = CONFIG_DIR / "control"


@dataclass(frozen=True)
class PidConfig:
    """``pid_track_descend``: lateral PI plus a lateral-gated constant-rate descent.

    Attributes:
        kp_xy_per_s: Proportional gain on the lateral error, (m/s)/m.
        ki_xy_per_s2: Integral gain on the lateral error, (m/s)/(m s).
        integral_limit_m_s: Anti-windup clamp on the integral term's contribution, per
            axis, metres per second model scale.
        lateral_speed_max_m_s: Cap on the horizontal speed the tracker commands, metres per
            second model scale.
        commit_radius_m: Lateral error below which descent is committed, metres model scale.
        release_factor: Descent is released (height held) when the lateral error exceeds
            ``release_factor * commit_radius_m``; dimensionless, ``>= 1`` (hysteresis).
        descent_rate_m_s: Commanded descent speed once committed, metres per second model
            scale, positive downward.
    """

    kp_xy_per_s: float
    ki_xy_per_s2: float
    integral_limit_m_s: float
    lateral_speed_max_m_s: float
    commit_radius_m: float
    release_factor: float
    descent_rate_m_s: float

    def __post_init__(self) -> None:
        """Validate ranges.

        Raises:
            ValueError: If a gain is negative, a radius, speed or rate is not positive, or
                the hysteresis factor is below 1.
        """
        if min(self.kp_xy_per_s, self.ki_xy_per_s2, self.integral_limit_m_s) < 0.0:
            raise ValueError("kp_xy_per_s, ki_xy_per_s2 and integral_limit_m_s must be >= 0")
        if min(self.lateral_speed_max_m_s, self.commit_radius_m, self.descent_rate_m_s) <= 0.0:
            raise ValueError(
                "lateral_speed_max_m_s, commit_radius_m and descent_rate_m_s must be positive"
            )
        if self.release_factor < 1.0:
            raise ValueError(f"release_factor must be >= 1, got {self.release_factor}")


@dataclass(frozen=True)
class FeedforwardConfig:
    """``pid_feedforward``: :class:`PidConfig` plus deck-point velocity feedforward.

    Attributes:
        pid: The tracker and descent gains.
        k_ff: Feedforward gain on the pad deck-point velocity, all three axes,
            dimensionless (1.0 = follow the pad exactly).
    """

    pid: PidConfig
    k_ff: float

    def __post_init__(self) -> None:
        """Validate ranges.

        Raises:
            ValueError: If ``k_ff`` is negative.
        """
        if self.k_ff < 0.0:
            raise ValueError(f"k_ff must be >= 0, got {self.k_ff}")


@dataclass(frozen=True)
class LowvzCutConfig(FeedforwardConfig):
    """``pid_feedforward_lowvz_cut``: lowvz's gains (by reference) plus the post-contact cut.

    A subclass of :class:`FeedforwardConfig` rather than a wrapper, so the controller's ``ff``
    attribute -- the config :mod:`rld.eval.controller_config` hashes -- carries the cut and
    the ``gains_from`` provenance too (the pattern of :class:`GatedForecastConfig`).

    Attributes:
        gains_from: The YAML the gains came from (``pid_feedforward_lowvz.yaml``), absolute.
        cut_speed_m_s: Descent commanded along the observed deck normal, on top of lowvz's
            setpoint, once contact has been observed; metres per second model scale,
            positive. The rule is ``docs/protocol.md`` P5-D2.
    """

    gains_from: Path
    cut_speed_m_s: float

    def __post_init__(self) -> None:
        """Validate ranges.

        Raises:
            ValueError: As :class:`FeedforwardConfig`, or if ``cut_speed_m_s`` is not
                positive.
        """
        super().__post_init__()
        if not self.cut_speed_m_s > 0.0:
            raise ValueError(f"cut_speed_m_s must be positive, got {self.cut_speed_m_s}")


@dataclass(frozen=True)
class GatedConfig:
    """``gated`` and ``oracle_gated``: hover above the deck, commit on a quiescence rule.

    Attributes:
        gains: ``pid_feedforward``'s gains, loaded from ``gains_from``.
        gains_from: The YAML the gains came from, for provenance.
        hover_height_m: Clearance held above the pad while waiting, metres model scale.
        hover_kp_per_s: Proportional gain on the hover-height error, (m/s)/m.
        hover_speed_max_m_s: Cap on the hover-height correction, metres per second model
            scale.
        hover_tolerance_m: The drone counts as "at hover" when within this of
            ``hover_height_m``, metres model scale; commits are only considered there.
        thresholds: dmf threshold-set name, ``"permissive"`` or ``"strict"``.
        fallback_commit_s: Episode time after which descent is committed regardless of the
            rule, seconds model scale, or ``None`` for never (the default: a never-quiescent
            episode then times out, and the timeout is reported).
    """

    gains: FeedforwardConfig
    gains_from: Path
    hover_height_m: float
    hover_kp_per_s: float
    hover_speed_max_m_s: float
    hover_tolerance_m: float
    thresholds: str
    fallback_commit_s: float | None

    def __post_init__(self) -> None:
        """Validate ranges.

        Raises:
            ValueError: If a height, gain, speed or tolerance is not positive, the threshold
                set is unknown, or the fallback time is negative.
        """
        values = (
            self.hover_height_m,
            self.hover_kp_per_s,
            self.hover_speed_max_m_s,
            self.hover_tolerance_m,
        )
        if min(values) <= 0.0:
            raise ValueError(f"hover parameters must be positive, got {values}")
        if self.thresholds not in THRESHOLD_SETS:
            raise ValueError(
                f"thresholds must be one of {sorted(THRESHOLD_SETS)}, got {self.thresholds!r}"
            )
        if self.fallback_commit_s is not None and self.fallback_commit_s < 0.0:
            raise ValueError(
                f"fallback_commit_s must be >= 0 or null, got {self.fallback_commit_s}"
            )


@dataclass(frozen=True)
class GatedForecastConfig(GatedConfig):
    """``gated_forecast`` / ``gated_forecast_tcn``: :class:`GatedConfig` plus the forecaster.

    Every inherited key has ``gated``'s meaning and, in the committed YAMLs, ``gated``'s
    value. A subclass rather than a wrapper, so the controller's ``gated`` attribute -- the
    config :mod:`rld.eval.controller_config` hashes -- carries the forecaster too.

    Attributes:
        forecaster_dir: The fitted dmf model directory written by
            :mod:`rld.deck.forecast_fit` (``artifacts/dmf/<model>/``), absolute. The YAML
            gives it relative to the repository root.
        interval: The band's quantile levels, dimensionless, ``(lo, hi)`` with
            ``0 < lo < hi < 1``; checked against the forecaster's ``interval_levels`` when
            the controller loads it, so the YAML cannot silently describe a different band
            from the one the rule is fed.
    """

    forecaster_dir: Path
    interval: tuple[float, float]

    def __post_init__(self) -> None:
        """Validate ranges.

        Raises:
            ValueError: As :class:`GatedConfig`, or if ``interval`` is not two levels with
                ``0 < lo < hi < 1``.
        """
        super().__post_init__()
        if len(self.interval) != 2 or not 0.0 < self.interval[0] < self.interval[1] < 1.0:
            raise ValueError(f"interval must be (lo, hi) with 0 < lo < hi < 1, got {self.interval}")

    def interval_matches(self, levels: tuple[float, float] | None) -> bool:
        """Return whether a forecaster's band levels are this config's ``interval``.

        Args:
            levels: The forecaster's ``interval_levels`` (None for a point model).

        Returns:
            True when both levels agree to ``1e-12``.
        """
        if levels is None or len(levels) != 2:
            return False
        return all(
            math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
            for a, b in zip(levels, self.interval, strict=True)
        )


#: The YAML keys of :class:`PidConfig`.
_PID_KEYS: frozenset[str] = frozenset(
    (
        "kp_xy_per_s",
        "ki_xy_per_s2",
        "integral_limit_m_s",
        "lateral_speed_max_m_s",
        "commit_radius_m",
        "release_factor",
        "descent_rate_m_s",
    )
)


def _pid_from(raw: dict[str, object], path: Path) -> PidConfig:
    """Build a :class:`PidConfig` from a parsed YAML mapping.

    Args:
        raw: The mapping.
        path: The file it came from, for error messages.

    Returns:
        The config.
    """
    return PidConfig(
        kp_xy_per_s=float(require(raw, "kp_xy_per_s", path)),
        ki_xy_per_s2=float(require(raw, "ki_xy_per_s2", path)),
        integral_limit_m_s=float(require(raw, "integral_limit_m_s", path)),
        lateral_speed_max_m_s=float(require(raw, "lateral_speed_max_m_s", path)),
        commit_radius_m=float(require(raw, "commit_radius_m", path)),
        release_factor=float(require(raw, "release_factor", path)),
        descent_rate_m_s=float(require(raw, "descent_rate_m_s", path)),
    )


def load_pid(path: Path) -> PidConfig:
    """Load ``configs/control/pid_track_descend.yaml``.

    Args:
        path: The YAML file.

    Returns:
        The parsed :class:`PidConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a key is missing or a value is out of range.
    """
    return _pid_from(load_yaml(path), path)


def load_feedforward(path: Path) -> FeedforwardConfig:
    """Load ``configs/control/pid_feedforward.yaml``.

    Args:
        path: The YAML file.

    Returns:
        The parsed :class:`FeedforwardConfig`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a key is missing or a value is out of range.
    """
    raw = load_yaml(path)
    return FeedforwardConfig(pid=_pid_from(raw, path), k_ff=float(require(raw, "k_ff", path)))


def load_lowvz_cut(path: Path) -> LowvzCutConfig:
    """Load ``configs/control/pid_feedforward_lowvz_cut.yaml``.

    ``gains_from`` is resolved relative to the YAML's own directory (as :func:`load_gated`),
    and every gain comes from that file; none is read from this one.

    Args:
        path: The YAML file.

    Returns:
        The parsed :class:`LowvzCutConfig`, with the referenced gains inlined.

    Raises:
        FileNotFoundError: If ``path`` or the referenced gains file does not exist.
        ValueError: If a key is missing, a value is out of range, or the file also carries a
            gain key (a copied gain could drift from the referenced one).
    """
    raw = load_yaml(path)
    copied = sorted(set(raw) & (_PID_KEYS | {"k_ff"}))
    if copied:
        raise ValueError(f"{path}: gains must come from gains_from, not be copied: {copied}")
    gains_from = (path.parent / str(require(raw, "gains_from", path))).resolve()
    gains = load_feedforward(gains_from)
    return LowvzCutConfig(
        pid=gains.pid,
        k_ff=gains.k_ff,
        gains_from=gains_from,
        cut_speed_m_s=float(require(raw, "cut_speed_m_s", path)),
    )


def load_gated(path: Path) -> GatedConfig:
    """Load ``configs/control/gated.yaml`` or ``oracle_gated.yaml``.

    ``gains_from`` is resolved relative to the YAML's own directory.

    Args:
        path: The YAML file.

    Returns:
        The parsed :class:`GatedConfig`, with ``pid_feedforward``'s gains inlined.

    Raises:
        FileNotFoundError: If ``path`` or the referenced gains file does not exist.
        ValueError: If a key is missing or a value is out of range.
    """
    raw = load_yaml(path)
    gains_from = (path.parent / str(require(raw, "gains_from", path))).resolve()
    fallback = require(raw, "fallback_commit_s", path)
    return GatedConfig(
        gains=load_feedforward(gains_from),
        gains_from=gains_from,
        hover_height_m=float(require(raw, "hover_height_m", path)),
        hover_kp_per_s=float(require(raw, "hover_kp_per_s", path)),
        hover_speed_max_m_s=float(require(raw, "hover_speed_max_m_s", path)),
        hover_tolerance_m=float(require(raw, "hover_tolerance_m", path)),
        thresholds=str(require(raw, "thresholds", path)),
        fallback_commit_s=None if fallback is None else float(fallback),
    )


def load_gated_forecast(path: Path) -> GatedForecastConfig:
    """Load ``configs/control/gated_forecast.yaml`` or ``gated_forecast_tcn.yaml``.

    ``gains_from`` is resolved relative to the YAML's own directory (as :func:`load_gated`);
    ``forecaster_dir`` relative to the repository root.

    Args:
        path: The YAML file.

    Returns:
        The parsed :class:`GatedForecastConfig`. The forecaster directory is **not** opened
        here; the controller loads it on first use.

    Raises:
        FileNotFoundError: If ``path`` or the referenced gains file does not exist.
        ValueError: If a key is missing or a value is out of range.
    """
    base = load_gated(path)
    raw = load_yaml(path)
    forecaster_dir = Path(str(require(raw, "forecaster_dir", path)))
    if not forecaster_dir.is_absolute():
        forecaster_dir = REPO_ROOT / forecaster_dir
    interval_raw = require(raw, "interval", path)
    if not isinstance(interval_raw, list | tuple) or len(interval_raw) != 2:
        raise ValueError(f"{path}: interval must be a two-element list, got {interval_raw!r}")
    return GatedForecastConfig(
        gains=base.gains,
        gains_from=base.gains_from,
        hover_height_m=base.hover_height_m,
        hover_kp_per_s=base.hover_kp_per_s,
        hover_speed_max_m_s=base.hover_speed_max_m_s,
        hover_tolerance_m=base.hover_tolerance_m,
        thresholds=base.thresholds,
        fallback_commit_s=base.fallback_commit_s,
        forecaster_dir=forecaster_dir.resolve(),
        interval=(float(interval_raw[0]), float(interval_raw[1])),
    )
