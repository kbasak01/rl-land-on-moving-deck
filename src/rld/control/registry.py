"""Controller registry: name -> (factory, config path, ``privileged``, ``needs_motion_feed``).

The evaluation runner resolves every controller through here, and it is the registry's
flags -- not the controller's own -- that decide what the runner hands to ``reset``: a
:class:`~rld.control.base.PrivilegedContext` only when ``privileged``, and a past-only
:class:`~rld.deck.forecast.ShipMotionFeed` only when ``needs_motion_feed``. Each
controller's test asserts that its own flags agree with the registry's.

``gated_forecast`` and ``gated_forecast_tcn`` are one class and law
(:class:`rld.control.gated_forecast.GatedForecast`) with two forecasters (dmf's
``residual_interval`` and ``tcn_quantile``). They need the feed and are **not** privileged:
the feed never reaches beyond the runner's clock.

``pid_feedforward_lowvz`` is ``pid_feedforward``'s class and law with a second gain set,
selected from the existing tuning log (P3-D3 amendment); it is not a new control law.

``oracle_gated`` is the only privileged entry. It is a commit-timing oracle (privileged), not
a bound on success or on landing quality, and must be marked as such in every table; it is
never a deployable result.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rld.control.base import Controller, ControlSpec
from rld.control.config import CONTROL_CONFIG_DIR
from rld.control.feedforward import make_pid_feedforward, make_pid_feedforward_lowvz
from rld.control.gated import make_gated
from rld.control.gated_forecast import make_gated_forecast, make_gated_forecast_tcn
from rld.control.oracle import make_oracle_gated
from rld.control.pid import make_pid_track_descend

__all__ = ["REGISTRY", "RegistryEntry", "entry", "make_controller"]

#: Factory signature: ``(config_path, spec) -> Controller``.
type Factory = Callable[[Path, ControlSpec], Controller]


@dataclass(frozen=True)
class RegistryEntry:
    """One registered controller.

    Attributes:
        name: Registry name; equals the controller's ``name``.
        factory: Builds a fresh controller from a config path and the environment facts.
        config_path: The committed YAML under ``configs/control/``.
        privileged: True if the controller reads the true future deck trajectory. The runner
            passes a context **only** to these.
        description: One line for tables and ``--help``.
        needs_motion_feed: True if ``reset`` requires the episode's past-only
            :class:`~rld.deck.forecast.ShipMotionFeed`. The runner builds and advances one
            **only** for these; every other controller raises if handed one.
    """

    name: str
    factory: Factory
    config_path: Path
    privileged: bool
    description: str
    needs_motion_feed: bool = False


#: Every registered controller, in the order tables print them.
REGISTRY: dict[str, RegistryEntry] = {
    item.name: item
    for item in (
        RegistryEntry(
            name="pid_track_descend",
            factory=make_pid_track_descend,
            config_path=CONTROL_CONFIG_DIR / "pid_track_descend.yaml",
            privileged=False,
            description="lateral PI, descend at a constant rate once laterally inside a radius",
        ),
        RegistryEntry(
            name="pid_feedforward",
            factory=make_pid_feedforward,
            config_path=CONTROL_CONFIG_DIR / "pid_feedforward.yaml",
            privileged=False,
            description=(
                "pid_track_descend + k_ff * pad velocity (ideal noise-free, zero-latency "
                "ship motion reference unit plus state estimate); residual base"
            ),
        ),
        RegistryEntry(
            name="pid_feedforward_lowvz",
            factory=make_pid_feedforward_lowvz,
            config_path=CONTROL_CONFIG_DIR / "pid_feedforward_lowvz.yaml",
            privileged=False,
            description=(
                "pid_feedforward's law with the lowest-p95-closing-speed gains within 0.02 of "
                "the best tune success (trial 10 of its log); H1 closing-speed reference"
            ),
        ),
        RegistryEntry(
            name="gated",
            factory=make_gated,
            config_path=CONTROL_CONFIG_DIR / "gated.yaml",
            privileged=False,
            description="hover 0.3 m above the pad, commit on current-state dmf quiescence",
        ),
        RegistryEntry(
            name="oracle_gated",
            factory=make_oracle_gated,
            config_path=CONTROL_CONFIG_DIR / "oracle_gated.yaml",
            privileged=True,
            description=(
                "commit-timing oracle (privileged): commit on the true future quiescent window"
            ),
        ),
        RegistryEntry(
            name="gated_forecast",
            factory=make_gated_forecast,
            config_path=CONTROL_CONFIG_DIR / "gated_forecast.yaml",
            privileged=False,
            needs_motion_feed=True,
            description=(
                "hover 0.3 m above the pad, commit when dmf residual_interval's 90 % band at "
                "the predicted touchdown is quiescent (past-only ship-motion feed)"
            ),
        ),
        RegistryEntry(
            name="gated_forecast_tcn",
            factory=make_gated_forecast_tcn,
            config_path=CONTROL_CONFIG_DIR / "gated_forecast_tcn.yaml",
            privileged=False,
            needs_motion_feed=True,
            description=(
                "gated_forecast with dmf tcn_quantile's 90 % band (past-only ship-motion feed)"
            ),
        ),
    )
}


def entry(name: str) -> RegistryEntry:
    """Return one registry entry.

    Args:
        name: Registry name.

    Returns:
        The entry.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown controller {name!r}, expected one of {list(REGISTRY)}")
    return REGISTRY[name]


def make_controller(
    name: str, spec: ControlSpec | None = None, config_path: Path | None = None
) -> Controller:
    """Build a registered controller.

    Args:
        name: Registry name.
        spec: Environment facts; the committed configs when omitted.
        config_path: Override of the committed YAML (tuning trials, tests).

    Returns:
        A fresh controller; call ``reset`` before use.
    """
    item = entry(name)
    return item.factory(
        item.config_path if config_path is None else config_path,
        ControlSpec.committed() if spec is None else spec,
    )
