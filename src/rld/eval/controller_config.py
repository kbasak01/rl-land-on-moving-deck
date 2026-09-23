"""The fully resolved configuration of a registered controller, and its SHA-256.

``summary.csv`` has always carried ``controller_config_sha256``: the hash of the controller's
own YAML **bytes**. That hash does not move when an **inherited** value moves. ``gated`` and
``oracle_gated`` load their gains from ``pid_feedforward.yaml`` through ``gains_from``, so
re-tuning ``pid_feedforward`` changes what they fly while leaving their YAML hash unchanged
(results-skeptic minor 5). ``resolved_gains_sha256`` closes that gap. It is the SHA-256 of a
canonical JSON of the configuration the controller **was built with**, after every
``gains_from`` has been followed, so it moves whenever any value the controller flies moves.

What is hashed
--------------
The controller is built exactly as the runner builds it (:func:`rld.control.registry.
make_controller` with the committed configs). The canonical document is:

* ``registry_name`` and ``class`` (``module.QualName``), so two controllers with identical
  numbers but different laws (``gated`` and ``oracle_gated``) never share a hash;
* ``config``: the most specific config dataclass the controller holds -- ``gated``
  (:class:`~rld.control.config.GatedConfig`, gains inlined) if present, else ``ff``
  (:class:`~rld.control.config.FeedforwardConfig`), else ``pid``
  (:class:`~rld.control.config.PidConfig`) -- as nested field dicts;
* ``quiescence_rule``, for the gated controllers: the Froude-converted limits and the rate
  (:class:`rld.control.quiescence.QuiescenceRule`), so a change in dmf's thresholds or in
  ``lam`` moves the hash too.

Canonicalisation: dataclasses become field dicts, paths are written relative to
``configs/control/`` (never absolute, so the hash does not depend on the checkout location),
floats are ``repr`` (shortest exact round trip), keys are sorted, separators are fixed.
:mod:`rld.control` is read, never edited.
"""

import hashlib
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from rld.control.base import ControlSpec
from rld.control.config import CONTROL_CONFIG_DIR
from rld.control.registry import REGISTRY, make_controller

__all__ = [
    "controller_provenance_columns",
    "resolved_config_json",
    "resolved_config_sha256",
]


def _canonical(value: Any) -> Any:
    """Return a JSON-ready canonical form of a config value.

    Args:
        value: A dataclass, path, mapping, sequence or scalar.

    Returns:
        Nested dicts, lists and scalars; a path relative to ``configs/control/`` when it
        lies there, else its file name.

    Raises:
        TypeError: On a value with no canonical form (a new config type must be added here
            deliberately, not hashed by ``repr``).
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _canonical(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        resolved = value.resolve()
        base = CONTROL_CONFIG_DIR.resolve()
        return (
            resolved.relative_to(base).as_posix()
            if resolved.is_relative_to(base)
            else resolved.name
        )
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"no canonical form for {type(value).__name__}: {value!r}")


def resolved_config_json(
    name: str, spec: ControlSpec | None = None, config_path: Path | None = None
) -> str:
    """Return the canonical JSON of a controller's fully resolved configuration.

    Args:
        name: Registry name.
        spec: Environment facts; the committed configs when omitted.
        config_path: Override of the committed YAML (tests only).

    Returns:
        Canonical JSON text (sorted keys, no whitespace), as described in the module
        docstring.

    Raises:
        TypeError: If the controller holds none of the known config attributes.
    """
    controller = make_controller(name, spec, config_path)
    for attr in ("gated", "ff", "pid"):
        config = getattr(controller, attr, None)
        if config is not None and is_dataclass(config):
            break
    else:
        raise TypeError(f"{name}: no gated/ff/pid config attribute to resolve")
    cls = type(controller)
    document: dict[str, Any] = {
        "registry_name": name,
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "config": _canonical(config),
    }
    rule = getattr(controller, "rule", None)
    if rule is not None:
        document["quiescence_rule"] = {
            "limits": _canonical(rule.limits),
            "rate_hz": float(rule.rate_hz),
            "n_samples": int(rule.n_samples),
        }
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def resolved_config_sha256(
    name: str, spec: ControlSpec | None = None, config_path: Path | None = None
) -> str:
    """Return the SHA-256 hex digest of :func:`resolved_config_json` (UTF-8).

    Args:
        name: Registry name.
        spec: Environment facts; the committed configs when omitted.
        config_path: Override of the committed YAML (tests only).

    Returns:
        64 hex characters.
    """
    text = resolved_config_json(name, spec, config_path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def controller_provenance_columns(
    names: list[str], spec: ControlSpec | None = None
) -> dict[str, dict[str, str]]:
    """Return the per-method provenance columns of ``summary.csv`` for registry controllers.

    Args:
        names: Registry names, in output order.
        spec: Environment facts; the committed configs when omitted.

    Returns:
        ``{name: {"controller_config_sha256": <YAML bytes>, "resolved_gains_sha256":
        <resolved config>}}``.
    """
    return {
        name: {
            "controller_config_sha256": hashlib.sha256(
                REGISTRY[name].config_path.read_bytes()
            ).hexdigest(),
            "resolved_gains_sha256": resolved_config_sha256(name, spec),
        }
        for name in names
    }
