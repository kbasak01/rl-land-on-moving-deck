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
  ``lam`` moves the hash too;
* ``forecaster``, for a controller whose config names a ``forecaster_dir``
  (``gated_forecast``, ``gated_forecast_tcn``): the directory's name, the SHA-256 of its
  ``meta.json``, its ``smoke`` flag, and the SHA-256 of **every** file ``meta.json`` lists
  under ``files`` -- today the checkpoint (``state_dict.pt``), the ONNX graph
  (``model.onnx``), the normalisation stats (``norm_stats.npz``) and the conformal pad-``v_z``
  calibration (``conformal_padvz.npz``), handled generically so a file the fit adds later is
  hashed without a code change. Each file is hashed from its bytes and checked against the
  digest ``meta.json`` records; a mismatch or a missing file raises, so a refit forecaster
  can never be reported under the old hash. The controllers without a forecaster are
  unaffected: their documents, and so their committed ``resolved_gains_sha256``, are
  unchanged.

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
    "PROVENANCE_COLUMNS",
    "controller_provenance_columns",
    "forecaster_provenance",
    "resolved_config_json",
    "resolved_config_sha256",
]

#: The per-method provenance columns of ``summary.csv``, in order.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "controller_config_sha256",
    "resolved_gains_sha256",
    "forecaster_files_sha256",
)


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def forecaster_provenance(model_dir: Path) -> dict[str, Any]:
    """Return the hashed identity of a fitted forecaster directory.

    Args:
        model_dir: ``artifacts/dmf/<model>/`` (or a smoke directory), holding ``meta.json``.

    Returns:
        ``{"dir": <name>, "meta_json_sha256": ..., "smoke": bool, "files": {name: sha256}}``
        with ``files`` sorted by name and covering every file ``meta.json`` lists.

    Raises:
        FileNotFoundError: If ``meta.json`` or a listed file is missing.
        ValueError: If a listed file's bytes do not match the digest in ``meta.json``, or
            ``meta.json`` lists no files.
    """
    meta_path = model_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{meta_path} missing: the forecaster is not fitted (make dmf-forecasters)"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    listed: dict[str, str] = dict(meta.get("files") or {})
    if not listed:
        raise ValueError(f"{meta_path} lists no files")
    files: dict[str, str] = {}
    for name in sorted(listed):
        digest = _sha256(model_dir / name)
        if digest != listed[name]:
            raise ValueError(f"{model_dir / name}: SHA-256 {digest} != meta.json {listed[name]}")
        files[name] = digest
    return {
        "dir": model_dir.name,
        "meta_json_sha256": _sha256(meta_path),
        "smoke": bool(meta.get("smoke", False)),
        "files": files,
    }


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
        FileNotFoundError: If the controller names a forecaster that is not fitted.
        ValueError: If a forecaster file does not match its ``meta.json`` digest.
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
    forecaster_dir = getattr(config, "forecaster_dir", None)
    if forecaster_dir is not None:
        document["forecaster"] = forecaster_provenance(Path(forecaster_dir))
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


def _files_column(name: str, spec: ControlSpec | None, config_path: Path | None) -> str:
    """Return ``forecaster_files_sha256``: ``"file=sha256|..."`` (sorted), or ``""``."""
    document = json.loads(resolved_config_json(name, spec, config_path))
    forecaster = document.get("forecaster")
    if forecaster is None:
        return ""
    return "|".join(f"{file}={digest}" for file, digest in forecaster["files"].items())


def controller_provenance_columns(
    names: list[str],
    spec: ControlSpec | None = None,
    config_paths: dict[str, Path] | None = None,
) -> dict[str, dict[str, str]]:
    """Return the per-method provenance columns of ``summary.csv`` for registry controllers.

    Args:
        names: Registry names, in output order.
        spec: Environment facts; the committed configs when omitted.
        config_paths: YAML overrides by name (scratch smoke runs only); the committed
            registry path otherwise. The YAML hash is of the file actually flown.

    Returns:
        ``{name: {"controller_config_sha256": <YAML bytes>, "resolved_gains_sha256":
        <resolved config>, "forecaster_files_sha256": <"file=sha256|...", or "" without a
        forecaster>}}`` (:data:`PROVENANCE_COLUMNS`).
    """
    overrides = config_paths or {}
    out: dict[str, dict[str, str]] = {}
    for name in names:
        path = overrides.get(name)
        out[name] = {
            "controller_config_sha256": _sha256(
                REGISTRY[name].config_path if path is None else path
            ),
            "resolved_gains_sha256": resolved_config_sha256(name, spec, path),
            "forecaster_files_sha256": _files_column(name, spec, path),
        }
    return out
