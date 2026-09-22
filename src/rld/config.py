"""YAML loading primitive and the repository's config paths.

Every config family in this project is a YAML file next to its siblings under
``configs/`` and a frozen dataclass next to the package that consumes it
(``rld.deck.config`` for Phase 1, ``rld.envs.config`` for Phase 2, and so on). This module
holds only the two things all of them share: the loader and the directory constants.

The loader mirrors ``dmf.config.load_yaml`` deliberately -- exists-check, mapping-check,
named missing keys -- so that a config error in this project reads the same way as one in
``deck-motion-forecast``.

No unit conversion happens here. Units are whatever the individual schema documents, and
every schema names model-scale versus full-scale for each time and length quantity.
"""

from pathlib import Path
from typing import Any

import yaml

__all__ = ["CONFIG_DIR", "REPO_ROOT", "load_yaml", "require"]

#: Repository root, resolved from this file so nothing depends on the working directory.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Root of the committed YAML configs, ``<repo>/configs``.
CONFIG_DIR: Path = REPO_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML config file into its top-level mapping.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed top-level mapping, keys coerced to ``str``. Scalar units are whatever
        the individual config schema documents; this function performs no unit conversion
        and no Froude scaling.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the document's top level is not a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level of a config file must be a mapping")
    return {str(key): value for key, value in raw.items()}


def require(raw: dict[str, Any], key: str, path: Path) -> Any:
    """Return one required config value, naming the file and the key when it is missing.

    Args:
        raw: The mapping returned by :func:`load_yaml`.
        key: Required key.
        path: The file ``raw`` came from, used only in the error message.

    Returns:
        The raw value, unconverted.

    Raises:
        ValueError: If ``key`` is absent.
    """
    if key not in raw:
        raise ValueError(f"{path}: missing required key {key!r}")
    return raw[key]
