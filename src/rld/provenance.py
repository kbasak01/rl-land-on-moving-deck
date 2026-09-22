"""Environment provenance: the conditions a committed number is only meaningful alongside.

Every CSV this project commits carries this block as its trailing columns, so that a row can
be read years later without guessing which numpy, which submodule commit or which thread
count produced it. Phase 0 wrote it beside the throughput measurement in
:mod:`rld.bench.throughput`; it lives here from Phase 1 on because Phases 1, 3, 5, 7 and 8
all need it and a deck script importing a helper out of a ``bench`` module is the wrong
layering. :mod:`rld.bench.throughput` re-exports it, so Phase 0's
``results/env_throughput.csv`` column set is unchanged.

``extra`` is inserted between the submodule SHAs and the host block, which is exactly where
the throughput rows' ``drone_model`` / ``pyb_freq_hz`` / ``ctrl_freq_hz`` columns sat, so
column *order* is preserved too, not merely the column set.

All values are strings; units, where a value has any, are named in the key
(``*_hz``, ``*_s``).
"""

import importlib.metadata as md
import os
import platform
import socket
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["environment_provenance"]


def _version(dist: str) -> str:
    """Return an installed distribution's version, or ``"unknown"``.

    Args:
        dist: Distribution name as it appears on PyPI.

    Returns:
        The version string.
    """
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "unknown"


def _submodule_sha(path: Path) -> str:
    """Return the checked-out commit of a git submodule, or ``"unknown"``.

    Args:
        path: Path to the submodule working tree.

    Returns:
        The full 40-character SHA.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def environment_provenance(
    repo_root: Path, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Collect the environment block that every committed CSV row carries.

    ``OMP_NUM_THREADS`` is recorded because it is set to 1 by ``.claude/settings.json``:
    with 16 single-threaded workers on a 36-thread host the workers do not contend for
    BLAS threads, and the same measurement on a host that leaves it unset would be slower,
    not faster.

    Args:
        repo_root: Repository root, used to locate the two submodules.
        extra: Measurement-specific fields, inserted after the submodule SHAs and before
            the host block. Values must already be strings.

    Returns:
        A mapping of provenance field to string value, all of which become CSV columns, in
        a stable order.
    """
    return {
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "torch": _version("torch"),
        "sb3": _version("stable-baselines3"),
        "gymnasium": _version("gymnasium"),
        "pybullet": _version("pybullet"),
        "dmf_sha": _submodule_sha(repo_root / "third_party" / "deck-motion-forecast"),
        "gpd_sha": _submodule_sha(repo_root / "third_party" / "gym-pybullet-drones"),
        **dict(extra or {}),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
        "cpu_count": str(os.cpu_count()),
        "host": socket.gethostname(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
