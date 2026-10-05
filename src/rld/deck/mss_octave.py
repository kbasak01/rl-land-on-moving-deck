"""Octave parity check for the MSS arm, run entirely outside ``third_party/``.

dmf's ``scripts/mss_octave_check.py`` cannot be pointed elsewhere. It drives Octave in
``Path("mss").resolve()`` (relative to the working directory), and its m-files
(``mss/run_case.m``, ``mss/check_patch.m``) find the toolbox at
``fileparts(mfilename('fullpath'))/upstream``, i.e. inside dmf's tree. This module
reproduces that script's logic through the ``dmf.mss`` library functions instead:

* **Staging.** The three m-files (``run_case.m``, ``check_patch.m``,
  ``waveMotionRAO_seeded.m``) are copied **byte for byte** from dmf's ``mss/`` into the
  directory that holds the clone (``artifacts/mss/``, with the clone at
  ``artifacts/mss/upstream``). Their own ``here/upstream`` then resolves to our clone, and
  no m-file is edited. The copies are SHA-256-checked against dmf's originals.
* **Step 1, patch equivalence.** On the first parity cell, stock ``waveMotionRAO.m`` and
  dmf's patched ``waveMotionRAO_seeded.m`` are run. The patched copy is fed the phases stock
  draws itself. They must agree exactly (``max_abs_diff == 0``).
* **Step 2, NumPy vs Octave.** dmf's cells, step count and tolerance are used:
  ``(180, 0), (180, 12), (135, 6), (135, 12)`` deg/kn, grid kind ``mss``, seed 0, 300
  steps at 10 Hz full scale from ``t_start_s``, tolerance 1 % relative to the Octave
  series' peak. ``synthesize_mss_motion`` is compared with ``run_case.m`` on 6 DOFs x
  (``eta``, ``eta_dot``), in MSS units and signs. Same row schema as dmf's
  ``results/mss/octave_parity.csv``.
* **Step 3, this project's source vs Octave (added here).** :class:`rld.deck.mss.MssDeckMotion`
  is compared with the same Octave run, after converting Octave to corpus units and signs by
  hand (roll and pitch ``rad -> deg``; heave negated). This covers the six forecast
  channels. It is the end-to-end check of our wave grid, time map and sign convention
  against MSS's own m-file.

Units: Octave returns metres and **radians** (SNAME, z down); step 3 compares degrees,
degrees per second, metres and metres per second, full scale. Times are seconds full scale.
Everything is simulation.
"""

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import scipy.io as sio
from dmf.mss.export import MSSRealizationSpec, _build_grid, realization_seed
from dmf.mss.synth import synthesize_mss_motion
from dmf.mss.vessel import DOF_NAMES

from rld.deck.config import PAD_CONFIG, load_pads
from rld.deck.mss import (
    KNOT_M_S,
    MSS_ROOT,
    MssDeckMotion,
    load_mss_vessel_cached,
    resolve_mat_path,
)
from rld.deck.scaling import FroudeScale

__all__ = [
    "DMF_MSS_DIR",
    "OCTAVE_FILES",
    "PARITY_CELLS",
    "OctaveUnavailableError",
    "run_parity",
    "stage_octave_files",
]

#: dmf's m-file directory, read only.
DMF_MSS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[3] / "third_party" / "deck-motion-forecast" / "mss"
)

#: The m-files staged beside the clone, verbatim.
OCTAVE_FILES: Final[tuple[str, ...]] = ("run_case.m", "check_patch.m", "waveMotionRAO_seeded.m")

#: dmf's ``scripts/mss_octave_check.py::PARITY_CELLS``: (heading deg, speed kn).
PARITY_CELLS: Final[tuple[tuple[float, float], ...]] = (
    (180.0, 0.0),
    (180.0, 12.0),
    (135.0, 6.0),
    (135.0, 12.0),
)

#: Corpus channel -> (Octave DOF index 1..6, uses eta_dot, sign, rad->deg).
_RLD_CHANNELS: Final[dict[str, tuple[int, bool, float, bool]]] = {
    "roll": (4, False, +1.0, True),
    "pitch": (5, False, +1.0, True),
    "heave": (3, False, -1.0, False),
    "roll_rate": (4, True, +1.0, True),
    "pitch_rate": (5, True, +1.0, True),
    "heave_rate": (3, True, -1.0, False),
}

#: ``MotionChannels`` field for each corpus channel name.
_FIELDS: Final[dict[str, str]] = {
    "roll": "roll_deg",
    "pitch": "pitch_deg",
    "heave": "heave_m",
    "roll_rate": "roll_rate_dps",
    "pitch_rate": "pitch_rate_dps",
    "heave_rate": "heave_rate_m_s",
}


class OctaveUnavailableError(RuntimeError):
    """Octave is not installed, so the parity check cannot run (reported as SKIPPED)."""


@dataclass(frozen=True)
class _Deviation:
    """One compared series: relative max deviation and RMS ratio, dimensionless."""

    rel: float
    rms_ratio: float


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage_octave_files(
    stage_dir: Path = MSS_ROOT, source_dir: Path = DMF_MSS_DIR
) -> dict[str, str]:
    """Copy dmf's m-files beside the clone, byte for byte, and verify the copies.

    Args:
        stage_dir: Directory holding the MSS clone as ``upstream/``.
        source_dir: dmf's ``mss/`` directory (read only).

    Returns:
        ``{file name: sha256}`` of the staged files, equal to dmf's originals.

    Raises:
        FileNotFoundError: If the clone or an m-file is missing.
        RuntimeError: If a staged copy differs from its original.
    """
    if not (stage_dir / "upstream" / "LIBRARY").is_dir():
        raise FileNotFoundError(f"no MSS clone at {stage_dir / 'upstream'}")
    digests: dict[str, str] = {}
    for name in OCTAVE_FILES:
        src, dst = source_dir / name, stage_dir / name
        shutil.copyfile(src, dst)
        want, got = _sha256(src), _sha256(dst)
        if want != got:
            raise RuntimeError(f"staged {dst} differs from {src}")
        digests[name] = got
    return digests


def _run_octave(octave: str, stage_dir: Path, func: str, case: Path, out: Path) -> None:
    """Run one staged m-file function, as dmf's ``_run_octave`` does."""
    cmd = [octave, "--no-gui", "--quiet", "--eval", f"cd('{stage_dir}'); {func}('{case}','{out}')"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"octave {func} failed:\n{proc.stdout}\n{proc.stderr}")


def _deviation(ours: np.ndarray[Any, Any], theirs: np.ndarray[Any, Any]) -> _Deviation:
    """Return dmf's metric, max |ours - theirs| / max |theirs|, and the RMS ratio."""
    scale = max(float(np.max(np.abs(theirs))), 1e-30)
    rel = float(np.max(np.abs(ours - theirs))) / scale
    ratio = float(np.sqrt(np.mean(ours**2)) / max(float(np.sqrt(np.mean(theirs**2))), 1e-30))
    return _Deviation(rel=rel, rms_ratio=ratio)


def run_parity(
    cfg: dict[str, Any],
    stage_dir: Path = MSS_ROOT,
    *,
    n_steps: int = 300,
    tolerance: float = 0.01,
    octave: str = "octave",
) -> pd.DataFrame:
    """Run the three-step parity check and return one row per compared series.

    Args:
        cfg: The arm's config (``configs/deck/mss_s175_ss5.yaml``).
        stage_dir: Directory with the clone and the staged m-files.
        n_steps: Samples per cell at ``record.fs_hz`` (10 Hz full scale).
        tolerance: Maximum relative pointwise deviation, dimensionless.
        octave: Octave executable.

    Returns:
        Columns ``check, heading_deg, speed_kn, dof, rel_deviation, rms_ratio, passed``.
        ``check`` is ``patch_equivalence``, ``numpy_vs_octave_{eta,eta_dot}`` (dmf's
        rows) or ``rld_vs_octave`` (step 3, ``dof`` = corpus channel name).

    Raises:
        OctaveUnavailableError: If ``octave`` is not on PATH.
        RuntimeError: If an Octave run fails.
    """
    exe = shutil.which(octave)
    if exe is None:
        raise OctaveUnavailableError(f"{octave!r} not found on PATH")
    stage_octave_files(stage_dir)
    vessel = load_mss_vessel_cached(resolve_mat_path(cfg))
    speed_index = int(cfg["vessel"]["speed_index"])
    fs = float(cfg["record"]["fs_hz"])
    t0 = float(cfg["record"]["t_start_s"])
    pads = load_pads(PAD_CONFIG)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i_cell, (heading, speed_kn) in enumerate(PARITY_CELLS):
            spec = MSSRealizationSpec("mss", heading, speed_kn, 0)
            grid = _build_grid(spec, cfg, realization_seed(spec))
            t = t0 + np.arange(n_steps) / fs
            case = tmpdir / f"case_{heading:.0f}_{speed_kn:.0f}.mat"
            sio.savemat(
                str(case),
                {
                    "Omega": grid.w_rad_s,
                    "Amp": grid.amplitude_m,
                    "phases": grid.phase_rad,
                    "t": t,
                    "U": speed_kn * KNOT_M_S,
                    "beta_wave": np.deg2rad(heading),
                },
            )
            if i_cell == 0:
                verdict = tmpdir / "patch_check.csv"
                _run_octave(exe, stage_dir, "check_patch", case, verdict)
                pc = pd.read_csv(verdict)
                rows.append(
                    {
                        "check": "patch_equivalence",
                        "heading_deg": heading,
                        "speed_kn": speed_kn,
                        "dof": "all",
                        "rel_deviation": float(pc["rel_diff"].iloc[0]),
                        "rms_ratio": float("nan"),
                        "passed": bool(pc["max_abs_diff"].iloc[0] == 0.0),
                    }
                )
            out = tmpdir / f"out_{heading:.0f}_{speed_kn:.0f}.csv"
            _run_octave(exe, stage_dir, "run_case", case, out)
            oct_df = pd.read_csv(out)
            motion = synthesize_mss_motion(
                vessel, grid, heading, speed_kn * KNOT_M_S, t, speed_index=speed_index
            )
            for i, dof in enumerate(DOF_NAMES):
                for label, ours, theirs in (
                    ("eta", motion.eta[i], oct_df[f"eta{i + 1}"].to_numpy()),
                    ("eta_dot", motion.eta_dot[i], oct_df[f"etadot{i + 1}"].to_numpy()),
                ):
                    dev = _deviation(ours, theirs)
                    rows.append(
                        {
                            "check": f"numpy_vs_octave_{label}",
                            "heading_deg": heading,
                            "speed_kn": speed_kn,
                            "dof": dof,
                            "rel_deviation": dev.rel,
                            "rms_ratio": dev.rms_ratio,
                            "passed": dev.rel <= tolerance,
                        }
                    )
            # Step 3: our source, in corpus units and signs, against the same Octave run.
            # lam = 1 so that model time is full time minus t_start_s.
            source = MssDeckMotion(spec, cfg, FroudeScale(lam=1.0), pads, vessel=vessel)
            ch = source.channels(t)
            for name, (dof_1, rate, sign, to_deg) in _RLD_CHANNELS.items():
                col = f"etadot{dof_1}" if rate else f"eta{dof_1}"
                theirs = sign * oct_df[col].to_numpy(dtype=np.float64)
                if to_deg:
                    theirs = np.rad2deg(theirs)
                dev = _deviation(np.asarray(getattr(ch, _FIELDS[name])), theirs)
                rows.append(
                    {
                        "check": "rld_vs_octave",
                        "heading_deg": heading,
                        "speed_kn": speed_kn,
                        "dof": name,
                        "rel_deviation": dev.rel,
                        "rms_ratio": dev.rms_ratio,
                        "passed": dev.rel <= tolerance,
                    }
                )
    return pd.DataFrame(rows)
