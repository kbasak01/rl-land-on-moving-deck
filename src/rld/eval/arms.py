"""The Phase 7 evaluation arms, as data: what each arm flies, where, and under which condition.

Authority: ``docs/protocol.md`` P7-D1 (committed alone at ``27934bf`` before any Phase 7
flight), on top of P3-D1 / P3-D4. Nothing here flies anything; :mod:`rld.eval.phase7` flies
the conditions, :mod:`rld.eval.hypotheses` scores them and :mod:`rld.eval.results_md` renders
them. Keeping the definitions in one import-light module lets all three read the same ones.

Arms and conditions
-------------------
=============  ===========================================================================
arm            what it flies (P7-D1 item)
=============  ===========================================================================
``matrix``     the full frozen matrix, aft pad, JONSWAP: every learned method x seed on all
               five lists (forecast runs skip the static list, reason recorded); the six
               baselines are **carried** line for line from ``results/e01`` and
               ``results/e01_lowvz_cut``
``cg``         pad at CG (§5): every learned method x seed and ``pid_feedforward_lowvz_cut``
               on the four non-static lists; the other five baselines' CG rows are carried
               from ``results/e02``
``sinusoid``   the H4 test leg (§1): every method (learned and the six baselines) on ``id``,
               aft pad, on the training builder's matched sinusoid
``noise/<c>``  the perception stand-in (§4): 11 non-clean (sigma_p, latency) conditions, every
               method, ``id``, aft; the clean condition is the matrix's rows
``lambda/<c>`` lambda in {1/15, 1/40} (§3): ``ppo`` x 5 seeds, ``pid_feedforward`` and the
               always-printed ``pid_track_descend`` and ``oracle_gated``, the four non-static
               lists, aft; ``t0`` re-drawn by the environment (start check ``"lambda"``)
``mss``        the optional MSS transfer arm (§6): every method, aft and CG, on
               ``results/episodes_mss/`` (``mss_transfer`` -> motion ``mss``,
               ``mss_transfer_corpus`` -> ``mss_corpus``)
=============  ===========================================================================

Units: sigmas metres and metres per second model scale; latency milliseconds model scale
(one control step = 33.3 ms model = 167 ms full scale at lambda = 1/25); lambda
dimensionless.
"""

from dataclasses import dataclass
from pathlib import Path

from rld.config import REPO_ROOT

__all__ = [
    "ALL_LISTS",
    "ALWAYS_PRINTED",
    "ARMS",
    "BASELINES",
    "E07_DIR",
    "EPISODES_MSS_DIR",
    "LAMBDA_INVERSES",
    "LEARNED_METHODS",
    "LEARNED_SEEDS",
    "MSS_REGIME_MOTION",
    "NOISE_GRID",
    "NONSTATIC_LISTS",
    "PROJECT_LAM_INVERSE",
    "RESULTS_DIR",
    "RUNS_ROOT",
    "TAU_V_MODEL_S",
    "Condition",
    "NoiseSetting",
    "conditions",
    "run_dir",
]

RESULTS_DIR: Path = REPO_ROOT / "results"

#: Phase 7 output root.
E07_DIR: Path = RESULTS_DIR / "e07"

#: The MSS transfer lists and their own manifest (P7-D1 §6; never the frozen MANIFEST).
EPISODES_MSS_DIR: Path = RESULTS_DIR / "episodes_mss"

#: Where the training runs live (gitignored); ``<method>/<seed>/final/`` is evaluated.
RUNS_ROOT: Path = REPO_ROOT / "artifacts" / "runs"

#: Every learned method, in table order (Phase 5 then Phase 6).
LEARNED_METHODS: tuple[str, ...] = (
    "ppo",
    "sac",
    "residual_ppo",
    "ppo_forecast",
    "residual_ppo_forecast",
    "ppo_sinusoid",
)

#: Training seeds of every learned method (P3-D1 §6; none is ever dropped).
LEARNED_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: The six classical baselines printed beside the learned methods, registry order.
BASELINES: tuple[str, ...] = (
    "pid_track_descend",
    "pid_feedforward",
    "pid_feedforward_lowvz",
    "pid_feedforward_lowvz_cut",
    "gated",
    "oracle_gated",
)

#: The baselines CLAUDE.md requires in every table.
ALWAYS_PRINTED: tuple[str, ...] = ("pid_track_descend", "pid_feedforward", "oracle_gated")

#: The frozen lists without a lambda- or pad-independent fixture.
NONSTATIC_LISTS: tuple[str, ...] = ("id", "unseen_seastate", "unseen_heading", "unseen_vessel")

#: Every frozen list, manifest order.
ALL_LISTS: tuple[str, ...] = (*NONSTATIC_LISTS, "static")

#: The project Froude scale, as ``1 / lambda`` (P1-D1); the main matrix.
PROJECT_LAM_INVERSE: float = 25.0

#: The lambda sensitivity points (P7-D1 §3), as ``1 / lambda``.
LAMBDA_INVERSES: tuple[float, ...] = (15.0, 40.0)

#: The velocity-noise time constant of P7-D1 §4: sigma_v = sigma_p / tau, seconds model
#: scale (1 s full scale at lambda = 1/25).
TAU_V_MODEL_S: float = 0.2

#: MSS list regime -> the motion kind it is flown under (P7-D1 §6).
MSS_REGIME_MOTION: dict[str, str] = {"mss_transfer": "mss", "mss_transfer_corpus": "mss_corpus"}


@dataclass(frozen=True)
class NoiseSetting:
    """One perception stand-in condition (P7-D1 §4).

    Attributes:
        sigma_p_m: Noise sigma on the perceived pad position, metres model scale (P7-D4).
        sigma_v_m_s: Noise sigma on the perceived pad velocity, metres per second model scale
            (sigma_p / tau; P7-D4).
        latency_ms: Configured latency, milliseconds model scale. 33.4 and 66.7 quantise
            **down** to exactly 1 and 2 control steps at 30 Hz (a literal 33 ms would be 0).
        latency_steps: The control steps that latency is (asserted by a test).
    """

    sigma_p_m: float
    sigma_v_m_s: float
    latency_ms: float
    latency_steps: int

    @property
    def name(self) -> str:
        """Directory name, e.g. ``sigma2cm_lat1step``."""
        return f"sigma{round(self.sigma_p_m * 100)}cm_lat{self.latency_steps}step"

    @property
    def clean(self) -> bool:
        """Whether this is the (0, 0) condition, which is the main matrix's rows."""
        return self.sigma_p_m == 0.0 and self.latency_steps == 0


#: Pad-position sigmas (m) and the matching pad-velocity sigmas (m/s), sigma_v = sigma_p / 0.2 s
#: (P7-D1 §4 grid; applied to the perceived deck sample, P7-D4).
_SIGMAS: tuple[tuple[float, float], ...] = ((0.0, 0.0), (0.01, 0.05), (0.02, 0.10), (0.04, 0.20))

#: Configured latencies (ms) and the control steps they quantise to.
_LATENCIES: tuple[tuple[float, int], ...] = ((0.0, 0), (33.4, 1), (66.7, 2))

#: The full 4 x 3 grid, clean condition first; :func:`conditions` flies the 11 others.
NOISE_GRID: tuple[NoiseSetting, ...] = tuple(
    NoiseSetting(sp, sv, lat, steps) for lat, steps in _LATENCIES for sp, sv in _SIGMAS
)


@dataclass(frozen=True)
class Condition:
    """One flown condition of one arm, written to its own directory.

    Attributes:
        arm: Arm name (:data:`ARMS`).
        name: Condition name inside a multi-condition arm, ``""`` otherwise.
        lists: List names flown (files ``<name>.parquet`` in ``episodes_dir``).
        pads: Pad overrides, in output order (``None`` = the listed aft pad).
        motion: ``"jonswap"``, ``"sinusoid"``, or ``"mss"`` (the MSS arm: per list regime,
            :data:`MSS_REGIME_MOTION`).
        learned: Learned methods flown, every seed in :data:`LEARNED_SEEDS`.
        baselines: Baselines flown here (the rest are carried or not part of the arm).
        carry: ``((source dir name under results/, methods, pad), ...)`` carried line for line.
        noise: The perception condition, or ``None`` (noise disabled, as committed).
        lam_inverse: ``1 / lambda`` flown (25 unless the lambda arm).
        start_check: ``"strict"`` or ``"lambda"``.
        episodes_dir: Where the lists and their manifest live.
        optional: An arm whose absence is recorded rather than failed (MSS).
    """

    arm: str
    name: str
    lists: tuple[str, ...]
    pads: tuple[str | None, ...]
    motion: str
    learned: tuple[str, ...]
    baselines: tuple[str, ...]
    carry: tuple[tuple[str, tuple[str, ...], str], ...]
    noise: NoiseSetting | None
    lam_inverse: float
    start_check: str
    episodes_dir: Path
    optional: bool = False

    @property
    def key(self) -> str:
        """``arm`` or ``arm/name``: the condition's path below the output root."""
        return f"{self.arm}/{self.name}" if self.name else self.arm

    def out_dir(self, root: Path = E07_DIR) -> Path:
        """Return the condition's output directory below ``root``."""
        return root / self.key

    def motion_for_regime(self, regime: str) -> str:
        """Return the motion kind a list regime is flown under in this condition."""
        if self.motion == "mss":
            return MSS_REGIME_MOTION[regime]
        return self.motion


#: Arm names, in the order ``--arm all`` flies them.
ARMS: tuple[str, ...] = ("matrix", "cg", "sinusoid", "lambda", "noise", "mss")

_E01_CARRY: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "e01",
        ("pid_track_descend", "pid_feedforward", "pid_feedforward_lowvz", "gated", "oracle_gated"),
        "aft",
    ),
    ("e01_lowvz_cut", ("pid_feedforward_lowvz_cut",), "aft"),
)

_E02_CG_CARRY: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "e02",
        ("pid_track_descend", "pid_feedforward", "pid_feedforward_lowvz", "gated", "oracle_gated"),
        "cg",
    ),
)

_EPISODES_DIR: Path = RESULTS_DIR / "episodes"


def _mss_lists(episodes_dir: Path) -> tuple[str, ...]:
    """Return the MSS list names in their manifest's order, or the P7-D1 names if absent."""
    manifest = episodes_dir / "MANIFEST.csv"
    if not manifest.is_file():
        return tuple(MSS_REGIME_MOTION)
    import csv

    with manifest.open(newline="", encoding="utf-8") as handle:
        return tuple(row["file"].removesuffix(".parquet") for row in csv.DictReader(handle))


def _jonswap(
    arm: str,
    lists: tuple[str, ...],
    pads: tuple[str | None, ...],
    learned: tuple[str, ...],
    baselines: tuple[str, ...],
    carry: tuple[tuple[str, tuple[str, ...], str], ...],
) -> Condition:
    """Return a clean, project-lambda, JONSWAP condition on the frozen lists."""
    return Condition(
        arm=arm,
        name="",
        lists=lists,
        pads=pads,
        motion="jonswap",
        learned=learned,
        baselines=baselines,
        carry=carry,
        noise=None,
        lam_inverse=PROJECT_LAM_INVERSE,
        start_check="strict",
        episodes_dir=_EPISODES_DIR,
    )


def conditions(arm: str, episodes_mss_dir: Path = EPISODES_MSS_DIR) -> list[Condition]:
    """Return every condition of ``arm``, in flight order.

    Args:
        arm: One of :data:`ARMS`.
        episodes_mss_dir: The MSS lists' directory.

    Returns:
        The conditions.

    Raises:
        ValueError: On an unknown arm.
    """
    if arm == "matrix":
        return [_jonswap("matrix", ALL_LISTS, (None,), LEARNED_METHODS, (), _E01_CARRY)]
    if arm == "cg":
        return [
            _jonswap(
                "cg",
                NONSTATIC_LISTS,
                ("cg",),
                LEARNED_METHODS,
                ("pid_feedforward_lowvz_cut",),
                _E02_CG_CARRY,
            )
        ]
    if arm == "sinusoid":
        return [
            Condition(
                arm="sinusoid",
                name="",
                lists=("id",),
                pads=(None,),
                motion="sinusoid",
                learned=LEARNED_METHODS,
                baselines=BASELINES,
                carry=(),
                noise=None,
                lam_inverse=PROJECT_LAM_INVERSE,
                start_check="strict",
                episodes_dir=_EPISODES_DIR,
            )
        ]
    if arm == "noise":
        return [
            Condition(
                arm="noise",
                name=setting.name,
                lists=("id",),
                pads=(None,),
                motion="jonswap",
                learned=LEARNED_METHODS,
                baselines=BASELINES,
                carry=(),
                noise=setting,
                lam_inverse=PROJECT_LAM_INVERSE,
                start_check="strict",
                episodes_dir=_EPISODES_DIR,
            )
            for setting in NOISE_GRID
            if not setting.clean
        ]
    if arm == "lambda":
        return [
            Condition(
                arm="lambda",
                name=f"lam{round(lam)}",
                lists=NONSTATIC_LISTS,
                pads=(None,),
                motion="jonswap",
                learned=("ppo",),
                baselines=("pid_track_descend", "pid_feedforward", "oracle_gated"),
                carry=(),
                noise=None,
                lam_inverse=lam,
                start_check="lambda",
                episodes_dir=_EPISODES_DIR,
            )
            for lam in LAMBDA_INVERSES
        ]
    if arm == "mss":
        return [
            Condition(
                arm="mss",
                name="",
                lists=_mss_lists(episodes_mss_dir),
                pads=(None, "cg"),
                motion="mss",
                learned=LEARNED_METHODS,
                baselines=BASELINES,
                carry=(),
                noise=None,
                lam_inverse=PROJECT_LAM_INVERSE,
                start_check="strict",
                episodes_dir=episodes_mss_dir,
                optional=True,
            )
        ]
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def run_dir(method: str, seed: int, root: Path = RUNS_ROOT) -> Path:
    """Return the training run directory of ``method`` at ``seed``."""
    return root / method / str(seed)
