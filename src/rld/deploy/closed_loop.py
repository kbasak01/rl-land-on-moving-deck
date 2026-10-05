"""Closed-loop parity: PyTorch vs ONNX on 50 frozen episodes (P8-D1 §7).

Episodes: the frozen ``id`` list, aft pad as listed, JONSWAP, noise off, lambda = 1/25 -- the
rows with listed ``index`` 0-12 of SS3 and SS4 and 0-11 of SS5 and SS6 (13 + 13 + 12 + 12 = 50),
in committed order (:func:`closed_loop_episodes`).

Each exported (method, seed) flies them twice through the unchanged :mod:`rld.eval.runner`:

* **PyTorch** -- :func:`rld.eval.learned.learned_spec`, exactly the e05/e06 spec;
* **ONNX** -- :func:`rld.deploy.onnx_policy.build_onnx_policy`: the same policy with only
  ``network_action`` replaced by the graph (ORT CPU, 1 thread).

Pass: every one of the 50 episodes has the same outcome class. Also reported: max |delta
touchdown ``rel_vz_normal_m_s``| (metres per second model scale), max |delta steps| (control
steps), how many full rows are byte-identical, and -- as a consistency check of the PyTorch path
-- how many PyTorch rows are byte-identical to the committed e05/e06 rows of the same episodes.

Post-hoc controls (added after the first closed-loop result was read; P8-D2)
--------------------------------------------------------------------------
Two more arms fly the same episodes, to say whether an outcome difference is an export defect or
the closed loop's own sensitivity to float32 rounding. Neither changes the P8-D1 §7 verdict.

* ``torch_folded`` -- the policy with ``network_action`` replaced by the folded torch module
  (:class:`rld.deploy.export.FoldedActor`): the graph's arithmetic, executed by PyTorch;
* ``torch_obs_plus_1ulp`` -- the unchanged SB3 path with every raw input entry nudged up by one
  float32 ulp (``np.nextafter``) before normalisation, with no ONNX involved. Its size was **not**
  matched to the export error: measured afterwards on the recorded inputs (P8-D6,
  ``posthoc_same_input.csv``), a one-ulp input perturbation moves the action several times more
  than the ONNX-vs-PyTorch difference does.

``closed_loop_controls.csv`` puts the ONNX arm and both controls side by side, each against the
PyTorch arm.

Units: speeds metres per second model scale; steps are control steps at 30 Hz model scale.
"""

import csv
import functools
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
from dmf.typedefs import FloatArray

from rld.config import REPO_ROOT
from rld.deploy.export import FoldedActor, folded_actor
from rld.deploy.onnx_policy import build_onnx_policy
from rld.eval.envs import EvalConfigs
from rld.eval.episodes import EPISODES_DIR, ListedEpisode, read_list
from rld.eval.runner import EvalArm, PolicySpec, callable_spec, episode_rows_csv, run_arms
from rld.rl.policy import LearnedPolicy

__all__ = [
    "ARMS",
    "CLOSED_LOOP_COLUMNS",
    "CONTROL_COLUMNS",
    "CLOSED_LOOP_EPISODE_COLUMNS",
    "CLOSED_LOOP_PER_SS",
    "COMMITTED_EPISODES",
    "ClosedLoopPair",
    "closed_loop_episodes",
    "build_control_policy",
    "closed_loop_tables",
    "controls_table",
    "committed_lines",
    "run_closed_loop",
]

#: Episodes per sea state of the ``id`` list (P8-D1 §7): the lowest listed indices.
CLOSED_LOOP_PER_SS: dict[str, int] = {"SS3": 13, "SS4": 13, "SS5": 12, "SS6": 12}

#: The committed episode rows the PyTorch path is compared with.
COMMITTED_EPISODES: dict[str, Path] = {
    "ppo": REPO_ROOT / "results" / "e05" / "episodes.csv",
    "residual_ppo_forecast": REPO_ROOT / "results" / "e06" / "episodes.csv",
}

#: Columns of ``closed_loop_episodes.csv``.
CLOSED_LOOP_EPISODE_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "regime",
    "ss",
    "index",
    "episode_seed",
    "outcome_torch",
    "outcome_onnx",
    "steps_torch",
    "steps_onnx",
    "rel_vz_normal_m_s_torch",
    "rel_vz_normal_m_s_onnx",
    "outcome_identical",
    "row_identical",
    "torch_row_matches_committed",
)

#: Columns of ``closed_loop_parity.csv``.
CLOSED_LOOP_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "n_episodes",
    "n_outcome_identical",
    "passed",
    "max_abs_d_rel_vz_normal_m_s",
    "max_abs_d_steps",
    "n_rows_identical",
    "n_torch_rows_match_committed",
    "committed_source",
    "outcomes_torch",
    "outcomes_onnx",
)


def closed_loop_episodes(episodes_dir: Path = EPISODES_DIR) -> list[ListedEpisode]:
    """Return the 50 closed-loop episodes, in committed order.

    Args:
        episodes_dir: Where the frozen lists live.

    Returns:
        The rows of ``id.parquet`` with ``index < CLOSED_LOOP_PER_SS[ss]``.

    Raises:
        ValueError: If the selection is not 50 aft episodes with the expected indices.
    """
    rows = read_list(episodes_dir / "id.parquet")
    picked = [r for r in rows if r.index < CLOSED_LOOP_PER_SS.get(r.ss, 0)]
    for ss, k in CLOSED_LOOP_PER_SS.items():
        got = sorted(r.index for r in picked if r.ss == ss)
        if got != list(range(k)):
            raise ValueError(f"id {ss}: indices {got}, expected 0..{k - 1}")
    if len(picked) != 50 or any(r.pad != "aft" or r.regime != "id" for r in picked):
        raise ValueError("the closed-loop selection must be 50 aft episodes of the id list")
    return picked


#: Arms flown per (method, seed), in output order: the two P8-D1 §7 arms, then the post-hoc
#: controls.
ARMS: tuple[str, ...] = ("torch", "onnx", "torch_folded", "torch_obs_plus_1ulp")

ControlKind = Literal["torch_folded", "torch_obs_plus_1ulp"]


class _Host(Protocol):
    """What the control mixins read from their host policy."""

    def policy_input(self, obs: FloatArray) -> FloatArray:
        """Return the raw network input."""
        ...


class FoldedTorchNetworkMixin:
    """``network_action`` = the folded torch module on the raw input (control arm)."""

    folded: FoldedActor

    def network_action(self: "_Host", obs: FloatArray) -> FloatArray:
        """Return the folded module's action, ``(3,)`` float32 in ``[-1, 1]``."""
        import torch

        x = np.ascontiguousarray(self.policy_input(obs).reshape(1, -1), dtype=np.float32)
        with torch.no_grad():
            out = cast(Any, self).folded(torch.from_numpy(x)).numpy().reshape(-1)
        action: FloatArray = np.clip(out, -1.0, 1.0).astype(np.float32)
        return action


class UlpNetworkMixin:
    """``network_action`` = the SB3 path on the raw input nudged up one float32 ulp (control)."""

    def network_action(self: "_Host", obs: FloatArray) -> FloatArray:
        """Return the SB3 action on ``nextafter(x, +inf)``, ``(3,)`` float32 in ``[-1, 1]``."""
        host = cast(Any, self)
        x = np.asarray(self.policy_input(obs), dtype=np.float32).reshape(1, -1)
        x = np.nextafter(x, np.float32(np.inf)).astype(np.float32)
        z = np.asarray(host.normalizer.normalize_obs(x), dtype=np.float32)
        action, _ = host.model.predict(z, deterministic=True)
        out: FloatArray = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        return out


_CONTROL_MIXINS: dict[str, type] = {
    "torch_folded": FoldedTorchNetworkMixin,
    "torch_obs_plus_1ulp": UlpNetworkMixin,
}


def build_control_policy(
    cfgs: EvalConfigs, *, run_dir: Path, kind: ControlKind, ckpt: str = "final"
) -> LearnedPolicy:
    """Picklable runner factory of a control arm: the e05/e06 policy, network overridden.

    Args:
        cfgs: The evaluation configs.
        run_dir: The run directory.
        kind: ``"torch_folded"`` or ``"torch_obs_plus_1ulp"``.
        ckpt: Checkpoint name.

    Returns:
        The policy, re-classed to a subclass with the control mixin first.
    """
    from rld.rl.train import build_policy

    policy = build_policy(cfgs, run_dir=run_dir, ckpt=ckpt)
    base = type(policy)
    policy.__class__ = type(f"{kind}_{base.__name__}", (_CONTROL_MIXINS[kind], base), {})
    if kind == "torch_folded":
        cast(FoldedTorchNetworkMixin, policy).folded = folded_actor(policy)
    return policy


@dataclass(frozen=True)
class ClosedLoopPair:
    """One exported (method, seed) and its specs, one per :data:`ARMS` entry.

    Attributes:
        method: Method label.
        seed: Training seed.
        specs: ``{arm: spec}``.
    """

    method: str
    seed: int
    specs: dict[str, PolicySpec]


def _pair(method: str, seed: int, run_dir: Path, onnx_path: Path) -> ClosedLoopPair:
    """Build every arm's spec of one run."""
    from rld.eval.learned import inspect_run, learned_spec

    run = inspect_run(method, run_dir)
    if run.seed != seed:
        raise ValueError(f"{run_dir}: recorded seed {run.seed} != {seed}")

    def spec(build: Any) -> PolicySpec:
        return callable_spec(
            method, build, run_seed=run.seed, needs_motion_feed=run.needs_motion_feed
        )

    specs = {
        "torch": learned_spec(run),
        "onnx": spec(
            functools.partial(build_onnx_policy, run_dir=run.run_dir, onnx_path=onnx_path)
        ),
        "torch_folded": spec(
            functools.partial(build_control_policy, run_dir=run.run_dir, kind="torch_folded")
        ),
        "torch_obs_plus_1ulp": spec(
            functools.partial(build_control_policy, run_dir=run.run_dir, kind="torch_obs_plus_1ulp")
        ),
    }
    return ClosedLoopPair(method, run.seed, specs)


def run_closed_loop(
    graphs: Sequence[tuple[str, int, Path, Path]],
    cfgs: EvalConfigs,
    episodes: Sequence[ListedEpisode],
    *,
    workers: int,
    chunk: int = 25,
) -> dict[tuple[str, int], dict[str, list[dict[str, Any]]]]:
    """Fly every (method, seed) on the same episodes in every :data:`ARMS` arm.

    Args:
        graphs: ``(method, seed, run_dir, onnx_path)`` per exported graph.
        cfgs: The committed configs (noise must be off).
        episodes: The closed-loop episodes.
        workers: Runner processes (rows do not depend on it).
        chunk: Episodes per runner task (rows do not depend on it).

    Returns:
        ``{(method, seed): {arm: rows}}``, each in episode order.

    Raises:
        ValueError: If the perception stand-in is enabled.
    """
    if cfgs.noise.enabled:
        raise ValueError("closed-loop parity flies noise off (P8-D1 §7)")
    pairs = [_pair(m, s, r, o) for m, s, r, o in graphs]
    arms = [EvalArm(p.specs[a], None, episodes) for p in pairs for a in ARMS]
    rows = run_arms(arms, cfgs, workers=workers, chunk=chunk)
    n = len(episodes)
    out: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}
    k = 0
    for p in pairs:
        out[(p.method, p.seed)] = {}
        for a in ARMS:
            out[(p.method, p.seed)][a] = rows[k * n : (k + 1) * n]
            k += 1
    return out


def committed_lines(path: Path, method: str, seed: int) -> dict[tuple[str, str, str], str]:
    """Index a committed ``episodes.csv``'s lines of one (method, seed) by (regime, ss, index).

    Args:
        path: The committed ``episodes.csv``.
        method: Method label.
        seed: Training seed (``run_seed``).

    Returns:
        ``{(regime, ss, index): the raw CSV line}``.
    """
    out: dict[tuple[str, str, str], str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\n").split(",")
        i_m, i_s = header.index("method"), header.index("run_seed")
        i_r, i_ss, i_i = header.index("regime"), header.index("ss"), header.index("index")
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if parts[i_m] == method and parts[i_s] == str(seed) and parts[i_r] == "id":
                out[(parts[i_r], parts[i_ss], parts[i_i])] = line.rstrip("\n")
    return out


def _lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Serialise runner rows exactly as the committed CSVs were (no header)."""
    return episode_rows_csv(rows).rstrip("\n").split("\n")[1:]


def _float(value: Any) -> float:
    """Return a row value as float (NaN for missing)."""
    return float("nan") if value is None else float(value)


def closed_loop_tables(
    results: Mapping[tuple[str, int], Mapping[str, list[dict[str, Any]]]],
    committed: Mapping[str, Path] = COMMITTED_EPISODES,
) -> tuple[str, str]:
    """Render ``closed_loop_parity.csv`` and ``closed_loop_episodes.csv``.

    Args:
        results: :func:`run_closed_loop`'s output.
        committed: The committed episode CSV per method.

    Returns:
        ``(summary CSV text, per-episode CSV text)``.
    """
    summary = io.StringIO()
    per_ep = io.StringIO()
    w_sum = csv.writer(summary, lineterminator="\n")
    w_ep = csv.writer(per_ep, lineterminator="\n")
    w_sum.writerow(CLOSED_LOOP_COLUMNS)
    w_ep.writerow(CLOSED_LOOP_EPISODE_COLUMNS)
    for (method, seed), arms in results.items():
        t_rows, o_rows = arms["torch"], arms["onnx"]
        source = committed[method]
        reference = committed_lines(source, method, seed)
        t_lines, o_lines = _lines(t_rows), _lines(o_rows)
        n_same = n_rows = n_match = 0
        d_vz = 0.0
        d_steps = 0
        for t, o, t_line, o_line in zip(t_rows, o_rows, t_lines, o_lines, strict=True):
            key = (str(t["regime"]), str(t["ss"]), str(t["index"]))
            same = t["outcome"] == o["outcome"]
            row_same = t_line == o_line
            match = reference.get(key) == t_line
            n_same += same
            n_rows += row_same
            n_match += match
            vt, vo = _float(t["rel_vz_normal_m_s"]), _float(o["rel_vz_normal_m_s"])
            if not (math.isnan(vt) and math.isnan(vo)):
                d_vz = max(d_vz, abs(vt - vo))
            d_steps = max(d_steps, abs(int(t["steps"]) - int(o["steps"])))
            w_ep.writerow(
                [
                    method,
                    seed,
                    t["regime"],
                    t["ss"],
                    t["index"],
                    t["episode_seed"],
                    t["outcome"],
                    o["outcome"],
                    t["steps"],
                    o["steps"],
                    repr(vt),
                    repr(vo),
                    same,
                    row_same,
                    match,
                ]
            )

        def _counts(rows: Sequence[Mapping[str, Any]]) -> str:
            names = sorted({str(r["outcome"]) for r in rows})
            return "|".join(f"{k}:{sum(1 for r in rows if r['outcome'] == k)}" for k in names)

        rel = source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source
        w_sum.writerow(
            [
                method,
                seed,
                len(t_rows),
                n_same,
                n_same == len(t_rows),
                repr(d_vz),
                d_steps,
                n_rows,
                n_match,
                str(rel),
                _counts(t_rows),
                _counts(o_rows),
            ]
        )
    return summary.getvalue(), per_ep.getvalue()


#: Columns of ``closed_loop_controls.csv``.
CONTROL_COLUMNS: tuple[str, ...] = (
    "method",
    "seed",
    "arm",
    "post_hoc",
    "n_episodes",
    "n_outcome_identical_to_torch",
    "flipped_episodes",
    "max_abs_d_rel_vz_normal_m_s",
    "max_abs_d_steps",
    "n_steps_identical",
    "n_rows_identical",
)


def controls_table(results: Mapping[tuple[str, int], Mapping[str, list[dict[str, Any]]]]) -> str:
    """Render ``closed_loop_controls.csv``: every non-PyTorch arm against the PyTorch arm.

    Args:
        results: :func:`run_closed_loop`'s output.

    Returns:
        The CSV text; ``flipped_episodes`` lists ``SS/index:torch->arm`` for each outcome
        difference.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CONTROL_COLUMNS)
    for (method, seed), arms in results.items():
        t_rows = arms["torch"]
        t_lines = _lines(t_rows)
        for arm in ARMS[1:]:
            o_rows = arms[arm]
            o_lines = _lines(o_rows)
            flips = [
                f"{t['ss']}/{t['index']}:{t['outcome']}->{o['outcome']}"
                for t, o in zip(t_rows, o_rows, strict=True)
                if t["outcome"] != o["outcome"]
            ]
            d_vz = 0.0
            for t, o in zip(t_rows, o_rows, strict=True):
                vt, vo = _float(t["rel_vz_normal_m_s"]), _float(o["rel_vz_normal_m_s"])
                if not (math.isnan(vt) and math.isnan(vo)):
                    d_vz = max(d_vz, abs(vt - vo))
            writer.writerow(
                [
                    method,
                    seed,
                    arm,
                    arm != "onnx",
                    len(t_rows),
                    len(t_rows) - len(flips),
                    " ".join(flips),
                    repr(d_vz),
                    max(
                        abs(int(t["steps"]) - int(o["steps"]))
                        for t, o in zip(t_rows, o_rows, strict=True)
                    ),
                    sum(
                        int(t["steps"]) == int(o["steps"])
                        for t, o in zip(t_rows, o_rows, strict=True)
                    ),
                    sum(a == b for a, b in zip(t_lines, o_lines, strict=True)),
                ]
            )
    return buffer.getvalue()
