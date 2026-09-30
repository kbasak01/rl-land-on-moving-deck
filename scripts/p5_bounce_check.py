"""P5-D1 diagnostic: are ``pid_feedforward_lowvz`` bounces physical or a solver artifact?

Diagnosis only (docs/protocol.md P5-D1). Nothing under ``src/rld/envs`` or ``configs/`` is
changed: a subclass of :class:`~rld.envs.landing_env.DeckLandingAviary` defined **here** logs
every physics substep, and the non-committed physics arms (480 Hz, ``contactERP`` = 0) are
built in memory for this script only.

Episodes: the P3-D2 **tune pool only** (``rld.deck.splits.dev_pool(cfg)[1]``) -- the committed
P3-D3 tune draw (``tuning_seed`` 20260923, 60 per sea state) plus an extra diagnostic draw on
the same pool with a different seed. Never ``results/episodes/``.

Physics arms:
    ``base``      the committed environment (240 Hz, PyBullet's default contact ERP).
    ``hz480``     physics at 480 Hz, control still 30 Hz; everything else committed.
    ``erp0``      240 Hz with ``setPhysicsEngineParameter(contactERP=0)``: Bullet's
                  penetration recovery then injects no separating velocity at all.

The discriminator. With the drone resting on the deck, the contact must supply
``N = m (a_deck - a_free) . n`` where ``a_free = (T b_z)/m + g`` is the drone's free-flight
acceleration from its applied rotor thrust. ``N < 0`` means the deck is accelerating away
from the drone faster than the thrust lets the drone follow: a physical lift-off. The
solver's own contribution is the relative normal velocity it leaves behind at the end of a
contact substep (``push-off``); with zero restitution that should be zero except for
Baumgarte penetration recovery (``contactERP * penetration / dt``).

Units: metres, metres per second, seconds MODEL scale (lambda = 1/25); newtons; degrees.

Usage::

    .venv/bin/python scripts/p5_bounce_check.py --out-dir results/p5_bounce_check
"""

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as pyb
from dmf.config import load_sim

from rld.control.registry import make_controller
from rld.control.tuning import TuningEpisode, draw_tuning_episodes, load_tuning
from rld.deck.splits import dev_pool
from rld.envs.landing_env import DeckLandingAviary
from rld.envs.touchdown import analytic_clearance_m
from rld.eval.envs import EvalConfigs, load_eval_configs, motion_for, pad_offset_for

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = "pid_feedforward_lowvz"
DIAG_SEED = 20260925  # extra tune-pool draw; != tuning_seed 20260923, != P2-D4 20260922
PRE_TD_S = 0.10  # seconds model scale logged before first contact

# Per-substep columns.
COLS = (
    "t_ep_s",
    "deck_z_m",
    "deck_vz_m_s",
    "deck_az_m_s2",
    "deck_an_m_s2",  # analytic deck acceleration along the deck normal
    "drone_vz_m_s",
    "rel_vn_analytic_m_s",  # (v_drone - v_deck_analytic) . n, + = separating
    "rel_vn_pyb_m_s",  # (v_drone_pyb - v_plate_pyb) . n after the step
    "cmd_vz_m_s",
    "thrust_n",
    "a_free_n_m_s2",  # drone free-flight accel along n from applied thrust + gravity
    "need_n_n",  # m (a_deck - a_free) . n: contact force needed to stay together
    "normal_force_n",  # sum of reported normal forces, drone-deck, unfiltered
    "n_points",
    "n_points_filtered",  # normalForce > contact_normal_force_min_n
    "min_dist_m",  # min contactDistance (negative = penetration)
    "clearance_m",  # analytic clearance of the lowest cylinder point
    "impulse_n_s",  # m (dv_drone - a_free dt) . n: inferred vertical contact impulse
    "in_contact",
    "contact_lost",
    "events",
    "rel_tilt_deg",
    "lateral_m",
    "omega_drone_rad_s",  # |drone angular velocity|
    "omega_rel_tilt_rad_s",  # d/dt of relative tilt from angular velocities, + = tilting away
    "cp_rel_vn_m_s",  # contact-POINT relative normal velocity after the step, + = separating
    "com_rel_vn_m_s",  # same as rel_vn_analytic (kept for clarity)
)


class DiagAviary(DeckLandingAviary):  # type: ignore[misc]
    """The committed environment plus a per-substep logger and optional engine overrides."""

    def __init__(self, *, engine_params: dict[str, float] | None = None, **kw: Any) -> None:
        """Build the committed env; ``engine_params`` go to ``setPhysicsEngineParameter``."""
        self._engine_params = dict(engine_params or {})
        self._cmd = np.zeros(3)
        self._rpm = np.zeros(4)
        self._v_pre = np.zeros(3)
        self.log: list[tuple[float, ...]] = []
        super().__init__(**kw)

    def _addObstacles(self) -> None:  # noqa: N802
        super()._addObstacles()
        if self._engine_params:
            pyb.setPhysicsEngineParameter(physicsClientId=self.CLIENT, **self._engine_params)

    def reset(self, **kw: Any) -> Any:
        """Clear the substep log, then reset as committed."""
        self.log = []
        return super().reset(**kw)

    def velocity_setpoint_m_s(self, action: Any) -> Any:
        """Record the commanded setpoint (m/s, world), then return it unchanged."""
        self._cmd = super().velocity_setpoint_m_s(action)
        return self._cmd

    def _physics(self, rpm: Any, nth_drone: int) -> None:
        self._rpm = np.asarray(rpm, dtype=np.float64).copy()
        self._v_pre = np.asarray(
            pyb.getBaseVelocity(self.DRONE_IDS[0], physicsClientId=self.CLIENT)[0]
        )
        super()._physics(rpm, nth_drone)

    def _poll_detectors(self) -> None:
        super()._poll_detectors()
        deck = self._deck_sample()
        n = np.asarray(deck.normal, dtype=np.float64)
        pos = np.asarray(self.pos[0], dtype=np.float64)
        vel = np.asarray(self.vel[0], dtype=np.float64)
        rot = self._drone_rotation(np.asarray(self.quat[0], dtype=np.float64))
        mass, g = float(self.M), float(self.G)
        thrust = float(np.sum(self._rpm**2) * self.KF)
        a_free = rot[:, 2] * thrust / mass + np.array([0.0, 0.0, -g])
        a_deck = np.asarray(deck.acceleration_m_s2, dtype=np.float64)
        dt = self.cfg.physics_dt_s
        impulse = mass * float(np.dot(vel - self._v_pre - a_free * dt, n))
        plate_v = np.asarray(
            pyb.getBaseVelocity(self._platform.body_id, physicsClientId=self.CLIENT)[0]
        )
        points = pyb.getContactPoints(
            bodyA=self.DRONE_IDS[0], bodyB=self._platform.body_id, physicsClientId=self.CLIENT
        )
        forces = [float(p[9]) for p in points]
        dists = [float(p[8]) for p in points]
        thr = self.success.contact_normal_force_min_n
        w_d = np.asarray(pyb.getBaseVelocity(self.DRONE_IDS[0], physicsClientId=self.CLIENT)[1])
        plate_pose = pyb.getBasePositionAndOrientation(
            self._platform.body_id, physicsClientId=self.CLIENT
        )
        w_p = np.asarray(
            pyb.getBaseVelocity(self._platform.body_id, physicsClientId=self.CLIENT)[1]
        )
        com = np.asarray(
            pyb.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.CLIENT)[0]
        )
        cp_v = float("nan")
        if points:
            # contact point on the drone with the largest force (or deepest)
            best = max(points, key=lambda q: (float(q[9]), -float(q[8])))
            pa = np.asarray(best[5])
            pb = np.asarray(best[6])
            va = vel + np.cross(w_d, pa - com)
            vb = plate_v + np.cross(w_p, pb - np.asarray(plate_pose[0]))
            cp_v = float(np.dot(va - vb, n))
        # relative tilt rate: component of (w_d - w_deck) that rotates body z away from n
        bz = rot[:, 2]
        axis = np.cross(n, bz)
        na = float(np.linalg.norm(axis))
        w_rel = w_d - np.asarray(deck.angular_velocity_rad_s)
        tilt_rate = float(np.dot(w_rel, axis / na)) if na > 1e-9 else 0.0
        offset = pos - np.asarray(deck.position_m)
        lateral = float(np.linalg.norm(offset - np.dot(offset, n) * n))
        tilt = float(np.degrees(np.arccos(np.clip(np.dot(rot[:, 2], n), -1.0, 1.0))))
        self.log.append(
            (
                self._substep * dt,
                float(deck.position_m[2]),
                float(deck.velocity_m_s[2]),
                float(a_deck[2]),
                float(np.dot(a_deck, n)),
                float(vel[2]),
                float(np.dot(vel - np.asarray(deck.velocity_m_s), n)),
                float(np.dot(vel - plate_v, n)),
                float(self._cmd[2]),
                thrust,
                float(np.dot(a_free, n)),
                mass * float(np.dot(a_deck - a_free, n)),
                float(sum(forces)),
                float(len(points)),
                float(sum(f > thr for f in forces)),
                float(min(dists)) if dists else float("nan"),
                analytic_clearance_m(pos, rot, deck, self._geometry),
                impulse,
                float(self._in_contact),
                float(self._contact_lost),
                float(self.record.contact_events),
                tilt,
                lateral,
                float(np.linalg.norm(w_d)),
                tilt_rate,
                cp_v,
                float(np.dot(vel - np.asarray(deck.velocity_m_s), n)),
            )
        )


@dataclass(frozen=True)
class Arm:
    """One physics configuration. Diagnostic only; none of these is committed."""

    name: str
    physics_hz: int
    engine_params: tuple[tuple[str, float], ...]
    grace_s: float | None = None


ARMS = {
    "base": Arm("base", 240, ()),
    "hz480": Arm("hz480", 480, ()),
    "erp0": Arm("erp0", 240, (("contactERP", 0.0),)),
    # 240 Hz, committed physics, but a 0.5 s contact-loss grace so no episode ends on a
    # release: the full dwell window is logged and any grace <= 0.5 s can be re-applied
    # post hoc from the substep log. Scoring-rule probe only; changes no physics.
    "grace_probe": Arm("grace_probe", 240, (), 0.5),
    "grace_probe480": Arm("grace_probe480", 480, (), 0.5),
    "grace_probe960": Arm("grace_probe960", 960, (), 0.5),
    "grace_probe_erp0": Arm("grace_probe_erp0", 240, (("contactERP", 0.0),), 0.5),
    "grace_probe960_erp0": Arm("grace_probe960_erp0", 960, (("contactERP", 0.0),), 0.5),
    "grace_probe1920": Arm("grace_probe1920", 1920, (), 0.5),
}


def draw_extra(cfgs: EvalConfigs, ss: str, n: int, pad: str) -> list[TuningEpisode]:
    """Extra tune-pool episodes, same procedure as ``draw_tuning_episodes``, another seed."""
    _, tune = dev_pool(cfgs.sim)
    cands = [s for s in tune if s.sea_state == ss]
    label = int.from_bytes(hashlib.sha256(f"p5diag|{ss}".encode()).digest()[:4], "big")
    rng = np.random.default_rng(np.random.SeedSequence([DIAG_SEED, label]))
    picks = rng.integers(0, len(cands), size=n)
    seeds = rng.integers(0, 2**31 - 1, size=n)
    return [
        TuningEpisode(
            ss=ss,
            index=i,
            pad=pad,
            vessel=cands[int(k)].vessel,
            heading_deg=float(cands[int(k)].heading_deg),
            speed_kn=float(cands[int(k)].speed_kn),
            realization_seed=int(cands[int(k)].seed),
            episode_seed=int(s),
        )
        for i, (k, s) in enumerate(zip(picks, seeds, strict=True))
    ]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, stop) index pairs of True runs."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def summarise(log: np.ndarray, env: DiagAviary) -> dict[str, Any]:
    """Per-episode features over the dwell window; see module docstring."""
    rec = env.record
    c = {k: i for i, k in enumerate(COLS)}
    row: dict[str, Any] = {**rec.as_row()}
    dt = env.cfg.physics_dt_s
    first = rec.contact_record
    if first is None or log.size == 0:
        return row
    t = log[:, c["t_ep_s"]]
    win = (t >= first.t_episode_s) & (t <= first.t_episode_s + env.success.contact_dwell_min_s)
    w = log[win]
    inc = w[:, c["n_points_filtered"]] > 0
    need = w[:, c["need_n_n"]]
    # first loss that became a release, or the first loss at all
    loss_idx = [s for s, _ in _runs(~inc)]
    row["n_loss_runs"] = len(loss_idx)
    row["min_need_n_n"] = float(need.min())
    neg_runs = _runs(need < 0.0)
    row["max_neg_need_run_s"] = max((b - a) * dt for a, b in neg_runs) if neg_runs else 0.0
    row["frac_neg_need"] = float(np.mean(need < 0.0))
    row["min_deck_an_m_s2"] = float(w[:, c["deck_an_m_s2"]].min())
    # solver push-off: separating relative velocity left at the end of a contact substep
    push = w[inc, c["rel_vn_pyb_m_s"]]
    row["max_pushoff_m_s"] = float(push.max()) if push.size else float("nan")
    row["max_penetration_window_m"] = float(-np.nanmin(w[:, c["min_dist_m"]]))
    row["max_clearance_window_m"] = float(w[:, c["clearance_m"]].max())
    row["max_rel_vn_sep_m_s"] = float(w[:, c["rel_vn_analytic_m_s"]].max())
    cpv = w[inc, c["cp_rel_vn_m_s"]]
    row["max_cp_pushoff_m_s"] = float(np.nanmax(cpv)) if cpv.size else float("nan")
    # longest gap in filtered contact inside the dwell window (for post-hoc grace)
    gaps = _runs(~inc)
    row["max_gap_s"] = max(((b - a) * dt for a, b in gaps), default=0.0)
    row["n_pts_first"] = int(w[0, c["n_points_filtered"]])
    row["com_closing_throughout_first_gap"] = (
        bool(np.all(w[gaps[0][0] : gaps[0][1], c["rel_vn_analytic_m_s"]] < 0.0)) if gaps else None
    )
    row["max_com_rel_vn_first_gap_m_s"] = (
        float(w[gaps[0][0] : gaps[0][1], c["rel_vn_analytic_m_s"]].max()) if gaps else None
    )
    row["first_gap_start_s"] = float(gaps[0][0] * dt) if gaps else None
    row["first_gap_len_s"] = float((gaps[0][1] - gaps[0][0]) * dt) if gaps else None
    row["max_clearance_first_gap_m"] = (
        float(w[gaps[0][0] : gaps[0][1], c["clearance_m"]].max()) if gaps else None
    )
    row["cp_rel_vn_before_first_gap_m_s"] = (
        float(w[gaps[0][0] - 1, c["cp_rel_vn_m_s"]]) if gaps and gaps[0][0] > 0 else None
    )
    row["min_need_before_first_gap_n"] = (
        float(need[: gaps[0][0]].min()) if gaps and gaps[0][0] > 0 else None
    )
    row["need_mean_first_gap_n"] = float(need[gaps[0][0] : gaps[0][1]].mean()) if gaps else None
    row["tilt_rate_after_td_rad_s"] = float(w[min(3, len(w) - 1), c["omega_rel_tilt_rad_s"]])
    row["dwell_end_in_contact"] = bool(inc[-1])
    if loss_idx:
        j = loss_idx[0]
        row["t_first_loss_after_td_s"] = float(w[j, c["t_ep_s"]] - first.t_episode_s)
        lo = max(0, j - int(round(0.02 / dt)))  # 20 ms before loss
        row["need_n_before_loss_mean_n"] = float(need[lo:j].mean()) if j > lo else float("nan")
        row["need_n_at_loss_n"] = float(need[j])
        row["pushoff_at_loss_m_s"] = float(w[j - 1, c["rel_vn_pyb_m_s"]]) if j > 0 else float("nan")
        row["deck_an_at_loss_m_s2"] = float(w[j, c["deck_an_m_s2"]])
        row["a_free_n_at_loss_m_s2"] = float(w[j, c["a_free_n_m_s2"]])
        row["pen_before_loss_m"] = float(-w[j - 1, c["min_dist_m"]]) if j > 0 else float("nan")
        # release: first substep with contact_lost set
        lost = np.nonzero(log[:, c["contact_lost"]] > 0)[0]
        if lost.size:
            k = int(lost[0])
            row["release_t_after_td_s"] = float(log[k, c["t_ep_s"]] - first.t_episode_s)
            row["release_clearance_m"] = float(log[k, c["clearance_m"]])
            row["release_rel_vn_m_s"] = float(log[k, c["rel_vn_analytic_m_s"]])
            gap = log[: k + 1][log[: k + 1, c["t_ep_s"]] >= w[j, c["t_ep_s"]]]
            row["analytic_gap_during_loss"] = bool(
                np.any(gap[:, c["clearance_m"]] > env.success.analytic_contact_margin_m)
            )
            row["max_clearance_during_loss_m"] = float(gap[:, c["clearance_m"]].max())
    return row


type Task = tuple[str, str, list[TuningEpisode], int]


def run_chunk(task: Task) -> list[dict[str, Any]]:
    """Fly one chunk of episodes in one arm; return summaries (+ logs of kept episodes)."""
    controller, arm_name, episodes, keep_success_logs = task
    arm = ARMS[arm_name]
    cfgs = load_eval_configs()
    landing = replace(cfgs.landing, physics_freq_hz=arm.physics_hz)
    cfgs = replace(cfgs, landing=landing)
    if arm.grace_s is not None:
        cfgs = replace(cfgs, success=replace(cfgs.success, contact_loss_grace_s=arm.grace_s))
    ctrl = make_controller(controller)
    first = episodes[0]
    env = DiagAviary(
        engine_params=dict(arm.engine_params),
        motion=motion_for(
            cfgs, first.vessel, first.ss, first.heading_deg, first.speed_kn, first.realization_seed
        ),
        pad=first.pad,
        pad_radius_m=cfgs.pads.radius_model_m,
        pad_offset_m=pad_offset_for(cfgs, first.vessel, first.pad),
        cfg=cfgs.landing,
        success=cfgs.success,
        obs_cfg=cfgs.observation,
        noise_cfg=cfgs.noise,
        reward_cfg=cfgs.reward,
        episode_seed=first.episode_seed,
    )
    out: list[dict[str, Any]] = []
    kept = 0
    max_steps = int(cfgs.landing.total_len_s * cfgs.landing.ctrl_freq_hz) + 1
    try:
        for ep in episodes:
            env.set_motion(
                motion_for(
                    cfgs, ep.vessel, ep.ss, ep.heading_deg, ep.speed_kn, ep.realization_seed
                ),
                ep.pad,
                pad_offset_for(cfgs, ep.vessel, ep.pad),
            )
            obs, _ = env.reset(seed=ep.episode_seed)
            ctrl.reset(ep.episode_seed)
            for _ in range(max_steps):
                obs, _, term, trunc, _ = env.step(ctrl.act(obs))
                if term or trunc:
                    break
            log = np.asarray(env.log, dtype=np.float64)
            row = {
                "controller": controller,
                "arm": arm_name,
                "ss": ep.ss,
                "index": ep.index,
                "heading_deg": ep.heading_deg,
                "speed_kn": ep.speed_kn,
                "realization_seed": ep.realization_seed,
                "episode_seed": ep.episode_seed,
                **summarise(log, env),
            }
            keep = (
                row["outcome"] == "bounce"
                or float(row.get("max_gap_s", 0.0)) > 0.03
                or (row["outcome"] == "success" and kept < keep_success_logs)
            )
            if keep and env.record.contact_record is not None:
                if row["outcome"] == "success":
                    kept += 1
                t_td = env.record.contact_record.t_episode_s
                sel = log[:, 0] >= t_td - PRE_TD_S
                row["_log"] = log[sel].tolist()
            out.append(row)
    finally:
        env.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "p5_bounce_check")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--controllers", nargs="+", default=[CONTROLLER])
    parser.add_argument("--extra-ss5", type=int, default=600)
    parser.add_argument("--extra-ss4", type=int, default=300)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--chunk", type=int, default=10)
    parser.add_argument("--logs-dir", type=Path, required=True, help="per-substep logs (large)")
    args = parser.parse_args()

    cfgs = load_eval_configs()
    tuning = load_tuning()
    sim = load_sim(cfgs.motion.sim_config_path)
    committed = draw_tuning_episodes(sim, tuning)
    extra = draw_extra(cfgs, "SS5", args.extra_ss5, tuning.pad) + draw_extra(
        cfgs, "SS4", args.extra_ss4, tuning.pad
    )
    extra = [replace(e, index=e.index + 1000) for e in extra]  # disjoint index space
    tasks: list[Task] = []
    for controller in args.controllers:
        for arm in args.arms:
            for eps in (committed, extra):
                for s in range(0, len(eps), args.chunk):
                    tasks.append((controller, arm, eps[s : s + args.chunk], 1))
    started = time.perf_counter()
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        rows = [r for chunk in pool.imap(run_chunk, tasks, chunksize=1) for r in chunk]
    wall = time.perf_counter() - started

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    logs = {}
    for r in rows:
        r["draw"] = "tune_final" if r["index"] < 1000 else "diag"
        if "_log" in r:
            key = f"{r['controller']}|{r['arm']}|{r['draw']}|{r['ss']}|{r['index']}|{r['outcome']}"
            logs[key] = r.pop("_log")
    (args.logs_dir / "substep_logs.json").write_text(json.dumps({"cols": COLS, "logs": logs}))
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (args.logs_dir / "episodes_all.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} episode-runs in {wall:.0f} s -> {args.logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
