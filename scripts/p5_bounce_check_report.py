"""P5-D1: reduce the ``p5_bounce_check.py`` logs to the small committed CSVs.

Reads one or more ``--logs-dir`` directories written by ``scripts/p5_bounce_check.py`` (each
holds ``episodes_all.csv`` and ``substep_logs.json``; large, not committed) and writes to
``--out-dir``:

* ``arms_summary.csv``   -- one row per (controller, arm): outcome counts, contact-gap counts
  in the 0.5 s dwell window, solver push-off and penetration percentiles.
* ``paired_vs_base.csv`` -- same-episode outcome transitions base -> hz480 and base -> erp0.
* ``bounce_triggers.csv`` -- one row per bounce in the terminating arms (base/hz480/erp0):
  what the drone and deck were doing across the gap that triggered the release, and a
  mechanism label.
* ``tau_bins.csv``       -- bounce rate against the rim-rocking time
  ``tau = 0.06 m * sin(rel_tilt) / closing_speed`` (base arm).

Units: metres, metres per second, seconds MODEL scale; newtons; degrees; m/s^2.
Tune pool only (P3-D2); see ``docs/protocol.md`` P5-D1.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GAP_THRESHOLDS_S = (0.0333, 0.05, 0.075, 0.1, 0.25)
ROTOR_RIM_M = 0.06  # CF2X collision-cylinder radius, metres (cf2x.urdf)


def arms_summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ctrl, arm), x in d.groupby(["controller", "arm"]):
        td = x[x.touchdown_contact.astype(bool)]
        r: dict[str, Any] = {"controller": ctrl, "arm": arm, "n": len(x), "n_touchdown": len(td)}
        for o in ("success", "bounce", "hard_landing", "off_pad", "crash", "timeout"):
            r[f"n_{o}"] = int((x.outcome == o).sum())
        for th in GAP_THRESHOLDS_S:
            r[f"n_gap_gt_{th * 1e3:.0f}ms"] = int((td.max_gap_s > th + 1e-9).sum())
        r["n_gap_gt_50ms_com_closing"] = int(
            (
                (td.max_gap_s > 0.05 + 1e-9)
                & (td.com_closing_throughout_first_gap.astype(str) == "True")
            ).sum()
        )
        r["dwell_end_in_contact"] = int(td.dwell_end_in_contact.astype(bool).sum())
        r["cp_pushoff_max_m_s_p50"] = float(td.max_cp_pushoff_m_s.median())
        r["cp_pushoff_max_m_s_p99"] = float(td.max_cp_pushoff_m_s.quantile(0.99))
        r["penetration_max_m_p99"] = float(td.max_penetration_window_m.quantile(0.99))
        r["clearance_max_m_p95"] = float(td.max_clearance_window_m.quantile(0.95))
        rows.append(r)
    return pd.DataFrame(rows)


def paired(d: pd.DataFrame) -> pd.DataFrame:
    key = ["controller", "ss", "index"]
    base = d[d.arm == "base"].set_index(key)[["outcome"]]
    out = []
    for arm in ("hz480", "erp0"):
        o = d[d.arm == arm].set_index(key)[["outcome"]]
        j = base.join(o, rsuffix="_arm", how="inner")
        t = j.groupby(["controller", "outcome", "outcome_arm"]).size().reset_index(name="n")
        t.insert(1, "arm", arm)
        out.append(t.rename(columns={"outcome": "outcome_base"}))
    return pd.concat(out, ignore_index=True)


def triggers(logs: dict[str, Any]) -> pd.DataFrame:
    cols = logs["cols"]
    c = {k: i for i, k in enumerate(cols)}
    rows = []
    for key, v in logs["logs"].items():
        ctrl, arm, draw, ss, idx, outcome = key.split("|")
        if outcome != "bounce" or arm.startswith("grace"):
            continue
        log = np.asarray(v, dtype=float)
        dt = log[1, 0] - log[0, 0]
        inc = log[:, c["n_points_filtered"]] > 0
        td = int(np.argmax(inc))
        k = int(np.nonzero(log[:, c["contact_lost"]] > 0)[0][0])
        last = td + int(np.nonzero(inc[td:k])[0].max())
        gap = log[last + 1 : k + 1]
        pre = log[max(td, last - int(round(0.02 / dt))) : last + 1]
        com = gap[:, c["rel_vn_analytic_m_s"]]
        start = (last + 1 - td) * dt
        receding = float(com.mean()) >= 0.0
        if not receding:
            mech = "rim_rocking_at_impact" if start <= 0.05 else "rocking_later"
        else:
            mech = "unloaded_rim_liftoff"
        rows.append(
            {
                "controller": ctrl,
                "arm": arm,
                "draw": draw,
                "ss": ss,
                "index": int(idx),
                "mechanism": mech,
                "td_closing_m_s": float(-log[td - 1, c["rel_vn_analytic_m_s"]]) if td else None,
                "td_rel_tilt_deg": float(log[td, c["rel_tilt_deg"]]),
                "td_contact_points": int(log[td, c["n_points_filtered"]]),
                "gap_start_after_td_s": start,
                "gap_len_to_release_s": float(len(gap) * dt),
                "cp_rel_vn_end_last_contact_m_s": float(log[last, c["cp_rel_vn_m_s"]]),
                "com_rel_vn_mean_in_gap_m_s": float(com.mean()),
                "com_rel_vn_max_in_gap_m_s": float(com.max()),
                "clearance_max_in_gap_m": float(gap[:, c["clearance_m"]].max()),
                "rel_tilt_rate_mean_in_gap_rad_s": float(gap[:, c["omega_rel_tilt_rad_s"]].mean()),
                "need_force_mean_20ms_before_gap_n": float(pre[:, c["need_n_n"]].mean()),
                "normal_force_mean_20ms_before_gap_n": float(pre[:, c["normal_force_n"]].mean()),
                "deck_accel_n_mean_in_gap_m_s2": float(gap[:, c["deck_an_m_s2"]].mean()),
                "drone_free_accel_n_mean_in_gap_m_s2": float(gap[:, c["a_free_n_m_s2"]].mean()),
                "thrust_mean_in_gap_n": float(gap[:, c["thrust_n"]].mean()),
                "penetration_max_td_to_gap_m": float(
                    -np.nanmin(log[td : last + 1, c["min_dist_m"]])
                ),
            }
        )
    return pd.DataFrame(rows)


def tau_bins(d: pd.DataFrame) -> pd.DataFrame:
    x = d[(d.arm == "base") & d.touchdown_contact.astype(bool)].copy()
    x["tau_s"] = (
        ROTOR_RIM_M
        * np.sin(np.radians(x.rel_tilt_deg))
        / x.closing_speed_normal_m_s.clip(lower=1e-3)
    )
    edges = [0.0, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 10.0]
    x["tau_bin_s"] = pd.cut(x.tau_s, edges).astype(str)
    return (
        x.groupby(["controller", "tau_bin_s"], observed=True)
        .agg(
            n_touchdown=("outcome", "size"),
            n_bounce=("outcome", lambda o: int((o == "bounce").sum())),
        )
        .reset_index()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    frames, trig = [], []
    for ld in args.logs_dir:
        frame = pd.read_csv(ld / "episodes_all.csv", low_memory=False)
        frames.append(frame)
        if frame.arm.str.startswith("grace").all():
            continue  # no terminating arm here, so no release to explain
        t = triggers(json.loads((ld / "substep_logs.json").read_text()))
        if len(t):
            trig.append(t)
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["controller", "arm", "ss", "index"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arms_summary(d).to_csv(args.out_dir / "arms_summary.csv", index=False)
    paired(d).to_csv(args.out_dir / "paired_vs_base.csv", index=False)
    pd.concat(trig, ignore_index=True).to_csv(args.out_dir / "bounce_triggers.csv", index=False)
    tau_bins(d).to_csv(args.out_dir / "tau_bins.csv", index=False)
    print(f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
