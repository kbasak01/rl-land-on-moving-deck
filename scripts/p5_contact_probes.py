"""P5-D1: two controller-free PyBullet contact probes on the CF2X and the deck plate.

Diagnosis only; builds its own bare PyBullet world (no environment code, no config change).

1. ``pushoff``: the drone placed with a given initial overlap on a static plate, thrust held
   exactly equal to weight, so the only vertical impulse is the contact's. The relative
   normal velocity left after the solve is Bullet's penetration-recovery push-off
   (Baumgarte, ``contactERP * penetration / dt``). Swept over dt and ``contactERP``.
2. ``rim_drop``: the drone released tilted onto a static plate at a closing speed, thrust a
   fixed fraction of weight along body z, no attitude control. The longest gap in
   force-filtered contact (``normalForce > 1e-4 N``, the environment's filter) after first
   contact is recorded, swept over dt, tilt, closing speed and thrust fraction.

Units: metres, metres per second, seconds MODEL scale; newtons; degrees.

Usage::

    .venv/bin/python scripts/p5_contact_probes.py --out results/p5_bounce_check/contact_probes.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import gym_pybullet_drones
import numpy as np
import pybullet as p

URDF = str(Path(gym_pybullet_drones.__file__).parent / "assets" / "cf2x.urdf")
FORCE_MIN_N = 1.0e-4  # configs/env/success.yaml contact_normal_force_min_n
HALF = (0.40, 0.26, 0.01)  # configs/env/landing.yaml platform.half_extents_m
G = 9.8


def _world(hz: int, erp: float | None) -> tuple[int, int]:
    c = p.connect(p.DIRECT)
    p.setGravity(0, 0, -G, physicsClientId=c)
    p.setTimeStep(1.0 / hz, physicsClientId=c)
    if erp is not None:
        p.setPhysicsEngineParameter(contactERP=erp, physicsClientId=c)
    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=list(HALF),
        collisionFramePosition=[0, 0, -HALF[2]],
        physicsClientId=c,
    )
    plate = p.createMultiBody(0.0, col, -1, [0, 0, 1.0], physicsClientId=c)
    p.changeDynamics(plate, -1, lateralFriction=0.5, restitution=0.0, physicsClientId=c)
    return c, plate


def pushoff(hz: int, overlap_m: float, erp: float | None) -> dict[str, Any]:
    c, plate = _world(hz, erp)
    d = p.loadURDF(
        URDF,
        [0, 0, 1.0 + 0.0125 - overlap_m],
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=c,
    )
    m = float(p.getDynamicsInfo(d, -1, physicsClientId=c)[0])
    vz, dist = [], []
    for _ in range(6):
        p.applyExternalForce(d, -1, [0, 0, m * G], [0, 0, 0], p.LINK_FRAME, physicsClientId=c)
        p.stepSimulation(physicsClientId=c)
        vz.append(float(p.getBaseVelocity(d, physicsClientId=c)[0][2]))
        cps = p.getContactPoints(d, plate, physicsClientId=c)
        dist.append(min((float(q[8]) for q in cps), default=float("nan")))
    p.disconnect(c)
    return {
        "probe": "pushoff",
        "physics_hz": hz,
        "contact_erp": "default" if erp is None else erp,
        "geometric_overlap_m": overlap_m,
        "contact_distance_step0_m": dist[0],
        "sep_velocity_step1_m_s": vz[1],
        "sep_velocity_step5_m_s": vz[5],
    }


def rim_drop(
    hz: int, tilt_deg: float, v_close: float, thrust_frac: float, erp: float | None = None
) -> dict[str, Any]:
    c, plate = _world(hz, erp)
    th = np.radians(tilt_deg)
    low = 0.0125 * np.cos(th) + 0.06 * np.sin(th)
    q = p.getQuaternionFromEuler([th, 0, 0])
    d = p.loadURDF(
        URDF, [0, 0, 1.0 + low + 0.004], q, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=c
    )
    m = float(p.getDynamicsInfo(d, -1, physicsClientId=c)[0])
    p.resetBaseVelocity(d, [0, 0, -v_close], [0, 0, 0], physicsClientId=c)
    inc = []
    for _ in range(int(0.4 * hz)):
        p.applyExternalForce(
            d, -1, [0, 0, thrust_frac * m * G], [0, 0, 0], p.LINK_FRAME, physicsClientId=c
        )
        p.stepSimulation(physicsClientId=c)
        inc.append(
            any(float(x[9]) > FORCE_MIN_N for x in p.getContactPoints(d, plate, physicsClientId=c))
        )
    p.disconnect(c)
    a = np.asarray(inc)
    gap, cur = 0, 0
    for v in a[int(np.argmax(a)) :] if a.any() else []:
        cur = 0 if v else cur + 1
        gap = max(gap, cur)
    return {
        "probe": "rim_drop",
        "physics_hz": hz,
        "contact_erp": "default" if erp is None else erp,
        "tilt_deg": tilt_deg,
        "closing_speed_m_s": v_close,
        "thrust_frac_of_weight": thrust_frac,
        "tau_rock_s": 0.06 * np.sin(th) / v_close,
        "max_contact_gap_s": gap / hz,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for hz in (240, 480):
        for overlap in (0.0, 0.001, 0.004):
            rows.append(pushoff(hz, overlap, None))
    for erp in (0.0, 0.05, 0.2):
        rows.append(pushoff(240, 0.001, erp))
    rates = (240, 480, 960, 1920)
    for tf in (1.0, 0.99, 0.98, 0.96):
        for tilt, v in ((3, 0.09), (6, 0.09), (9, 0.09), (6, 0.2), (12, 0.2)):
            for hz in rates:
                rows.append(rim_drop(hz, tilt, v, tf))
    for hz in rates:
        rows.append(rim_drop(hz, 6, 0.09, 1.0, erp=0.0))
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
