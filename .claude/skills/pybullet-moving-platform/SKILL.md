---
name: pybullet-moving-platform
description: Hard-won guidance for moving landing platforms, contact and touchdown detection in PyBullet and gym-pybullet-drones. Use whenever writing or debugging the platform body, constraints, contact points, collision filtering, the landing environment's step loop, or touchdown classification.
---

# Moving platform in PyBullet

## Driving the platform
- Do **not** call `resetBasePositionAndOrientation` every step on its own: the solver sees a
  zero-velocity body, so relative touchdown velocity is wrong and the drone can tunnel.
- Preferred: dynamic box (large mass, e.g. 1 000 kg), `createConstraint(body, -1, -1, -1,
  JOINT_FIXED, ...)` to world, then each **physics** step `changeConstraint(cid, jointChildPivot=p,
  jointChildFrameOrientation=q, maxForce=1e6)`. Tune `maxForce`, check tracking.
- Alternative: reset pose **and** `resetBaseVelocity(linear, angular)` from analytic deck velocity
  every physics step. Must pass the same tests.
- Update at the physics rate (240 Hz), not the control rate.

## Tests that must pass
- Pose tracking error ≤ 1 mm over 10 s at SS6.
- `getBaseVelocity` within 2 % of analytic deck-point velocity.
- Drone dropped on a static pad: exactly one touchdown event.

## Contact and touchdown
- Use `getContactPoints(drone, platform)`; filter out drone-ground contacts.
- Record first-contact state: relative velocity along deck normal, lateral offset in deck frame,
  relative tilt. Compute the same from analytic states; log disagreement.
- Penetration depth (`contactDistance` < 0) above ~5 mm indicates tunnelling: flag it.

## gym-pybullet-drones notes
- Use `DSLPIDControl` for velocity-setpoint tracking; keep `ctrl_freq` dividing `pyb_freq`.
- `DIRECT` for training (never GUI in workers); GUI or offscreen rendering only for GIFs.
- Remove the default ground-plane interaction from success logic; landing on the ground is `crash`.
