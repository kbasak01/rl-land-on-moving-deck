---
name: pybullet-moving-platform
description: Hard-won guidance for moving landing platforms, contact and touchdown detection in PyBullet and gym-pybullet-drones. Use whenever writing or debugging the platform body, constraints, contact points, collision filtering, the landing environment's step loop, or touchdown classification.
---

# Moving platform in PyBullet

## Driving the platform
- Do **not** call `resetBasePositionAndOrientation` every step on its own: the solver sees a
  zero-velocity body, so relative touchdown velocity is wrong and the drone can tunnel.
- **Preferred (measured, P2-D1): `kinematic`.** Dynamic box (1 000 kg) with its weight cancelled
  by `applyExternalForce(..., WORLD_FRAME, posObj=<the body's world CoM>)`, then each **physics**
  step `resetBasePositionAndOrientation` **and** `resetBaseVelocity(linear, angular)` from the
  analytic deck state. Pose error 0.0 mm, `getBaseVelocity` error 0.0 %.
  - The weight cancellation is not optional: PyBullet integrates gravity into the body's velocity
    *before* the solver runs, so without it the plate enters every contact solve at `v - g*dt`
    (40.8 mm/s, ~7 % of the SS6 deck's own peak |v_z|).
  - `WORLD_FRAME` with `posObj=[0,0,0]` is **not** the same as "at the CoM": it applies the force
    at the world origin and adds a large spurious torque.
- **Rejected (measured): `constraint`.** `createConstraint(body, -1, -1, -1, JOINT_FIXED, ...)` to
  world plus `changeConstraint(cid, jointChildPivot=p, jointChildFrameOrientation=q, maxForce=1e6)`
  every physics step. It is a velocity-level servo and trails its target by a few steps: 11.4 mm
  pose error, 14.8 mm at the plate corners, 8.5 mrad, `getBaseVelocity` p99 error 416 %. **It
  cannot be tuned out** -- `maxForce`, `erp`, `numSolverIterations` and plate mass were all swept
  and returned four identical digits. (This file previously called it "Preferred"; it is not.)
- Update at the physics rate (240 Hz), not the control rate.

## Tests that must pass
- Pose tracking error ≤ 1 mm over 10 s at SS6, measured **before** `stepSimulation` (the state the
  solver is about to use) and at the plate's **corners** as well as its origin -- a plate rotated
  about its own origin has zero origin error.
- `getBaseVelocity` within 2 % of analytic deck-point velocity. **Necessary but not sufficient**:
  under a kinematic drive it reads back what was written, whether or not the solver used it.
- **The conveyor test is the decisive one.** A drone resting on a plate translating at 0.5 m/s must
  be carried along; a teleport-only drive must be asserted to *fail* it (measured: the drone stays
  put and slides off, v_x = -0.04 m/s against the plate's +0.50 m/s).
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
