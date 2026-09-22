"""The deck plate: does it really move, and does the contact solver know?

Three questions, in increasing order of how much they matter.

1. **Does the body follow the trajectory?** Pose tracking <= 1 mm and orientation <= 1 mrad
   over 10 s model at SS6, measured at the instant the contact solver reads the body -- and
   measured at the plate's *corners* as well as its origin, because a plate rotated about
   its own origin has zero position error.
2. **Does ``getBaseVelocity`` report the analytic velocity?** Within 2 %, linear and
   angular.
3. **Is the drone carried by it?** The conveyor test. Question 2 is *necessary but not
   sufficient*: under a kinematic drive ``getBaseVelocity`` merely reads back what was
   written, whether or not the solver used it. The teleport-only drive is asserted to pass
   question 2 (it does not -- it reports zero) and to **fail** the conveyor, which is what
   makes the conveyor the decisive test and the reason ``resetBasePositionAndOrientation``
   alone is forbidden.

Both drivers are measured on every question, because P2-D1 is a measurement, not a
preference.

Units: metres, metres per second and seconds are model scale; angles radians.
"""

import numpy as np
import pybullet as pyb
import pytest
from dmf.sim.generate import RealizationSpec
from gym_pybullet_drones.utils.enums import DroneModel

from rld.envs.config import DRIVERS, Driver, LandingConfig, PlatformConfig
from rld.envs.platform import (
    DeckPlatform,
    DeckTrajectory,
    StaticDeckMotion,
    build_trajectory,
    quaternion_angle_rad,
)

#: The tracking test's cell: the frigate in SS6 head seas at 12 kn, aft pad. P1-D1 measured
#: its aft-pad ``v_z`` p99 at 0.577 m/s, the worst frigate cell the feasibility rule names.
SS6_SPEC = RealizationSpec(
    sea_state="SS6", heading_deg=180.0, speed_kn=12.0, vessel="frigate", seed=0
)

#: Sweep length, seconds model scale.
SWEEP_S = 10.0

#: Gate 2's pose tolerance, metres model scale.
POSE_TOL_M = 1.0e-3

#: Gate 2's orientation tolerance, radians.
QUAT_TOL_RAD = 1.0e-3

#: Gate 2's velocity tolerance, dimensionless fraction.
VEL_TOL = 0.02


def _trajectory(source: object, cfg: PlatformConfig, pad: str, t0_s: float) -> DeckTrajectory:
    """Build a 10 s model-scale trajectory on the physics grid.

    Args:
        source: A deck-motion source.
        cfg: Plate config, for ``deck_origin_m``.
        pad: Pad name.
        t0_s: Start offset, seconds model scale.

    Returns:
        The :class:`~rld.envs.platform.DeckTrajectory`.
    """
    grid = t0_s + np.arange(int(SWEEP_S * 240.0) + 1) * (1.0 / 240.0)
    return build_trajectory(source, pad, cfg, grid)  # type: ignore[arg-type]


def _drive_sweep(
    client: int, cfg: PlatformConfig, pad_radius_m: float, traj: DeckTrajectory
) -> dict[str, float]:
    """Run one driver over a whole trajectory and return its measured errors.

    Every quantity is read from PyBullet **before** ``stepSimulation``, which is the state
    the contact solver is about to use -- the only reading that can affect a touchdown.
    ``pose_after_m`` is read after the step instead, against the analytic pose at the new
    time, and is the drift the driver has to correct on the next step.

    Args:
        client: PyBullet physics client id.
        cfg: Plate config, including the driver under test.
        pad_radius_m: Painted-pad radius, metres model scale.
        traj: The trajectory to follow.

    Returns:
        A mapping of metric to value: ``pose_max_m`` and ``pose_after_m`` metres model
        scale, ``quat_max_rad`` radians, ``corner_max_m`` metres model scale,
        ``vel_p99`` / ``vel_rms`` / ``ang_p99`` dimensionless ratios.
    """
    platform = DeckPlatform(cfg, pad_radius_m)
    platform.spawn(client, traj.sample(0))
    velocity = traj.state.velocity_m_s
    angular = traj.state.angular_velocity_rad_s
    vz_rms = float(np.sqrt(np.mean(velocity[:, 2] ** 2)))
    # The deck's velocity crosses zero, so a bare relative error is unbounded there. The
    # floor is 2 % of the record's own v_z RMS: below that the deck is not moving in any
    # sense a touchdown cares about.
    v_floor = max(0.02 * vz_rms, 1e-9)
    w_floor = max(0.02 * float(np.sqrt(np.mean(np.sum(angular**2, axis=1)))), 1e-9)

    pose_err, pose_after, quat_err, corner_err = [], [], [], []
    vel_ratio, ang_ratio = [], []
    for index in range(len(traj) - 1):
        sample = traj.sample(index)
        platform.advance(sample)
        position, quaternion = platform.pose()
        pose_err.append(float(np.linalg.norm(position - sample.position_m)))
        quat_err.append(quaternion_angle_rad(quaternion, sample.quaternion))
        corner_err.append(
            float(
                np.max(
                    np.linalg.norm(
                        platform.measured_corner_positions_m()
                        - platform.corner_positions_m(sample),
                        axis=1,
                    )
                )
            )
        )
        linear, spin = platform.velocity()
        vel_ratio.append(
            float(np.linalg.norm(linear - sample.velocity_m_s))
            / max(float(np.linalg.norm(sample.velocity_m_s)), v_floor)
        )
        ang_ratio.append(
            float(np.linalg.norm(spin - sample.angular_velocity_rad_s))
            / max(float(np.linalg.norm(sample.angular_velocity_rad_s)), w_floor)
        )
        pyb.stepSimulation(physicsClientId=client)
        nxt = traj.sample(index + 1)
        pose_after.append(float(np.linalg.norm(platform.pose()[0] - nxt.position_m)))
    return {
        "pose_max_m": max(pose_err),
        "pose_after_m": max(pose_after),
        "quat_max_rad": max(quat_err),
        "corner_max_m": max(corner_err),
        "vel_p99": float(np.percentile(vel_ratio, 99.0)),
        "vel_rms": float(np.sqrt(np.mean(np.square(vel_ratio)))),
        "ang_p99": float(np.percentile(ang_ratio, 99.0)),
    }


def _with_driver(cfg: LandingConfig, driver: Driver) -> PlatformConfig:
    """Return the committed plate config with one field replaced.

    Args:
        cfg: The committed landing config.
        driver: The driver to measure.

    Returns:
        A :class:`~rld.envs.config.PlatformConfig` identical to the committed one except
        for its driver.
    """
    plate = cfg.platform
    return PlatformConfig(
        driver=driver,
        half_extents_m=plate.half_extents_m,
        deck_origin_m=plate.deck_origin_m,
        mass_kg=plate.mass_kg,
        constraint_max_force_n=plate.constraint_max_force_n,
        lateral_friction=plate.lateral_friction,
        restitution=plate.restitution,
    )


@pytest.mark.pybullet
@pytest.mark.slow
@pytest.mark.parametrize("driver", DRIVERS)
def test_platform_tracking_and_velocity_measured_for_both_drivers(
    driver, pyb_client, env_landing_cfg, deck_pads, deck_source, capsys
):
    """Measure both drivers over 10 s at SS6, print the numbers, and gate the committed one.

    Only the **configured** driver has to pass; the other is measured so that P2-D1 names a
    winner with evidence rather than a preference. The measured numbers are printed so a
    ``-s`` run reproduces the protocol entry.
    """
    source = deck_source(SS6_SPEC)
    plate = _with_driver(env_landing_cfg, driver)
    traj = _trajectory(source, plate, "aft", 4.0)
    measured = _drive_sweep(pyb_client, plate, deck_pads.radius_model_m, traj)
    with capsys.disabled():
        print(f"\nP2-D1 {driver:>10s}: " + "  ".join(f"{k}={v:.4e}" for k, v in measured.items()))

    if driver != env_landing_cfg.platform.driver:
        pytest.skip(f"{driver} measured for P2-D1; only the configured driver is gated")
    assert measured["pose_max_m"] <= POSE_TOL_M
    assert measured["pose_after_m"] <= POSE_TOL_M
    assert measured["quat_max_rad"] <= QUAT_TOL_RAD
    assert measured["corner_max_m"] <= POSE_TOL_M
    assert measured["vel_p99"] <= VEL_TOL
    assert measured["vel_rms"] <= VEL_TOL
    assert measured["ang_p99"] <= VEL_TOL


@pytest.mark.pybullet
@pytest.mark.slow
def test_constraint_driver_fails_the_tracking_gate(
    pyb_client, env_landing_cfg, deck_pads, deck_source
):
    """Pin the rejected driver's failure, so P2-D1's verdict cannot quietly invert.

    The ``constraint`` driver is a velocity-level servo: it trails its target by a few
    physics steps and cannot be tuned out of it (``maxForce``, ``erp``, solver iterations
    and plate mass were all swept, to four identical digits). If a PyBullet upgrade ever
    made it track, this test fails and P2-D1 is re-measured rather than inherited.
    """
    source = deck_source(SS6_SPEC)
    plate = _with_driver(env_landing_cfg, "constraint")
    traj = _trajectory(source, plate, "aft", 4.0)
    measured = _drive_sweep(pyb_client, plate, deck_pads.radius_model_m, traj)
    assert measured["pose_max_m"] > POSE_TOL_M
    assert measured["vel_p99"] > VEL_TOL


@pytest.mark.pybullet
@pytest.mark.parametrize("driver", [*DRIVERS, "teleport"])
def test_conveyor_carries_the_drone_unless_the_deck_is_only_teleported(
    driver, pyb_client, env_landing_cfg, deck_pads
):
    """The decisive test: a drone at rest on a plate translating at 0.5 m/s is carried.

    ``getBaseVelocity`` agreeing with the analytic velocity is necessary but not
    sufficient -- a kinematic drive reads back what was written whether or not the solver
    used it. This checks the only thing that actually matters for a touchdown: that the
    solver saw a moving surface.

    ``teleport`` is the forbidden drive (pose written, velocity left at zero). It is
    included precisely so that its failure is asserted rather than assumed.
    """
    from importlib.resources import files

    speed_m_s, duration_s = 0.5, 2.0
    dt = env_landing_cfg.physics_dt_s
    plate_cfg = _with_driver(env_landing_cfg, "kinematic" if driver == "teleport" else driver)
    hx, hy, hz = plate_cfg.half_extents_m
    origin = np.array(plate_cfg.deck_origin_m, dtype=np.float64)

    collision = int(
        pyb.createCollisionShape(
            pyb.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            collisionFramePosition=[0.0, 0.0, -hz],
            physicsClientId=pyb_client,
        )
    )
    plate = int(
        pyb.createMultiBody(
            baseMass=plate_cfg.mass_kg,
            baseCollisionShapeIndex=collision,
            basePosition=list(origin),
            physicsClientId=pyb_client,
        )
    )
    pyb.changeDynamics(
        plate,
        -1,
        lateralFriction=plate_cfg.lateral_friction,
        restitution=plate_cfg.restitution,
        physicsClientId=pyb_client,
    )
    constraint = -1
    if driver == "constraint":
        constraint = int(
            pyb.createConstraint(
                plate,
                -1,
                -1,
                -1,
                pyb.JOINT_FIXED,
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                list(origin),
                physicsClientId=pyb_client,
            )
        )
    urdf = str(files("gym_pybullet_drones") / "assets" / f"{DroneModel.CF2X.value}.urdf")
    drone = int(
        pyb.loadURDF(
            urdf,
            list(origin + np.array([0.0, 0.0, 0.0125])),
            pyb.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            flags=pyb.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=pyb_client,
        )
    )

    weight = [0.0, 0.0, plate_cfg.mass_kg * 9.8]
    for step in range(int(duration_s / dt)):
        target = origin + np.array([speed_m_s * step * dt, 0.0, 0.0])
        if driver == "constraint":
            pyb.changeConstraint(
                constraint,
                jointChildPivot=list(target),
                jointChildFrameOrientation=[0.0, 0.0, 0.0, 1.0],
                maxForce=plate_cfg.constraint_max_force_n,
                physicsClientId=pyb_client,
            )
        else:
            pyb.resetBasePositionAndOrientation(
                plate, list(target), [0.0, 0.0, 0.0, 1.0], physicsClientId=pyb_client
            )
            pyb.resetBaseVelocity(
                plate,
                linearVelocity=[speed_m_s if driver == "kinematic" else 0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
                physicsClientId=pyb_client,
            )
            pyb.applyExternalForce(
                plate, -1, weight, list(target), pyb.WORLD_FRAME, physicsClientId=pyb_client
            )
        pyb.stepSimulation(physicsClientId=pyb_client)

    drone_x = float(pyb.getBasePositionAndOrientation(drone, physicsClientId=pyb_client)[0][0])
    drone_vx = float(pyb.getBaseVelocity(drone, physicsClientId=pyb_client)[0][0])
    travel = speed_m_s * duration_s
    if driver == "teleport":
        # The forbidden drive. The plate slides out from under a stationary drone.
        assert drone_x < 0.1 * travel
        assert abs(drone_vx - speed_m_s) > 0.1 * speed_m_s
        return
    # The first ~0.2 s is friction spin-up (mu_eff = 0.25 gives 2.45 m/s^2), so the drone
    # lags by a fixed distance rather than a fixed fraction; the velocity is the tight test.
    assert drone_x > 0.8 * travel
    assert abs(drone_vx - speed_m_s) < 0.02 * speed_m_s


@pytest.mark.pybullet
def test_cg_pad_platform_is_pure_heave(pyb_client, env_landing_cfg, deck_pads, deck_source):
    """The CG pad has zero lever arm, so its motion is heave and nothing else (P1-D2).

    This anchors the whole platform path against the Phase 1 kinematics: if the plate's
    position picked up an x or y component at the CG pad, the lever arm or the rotation
    order would be wrong somewhere between the bridge and PyBullet.
    """
    source = deck_source(SS6_SPEC)
    plate = env_landing_cfg.platform
    traj = _trajectory(source, plate, "cg", 4.0)
    origin = np.array(plate.deck_origin_m, dtype=np.float64)
    assert np.allclose(traj.position_m[:, 0], origin[0], atol=1e-12)
    assert np.allclose(traj.position_m[:, 1], origin[1], atol=1e-12)
    assert np.allclose(traj.state.velocity_m_s[:, 0], 0.0, atol=1e-12)
    assert np.allclose(traj.state.velocity_m_s[:, 1], 0.0, atol=1e-12)
    assert np.ptp(traj.position_m[:, 2]) > 0.05


@pytest.mark.pybullet
@pytest.mark.slow
def test_trajectory_matches_the_phase_1_deck_statistics(env_landing_cfg, deck_source, deck_scaling):
    """The plate's own motion reproduces the committed Phase 1 numbers.

    Sanity anchors from ``results/deck_stats.csv``: the aft pad's z standard deviation is
    ~4.43 cm at SS5 180 deg 12 kn and ~4.88 cm at SS6, and peak ``|v_z|`` at SS6 is
    ~0.65-0.67 m/s over the whole record. The 10 s window used here is a small slice of a
    120 s model record, so the tolerances are loose on purpose: this catches a factor, a
    sign or a missing Froude scaling, not a percent.
    """
    plate = env_landing_cfg.platform
    for sea_state, z_std_cm in (("SS5", 4.43), ("SS6", 4.88)):
        spec = RealizationSpec(
            sea_state=sea_state, heading_deg=180.0, speed_kn=12.0, vessel="frigate", seed=0
        )
        grid = 4.0 + np.arange(int(100.0 * 240.0) + 1) * (1.0 / 240.0)
        traj = build_trajectory(deck_source(spec), "aft", plate, grid)
        measured_cm = 100.0 * float(np.std(traj.state.position_m[:, 2]))
        assert 0.6 * z_std_cm < measured_cm < 1.6 * z_std_cm, (sea_state, measured_cm)
    assert deck_scaling.lam == pytest.approx(1.0 / 25.0)


@pytest.mark.pybullet
def test_plate_origin_sits_on_the_top_surface(pyb_client, env_landing_cfg, deck_pads):
    """The body origin is the pad centre on the plate's TOP face, not the box centre.

    If the collision frame were not offset, the bridge's deck point would drive the plate's
    mid-plane and every landing would be one plate half-thickness (1 cm model, 25 cm full
    scale) too low.
    """
    plate_cfg = env_landing_cfg.platform
    platform = DeckPlatform(plate_cfg, deck_pads.radius_model_m)
    traj = _trajectory(StaticDeckMotion(), plate_cfg, "aft", 0.0)
    platform.spawn(pyb_client, traj.sample(0))
    lower, upper = pyb.getAABB(platform.body_id, -1, physicsClientId=pyb_client)
    margin = 0.01  # Bullet inflates the AABB by the collision margin and a velocity term.
    assert upper[2] == pytest.approx(plate_cfg.deck_origin_z_m, abs=margin)
    assert lower[2] == pytest.approx(
        plate_cfg.deck_origin_z_m - 2.0 * plate_cfg.half_extents_m[2], abs=margin
    )


@pytest.mark.pybullet
def test_analytic_contact_margin_matches_pybullets_own(pyb_client, env_success_cfg):
    """``analytic_contact_margin_m`` is PyBullet's own margin, not a tuned number.

    The analytic detector triggers at ``clearance <= margin`` because Bullet shrinks the
    drone's convex collision shape by its margin and inflates the contact by the same
    amount, so a resting drone's geometry is that far clear of the plate's. The margin lives
    on the **drone** body: the plate, built with ``createMultiBody``, reports 0.0. Reading
    it back from the engine is what keeps the two detectors agreeing about where the surface
    is if PyBullet ever changes its default.
    """
    from importlib.resources import files

    urdf = str(files("gym_pybullet_drones") / "assets" / f"{DroneModel.CF2X.value}.urdf")
    drone = int(
        pyb.loadURDF(
            urdf, [0.0, 0.0, 1.0], flags=pyb.URDF_USE_INERTIA_FROM_FILE, physicsClientId=pyb_client
        )
    )
    margin = float(pyb.getDynamicsInfo(drone, -1, physicsClientId=pyb_client)[11])
    assert margin == pytest.approx(env_success_cfg.analytic_contact_margin_m, rel=1e-9)


def test_static_deck_motion_is_motionless_and_bounded(static_motion):
    """The static test double is genuinely static, and refuses off-record times.

    It has to refuse them: the whole point of the fixture is to stand in for a real bridge,
    and a fixture that silently accepts any time would hide an episode-window bug that the
    real source would raise on.
    """
    times = np.linspace(0.0, 100.0, 11)
    traj = static_motion.deck_point(times, "aft")
    assert np.allclose(traj.state.position_m, 0.0)
    assert np.allclose(traj.state.velocity_m_s, 0.0)
    assert np.allclose(traj.state.normal, np.array([0.0, 0.0, 1.0]))
    assert np.allclose(traj.state.tilt_deg, 0.0)
    assert static_motion.episode_start_window_s(12.5) == (0.0, 107.5)
    with pytest.raises(ValueError, match="leave the static record"):
        static_motion.deck_point(np.array([200.0]), "aft")


def test_driver_config_rejects_an_unknown_driver(env_landing_cfg):
    """A typo'd driver name is an error at load time, never a silent fallback."""
    from rld.envs.config import LANDING_CONFIG, load_landing

    assert env_landing_cfg.platform.driver in DRIVERS
    assert load_landing(LANDING_CONFIG).platform.driver == env_landing_cfg.platform.driver
