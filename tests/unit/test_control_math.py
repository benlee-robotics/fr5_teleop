import math
from dataclasses import replace

import pytest

from teleop.config import OrientationConfig, TeleopConfig, WorkspaceBounds
from teleop.control_math import (
    MIN_JERK_PEAK_FACTOR,
    HomeTrajectory,
    MotionPlanner,
    Quaternion,
    axis_angle_to_quat,
    clamp_angular_step,
    clamp_orientation_deviation,
    clamp_step,
    compose_delta_orientation,
    euler_deg_to_quat,
    limit_home_step,
    min_jerk,
    quat_angle_deg,
    quat_axis_angle,
    quat_conjugate,
    quat_delta,
    quat_dot,
    quat_multiply,
    quat_normalize,
    quat_slerp,
    quat_to_euler_deg,
    remap_axis_vr_to_robot,
    remap_quat_vr_to_robot,
    vr_to_robot_delta,
)


IDENTITY_Q: Quaternion = (0.0, 0.0, 0.0, 1.0)


def _web_config(tmp_path, **overrides) -> TeleopConfig:
    web_dir = tmp_path / "web"
    web_dir.mkdir(exist_ok=True)
    web_dir.joinpath("index.html").write_text("", encoding="utf-8")
    web_dir.joinpath("monitor.html").write_text("", encoding="utf-8")
    return TeleopConfig(
        web_dir=web_dir,
        tls_cert_path=None,
        tls_key_path=None,
        **overrides,
    )


def test_vr_axis_mapping() -> None:
    assert vr_to_robot_delta((1.0, 2.0, 3.0), 500.0) == (
        1500.0,
        500.0,
        1000.0,
    )


def test_remap_axis_vr_to_robot_matches_position_mapping() -> None:
    assert remap_axis_vr_to_robot((1.0, 2.0, 3.0)) == (3.0, 1.0, 2.0)


def test_step_is_limited_by_vector_length() -> None:
    assert clamp_step((3.0, 4.0, 0.0), 2.5) == pytest.approx((1.5, 2.0, 0.0))


def test_engage_resets_previous_target_and_clamps_workspace(tmp_path) -> None:
    config = _web_config(
        tmp_path,
        ema_alpha=1.0,
        max_step_mm=1000.0,
        workspace=WorkspaceBounds(x=(-10, 310), y=(-20, 20), z=(50, 450)),
    )
    planner = MotionPlanner(config)
    origin = (300.0, 0.0, 400.0, 180.0, 0.0, 0.0)
    planner.engage((0.0, 0.0, 0.0), IDENTITY_Q, origin)
    assert planner.target_for((-1.0, 1.0, 0.0), IDENTITY_Q) == (
        300,
        -20,
        450,
        180,
        0,
        0,
    )
    planner.engage((5.0, 5.0, 5.0), IDENTITY_Q, origin)
    assert planner.last_target == origin


def test_target_accepts_time_based_step_below_absolute_cap(tmp_path) -> None:
    config = _web_config(
        tmp_path,
        ema_alpha=1.0,
        max_velocity_mm_s=50.0,
        max_step_mm=1.0,
    )
    planner = MotionPlanner(config)
    origin = (300.0, 0.0, 400.0, 180.0, 0.0, 0.0)
    planner.engage((0.0, 0.0, 0.0), IDENTITY_Q, origin)

    target = planner.target_for(
        (0.0, 0.0, 1.0),
        IDENTITY_Q,
        step_limit_mm=0.4,
    )

    assert target[:3] == pytest.approx((300.4, 0.0, 400.0))


def test_target_rejects_step_above_absolute_cap(tmp_path) -> None:
    config = _web_config(tmp_path, max_velocity_mm_s=50.0, max_step_mm=1.0)
    planner = MotionPlanner(config)
    planner.engage(
        (0.0, 0.0, 0.0),
        IDENTITY_Q,
        (300.0, 0.0, 400.0, 180.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="within max_step_mm"):
        planner.target_for((0.0, 0.0, -1.0), IDENTITY_Q, step_limit_mm=1.1)


def test_invalid_timing_config_is_rejected(tmp_path) -> None:
    config = replace(_web_config(tmp_path), pose_timeout_s=0.001)
    with pytest.raises(ValueError):
        config.validate()


# --- Quaternion primitives -------------------------------------------


def test_quat_multiply_identity() -> None:
    q = (0.1, 0.2, 0.3, math.sqrt(1 - 0.14))
    assert quat_multiply(IDENTITY_Q, q) == pytest.approx(q)
    assert quat_multiply(q, IDENTITY_Q) == pytest.approx(q)


def test_quat_multiply_composes_two_90_degree_rotations() -> None:
    # 90 deg about X then 90 deg about X == 180 deg about X.
    q90 = axis_angle_to_quat((1.0, 0.0, 0.0), math.radians(90))
    q180 = quat_multiply(q90, q90)
    axis, angle = quat_axis_angle(q180)
    assert math.degrees(angle) == pytest.approx(180.0)
    assert axis == pytest.approx((1.0, 0.0, 0.0))


def test_quat_conjugate_is_inverse_for_unit_quaternion() -> None:
    q = quat_normalize((0.1, 0.2, 0.3, 0.9))
    result = quat_multiply(q, quat_conjugate(q))
    assert result == pytest.approx(IDENTITY_Q)


def test_quat_delta_round_trip() -> None:
    origin = axis_angle_to_quat((0.0, 1.0, 0.0), math.radians(20))
    current = axis_angle_to_quat((0.0, 1.0, 0.0), math.radians(50))
    delta = quat_delta(origin, current)
    rebuilt = quat_normalize(quat_multiply(delta, origin))
    dot = abs(quat_dot(rebuilt, current))
    assert dot == pytest.approx(1.0, abs=1e-6)


def test_quat_slerp_endpoints_and_midpoint() -> None:
    a = axis_angle_to_quat((0.0, 0.0, 1.0), 0.0)
    b = axis_angle_to_quat((0.0, 0.0, 1.0), math.radians(90))
    assert quat_slerp(a, b, 0.0) == pytest.approx(a)
    assert abs(quat_dot(quat_slerp(a, b, 1.0), b)) == pytest.approx(1.0, abs=1e-6)
    mid = quat_slerp(a, b, 0.5)
    _, angle = quat_axis_angle(quat_delta(a, mid))
    assert math.degrees(angle) == pytest.approx(45.0, abs=1e-4)


def test_euler_round_trip_simple_cases() -> None:
    for rx, ry, rz in [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, 20.0, 0.0), (0.0, 0.0, -40.0), (10.0, -15.0, 25.0)]:
        q = euler_deg_to_quat(rx, ry, rz)
        out_rx, out_ry, out_rz = quat_to_euler_deg(q)
        assert (out_rx, out_ry, out_rz) == pytest.approx((rx, ry, rz), abs=1e-6)


def test_euler_round_trip_gimbal_lock_is_finite_and_quaternion_equivalent() -> None:
    q = euler_deg_to_quat(10.0, 90.0, 20.0)
    rx, ry, rz = quat_to_euler_deg(q)
    assert all(math.isfinite(v) for v in (rx, ry, rz))
    # Re-encoding the extracted Euler angles must reproduce the same rotation
    # even though the individual rx/rz split is not unique at ry=90.
    rebuilt = euler_deg_to_quat(rx, ry, rz)
    assert abs(quat_dot(rebuilt, q)) == pytest.approx(1.0, abs=1e-6)


def test_remap_quat_vr_to_robot_preserves_angle_and_swaps_axis() -> None:
    q = axis_angle_to_quat((0.0, 1.0, 0.0), math.radians(30))  # VR "up" axis
    remapped = remap_quat_vr_to_robot(q)
    axis, angle = quat_axis_angle(remapped)
    assert math.degrees(angle) == pytest.approx(30.0)
    assert axis == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)


def test_clamp_angular_step_limits_large_rotation() -> None:
    current = IDENTITY_Q
    desired = axis_angle_to_quat((1.0, 0.0, 0.0), math.radians(10))
    clamped = clamp_angular_step(current, desired, 2.0)
    _, angle = quat_axis_angle(quat_delta(current, clamped))
    assert math.degrees(angle) == pytest.approx(2.0, abs=1e-6)


def test_clamp_angular_step_passes_through_small_rotation() -> None:
    current = IDENTITY_Q
    desired = axis_angle_to_quat((1.0, 0.0, 0.0), math.radians(0.5))
    assert clamp_angular_step(current, desired, 2.0) == pytest.approx(desired)


def test_clamp_orientation_deviation_bounds_total_offset() -> None:
    origin = IDENTITY_Q
    desired = axis_angle_to_quat((0.0, 0.0, 1.0), math.radians(45))
    clamped = clamp_orientation_deviation(origin, desired, 20.0)
    _, angle = quat_axis_angle(quat_delta(origin, clamped))
    assert math.degrees(angle) == pytest.approx(20.0, abs=1e-6)


def test_compose_delta_orientation_world_frame() -> None:
    origin = axis_angle_to_quat((0.0, 0.0, 1.0), math.radians(10))
    delta = axis_angle_to_quat((0.0, 0.0, 1.0), math.radians(5))
    composed = compose_delta_orientation(origin, delta)
    _, angle = quat_axis_angle(composed)
    assert math.degrees(angle) == pytest.approx(15.0)


# --- MotionPlanner orientation behavior --------------------------------


def test_orientation_disabled_preserves_existing_position_only_behavior(tmp_path) -> None:
    config = _web_config(
        tmp_path,
        ema_alpha=1.0,
        max_step_mm=1000.0,
        workspace=WorkspaceBounds(x=(-10, 310), y=(-20, 20), z=(50, 450)),
    )
    assert config.orientation.enabled is False
    planner = MotionPlanner(config)
    origin = (300.0, 0.0, 400.0, 180.0, 0.0, 0.0)
    non_identity_orientation = axis_angle_to_quat((0.0, 1.0, 0.0), math.radians(45))
    planner.engage((0.0, 0.0, 0.0), non_identity_orientation, origin)
    target = planner.target_for(
        (-1.0, 1.0, 0.0), axis_angle_to_quat((1.0, 0.0, 0.0), math.radians(30))
    )
    assert target == (300, -20, 450, 180, 0, 0)


def test_orientation_enabled_zero_rotation_matches_origin(tmp_path) -> None:
    config = _web_config(
        tmp_path,
        orientation=OrientationConfig(enabled=True, ema_alpha=1.0),
    )
    planner = MotionPlanner(config)
    origin = (300.0, 0.0, 400.0, 12.0, -5.0, 8.0)
    vr_orientation = axis_angle_to_quat((0.0, 1.0, 0.0), math.radians(15))
    planner.engage((0.0, 0.0, 0.0), vr_orientation, origin)
    target = planner.target_for((0.0, 0.0, 0.0), vr_orientation)
    assert target[3:] == pytest.approx((12.0, -5.0, 8.0), abs=1e-6)


def test_orientation_enabled_single_axis_rotation_is_remapped(tmp_path) -> None:
    config = _web_config(
        tmp_path,
        orientation=OrientationConfig(
            enabled=True, ema_alpha=1.0, max_step_deg=90.0, max_deviation_deg=90.0
        ),
    )
    planner = MotionPlanner(config)
    origin = (300.0, 0.0, 400.0, 0.0, 0.0, 0.0)
    vr_origin_q = IDENTITY_Q
    planner.engage((0.0, 0.0, 0.0), vr_origin_q, origin)

    vr_rotated_q = axis_angle_to_quat((0.0, 1.0, 0.0), math.radians(20))
    target = planner.target_for(
        (0.0, 0.0, 0.0),
        vr_rotated_q,
        angular_step_limit_deg=90.0,
    )
    target_q = euler_deg_to_quat(*target[3:])
    axis, angle = quat_axis_angle(target_q)
    assert math.degrees(angle) == pytest.approx(20.0, abs=1e-3)
    assert axis == pytest.approx((0.0, 0.0, 1.0), abs=1e-3)


def test_orientation_angular_step_clamp_limits_single_tick(tmp_path) -> None:
    config = _web_config(
        tmp_path,
        orientation=OrientationConfig(
            enabled=True, ema_alpha=1.0, max_step_deg=1.0, max_deviation_deg=90.0
        ),
    )
    planner = MotionPlanner(config)
    origin = (300.0, 0.0, 400.0, 0.0, 0.0, 0.0)
    planner.engage((0.0, 0.0, 0.0), IDENTITY_Q, origin)

    big_rotation = axis_angle_to_quat((0.0, 1.0, 0.0), math.radians(45))
    target = planner.target_for((0.0, 0.0, 0.0), big_rotation, angular_step_limit_deg=1.0)
    target_q = euler_deg_to_quat(*target[3:])
    _, angle = quat_axis_angle(quat_delta(euler_deg_to_quat(0, 0, 0), target_q))
    assert math.degrees(angle) <= 1.0 + 1e-6


def test_orientation_deviation_clamp_bounds_sustained_rotation(tmp_path) -> None:
    config = _web_config(
        tmp_path,
        orientation=OrientationConfig(
            enabled=True,
            ema_alpha=1.0,
            max_step_deg=45.0,
            max_angular_velocity_deg_s=1000.0,
            max_deviation_deg=10.0,
        ),
    )
    planner = MotionPlanner(config)
    origin = (300.0, 0.0, 400.0, 0.0, 0.0, 0.0)
    planner.engage((0.0, 0.0, 0.0), IDENTITY_Q, origin)

    held_rotation = axis_angle_to_quat((0.0, 1.0, 0.0), math.radians(45))
    target = None
    for _ in range(50):
        target = planner.target_for(
            (0.0, 0.0, 0.0), held_rotation, angular_step_limit_deg=45.0
        )
    target_q = euler_deg_to_quat(*target[3:])
    _, angle = quat_axis_angle(quat_delta(euler_deg_to_quat(0, 0, 0), target_q))
    assert math.degrees(angle) <= 10.0 + 1e-3


def test_orientation_handles_near_180_degree_rotation(tmp_path) -> None:
    config = _web_config(
        tmp_path,
        orientation=OrientationConfig(enabled=True, ema_alpha=1.0, max_step_deg=180.0, max_deviation_deg=90.0),
    )
    planner = MotionPlanner(config)
    origin = (300.0, 0.0, 400.0, 0.0, 0.0, 0.0)
    planner.engage((0.0, 0.0, 0.0), IDENTITY_Q, origin)

    near_flip = axis_angle_to_quat((1.0, 0.0, 0.0), math.radians(179.9))
    target = planner.target_for((0.0, 0.0, 0.0), near_flip, angular_step_limit_deg=180.0)
    assert all(math.isfinite(v) for v in target)


# --- Home return ------------------------------------------------------


def test_min_jerk_profile_is_smooth_and_bounded() -> None:
    assert min_jerk(0.0) == 0.0 and min_jerk(1.0) == 1.0
    assert min_jerk(-1.0) == 0.0 and min_jerk(2.0) == 1.0
    assert min_jerk(0.5) == pytest.approx(0.5)
    # Peak slope (at s = 0.5) is the factor used to size the duration.
    h = 1e-6
    slope = (min_jerk(0.5 + h) - min_jerk(0.5 - h)) / (2 * h)
    assert slope == pytest.approx(MIN_JERK_PEAK_FACTOR)


def sampled_speeds(trajectory: HomeTrajectory, dt: float = 0.001):
    steps = int(trajectory.duration_s / dt) + 2
    previous = trajectory.sample(0.0)
    max_mm_s = max_deg_s = 0.0
    for k in range(1, steps):
        current = trajectory.sample(k * dt)
        distance = math.dist(current[0], previous[0])
        angle = quat_angle_deg(previous[1], current[1])
        max_mm_s = max(max_mm_s, distance / dt)
        max_deg_s = max(max_deg_s, angle / dt)
        previous = current
    return max_mm_s, max_deg_s


def test_home_trajectory_respects_peak_speeds_and_ends_at_goal() -> None:
    start = (300.0, 0.0, 400.0, 180.0, 0.0, 0.0)
    goal = (400.0, -100.0, 500.0, 180.0, 0.0, 30.0)
    trajectory = HomeTrajectory.plan(
        start, goal, max_velocity_mm_s=100.0, max_angular_velocity_deg_s=30.0
    )
    distance = math.dist(start[:3], goal[:3])
    # Position dominates: 173 mm at 100 mm/s peak.
    assert trajectory.duration_s == pytest.approx(MIN_JERK_PEAK_FACTOR * distance / 100.0)
    max_mm_s, max_deg_s = sampled_speeds(trajectory)
    assert max_mm_s <= 100.0 * 1.001
    assert max_mm_s >= 99.0
    assert max_deg_s <= 30.0 * 1.001
    position, orientation = trajectory.sample(trajectory.duration_s + 1.0)
    assert position == pytest.approx(goal[:3])
    assert quat_angle_deg(orientation, euler_deg_to_quat(*goal[3:])) < 1e-6
    assert trajectory.sample(0.0)[0] == pytest.approx(start[:3])


def test_home_trajectory_rotation_takes_the_short_way_across_180() -> None:
    # rx = 179 to rx = -179 is a 2 deg turn, not 358 deg.
    start = (300.0, 0.0, 400.0, 179.0, 0.0, 0.0)
    goal = (300.0, 0.0, 400.0, -179.0, 0.0, 0.0)
    trajectory = HomeTrajectory.plan(
        start, goal, max_velocity_mm_s=100.0, max_angular_velocity_deg_s=30.0
    )
    assert trajectory.duration_s == pytest.approx(MIN_JERK_PEAK_FACTOR * 2.0 / 30.0)
    start_q = euler_deg_to_quat(*start[3:])
    for k in range(11):
        _, orientation = trajectory.sample(trajectory.duration_s * k / 10)
        assert quat_angle_deg(start_q, orientation) <= 2.0 + 1e-6


def test_home_trajectory_already_at_goal_has_zero_duration() -> None:
    pose = (300.0, 0.0, 400.0, 180.0, 0.0, 0.0)
    trajectory = HomeTrajectory.plan(
        pose, pose, max_velocity_mm_s=100.0, max_angular_velocity_deg_s=30.0
    )
    assert trajectory.duration_s == pytest.approx(0.0, abs=1e-6)
    assert trajectory.sample(0.0)[0] == pytest.approx(pose[:3])


def test_limit_home_step_applies_step_and_workspace_limits() -> None:
    config = TeleopConfig(workspace=WorkspaceBounds(z=(50.0, 450.0)))
    previous_q = euler_deg_to_quat(180.0, 0.0, 0.0)
    target, orientation = limit_home_step(
        (300.0, 0.0, 449.0),
        previous_q,
        (300.0, 0.0, 600.0),
        euler_deg_to_quat(180.0, 0.0, 10.0),
        step_limit_mm=5.0,
        angular_step_limit_deg=1.0,
        config=config,
    )
    # 5 mm step up, then clamped to the workspace ceiling.
    assert target[2] == pytest.approx(450.0)
    assert quat_angle_deg(previous_q, orientation) == pytest.approx(1.0)
