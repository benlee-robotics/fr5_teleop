from __future__ import annotations

import time
from dataclasses import dataclass

from teleop.control_math import TcpPose
from teleop.robot.client import (
    GripperMotionState,
    RobotClientError,
    RobotStateSnapshot,
)


FAKE_JOINTS = (0.0, -90.0, 90.0, -90.0, 90.0, 0.0)

@dataclass(frozen=True)
class FakeRobotBehavior:
    fail_on: str | None = None
    fail_code: int = 99
    hang_on: str | None = None
    hang_seconds: float = 0.0
    gripper_motion_s: float = 0.0
    gripper_fault: int = 0
    # Reported joint angles; FAKE_JOINTS when None.
    joints_deg: tuple[float, float, float, float, float, float] | None = None


class FakeRobotClient:
    def __init__(self, behavior: FakeRobotBehavior | None = None) -> None:
        self.behavior = behavior or FakeRobotBehavior()
        self.connected = False
        self.servo_active = False
        self.current_tcp: TcpPose = (300.0, 0.0, 400.0, 180.0, 0.0, 0.0)
        self.gripper_active = False
        self.gripper_position = 0
        self._gripper_done_ns = 0
        self._frame_cnt = 0
        self.joints = self.behavior.joints_deg or FAKE_JOINTS

    def _before(self, operation: str) -> None:
        if self.behavior.hang_on == operation:
            time.sleep(self.behavior.hang_seconds)
        if self.behavior.fail_on == operation:
            raise RobotClientError(operation, self.behavior.fail_code, "injected failure")

    def connect(self) -> None:
        self._before("connect")
        self.connected = True

    def initialize(self) -> None:
        self._before("initialize")
        if not self.connected:
            raise RobotClientError("initialize", None, "not connected")

    def get_current_tcp(self) -> TcpPose:
        self._before("get_current_tcp")
        return self.current_tcp

    def get_current_joints(self) -> tuple[float, float, float, float, float, float]:
        self._before("get_current_joints")
        return self.joints

    def get_state_snapshot(self) -> RobotStateSnapshot | None:
        self._before("get_state_snapshot")
        self._frame_cnt = (self._frame_cnt + 1) % 256
        return RobotStateSnapshot(
            joint_pos_deg=self.joints,
            joint_vel_deg_s=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            tcp=self.current_tcp,
            gripper_position=float(self.gripper_position),
            frame_cnt=self._frame_cnt,
            robot_time_ms=time.monotonic_ns() // 1_000_000,
        )

    def servo_start(self) -> None:
        self._before("servo_start")
        self.servo_active = True

    def servo_cart(self, target: TcpPose) -> None:
        self._before("servo_cart")
        if not self.servo_active:
            raise RobotClientError("servo_cart", None, "servo is not active")
        self.current_tcp = tuple(target)

    def servo_end(self) -> None:
        self._before("servo_end")
        self.servo_active = False

    def activate_gripper(self) -> None:
        self._before("activate_gripper")
        self.gripper_active = True

    def move_gripper(self, position: int) -> None:
        self._before("move_gripper")
        if self.servo_active:
            raise RobotClientError(
                "move_gripper",
                None,
                "servo must be stopped before gripper movement",
            )
        self.gripper_position = position
        self._gripper_done_ns = (
            time.monotonic_ns()
            + int(self.behavior.gripper_motion_s * 1_000_000_000)
        )

    def get_gripper_motion_state(self) -> GripperMotionState:
        self._before("get_gripper_motion_state")
        return GripperMotionState(
            fault=self.behavior.gripper_fault,
            done=time.monotonic_ns() >= self._gripper_done_ns,
        )

    def reset_fault(self) -> None:
        self._before("reset_fault")

    def close(self) -> None:
        self._before("close")
        self.connected = False
