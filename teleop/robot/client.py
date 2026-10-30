from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from teleop.control_math import TcpPose


class RobotClientError(RuntimeError):
    def __init__(self, operation: str, code: int | None, detail: str = "") -> None:
        self.operation = operation
        self.code = code
        self.detail = detail
        message = f"{operation} failed"
        if code is not None:
            message += f" with code {code}"
        if detail:
            message += f": {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class GripperMotionState:
    fault: int
    done: bool


@dataclass(frozen=True)
class RobotStateSnapshot:
    """Controller-pushed robot state read from a local cache, never an RPC."""

    joint_pos_deg: tuple[float, float, float, float, float, float]
    joint_vel_deg_s: tuple[float, float, float, float, float, float]
    tcp: TcpPose
    gripper_position: float
    frame_cnt: int
    robot_time_ms: int


class RobotClient(Protocol):
    def connect(self) -> None: ...

    def initialize(self) -> None: ...

    def get_current_tcp(self) -> TcpPose: ...

    def get_current_joints(self) -> tuple[float, float, float, float, float, float]: ...

    def get_state_snapshot(self) -> RobotStateSnapshot | None: ...

    def servo_start(self) -> None: ...

    def servo_cart(self, target: TcpPose) -> None: ...

    def servo_end(self) -> None: ...

    def activate_gripper(self) -> None: ...

    def move_gripper(self, position: int) -> None: ...

    def get_gripper_motion_state(self) -> GripperMotionState: ...

    def reset_fault(self) -> None: ...

    def close(self) -> None: ...
