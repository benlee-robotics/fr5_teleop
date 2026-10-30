from __future__ import annotations

import ctypes
import importlib
import inspect
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

from teleop.config import TeleopConfig
from teleop.control_math import TcpPose
from teleop.robot.client import (
    GripperMotionState,
    RobotClientError,
    RobotStateSnapshot,
)


log = logging.getLogger(__name__)


class FairinoRobotClient:
    def __init__(self, config: TeleopConfig) -> None:
        if not config.robot_ip:
            raise ValueError("robot_ip is required for FairinoRobotClient")
        self.config = config
        self._robot_module: Any = None
        self._robot: Any = None
        self._servo_cart_supports_exaxis: bool | None = None
        self._move_gripper_supports_extended_args: bool | None = None

    @staticmethod
    def _code(result: Any, operation: str) -> int:
        value = result[0] if isinstance(result, (tuple, list)) and result else result
        if isinstance(value, bool) or not isinstance(value, int):
            raise RobotClientError(
                operation, None, f"unexpected SDK result: {result!r}"
            )
        return value

    @classmethod
    def _expect_zero(cls, result: Any, operation: str) -> None:
        code = cls._code(result, operation)
        if code != 0:
            raise RobotClientError(operation, code)

    def _call_zero(self, operation: str, *args: Any, **kwargs: Any) -> None:
        if self._robot is None:
            raise RobotClientError(operation, None, "robot is not connected")
        method = getattr(self._robot, operation)
        self._expect_zero(method(*args, **kwargs), operation)

    def _call_allow(
        self, operation: str, allowed_codes: set[int], *args: Any, **kwargs: Any
    ) -> None:
        if self._robot is None:
            raise RobotClientError(operation, None, "robot is not connected")
        method = getattr(self._robot, operation)
        code = self._code(method(*args, **kwargs), operation)
        if code != 0 and code not in allowed_codes:
            raise RobotClientError(operation, code)
        if code != 0:
            log.warning(
                "%s returned code %d (permitted allowed code)", operation, code
            )

    def _method_parameters(self, operation: str) -> set[str]:
        if self._robot is None:
            raise RobotClientError(operation, None, "robot is not connected")
        method = getattr(self._robot, operation, None)
        if method is None:
            raise RobotClientError(operation, None, "SDK method is unavailable")
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError) as exc:
            raise RobotClientError(
                operation,
                None,
                "cannot inspect SDK signature safely",
            ) from exc
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            raise RobotClientError(
                operation,
                None,
                "ambiguous SDK **kwargs signature",
            )
        log.info("Fairino SDK signature operation=%s signature=%s", operation, signature)
        return set(signature.parameters)

    def _inspect_sdk_signatures(self) -> None:
        servo_parameters = self._method_parameters("ServoCart")
        servo_required = {
            "mode",
            "desc_pos",
            "pos_gain",
            "acc",
            "vel",
            "cmdT",
            "filterT",
            "gain",
        }
        missing_servo = servo_required - servo_parameters
        if missing_servo:
            raise RobotClientError(
                "ServoCart",
                None,
                "unsupported SDK signature; missing parameters: "
                + ", ".join(sorted(missing_servo)),
            )
        self._servo_cart_supports_exaxis = "exaxis" in servo_parameters

        if not self.config.gripper.enabled:
            return
        gripper_parameters = self._method_parameters("MoveGripper")
        gripper_required = {
            "index",
            "pos",
            "vel",
            "force",
            "maxtime",
            "block",
        }
        missing_gripper = gripper_required - gripper_parameters
        if missing_gripper:
            raise RobotClientError(
                "MoveGripper",
                None,
                "unsupported SDK signature; missing parameters: "
                + ", ".join(sorted(missing_gripper)),
            )
        extended = {"type", "rotNum", "rotVel", "rotTorque"}
        present = extended.intersection(gripper_parameters)
        if present and present != extended:
            raise RobotClientError(
                "MoveGripper",
                None,
                "partially supported extended gripper signature",
            )
        self._move_gripper_supports_extended_args = present == extended
        self._method_parameters("GetGripperMotionDone")
        if self.config.gripper.activate_on_start:
            self._method_parameters("ActGripper")

    @staticmethod
    def _connection_state(rpc_type: Any) -> bool | None:
        states = [
            bool(getattr(rpc_type, name))
            for name in ("is_connect", "is_conect")
            if hasattr(rpc_type, name)
        ]
        return any(states) if states else None

    def connect(self) -> None:
        if platform.system() != "Linux":
            raise RobotClientError("connect", None, "Fairino target requires Linux")
        if platform.machine() not in {"x86_64", "AMD64"}:
            raise RobotClientError(
                "connect", None, f"unsupported architecture: {platform.machine()}"
            )

        sdk_path = Path(self.config.sdk_path).resolve()
        robot_source = sdk_path / "fairino" / "Robot.py"
        if not robot_source.is_file():
            raise RobotClientError("connect", None, f"SDK not found: {robot_source}")
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))

        self._robot_module = importlib.import_module("fairino.Robot")
        self._robot = self._robot_module.RPC(self.config.robot_ip)
        self._inspect_sdk_signatures()
        connection_state = self._connection_state(self._robot_module.RPC)
        if connection_state is False:
            raise RobotClientError(
                "connect",
                -4,
                "SDK connection flag is false (is_connect/is_conect)",
            )

        version_result = self._robot.GetSDKVersion()
        self._expect_zero(version_result, "GetSDKVersion")
        versions = version_result[1] if isinstance(version_result, tuple) else None
        if connection_state is None:
            log.warning(
                "Fairino SDK exposes no connection flag; GetSDKVersion succeeded"
            )
        log.info("Fairino connection ready: ip=%s versions=%s", self.config.robot_ip, versions)

    def initialize(self) -> None:
        self._call_allow("ResetAllError", {14})
        self._call_allow("Mode", {14}, 0)
        self._call_allow("DragTeachSwitch", {14}, 0)
        self._call_allow("RobotEnable", {14}, 1)
        self._call_allow("SetSpeed", {14}, 100)
        if self.config.gripper.enabled and self.config.gripper.activate_on_start:
            self.activate_gripper()

    def get_current_tcp(self) -> TcpPose:
        if self._robot is None:
            raise RobotClientError("GetActualTCPPose", None, "robot is not connected")
        result = self._robot.GetActualTCPPose()
        code = self._code(result, "GetActualTCPPose")
        if code != 0:
            raise RobotClientError("GetActualTCPPose", code)
        if (
            not isinstance(result, (tuple, list))
            or len(result) < 2
            or not isinstance(result[1], (tuple, list))
            or len(result[1]) != 6
        ):
            raise RobotClientError(
                "GetActualTCPPose", None, f"unexpected SDK result: {result!r}"
            )
        pose = tuple(float(value) for value in result[1])
        return (pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])

    def get_current_joints(self) -> tuple[float, float, float, float, float, float]:
        if self._robot is None:
            raise RobotClientError("GetActualJointPosDegree", None, "robot is not connected")
        result = self._robot.GetActualJointPosDegree()
        code = self._code(result, "GetActualJointPosDegree")
        if code != 0:
            raise RobotClientError("GetActualJointPosDegree", code)
        if (
            not isinstance(result, (tuple, list))
            or len(result) < 2
            or not isinstance(result[1], (tuple, list))
            or len(result[1]) != 6
        ):
            raise RobotClientError(
                "GetActualJointPosDegree", None, f"unexpected SDK result: {result!r}"
            )
        joints = tuple(float(value) for value in result[1])
        return (joints[0], joints[1], joints[2], joints[3], joints[4], joints[5])

    def get_state_snapshot(self) -> RobotStateSnapshot | None:
        """Copy the SDK's controller-pushed state cache without any RPC.

        SDK 2.2.x fills ``robot_state_pkg`` from the 20005 CNDE stream (8 ms by
        default). Older SDKs keep the bare class until the first packet, so a
        non-instance means no state has arrived yet. Fields are updated one by
        one by the SDK thread, so joints and TCP may come from adjacent packets.
        """
        if self._robot is None:
            return None
        pkg = getattr(self._robot, "robot_state_pkg", None)
        if not isinstance(pkg, ctypes.Structure):
            return None
        try:
            joints = tuple(float(value) for value in pkg.jt_cur_pos)
            velocities = tuple(float(value) for value in pkg.actual_qd)
            tcp = tuple(float(value) for value in pkg.tl_cur_pos)
            robot_time_ms = (
                ((int(pkg.hour) * 60 + int(pkg.minute)) * 60 + int(pkg.second))
                * 1000
                + int(pkg.millisecond)
            )
            return RobotStateSnapshot(
                joint_pos_deg=joints,  # type: ignore[arg-type]
                joint_vel_deg_s=velocities,  # type: ignore[arg-type]
                tcp=tcp,  # type: ignore[arg-type]
                gripper_position=float(pkg.gripper_position),
                frame_cnt=int(pkg.frame_cnt),
                robot_time_ms=robot_time_ms,
            )
        except AttributeError as exc:
            raise RobotClientError(
                "robot_state_pkg", None, f"unexpected SDK state layout: {exc}"
            ) from exc

    def servo_start(self) -> None:
        self._call_zero("ServoMoveStart")

    def servo_cart(self, target: TcpPose) -> None:
        if self._robot is None:
            raise RobotClientError("ServoCart", None, "robot is not connected")
        arguments = {
            "mode": 0,
            "desc_pos": list(target),
            "pos_gain": [1.0] * 6,
            "acc": 0.0 if self._servo_cart_supports_exaxis else 100.0,
            "vel": 0.0 if self._servo_cart_supports_exaxis else 100.0,
            "cmdT": self.config.servo_period_s,
            "filterT": 0.0,
            "gain": 0.0,
        }
        if self._servo_cart_supports_exaxis is None:
            raise RobotClientError(
                "ServoCart",
                None,
                "SDK signature was not inspected during connect",
            )
        if self._servo_cart_supports_exaxis:
            arguments["exaxis"] = list(self.config.exaxis_default)
        result = self._robot.ServoCart(**arguments)
        self._expect_zero(result, "ServoCart")

    def servo_end(self) -> None:
        self._call_zero("ServoMoveEnd")

    def activate_gripper(self) -> None:
        self._call_zero("ActGripper", self.config.gripper.index, 1)

    def move_gripper(self, position: int) -> None:
        if self._robot is None:
            raise RobotClientError("MoveGripper", None, "robot is not connected")
        if self._move_gripper_supports_extended_args is None:
            raise RobotClientError(
                "MoveGripper",
                None,
                "gripper SDK signature was not inspected during connect",
            )
        arguments: dict[str, Any] = {
            "index": self.config.gripper.index,
            "pos": position,
            "vel": self.config.gripper.velocity,
            "force": self.config.gripper.force,
            "maxtime": self.config.gripper.command_max_time_ms,
            "block": 1,
        }
        if self._move_gripper_supports_extended_args:
            arguments.update(
                {
                    "type": 0,
                    "rotNum": 0.0,
                    "rotVel": 0,
                    "rotTorque": 0,
                }
            )
        self._expect_zero(self._robot.MoveGripper(**arguments), "MoveGripper")

    def get_gripper_motion_state(self) -> GripperMotionState:
        if self._robot is None:
            raise RobotClientError(
                "GetGripperMotionDone",
                None,
                "robot is not connected",
            )
        result = self._robot.GetGripperMotionDone()
        code = self._code(result, "GetGripperMotionDone")
        if code != 0:
            raise RobotClientError("GetGripperMotionDone", code)
        # MoveGripper is sent with block=1 (non-blocking), so completion is
        # polled here. The SDK documents the result as [fault, status] with
        # status 1 = done, but the field FR5 + Robotiq returns [1, status]
        # with a gOBJ-like status: 0 moving, 2 stopped on an object, 3 at
        # the target (1, "stopped on an object while opening", was not
        # seen). So any non-zero status is done, and the first value, which
        # stays 1 without any fault, is ignored in favor of the CNDE
        # gripper_fault (0 none, 1 485 timeout, 2 bad command, 3 object
        # dropped, else gripper fault code).
        try:
            _, status = result[1]
        except (TypeError, ValueError) as exc:
            raise RobotClientError(
                "GetGripperMotionDone", None, f"unexpected SDK result: {result!r}"
            ) from exc
        return GripperMotionState(fault=self._gripper_fault(), done=int(status) != 0)

    def _gripper_fault(self) -> int:
        """CNDE gripper_fault; 0 until the first state packet arrived."""
        pkg = getattr(self._robot, "robot_state_pkg", None)
        if not isinstance(pkg, ctypes.Structure):
            return 0
        try:
            return int(pkg.gripper_fault)
        except AttributeError as exc:
            raise RobotClientError(
                "robot_state_pkg", None, f"unexpected SDK state layout: {exc}"
            ) from exc

    def reset_fault(self) -> None:
        self._call_allow("ResetAllError", {14})
        time.sleep(0.1)
        self._call_allow("Mode", {14}, 0)
        time.sleep(0.1)
        self._call_allow("RobotEnable", {14}, 1)
        time.sleep(0.2)

    def close(self) -> None:
        if self._robot is None:
            return
        try:
            self._robot.CloseRPC()
        finally:
            self._robot = None
