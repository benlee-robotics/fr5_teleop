import ctypes
from pathlib import Path

import pytest

from teleop.config import GripperConfig, TeleopConfig
from teleop.robot.client import RobotClientError
from teleop.robot.fairino_client import FairinoRobotClient


class StubRobot:
    def __init__(self) -> None:
        self.servo_cart_call = None
        self.tcp_result = (0, [1, 2, 3, 4, 5, 6])

        self.gripper_call = None
        self.gripper_result = (0, [0, 1])

    def ServoCart(
        self,
        mode,
        desc_pos,
        exaxis,
        pos_gain,
        acc,
        vel,
        cmdT,
        filterT,
        gain,
    ):
        self.servo_cart_call = {
            "mode": mode,
            "desc_pos": desc_pos,
            "exaxis": exaxis,
            "pos_gain": pos_gain,
            "acc": acc,
            "vel": vel,
            "cmdT": cmdT,
            "filterT": filterT,
            "gain": gain,
        }
        return 0

    def GetActualTCPPose(self):
        return self.tcp_result

    def MoveGripper(
        self,
        index,
        pos,
        vel,
        force,
        maxtime,
        block,
        type,
        rotNum,
        rotVel,
        rotTorque,
    ):
        self.gripper_call = {
            "index": index,
            "pos": pos,
            "vel": vel,
            "force": force,
            "maxtime": maxtime,
            "block": block,
            "type": type,
            "rotNum": rotNum,
            "rotVel": rotVel,
            "rotTorque": rotTorque,
        }
        return 0

    def GetGripperMotionDone(self):
        return self.gripper_result


class LegacyStubRobot:
    def __init__(self) -> None:
        self.servo_cart_calls = []
        self.gripper_call = None

    def ServoCart(
        self,
        mode,
        desc_pos,
        pos_gain,
        acc,
        vel,
        cmdT,
        filterT,
        gain,
    ):
        self.servo_cart_calls.append(
            {
                "mode": mode,
                "desc_pos": desc_pos,
                "pos_gain": pos_gain,
                "acc": acc,
                "vel": vel,
                "cmdT": cmdT,
                "filterT": filterT,
                "gain": gain,
            }
        )
        return 0

    def MoveGripper(self, index, pos, vel, force, maxtime, block):
        self.gripper_call = {
            "index": index,
            "pos": pos,
            "vel": vel,
            "force": force,
            "maxtime": maxtime,
            "block": block,
        }
        return 0

    def GetGripperMotionDone(self):
        return 0, [0, 1]


def client_with_stub(
    *, gripper: GripperConfig | None = None
) -> tuple[FairinoRobotClient, StubRobot]:
    config = TeleopConfig(
        dry_run=False,
        robot_ip="192.0.2.1",
        sdk_path=Path("unused"),
        web_dir=Path(__file__).resolve().parents[2] / "web",
        tls_cert_path=None,
        tls_key_path=None,
        gripper=gripper or GripperConfig(),
    )
    client = FairinoRobotClient(config)
    stub = StubRobot()
    client._robot = stub
    client._inspect_sdk_signatures()
    return client, stub


def test_servo_cart_supplies_required_exaxis() -> None:
    client, stub = client_with_stub()
    target = (1.0, 2.0, 3.0, 180.0, 0.0, 0.0)
    client.servo_cart(target)
    assert stub.servo_cart_call["desc_pos"] == list(target)
    assert stub.servo_cart_call["exaxis"] == [0.0, 0.0, 0.0, 0.0]
    assert stub.servo_cart_call == {
        "mode": 0,
        "desc_pos": list(target),
        "exaxis": [0.0, 0.0, 0.0, 0.0],
        "pos_gain": [1.0] * 6,
        "acc": 0.0,
        "vel": 0.0,
        "cmdT": 0.008,
        "filterT": 0.0,
        "gain": 0.0,
    }


def test_integer_error_from_pose_query_is_normalized() -> None:
    client, stub = client_with_stub()
    stub.tcp_result = -4
    with pytest.raises(RobotClientError) as error:
        client.get_current_tcp()
    assert error.value.code == -4


def test_legacy_servo_cart_without_exaxis_is_detected_once() -> None:
    client, _ = client_with_stub()
    legacy = LegacyStubRobot()
    client._robot = legacy
    client._inspect_sdk_signatures()
    target = (1.0, 2.0, 3.0, 180.0, 0.0, 0.0)

    client.servo_cart(target)
    client.servo_cart(target)

    assert client._servo_cart_supports_exaxis is False
    assert len(legacy.servo_cart_calls) == 2
    assert legacy.servo_cart_calls[0] == {
        "mode": 0,
        "desc_pos": list(target),
        "pos_gain": [1.0] * 6,
        "acc": 100.0,
        "vel": 100.0,
        "cmdT": 0.008,
        "filterT": 0.0,
        "gain": 0.0,
    }


def test_gripper_uses_extended_signature_without_trial_call() -> None:
    client, stub = client_with_stub(gripper=GripperConfig(enabled=True))

    client.move_gripper(100)

    assert stub.gripper_call == {
        "index": 1,
        "pos": 100,
        "vel": 50,
        "force": 50,
        "maxtime": 3000,
        "block": 1,
        "type": 0,
        "rotNum": 0.0,
        "rotVel": 0,
        "rotTorque": 0,
    }


class StubGripperPkg(ctypes.Structure):
    _fields_ = [("gripper_fault", ctypes.c_uint16)]


# GetGripperMotionDone as returned on the field FR5 + Robotiq (2026-10-29):
# the first value is always 1, the status is 0 while moving, 2 when stopped
# on an object and 3 at the target.
@pytest.mark.parametrize(
    ("result", "done"),
    [((0, [1, 0]), False), ((0, [1, 2]), True), ((0, [1, 3]), True), ((0, [1, 1]), True)],
)
def test_gripper_motion_state_is_done_for_any_stopped_status(result, done) -> None:
    client, stub = client_with_stub(gripper=GripperConfig(enabled=True))
    stub.robot_state_pkg = StubGripperPkg()
    stub.gripper_result = result

    state = client.get_gripper_motion_state()

    assert state.done is done
    assert state.fault == 0  # the leading 1 is not a fault


def test_gripper_motion_state_reports_the_cnde_gripper_fault() -> None:
    client, stub = client_with_stub(gripper=GripperConfig(enabled=True))
    pkg = StubGripperPkg()
    pkg.gripper_fault = 3  # object dropped
    stub.robot_state_pkg = pkg
    stub.gripper_result = (0, [1, 3])

    assert client.get_gripper_motion_state().fault == 3


def test_gripper_motion_state_without_state_packets_reports_no_fault() -> None:
    client, stub = client_with_stub(gripper=GripperConfig(enabled=True))
    stub.robot_state_pkg = StubGripperPkg  # the bare class: no packet yet
    stub.gripper_result = (0, [1, 0])

    state = client.get_gripper_motion_state()

    assert (state.fault, state.done) == (0, False)


def test_gripper_motion_state_rejects_malformed_result() -> None:
    client, stub = client_with_stub(gripper=GripperConfig(enabled=True))
    stub.gripper_result = (0, None)

    with pytest.raises(RobotClientError):
        client.get_gripper_motion_state()


def test_legacy_gripper_signature_omits_extended_arguments() -> None:
    client, _ = client_with_stub(gripper=GripperConfig(enabled=True))
    legacy = LegacyStubRobot()
    client._robot = legacy
    client._inspect_sdk_signatures()

    client.move_gripper(100)

    assert client._move_gripper_supports_extended_args is False
    assert legacy.gripper_call == {
        "index": 1,
        "pos": 100,
        "vel": 50,
        "force": 50,
        "maxtime": 3000,
        "block": 1,
    }


def test_ambiguous_servo_signature_fails_closed_before_motion() -> None:
    class AmbiguousRobot:
        calls = 0

        def ServoCart(self, **kwargs):
            self.calls += 1
            return 0

    client, _ = client_with_stub()
    ambiguous = AmbiguousRobot()
    client._robot = ambiguous

    with pytest.raises(RobotClientError, match="ambiguous SDK"):
        client._inspect_sdk_signatures()

    assert ambiguous.calls == 0


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({"is_connect": True}, True),
        ({"is_conect": True}, True),
        ({"is_connect": False, "is_conect": True}, True),
        ({"is_connect": False}, False),
        ({}, None),
    ],
)
def test_connection_state_supports_legacy_spelling(
    attributes: dict[str, bool], expected: bool | None
) -> None:
    rpc_type = type("RpcType", (), attributes)
    assert FairinoRobotClient._connection_state(rpc_type) is expected


class StubStatePkg(ctypes.Structure):
    _fields_ = [
        ("frame_cnt", ctypes.c_uint8),
        ("jt_cur_pos", ctypes.c_double * 6),
        ("tl_cur_pos", ctypes.c_double * 6),
        ("actual_qd", ctypes.c_double * 6),
        ("gripper_position", ctypes.c_uint8),
        ("hour", ctypes.c_uint8),
        ("minute", ctypes.c_uint8),
        ("second", ctypes.c_uint8),
        ("millisecond", ctypes.c_uint16),
    ]


def test_state_snapshot_reads_cnde_cache_without_rpc() -> None:
    client = FairinoRobotClient(TeleopConfig(robot_ip="192.168.58.2"))
    robot = StubRobot()
    client._robot = robot

    # Older SDKs keep the bare class until the first state packet arrives.
    robot.robot_state_pkg = StubStatePkg
    assert client.get_state_snapshot() is None

    pkg = StubStatePkg()
    pkg.frame_cnt = 17
    pkg.jt_cur_pos[:] = [1, 2, 3, 4, 5, 6]
    pkg.tl_cur_pos[:] = [300, 0, 400, 180, 0, 0]
    pkg.actual_qd[:] = [0.5] * 6
    pkg.gripper_position = 42
    pkg.hour, pkg.minute, pkg.second, pkg.millisecond = 1, 2, 3, 456
    robot.robot_state_pkg = pkg

    snapshot = client.get_state_snapshot()
    assert snapshot.joint_pos_deg == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert snapshot.tcp == (300.0, 0.0, 400.0, 180.0, 0.0, 0.0)
    assert snapshot.joint_vel_deg_s == (0.5,) * 6
    assert snapshot.gripper_position == 42.0
    assert snapshot.frame_cnt == 17
    assert snapshot.robot_time_ms == ((1 * 60 + 2) * 60 + 3) * 1000 + 456


def test_state_snapshot_rejects_unknown_state_layout() -> None:
    class OtherPkg(ctypes.Structure):
        _fields_ = [("frame_cnt", ctypes.c_uint8)]

    client = FairinoRobotClient(TeleopConfig(robot_ip="192.168.58.2"))
    client._robot = StubRobot()
    client._robot.robot_state_pkg = OtherPkg()
    with pytest.raises(RobotClientError, match="unexpected SDK state layout"):
        client.get_state_snapshot()
