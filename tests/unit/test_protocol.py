import math

import pytest

from teleop.protocol import (
    ClientTelemetry,
    ControlEvent,
    PoseMessage,
    ProtocolError,
    parse_client_message,
)


def valid_pose() -> dict:
    return {
        "version": 1,
        "type": "pose",
        "session_id": "session",
        "seq": 1,
        "client_time_ms": 12.5,
        "hand": "right",
        "position_m": [0.1, 1.2, -0.3],
        "orientation_xyzw": [0.0, 0.0, 0.0, 2.0],
        "grip": True,
        "trigger": False,
    }


def valid_telemetry() -> dict:
    return {
        "version": 1,
        "type": "telemetry",
        "session_id": "session",
        "seq": 3,
        "client_time_ms": 1000.0,
        "xr_frame_count": 90,
        "valid_pose_count": 88,
        "pose_send_count": 88,
        "pose_drop_count": 1,
        "tracking_loss_count": 2,
        "max_xr_frame_gap_ms": 15.0,
        "max_pose_gap_ms": 25.0,
        "ws_buffered_amount": 0,
        "last_rtt_ms": 4.5,
        "max_rtt_ms": 8.0,
    }


def test_pose_is_validated_and_quaternion_is_normalized() -> None:
    message = parse_client_message(valid_pose(), 123)
    assert isinstance(message, PoseMessage)
    assert message.received_ns == 123
    assert message.orientation_xyzw == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("position_m", [0, 0]),
        ("position_m", [math.nan, 0, 0]),
        ("orientation_xyzw", [0, 0, 0, 0]),
        ("grip", 1),
        ("seq", -1),
        ("version", True),
    ],
)
def test_invalid_pose_is_rejected(field: str, value: object) -> None:
    payload = valid_pose()
    payload[field] = value
    with pytest.raises(ProtocolError):
        parse_client_message(payload, 123)


@pytest.mark.parametrize("event", ["release_control", "go_home", "stop_home"])
def test_control_event_is_parsed(event: str) -> None:
    message = parse_client_message(
        {
            "version": 1,
            "type": "event",
            "session_id": "session",
            "event": event,
        },
        123,
    )
    assert message == ControlEvent("session", event)


def test_client_telemetry_is_validated() -> None:
    message = parse_client_message(valid_telemetry(), 456)

    assert isinstance(message, ClientTelemetry)
    assert message.received_ns == 456
    assert message.pose_drop_count == 1
    assert message.max_rtt_ms == 8.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seq", -1),
        ("xr_frame_count", True),
        ("tracking_loss_count", -1),
        ("max_pose_gap_ms", math.inf),
        ("last_rtt_ms", -0.1),
        ("ws_buffered_amount", 1.5),
    ],
)
def test_invalid_client_telemetry_is_rejected(
    field: str,
    value: object,
) -> None:
    payload = valid_telemetry()
    payload[field] = value

    with pytest.raises(ProtocolError):
        parse_client_message(payload, 456)
