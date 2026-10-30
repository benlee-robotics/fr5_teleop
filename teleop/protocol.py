from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


PROTOCOL_VERSION = 1
MAX_POSITION_METERS = 10.0


class ProtocolError(ValueError):
    """A client message violates the teleop protocol."""


@dataclass(frozen=True)
class PoseMessage:
    session_id: str
    seq: int
    client_time_ms: float
    hand: str
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    grip: bool
    trigger: bool
    received_ns: int


@dataclass(frozen=True)
class ControlEvent:
    session_id: str
    event: str


@dataclass(frozen=True)
class ClientTelemetry:
    session_id: str
    seq: int
    client_time_ms: float
    xr_frame_count: int
    valid_pose_count: int
    pose_send_count: int
    pose_drop_count: int
    tracking_loss_count: int
    max_xr_frame_gap_ms: float
    max_pose_gap_ms: float
    ws_buffered_amount: int
    last_rtt_ms: float | None
    max_rtt_ms: float | None
    received_ns: int

    def to_status_dict(self) -> dict[str, int | float | None]:
        return {
            "seq": self.seq,
            "xr_frame_count": self.xr_frame_count,
            "valid_pose_count": self.valid_pose_count,
            "pose_send_count": self.pose_send_count,
            "pose_drop_count": self.pose_drop_count,
            "tracking_loss_count": self.tracking_loss_count,
            "max_xr_frame_gap_ms": self.max_xr_frame_gap_ms,
            "max_pose_gap_ms": self.max_pose_gap_ms,
            "ws_buffered_amount": self.ws_buffered_amount,
            "last_rtt_ms": self.last_rtt_ms,
            "max_rtt_ms": self.max_rtt_ms,
        }


@dataclass(frozen=True)
class RecordCommand:
    """Episode recording control. Never moves the robot."""

    session_id: str
    action: str
    task: str | None = None
    outcome: str | None = None


MAX_TASK_CHARS = 1000

ClientMessage = PoseMessage | ControlEvent | ClientTelemetry | RecordCommand


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ProtocolError(f"{name} must contain {length} numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ProtocolError(f"{name} must contain only numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ProtocolError(f"{name} contains a non-finite number")
        result.append(number)
    return tuple(result)


def parse_client_message(payload: Mapping[str, Any], received_ns: int) -> ClientMessage:
    version = payload.get("version")
    if isinstance(version, bool) or version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {payload.get('version')!r}")

    message_type = payload.get("type")
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id or len(session_id) > 128:
        raise ProtocolError("session_id is required")

    if message_type == "event":
        event = payload.get("event")
        allowed = {
            "webxr_started",
            "webxr_ended",
            "claim_control",
            "release_control",
            "fault_reset",
            "go_home",
            "stop_home",
        }
        if event not in allowed:
            raise ProtocolError(f"unsupported event: {event!r}")
        return ControlEvent(session_id=session_id, event=event)

    if message_type == "telemetry":
        return _parse_client_telemetry(payload, session_id, received_ns)

    if message_type == "record":
        return _parse_record_command(payload, session_id)

    if message_type != "pose":
        raise ProtocolError(f"unsupported message type: {message_type!r}")

    seq = payload.get("seq")
    if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
        raise ProtocolError("seq must be a non-negative integer")
    client_time_ms = payload.get("client_time_ms")
    if isinstance(client_time_ms, bool) or not isinstance(client_time_ms, (int, float)):
        raise ProtocolError("client_time_ms must be a number")
    client_time_ms = float(client_time_ms)
    if not math.isfinite(client_time_ms) or client_time_ms < 0:
        raise ProtocolError("client_time_ms must be finite and non-negative")

    hand = payload.get("hand")
    if hand not in {"left", "right", "none"}:
        raise ProtocolError("hand must be left, right, or none")

    position = _finite_vector(payload.get("position_m"), 3, "position_m")
    if any(abs(value) > MAX_POSITION_METERS for value in position):
        raise ProtocolError("position_m exceeds the accepted input range")

    orientation = _finite_vector(
        payload.get("orientation_xyzw"), 4, "orientation_xyzw"
    )
    norm = math.sqrt(sum(value * value for value in orientation))
    if norm < 1e-9:
        raise ProtocolError("orientation quaternion has zero length")
    normalized = tuple(value / norm for value in orientation)

    grip = payload.get("grip")
    trigger = payload.get("trigger")
    if not isinstance(grip, bool) or not isinstance(trigger, bool):
        raise ProtocolError("grip and trigger must be booleans")

    return PoseMessage(
        session_id=session_id,
        seq=seq,
        client_time_ms=client_time_ms,
        hand=hand,
        position_m=(position[0], position[1], position[2]),
        orientation_xyzw=(
            normalized[0],
            normalized[1],
            normalized[2],
            normalized[3],
        ),
        grip=grip,
        trigger=trigger,
        received_ns=received_ns,
    )


def _parse_client_telemetry(
    payload: Mapping[str, Any],
    session_id: str,
    received_ns: int,
) -> ClientTelemetry:
    seq = _non_negative_integer(payload.get("seq"), "seq")
    client_time_ms = _non_negative_number(
        payload.get("client_time_ms"),
        "client_time_ms",
    )
    counters = {
        name: _non_negative_integer(payload.get(name), name)
        for name in (
            "xr_frame_count",
            "valid_pose_count",
            "pose_send_count",
            "pose_drop_count",
            "tracking_loss_count",
            "ws_buffered_amount",
        )
    }
    max_xr_frame_gap_ms = _non_negative_number(
        payload.get("max_xr_frame_gap_ms"),
        "max_xr_frame_gap_ms",
    )
    max_pose_gap_ms = _non_negative_number(
        payload.get("max_pose_gap_ms"),
        "max_pose_gap_ms",
    )
    last_rtt_ms = _optional_non_negative_number(
        payload.get("last_rtt_ms"),
        "last_rtt_ms",
    )
    max_rtt_ms = _optional_non_negative_number(
        payload.get("max_rtt_ms"),
        "max_rtt_ms",
    )
    return ClientTelemetry(
        session_id=session_id,
        seq=seq,
        client_time_ms=client_time_ms,
        xr_frame_count=counters["xr_frame_count"],
        valid_pose_count=counters["valid_pose_count"],
        pose_send_count=counters["pose_send_count"],
        pose_drop_count=counters["pose_drop_count"],
        tracking_loss_count=counters["tracking_loss_count"],
        max_xr_frame_gap_ms=max_xr_frame_gap_ms,
        max_pose_gap_ms=max_pose_gap_ms,
        ws_buffered_amount=counters["ws_buffered_amount"],
        last_rtt_ms=last_rtt_ms,
        max_rtt_ms=max_rtt_ms,
        received_ns=received_ns,
    )


def _parse_record_command(
    payload: Mapping[str, Any],
    session_id: str,
) -> RecordCommand:
    action = payload.get("action")
    if action not in {"start", "stop", "discard", "set_task"}:
        raise ProtocolError(f"unsupported record action: {action!r}")
    task = payload.get("task")
    if task is not None:
        if not isinstance(task, str):
            raise ProtocolError("task must be a string")
        task = task.strip()
        if len(task) > MAX_TASK_CHARS:
            raise ProtocolError(f"task must be at most {MAX_TASK_CHARS} characters")
        task = task or None
    outcome = payload.get("outcome")
    if action == "stop":
        if outcome not in {"success", "failure"}:
            raise ProtocolError("stop outcome must be success or failure")
    elif outcome is not None:
        raise ProtocolError("outcome is only valid for stop")
    if action == "set_task":
        if task is None:
            raise ProtocolError("set_task requires a non-empty task")
    elif action != "start" and task is not None:
        raise ProtocolError("task is only valid for start and set_task")
    return RecordCommand(
        session_id=session_id,
        action=action,
        task=task,
        outcome=outcome,
    )


def _non_negative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{name} must be a non-negative integer")
    return value


def _non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ProtocolError(f"{name} must be finite and non-negative")
    return result


def _optional_non_negative_number(
    value: Any,
    name: str,
) -> float | None:
    if value is None:
        return None
    return _non_negative_number(value, name)
