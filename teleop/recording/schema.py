"""Fixed binary layouts used by raw episode recordings.

The worker emits one robot sample per servo tick, and the camera writer one
frame record per recorded video frame. Both are packed with ``struct`` so the
teleop runtime needs no numpy; offline tools read the same bytes with
``numpy.dtype(layout.numpy_dtype_descr())``.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from teleop.robot.state import WorkerState


RAW_FORMAT = "fr5_teleop_raw"
RAW_FORMAT_VERSION = 1

NAN = math.nan
NAN3 = (NAN, NAN, NAN)
NAN4 = (NAN, NAN, NAN, NAN)
NAN6 = (NAN, NAN, NAN, NAN, NAN, NAN)

STATE_NAMES: tuple[str, ...] = tuple(state.value for state in WorkerState)
STATE_CODES: dict[WorkerState, int] = {
    state: index for index, state in enumerate(WorkerState)
}

_NUMPY_CODES = {"q": "<i8", "i": "<i4", "B": "|u1", "d": "<f8"}


class BinaryLayout:
    """Little-endian, unpadded record layout of (name, struct code, count)."""

    def __init__(self, fields: tuple[tuple[str, str, int], ...]) -> None:
        self.fields = fields
        self.struct_format = "<" + "".join(
            f"{count}{code}" if count > 1 else code for _, code, count in fields
        )
        self.struct = struct.Struct(self.struct_format)
        self.size = self.struct.size

    def pack(self, values: tuple[Any, ...]) -> bytes:
        return self.struct.pack(*values)

    def numpy_dtype_descr(self) -> list[tuple[Any, ...]]:
        descr: list[tuple[Any, ...]] = []
        for name, code, count in self.fields:
            if count > 1:
                descr.append((name, _NUMPY_CODES[code], (count,)))
            else:
                descr.append((name, _NUMPY_CODES[code]))
        return descr

    def metadata(self) -> dict[str, Any]:
        return {
            "struct_format": self.struct_format,
            "record_size": self.size,
            "fields": [
                {"name": name, "code": code, "count": count}
                for name, code, count in self.fields
            ],
        }

    def iter_records(self, path: Path) -> Iterator[dict[str, Any]]:
        """Decode a file of records; a torn trailing record is ignored."""
        data = path.read_bytes()
        usable = len(data) - len(data) % self.size
        for values in self.struct.iter_unpack(data[:usable]):
            record: dict[str, Any] = {}
            index = 0
            for name, _, count in self.fields:
                if count == 1:
                    record[name] = values[index]
                else:
                    record[name] = tuple(values[index:index + count])
                index += count
            yield record


# One robot sample per servo tick, in on-disk order.
ROBOT_LAYOUT = BinaryLayout(
    (
        # Timing. t_ns is the worker monotonic clock when robot state was read.
        ("t_ns", "q", 1),
        ("tick", "q", 1),
        ("generation", "q", 1),
        # Worker state machine.
        ("state", "B", 1),
        ("fresh", "B", 1),
        ("servo_active", "B", 1),
        ("clutch_id", "i", 1),
        # Action: the exact ServoCart target sent on this tick (NaN if none).
        ("cmd_valid", "B", 1),
        ("cmd_tcp", "d", 6),
        ("gripper_target", "d", 1),
        ("gripper_busy", "B", 1),
        # Raw teleop input from the latest pose mailbox snapshot.
        ("vr_valid", "B", 1),
        ("vr_seq", "q", 1),
        ("vr_received_ns", "q", 1),
        ("vr_position_m", "d", 3),
        ("vr_orientation_xyzw", "d", 4),
        ("grip", "B", 1),
        ("trigger", "B", 1),
        # Observation: controller-reported robot state (NaN if unavailable).
        ("state_valid", "B", 1),
        ("joint_pos_deg", "d", 6),
        ("joint_vel_deg_s", "d", 6),
        ("tcp", "d", 6),
        ("gripper_position", "d", 1),
        ("robot_frame_cnt", "i", 1),
        ("robot_time_ms", "q", 1),
    )
)

# One record per frame written to the episode video, in video frame order.
CAMERA_FRAME_LAYOUT = BinaryLayout(
    (
        # Index of this frame in the episode video (constant-rate pts).
        ("frame_index", "q", 1),
        # Host CLOCK_MONOTONIC when the frameset was delivered; same clock as
        # robot t_ns. Includes exposure-to-host transfer latency.
        ("arrival_mono_ns", "q", 1),
        # Host wall clock at the same moment, to relate device timestamps.
        ("arrival_wall_ns", "q", 1),
        # librealsense frame timestamp (ms) in the domain stored in meta.json.
        ("device_ts_ms", "d", 1),
        # Camera frame counter; gaps mean frames dropped inside the camera/USB.
        ("frame_number", "q", 1),
    )
)

# Backward-compatible names for the robot layout.
SAMPLE_FIELDS = ROBOT_LAYOUT.fields
STRUCT_FORMAT = ROBOT_LAYOUT.struct_format
SAMPLE_STRUCT = ROBOT_LAYOUT.struct
SAMPLE_SIZE = ROBOT_LAYOUT.size


def build_sample(
    *,
    t_ns: int,
    tick: int,
    generation: int,
    state: WorkerState,
    fresh: bool,
    servo_active: bool,
    clutch_id: int,
    cmd_tcp: tuple[float, ...] | None,
    gripper_target: float,
    gripper_busy: bool,
    vr_valid: bool,
    vr_seq: int,
    vr_received_ns: int,
    vr_position_m: tuple[float, float, float],
    vr_orientation_xyzw: tuple[float, float, float, float],
    grip: bool,
    trigger: bool,
    joint_pos_deg: tuple[float, ...] | None,
    joint_vel_deg_s: tuple[float, ...] | None,
    tcp: tuple[float, ...] | None,
    gripper_position: float | None,
    robot_frame_cnt: int,
    robot_time_ms: int,
) -> tuple[Any, ...]:
    """Return one flat sample tuple in ``ROBOT_LAYOUT`` order."""
    state_valid = tcp is not None and joint_pos_deg is not None
    return (
        t_ns,
        tick,
        generation,
        STATE_CODES[state],
        int(fresh),
        int(servo_active),
        clutch_id,
        int(cmd_tcp is not None),
        *(cmd_tcp if cmd_tcp is not None else NAN6),
        gripper_target,
        int(gripper_busy),
        int(vr_valid),
        vr_seq,
        vr_received_ns,
        *(vr_position_m if vr_valid else NAN3),
        *(vr_orientation_xyzw if vr_valid else NAN4),
        int(grip),
        int(trigger),
        int(state_valid),
        *(joint_pos_deg if joint_pos_deg is not None else NAN6),
        *(joint_vel_deg_s if joint_vel_deg_s is not None else NAN6),
        *(tcp if tcp is not None else NAN6),
        gripper_position if gripper_position is not None else NAN,
        robot_frame_cnt,
        robot_time_ms,
    )


def pack_sample(sample: tuple[Any, ...]) -> bytes:
    return ROBOT_LAYOUT.pack(sample)


def numpy_dtype_descr() -> list[tuple[Any, ...]]:
    """Packed numpy structured dtype of one robot sample."""
    return ROBOT_LAYOUT.numpy_dtype_descr()


def schema_metadata() -> dict[str, Any]:
    metadata = ROBOT_LAYOUT.metadata()
    metadata["sample_size"] = ROBOT_LAYOUT.size
    metadata["state_names"] = list(STATE_NAMES)
    return metadata


def iter_robot_samples(path: Path) -> Iterator[dict[str, Any]]:
    """Decode a raw ``robot_<hand>.bin`` file without numpy."""
    return ROBOT_LAYOUT.iter_records(path)
