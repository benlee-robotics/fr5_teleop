from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any

from teleop.protocol import PoseMessage


@dataclass(frozen=True)
class SharedPose:
    generation: int
    seq: int
    received_ns: int
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    grip: bool
    trigger: bool
    valid: bool


class LatestPoseMailbox:
    """One fixed-size, overwrite-only pose slot shared with the robot process."""

    def __init__(self, context: BaseContext) -> None:
        self._lock = context.Lock()
        self._generation = context.Value("Q", 0, lock=False)
        self._seq = context.Value("Q", 0, lock=False)
        self._received_ns = context.Value("Q", 0, lock=False)
        self._pose = context.Array("d", 7, lock=False)
        self._flags = context.Array("b", 3, lock=False)

    def publish(self, message: PoseMessage, generation: int) -> None:
        values = (*message.position_m, *message.orientation_xyzw)
        with self._lock:
            self._generation.value = generation
            self._seq.value = message.seq
            self._received_ns.value = message.received_ns
            for index, value in enumerate(values):
                self._pose[index] = value
            self._flags[0] = int(message.grip)
            self._flags[1] = int(message.trigger)
            self._flags[2] = 1

    def invalidate(self) -> None:
        with self._lock:
            self._flags[2] = 0

    def read(self) -> SharedPose:
        with self._lock:
            values = tuple(float(self._pose[index]) for index in range(7))
            return SharedPose(
                generation=int(self._generation.value),
                seq=int(self._seq.value),
                received_ns=int(self._received_ns.value),
                position_m=(values[0], values[1], values[2]),
                orientation_xyzw=(values[3], values[4], values[5], values[6]),
                grip=bool(self._flags[0]),
                trigger=bool(self._flags[1]),
                valid=bool(self._flags[2]),
            )


@dataclass
class WorkerIpc:
    mailbox: LatestPoseMailbox
    control_receive: Any
    control_send: Any
    status_queue: Any
    heartbeat_ns: Any
    # Worker -> parent batches of recording samples; None when recording is off.
    sample_queue: Any = None


# About 5 s of 10-tick sample batches at the 8 ms servo period.
SAMPLE_QUEUE_BATCHES = 64


def create_worker_ipc(
    context: BaseContext,
    *,
    with_samples: bool = False,
) -> WorkerIpc:
    control_receive, control_send = context.Pipe(duplex=False)
    return WorkerIpc(
        mailbox=LatestPoseMailbox(context),
        control_receive=control_receive,
        control_send=control_send,
        status_queue=context.Queue(maxsize=32),
        heartbeat_ns=context.Value("Q", 0, lock=False),
        sample_queue=(
            context.Queue(maxsize=SAMPLE_QUEUE_BATCHES) if with_samples else None
        ),
    )
