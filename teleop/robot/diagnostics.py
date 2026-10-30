from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from teleop.config import TeleopConfig
from teleop.ipc import LatestPoseMailbox, SharedPose
from teleop.robot.state import (
    ServoTransitionMonitor,
    WorkerCounters,
    WorkerState,
    WorkerStatus,
)


log = logging.getLogger(__name__)


@dataclass
class WorkerDiagnostics:
    """Rate-limited worker timing and lifecycle diagnostics."""

    config: TeleopConfig
    generation: int
    mailbox: LatestPoseMailbox
    transition_monitor: ServoTransitionMonitor = field(init=False)

    last_jitter_ms: float = 0.0
    last_sdk_call_ms: float = 0.0
    last_sdk_operation: str | None = None
    last_logged_transition: tuple[WorkerState, str | None] | None = None

    slow_sdk_call_count: int = 0
    slow_sdk_call_max_ms: float = 0.0
    last_slow_sdk_report_ns: int = 0
    missed_ticks_since_report: int = 0
    max_deadline_overrun_ms: float = 0.0
    last_missed_tick_report_ns: int = 0
    sdk_samples_ms: dict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=2_000))
    )
    sdk_metrics_window_started_ns: int = 0

    def __post_init__(self) -> None:
        self.transition_monitor = ServoTransitionMonitor(
            window_ns=int(
                self.config.servo_transition_window_s * 1_000_000_000
            ),
            limit=self.config.servo_transition_limit,
        )

    def timed_call(
        self,
        operation: Any,
        *args: Any,
        state: WorkerState,
        tick: int,
    ) -> Any:
        operation_name = getattr(
            operation,
            "__name__",
            type(operation).__name__,
        )
        started_ns = time.monotonic_ns()
        try:
            return operation(*args)
        finally:
            finished_ns = time.monotonic_ns()
            self.last_sdk_call_ms = (
                finished_ns - started_ns
            ) / 1_000_000
            self.last_sdk_operation = operation_name
            self._record_sdk_sample(
                finished_ns,
                operation_name,
            )
            self._record_slow_sdk_call(
                finished_ns,
                operation_name,
                state,
                tick,
            )

    def log_state_transition(
        self,
        status: WorkerStatus,
        *,
        servo_started: bool,
        pose_seq: int | None,
    ) -> None:
        transition = (status.state, status.reason)
        if transition == self.last_logged_transition:
            return
        logger = log.error if status.state == WorkerState.FAULT else log.info
        logger(
            "worker state generation=%d state=%s reason=%s "
            "servo_started=%s rearm_required=%s pose_seq=%s "
            "input_age_ms=%s sdk_operation=%s sdk_call_ms=%.3f "
            "start_count=%d cart_count=%d end_count=%d missed_ticks=%d "
            "idle_overrun_ticks=%d gripper_commands=%d gripper_completes=%d "
            "gripper_position=%s fault=%s",
            self.generation,
            status.state.value,
            status.reason,
            servo_started,
            status.rearm_required,
            pose_seq,
            (
                f"{status.input_age_ms:.1f}"
                if status.input_age_ms is not None
                else None
            ),
            self.last_sdk_operation,
            status.sdk_call_ms,
            status.counters.servo_start_count,
            status.counters.servo_cart_count,
            status.counters.servo_end_count,
            status.counters.missed_ticks,
            status.counters.idle_overrun_ticks,
            status.counters.gripper_command_count,
            status.counters.gripper_complete_count,
            status.gripper_position,
            status.fault,
        )
        self.last_logged_transition = transition

    def record_servo_transition(
        self,
        now_ns: int,
        transition: str,
        reason: str,
        counters: WorkerCounters,
    ) -> None:
        event_count = self.transition_monitor.record(now_ns)
        if event_count is None:
            return
        pose = self.mailbox.read()
        input_age_ms = self._pose_age_ms(now_ns, pose)
        log.error(
            "rapid servo start/end transitions detected: events=%d "
            "window_s=%.3f limit=%d latest=%s reason=%s "
            "start_count=%d end_count=%d pose_seq=%s input_age_ms=%s",
            event_count,
            self.config.servo_transition_window_s,
            self.config.servo_transition_limit,
            transition,
            reason,
            counters.servo_start_count,
            counters.servo_end_count,
            pose.seq if pose.valid else None,
            f"{input_age_ms:.1f}" if input_age_ms is not None else None,
        )

    def log_pose_timeout(
        self,
        now_ns: int,
        pose: SharedPose,
        counters: WorkerCounters,
    ) -> None:
        pose_age_ms = self._pose_age_ms(now_ns, pose)
        log.warning(
            "pose timeout stopping servo: generation=%d pose_seq=%s "
            "input_age_ms=%s timeout_ms=%.1f sdk_operation=%s "
            "last_sdk_call_ms=%.3f missed_ticks=%d",
            self.generation,
            pose.seq if pose.valid else None,
            f"{pose_age_ms:.1f}" if pose_age_ms is not None else None,
            self.config.pose_timeout_s * 1000,
            self.last_sdk_operation,
            self.last_sdk_call_ms,
            counters.missed_ticks,
        )

    def record_deadline_miss(
        self,
        now_ns: int,
        *,
        skipped: int,
        overrun_ms: float,
        counters: WorkerCounters,
        state: WorkerState,
        tick: int,
    ) -> None:
        self.missed_ticks_since_report += skipped
        self.max_deadline_overrun_ms = max(
            self.max_deadline_overrun_ms,
            overrun_ms,
        )
        if (
            self.last_missed_tick_report_ns != 0
            and now_ns - self.last_missed_tick_report_ns < 1_000_000_000
        ):
            return
        log.warning(
            "servo deadline miss summary generation=%d state=%s "
            "missed_since_report=%d missed_total=%d max_overrun_ms=%.3f "
            "last_jitter_ms=%.3f sdk_operation=%s sdk_call_ms=%.3f tick=%d",
            self.generation,
            state.value,
            self.missed_ticks_since_report,
            counters.missed_ticks,
            self.max_deadline_overrun_ms,
            self.last_jitter_ms,
            self.last_sdk_operation,
            self.last_sdk_call_ms,
            tick,
        )
        self.missed_ticks_since_report = 0
        self.max_deadline_overrun_ms = 0.0
        self.last_missed_tick_report_ns = now_ns

    def _record_slow_sdk_call(
        self,
        finished_ns: int,
        operation_name: str,
        state: WorkerState,
        tick: int,
    ) -> None:
        if self.last_sdk_call_ms <= self.config.servo_period_s * 1000:
            return
        self.slow_sdk_call_count += 1
        self.slow_sdk_call_max_ms = max(
            self.slow_sdk_call_max_ms,
            self.last_sdk_call_ms,
        )
        if (
            self.last_slow_sdk_report_ns != 0
            and finished_ns - self.last_slow_sdk_report_ns < 1_000_000_000
        ):
            return
        log.warning(
            "slow SDK call summary generation=%d state=%s operation=%s "
            "last_ms=%.3f max_ms=%.3f count_since_report=%d "
            "servo_period_ms=%.3f tick=%d",
            self.generation,
            state.value,
            operation_name,
            self.last_sdk_call_ms,
            self.slow_sdk_call_max_ms,
            self.slow_sdk_call_count,
            self.config.servo_period_s * 1000,
            tick,
        )
        self.slow_sdk_call_count = 0
        self.slow_sdk_call_max_ms = 0.0
        self.last_slow_sdk_report_ns = finished_ns

    def _record_sdk_sample(
        self,
        finished_ns: int,
        operation_name: str,
    ) -> None:
        self.sdk_samples_ms[operation_name].append(self.last_sdk_call_ms)
        if self.sdk_metrics_window_started_ns == 0:
            self.sdk_metrics_window_started_ns = finished_ns
            return
        if (
            finished_ns - self.sdk_metrics_window_started_ns
            < 10_000_000_000
        ):
            return
        window_s = (
            finished_ns - self.sdk_metrics_window_started_ns
        ) / 1_000_000_000
        for name, samples in sorted(self.sdk_samples_ms.items()):
            if not samples:
                continue
            ordered = sorted(samples)
            log.info(
                "SDK call timing generation=%d operation=%s count=%d "
                "p50_ms=%.3f p95_ms=%.3f p99_ms=%.3f max_ms=%.3f "
                "window_s=%.1f",
                self.generation,
                name,
                len(ordered),
                self._percentile(ordered, 0.50),
                self._percentile(ordered, 0.95),
                self._percentile(ordered, 0.99),
                ordered[-1],
                window_s,
            )
        self.sdk_samples_ms.clear()
        self.sdk_metrics_window_started_ns = finished_ns

    @staticmethod
    def _percentile(ordered: list[float], fraction: float) -> float:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    @staticmethod
    def _pose_age_ms(
        now_ns: int,
        pose: SharedPose,
    ) -> float | None:
        if not pose.valid or not pose.received_ns:
            return None
        return max(0.0, (now_ns - pose.received_ns) / 1_000_000)
