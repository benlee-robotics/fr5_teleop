from __future__ import annotations

from teleop.robot.state import ServoTransitionMonitor


def test_reports_when_transition_limit_is_reached() -> None:
    monitor = ServoTransitionMonitor(window_ns=1_000, limit=4)

    assert monitor.record(100) is None
    assert monitor.record(200) is None
    assert monitor.record(300) is None
    assert monitor.record(400) == 4


def test_report_is_rate_limited_to_once_per_window() -> None:
    monitor = ServoTransitionMonitor(window_ns=1_000, limit=3)

    assert monitor.record(100) is None
    assert monitor.record(200) is None
    assert monitor.record(300) == 3
    assert monitor.record(400) is None
    assert monitor.record(500) is None


def test_old_transitions_expire_from_window() -> None:
    monitor = ServoTransitionMonitor(window_ns=1_000, limit=3)

    assert monitor.record(100) is None
    assert monitor.record(200) is None
    assert monitor.record(1_201) is None
    assert monitor.record(1_202) is None
    assert monitor.record(1_203) == 3
