from __future__ import annotations

import logging
import time
from pathlib import Path

from teleop.config import TeleopConfig
from teleop.robot.diagnostics import WorkerDiagnostics
from teleop.robot.state import WorkerState


def test_sdk_timing_summary_is_grouped_by_operation(
    caplog,
) -> None:
    config = TeleopConfig(
        web_dir=Path(__file__).resolve().parents[2] / "web",
        tls_cert_path=None,
        tls_key_path=None,
    )
    diagnostics = WorkerDiagnostics(
        config=config,
        generation=3,
        mailbox=object(),  # type: ignore[arg-type]
    )
    diagnostics.sdk_metrics_window_started_ns = (
        time.monotonic_ns() - 10_000_000_001
    )

    with caplog.at_level(logging.INFO):
        result = diagnostics.timed_call(
            lambda: 7,
            state=WorkerState.ACTIVE,
            tick=12,
        )

    assert result == 7
    assert "SDK call timing generation=3 operation=<lambda>" in caplog.text
    assert "p95_ms=" in caplog.text
