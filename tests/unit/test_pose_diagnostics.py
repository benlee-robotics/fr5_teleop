from teleop.app import PoseStreamDiagnostics
from teleop.protocol import PoseMessage


def pose(seq: int, received_ms: float, client_ms: float) -> PoseMessage:
    return PoseMessage(
        session_id="session",
        seq=seq,
        client_time_ms=client_ms,
        hand="right",
        position_m=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        grip=False,
        trigger=False,
        received_ns=int(received_ms * 1_000_000),
    )


def test_pose_stream_diagnostics_separates_client_and_receive_gaps() -> None:
    diagnostics = PoseStreamDiagnostics()

    assert diagnostics.observe(pose(10, 1_000.0, 500.0)) == (None, None, 0)
    receive_gap, client_gap, missing = diagnostics.observe(
        pose(13, 1_120.0, 516.0)
    )

    assert receive_gap == 120.0
    assert client_gap == 16.0
    assert missing == 2


def test_pose_stream_summary_resets_only_window_counters() -> None:
    diagnostics = PoseStreamDiagnostics()
    diagnostics.observe(pose(1, 1_000.0, 100.0))
    diagnostics.observe(pose(2, 1_020.0, 120.0))

    summary = diagnostics.take_summary(
        1_020_000_000,
        force=True,
    )

    assert summary == {
        "samples": 2,
        "missing_sequences": 0,
        "max_receive_gap_ms": 20.0,
        "max_client_gap_ms": 20.0,
        "window_ms": 20.0,
    }
    assert diagnostics.sample_count == 0
    assert diagnostics.last_seq == 2
