from __future__ import annotations

import json
import math
import queue
from pathlib import Path

import pytest

from teleop.config import TeleopConfig
from teleop.protocol import ProtocolError, RecordCommand, parse_client_message
from teleop.recording.recorder import (
    EpisodeOutcome,
    RecorderRuntime,
    RecorderSpec,
    RecorderState,
)
from teleop.recording.schema import (
    SAMPLE_SIZE,
    STATE_NAMES,
    build_sample,
    iter_robot_samples,
    numpy_dtype_descr,
    pack_sample,
)
from teleop.robot.state import WorkerState


def sample(t_ns: int, *, cmd: bool = True, state_valid: bool = True) -> tuple:
    return build_sample(
        t_ns=t_ns,
        tick=t_ns // 8,
        generation=1,
        state=WorkerState.ACTIVE,
        fresh=True,
        servo_active=True,
        clutch_id=2,
        cmd_tcp=(300.0, 1.0, 400.0, 180.0, 0.0, 0.0) if cmd else None,
        gripper_target=100.0,
        gripper_busy=False,
        vr_valid=True,
        vr_seq=7,
        vr_received_ns=t_ns - 3,
        vr_position_m=(0.1, 1.2, -0.3),
        vr_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        grip=True,
        trigger=False,
        joint_pos_deg=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) if state_valid else None,
        joint_vel_deg_s=(0.0,) * 6 if state_valid else None,
        tcp=(299.0, 1.0, 400.0, 180.0, 0.0, 0.0) if state_valid else None,
        gripper_position=99.0 if state_valid else None,
        robot_frame_cnt=12 if state_valid else -1,
        robot_time_ms=123456 if state_valid else -1,
    )


def test_sample_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "robot_right.bin"
    path.write_bytes(
        pack_sample(sample(10))
        + pack_sample(sample(20, cmd=False, state_valid=False))
        # A torn trailing record from a crash is ignored.
        + b"\x00" * (SAMPLE_SIZE // 2)
    )
    records = list(iter_robot_samples(path))
    assert len(records) == 2
    first, second = records
    assert first["t_ns"] == 10
    assert STATE_NAMES[first["state"]] == "ACTIVE"
    assert first["cmd_valid"] == 1
    assert first["cmd_tcp"] == (300.0, 1.0, 400.0, 180.0, 0.0, 0.0)
    assert first["joint_pos_deg"] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert first["vr_orientation_xyzw"] == (0.0, 0.0, 0.0, 1.0)
    assert second["cmd_valid"] == 0
    assert all(math.isnan(value) for value in second["cmd_tcp"])
    assert second["state_valid"] == 0
    assert all(math.isnan(value) for value in second["tcp"])


def test_numpy_dtype_matches_struct_layout() -> None:
    numpy = pytest.importorskip("numpy")
    dtype = numpy.dtype(numpy_dtype_descr())
    assert dtype.itemsize == SAMPLE_SIZE
    data = numpy.frombuffer(pack_sample(sample(42)), dtype=dtype)
    assert data["t_ns"][0] == 42
    assert data["cmd_tcp"][0].tolist() == [300.0, 1.0, 400.0, 180.0, 0.0, 0.0]


def test_record_command_parsing() -> None:
    base = {"version": 1, "type": "record", "session_id": "abc"}
    start = parse_client_message(
        {**base, "action": "start", "task": "  pick up the red cube  "}, 0
    )
    assert start == RecordCommand("abc", "start", "pick up the red cube", None)
    stop = parse_client_message({**base, "action": "stop", "outcome": "failure"}, 0)
    assert stop == RecordCommand("abc", "stop", None, "failure")
    set_task = parse_client_message({**base, "action": "set_task", "task": "wipe"}, 0)
    assert set_task == RecordCommand("abc", "set_task", "wipe", None)
    for bad in (
        {**base, "action": "pause"},
        {**base, "action": "stop"},
        {**base, "action": "start", "task": 3},
        {**base, "action": "start", "task": "x" * 1001},
        {**base, "action": "discard", "outcome": "success"},
        {**base, "action": "stop", "outcome": "success", "task": "t"},
        {**base, "action": "set_task"},
        {**base, "action": "set_task", "task": "   "},
    ):
        with pytest.raises(ProtocolError):
            parse_client_message(bad, 0)


def test_recording_config_from_yaml(tmp_path: Path) -> None:
    web = Path(__file__).resolve().parents[2] / "web"
    path = tmp_path / "config.yaml"
    path.write_text(
        "tls: {enabled: false}\n"
        f"server: {{web_dir: {web}}}\n"
        "recording: {enabled: true, root_dir: data/raw}\n",
        encoding="utf-8",
    )
    config = TeleopConfig.from_yaml(path)
    assert config.recording.enabled
    assert config.recording.root_dir == tmp_path / "data" / "raw"

    path.write_text("recording: {enabled: true, fps: 30}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown recording key"):
        TeleopConfig.from_yaml(path, validate=False)


def make_recorder(root: Path, hands: tuple[str, ...] = ("right",)) -> RecorderRuntime:
    runtime = RecorderRuntime(
        RecorderSpec(root_dir=root, hands=hands, session_metadata={"dry_run": True}),
        queue.Queue(),
        queue.Queue(),
    )
    runtime._prepare_root()
    return runtime


def test_recorder_episode_keeps_only_samples_inside_window(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder._handle(("samples", "right", 1, 0, 0, [sample(50)]))
    recorder._handle(("start", 100, "pick up the red cube"))
    assert recorder.state == RecorderState.RECORDING
    recorder._handle(
        ("samples", "right", 1, 0, 0, [sample(90), sample(100), sample(150)])
    )
    recorder._handle(("stop", 200, "success"))
    assert recorder.state == RecorderState.FINALIZING
    # Tail samples after stop are cut, and seeing them completes the episode.
    recorder._handle(("samples", "right", 1, 0, 0, [sample(190), sample(210)]))
    recorder._handle(("tick",))
    assert recorder.state == RecorderState.IDLE

    episode_dir = tmp_path / "episode_000000"
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["task"] == "pick up the red cube"
    assert meta["outcome"] == "success"
    assert meta["complete"] is True
    assert meta["hands"]["right"]["samples"] == 3
    assert meta["sample_schema"]["state_names"] == list(STATE_NAMES)
    times = [r["t_ns"] for r in iter_robot_samples(episode_dir / "robot_right.bin")]
    assert times == [100, 150, 190]
    events = [
        json.loads(line)["event"]
        for line in (episode_dir / "events.jsonl").read_text().splitlines()
    ]
    assert events == ["start", "stop"]
    assert recorder.last_saved["episode_index"] == 0


def test_recorder_requires_task_and_reuses_last_task(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder._handle(("start", 100, None))
    assert recorder.state == RecorderState.IDLE
    assert "task is required" in recorder.last_error

    recorder._handle(("start", 100, "open the drawer"))
    recorder._handle(("discard",))
    assert not list(tmp_path.iterdir())
    recorder._handle(("start", 300, None))
    assert recorder.episode is not None
    assert recorder.episode.task == "open the drawer"
    # Discarded takes do not leave a numbering gap.
    assert recorder.episode.index == 0


def test_recorder_marks_drops_and_worker_restart_incomplete(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder._handle(("start", 0, "task"))
    recorder._handle(("samples", "right", 1, 0, 0, [sample(10)]))
    recorder._handle(("samples", "right", 1, 2, 0, [sample(20)]))
    recorder._handle(("samples", "right", 2, 0, 1, [sample(30)]))
    recorder._handle(("stop", 30, "failure"))
    recorder._handle(("tick",))
    meta = json.loads((tmp_path / "episode_000000" / "meta.json").read_text())
    assert meta["outcome"] == "failure"
    assert meta["complete"] is False
    assert meta["hands"]["right"]["worker_dropped_batches"] == 2
    assert meta["hands"]["right"]["generations"] == [1, 2]
    assert meta["forward_dropped_batches"] == 1


def test_recorder_finalizes_after_tail_timeout_and_on_shutdown(
    tmp_path: Path,
) -> None:
    recorder = make_recorder(tmp_path, hands=("left", "right"))
    recorder._handle(("start", 0, "task"))
    recorder._handle(("samples", "right", 1, 0, 0, [sample(10)]))
    recorder._handle(("stop", 100, "success"))
    recorder._handle(("tick",))
    # The left worker never reported, so the recorder waits for the deadline.
    assert recorder.state == RecorderState.FINALIZING
    recorder.episode.finalize_deadline_ns = 0
    recorder._handle(("tick",))
    meta = json.loads((tmp_path / "episode_000000" / "meta.json").read_text())
    assert meta["missing_hands"] == ["left"]
    assert meta["complete"] is False

    recorder._handle(("start", 200, "task"))
    recorder._handle(("shutdown_close",))
    meta = json.loads((tmp_path / "episode_000001" / "meta.json").read_text())
    assert meta["outcome"] == EpisodeOutcome.INTERRUPTED.value


def test_recorder_renames_crashed_tmp_episode(tmp_path: Path) -> None:
    (tmp_path / "episode_000004.tmp").mkdir()
    (tmp_path / "episode_000002").mkdir()
    recorder = make_recorder(tmp_path)
    assert (tmp_path / "episode_000004.incomplete").is_dir()
    assert recorder.next_index == 5


def test_set_task_is_used_by_start_without_task(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder._handle(("set_task", "  stack the cups  "))
    assert recorder.last_task == "stack the cups"
    recorder._handle(("start", 0, None))
    assert recorder.episode.task == "stack the cups"
    # Changing the task mid-episode only affects the next episode.
    recorder._handle(("set_task", "open the drawer"))
    assert recorder.episode.task == "stack the cups"
    recorder._handle(("stop", 0, "success"))
    recorder.episode.finalize_deadline_ns = 0
    recorder._handle(("tick",))
    status = recorder.status_queue.get_nowait()
    while not recorder.status_queue.empty():
        status = recorder.status_queue.get_nowait()
    assert status["next_task"] == "open the drawer"


def test_recorder_restores_last_task_after_restart(tmp_path: Path) -> None:
    recorder = make_recorder(tmp_path)
    recorder._handle(("start", 0, "fold the towel"))
    recorder._handle(("stop", 0, "success"))
    recorder.episode.finalize_deadline_ns = 0
    recorder._handle(("tick",))
    (tmp_path / "episode_000007.incomplete").mkdir()

    restarted = make_recorder(tmp_path)
    assert restarted.last_task == "fold the towel"
    assert restarted.next_index == 8
    restarted._handle(("start", 0, None))
    assert restarted.episode.task == "fold the towel"
