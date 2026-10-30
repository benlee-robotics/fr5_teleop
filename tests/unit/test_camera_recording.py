from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import pytest

from teleop.config import CameraConfig, RecordingConfig, TeleopConfig
from teleop.recording.camera import (
    CameraRecorder,
    CameraUnavailable,
    FakeCameraSource,
)
from teleop.recording.recorder import RecorderRuntime, RecorderSpec, RecorderState
from teleop.recording.schema import CAMERA_FRAME_LAYOUT

from tests.unit.test_recording import sample as robot_sample


av = pytest.importorskip("av")
numpy = pytest.importorskip("numpy")


def fake_camera(**overrides) -> CameraConfig:
    values = {
        "enabled": True,
        "source": "fake",
        "width": 64,
        "height": 48,
        "fps": 30,
    }
    values.update(overrides)
    return CameraConfig(**values)


def wait_until(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def decode_frames(path: Path, pixel_format: str) -> list:
    with av.open(str(path)) as container:
        return [
            frame.to_ndarray(format=pixel_format)
            for frame in container.decode(video=0)
        ]


@pytest.mark.parametrize("depth", [False, True])
def test_camera_episode_video_matches_frame_timestamps(
    tmp_path: Path, depth: bool
) -> None:
    config = fake_camera(depth=depth)
    camera = CameraRecorder(config, FakeCameraSource(config))
    camera.start()
    try:
        wait_until(lambda: camera.streaming)
        start_ns = time.monotonic_ns()
        camera.begin_episode(tmp_path, start_ns)
        time.sleep(0.4)
        stop_ns = time.monotonic_ns()
        camera.end_episode(stop_ns)
        wait_until(lambda: camera.caught_up(stop_ns))
        meta = camera.finish_episode()
    finally:
        camera.stop()

    assert meta["error"] is None
    assert meta["dropped_frames"] == 0
    # 0.4 s at 30 fps, allowing for scheduling slack.
    assert 8 <= meta["frames"] <= 14
    assert meta["timestamp_domain"] == "fake_monotonic"

    records = list(CAMERA_FRAME_LAYOUT.iter_records(tmp_path / meta["frames_file"]))
    assert [r["frame_index"] for r in records] == list(range(meta["frames"]))
    arrivals = [r["arrival_mono_ns"] for r in records]
    assert arrivals == sorted(arrivals)
    assert start_ns <= arrivals[0] and arrivals[-1] <= stop_ns
    numbers = [r["frame_number"] for r in records]
    assert numbers == list(range(numbers[0], numbers[0] + len(numbers)))

    color = decode_frames(tmp_path / meta["video"], "rgb24")
    assert len(color) == meta["frames"]
    assert color[0].shape == (48, 64, 3)

    if depth:
        depth_frames = decode_frames(tmp_path / meta["depth_video"], "gray16le")
        assert len(depth_frames) == meta["frames"]
        # FFV1 is lossless: the fake writes 500 + frame_number everywhere.
        for frame, number in zip(depth_frames, numbers):
            assert frame.dtype == numpy.uint16
            assert (frame == 500 + number).all()
    else:
        assert meta["depth_video"] is None


class BrokenSource(FakeCameraSource):
    def start(self):
        raise RuntimeError("No device connected")


def make_recorder(
    root: Path, cameras: list[tuple[CameraConfig, object]]
) -> RecorderRuntime:
    runtime = RecorderRuntime(
        RecorderSpec(
            root_dir=root,
            hands=("right",),
            session_metadata={},
            cameras=tuple(config for config, _ in cameras),
        ),
        queue.Queue(),
        queue.Queue(),
    )
    runtime._prepare_root()
    for config, source in cameras:
        camera = CameraRecorder(config, source)
        camera.start()
        runtime.cameras[config.name] = camera
    return runtime


def stop_cameras(recorder: RecorderRuntime) -> None:
    for camera in recorder.cameras.values():
        camera.stop()


def test_recording_refuses_to_start_without_camera_stream(tmp_path: Path) -> None:
    config = fake_camera()
    recorder = make_recorder(tmp_path, [(config, BrokenSource(config))])
    try:
        wait_until(lambda: recorder.cameras["wrist"].last_error is not None)
        recorder._handle(("start", time.monotonic_ns(), "task"))
        assert recorder.state == RecorderState.IDLE
        assert "No device connected" in recorder.last_error
        assert not list(tmp_path.iterdir())
        status = recorder.cameras["wrist"].status()
        assert status["streaming"] is False
    finally:
        stop_cameras(recorder)


def test_recording_refuses_to_start_if_any_camera_is_down(tmp_path: Path) -> None:
    wrist = fake_camera()
    top = fake_camera(name="top")
    recorder = make_recorder(
        tmp_path,
        [(wrist, FakeCameraSource(wrist)), (top, BrokenSource(top))],
    )
    try:
        wait_until(lambda: recorder.cameras["wrist"].streaming)
        wait_until(lambda: recorder.cameras["top"].last_error is not None)
        recorder._handle(("start", time.monotonic_ns(), "task"))
        assert recorder.state == RecorderState.IDLE
        assert "camera top" in recorder.last_error
        # The writer already opened for the healthy camera is closed again.
        assert recorder.cameras["wrist"].status()["episode_frames"] is None
        assert not list(tmp_path.iterdir())
    finally:
        stop_cameras(recorder)


@pytest.mark.parametrize("names", [("wrist",), ("wrist", "top")])
def test_recorder_episode_includes_camera_metadata(
    tmp_path: Path, names: tuple[str, ...]
) -> None:
    configs = [fake_camera(name=name) for name in names]
    recorder = make_recorder(
        tmp_path, [(config, FakeCameraSource(config)) for config in configs]
    )
    try:
        wait_until(lambda: all(c.streaming for c in recorder.cameras.values()))
        start_ns = time.monotonic_ns()
        recorder._handle(("start", start_ns, "wipe the table"))
        time.sleep(0.2)
        stop_ns = time.monotonic_ns()
        recorder._handle(("samples", "right", 1, 0, 0, [robot_sample(stop_ns)]))
        recorder._handle(("stop", stop_ns, "success"))
        wait_until(
            lambda: all(c.caught_up(stop_ns) for c in recorder.cameras.values())
        )
        recorder._handle(("tick",))
        assert recorder.state == RecorderState.IDLE
    finally:
        stop_cameras(recorder)
    meta = json.loads((tmp_path / "episode_000000" / "meta.json").read_text())
    assert sorted(meta["cameras"]) == sorted(names)
    for name in names:
        camera_meta = meta["cameras"][name]
        assert camera_meta["frames"] > 0
        assert camera_meta["video"] == f"camera_{name}.mp4"
        assert camera_meta["frame_layout"]["record_size"] == CAMERA_FRAME_LAYOUT.size
    assert meta["complete"] is True
    assert recorder.last_saved["camera_frames"] == {
        name: meta["cameras"][name]["frames"] for name in names
    }


def test_camera_begin_episode_requires_streaming(tmp_path: Path) -> None:
    config = fake_camera()
    camera = CameraRecorder(config, FakeCameraSource(config))
    with pytest.raises(CameraUnavailable, match="not streaming"):
        camera.begin_episode(tmp_path, 0)


def test_camera_config_parsing_and_validation(tmp_path: Path) -> None:
    web = Path(__file__).resolve().parents[2] / "web"
    path = tmp_path / "config.yaml"
    path.write_text(
        "tls: {enabled: false}\n"
        f"server: {{web_dir: {web}}}\n"
        "recording:\n"
        "  enabled: true\n"
        "  cameras:\n"
        "    - {enabled: true, source: fake, serial: 123456789012,\n"
        "       width: 1280, height: 720, fps: 15, depth: true}\n"
        "    - {enabled: false, name: top}\n",
        encoding="utf-8",
    )
    recording = TeleopConfig.from_yaml(path).recording
    camera, top = recording.cameras
    assert camera.enabled and camera.depth
    assert camera.name == "wrist"
    assert camera.serial == "123456789012"
    assert (camera.width, camera.height, camera.fps) == (1280, 720, 15)
    assert not top.enabled
    assert recording.enabled_cameras == (camera,)

    with pytest.raises(ValueError, match="not a D435 color resolution"):
        CameraConfig(width=800, height=600).validate()
    with pytest.raises(ValueError, match="fps"):
        CameraConfig(fps=25).validate()
    with pytest.raises(ValueError, match="source"):
        CameraConfig(source="webcam").validate()
    with pytest.raises(ValueError, match="name"):
        CameraConfig(name="wrist cam").validate()
    # A disabled recording section never requires camera packages.
    RecordingConfig(enabled=False, cameras=(CameraConfig(enabled=True),)).validate()


def test_legacy_single_camera_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "recording:\n  camera: {enabled: true, source: fake}\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="replaced by the recording.cameras list"):
        TeleopConfig.from_yaml(path)


def test_multiple_cameras_need_unique_names_and_serials() -> None:
    def recording(*cameras: CameraConfig) -> RecordingConfig:
        return RecordingConfig(enabled=True, cameras=cameras)

    with pytest.raises(ValueError, match="names must be unique: wrist"):
        recording(fake_camera(), fake_camera()).validate()
    # Names must be unique even while recording is disabled.
    with pytest.raises(ValueError, match="names must be unique"):
        RecordingConfig(cameras=(fake_camera(), fake_camera())).validate()

    wrist = CameraConfig(enabled=True, name="wrist", serial="111")
    top = CameraConfig(enabled=True, name="top")
    with pytest.raises(ValueError, match="need a serial .*: top"):
        recording(wrist, top).validate()
    with pytest.raises(ValueError, match="serials must be unique"):
        recording(wrist, CameraConfig(enabled=True, name="top", serial="111")).validate()
    # A disabled second RealSense does not need a serial.
    recording(wrist, CameraConfig(enabled=False, name="top")).validate()
    # Fake cameras never need one.
    recording(fake_camera(), fake_camera(name="top")).validate()


def test_camera_process_records_episode(tmp_path: Path) -> None:
    from teleop.recording.camera import CameraProcessHandle

    handle = CameraProcessHandle(fake_camera(depth=True))
    handle.start()
    try:
        wait_until(lambda: handle.streaming, timeout=10.0)
        start_ns = time.monotonic_ns()
        handle.begin_episode(tmp_path, start_ns)
        time.sleep(0.3)
        stop_ns = time.monotonic_ns()
        handle.end_episode(stop_ns)
        wait_until(lambda: handle.caught_up(stop_ns))
        meta = handle.finish_episode()
        assert handle.status()["streaming"] is True
    finally:
        handle.stop()
    assert meta["error"] is None
    assert meta["frames"] >= 5
    assert len(decode_frames(tmp_path / meta["video"], "rgb24")) == meta["frames"]
    assert len(decode_frames(tmp_path / meta["depth_video"], "gray16le")) == meta["frames"]
    assert handle.process is None


def test_realsense_without_device_reports_error_without_hanging(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyrealsense2")
    from teleop.recording.camera import CameraProcessHandle

    import pyrealsense2 as rs

    if len(rs.context().query_devices()) > 0:
        pytest.skip("a RealSense device is connected")
    handle = CameraProcessHandle(CameraConfig(enabled=True, source="realsense"))
    handle.start()
    try:
        wait_until(lambda: handle.last_error is not None, timeout=10.0)
        assert "no RealSense device connected" in handle.last_error
        with pytest.raises(CameraUnavailable, match="no RealSense device"):
            handle.begin_episode(tmp_path, time.monotonic_ns())
        started = time.monotonic()
    finally:
        handle.stop()
    assert time.monotonic() - started < 3.0
