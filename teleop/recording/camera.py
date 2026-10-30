"""Wrist camera capture and per-episode encoding in a dedicated process.

librealsense calls such as ``pipeline.start()`` can block for seconds while
holding the GIL, so capture and encoding run in their own Camera process,
spawned by the recorder. Frames never cross a process boundary; only small
commands, replies and status do (``CameraProcessHandle``).

The camera streams continuously while the server runs so auto exposure is
settled before an episode starts. Frames whose host arrival time falls inside
the episode window are encoded by a dedicated thread:

    camera_<name>.mp4         RGB, H.264 (libx264, CRF 18), constant-rate pts
    camera_<name>_depth.mkv   optional uint16 depth aligned to color, FFV1 lossless
    camera_<name>_frames.bin  per-frame timestamps, see CAMERA_FRAME_LAYOUT

Video pts are frame indices; the real capture time of frame ``i`` is record
``i`` of the frames file. ``av``, ``numpy`` and ``pyrealsense2`` are imported
only when a camera is enabled, so the base teleop install does not need them.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from teleop.config import CameraConfig
from teleop.recording.schema import CAMERA_FRAME_LAYOUT


log = logging.getLogger(__name__)

# Two seconds of 30 fps frames may wait for the encoder before frames drop.
ENCODE_QUEUE_FRAMES = 60
# Frames older than this mean the camera is not streaming.
STREAM_STALE_S = 1.0
RESTART_DELAY_S = 2.0
# Keep x264/FFV1 from taking every core next to the 8 ms servo worker.
ENCODER_THREADS = 2
COLOR_CRF = "18"
DEPTH_SIZE = (640, 480)


class CameraUnavailable(RuntimeError):
    """The camera cannot record an episode right now."""


@dataclass(frozen=True)
class CameraFrame:
    color: Any  # numpy uint8 (H, W, 3) RGB
    depth: Any | None  # numpy uint16 (H, W), aligned to color
    arrival_mono_ns: int
    arrival_wall_ns: int
    device_ts_ms: float
    frame_number: int


class FrameSource(Protocol):
    def start(self) -> dict[str, Any]:
        """Open the device and return metadata (model, serial, intrinsics)."""

    def read(self, timeout_s: float) -> CameraFrame | None: ...

    def stop(self) -> None: ...


class RealSenseSource:
    """RealSense D435/D435i color (+ optional depth aligned to color)."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._rs: Any = None
        self._np: Any = None
        self._pipeline: Any = None
        self._align: Any = None

    def start(self) -> dict[str, Any]:
        import numpy
        import pyrealsense2 as rs

        self._rs = rs
        self._np = numpy
        # pipeline.start() without a device blocks ~15 s holding the GIL.
        # query_devices() returns immediately, so check it first.
        serials = [
            device.get_info(rs.camera_info.serial_number)
            for device in rs.context().query_devices()
        ]
        if not serials:
            raise RuntimeError("no RealSense device connected")
        if self.config.serial and self.config.serial not in serials:
            raise RuntimeError(
                f"RealSense serial {self.config.serial} not found; "
                f"connected: {', '.join(serials)}"
            )
        pipeline = rs.pipeline()
        rs_config = rs.config()
        if self.config.serial:
            rs_config.enable_device(self.config.serial)
        rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            self.config.fps,
        )
        if self.config.depth:
            rs_config.enable_stream(
                rs.stream.depth, DEPTH_SIZE[0], DEPTH_SIZE[1], rs.format.z16,
                self.config.fps,
            )
        profile = pipeline.start(rs_config)
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color) if self.config.depth else None

        device = profile.get_device()
        intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        info: dict[str, Any] = {
            "model": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "librealsense": rs.__version__,
            "color_intrinsics": {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "ppx": intrinsics.ppx,
                "ppy": intrinsics.ppy,
                "model": str(intrinsics.model),
                "coeffs": list(intrinsics.coeffs),
            },
        }
        if self.config.depth:
            info["depth_scale_m"] = device.first_depth_sensor().get_depth_scale()
            info["depth_aligned_to"] = "color"
        return info

    def read(self, timeout_s: float) -> CameraFrame | None:
        frames = self._pipeline.wait_for_frames(int(timeout_s * 1000))
        arrival_mono_ns = time.monotonic_ns()
        arrival_wall_ns = time.time_ns()
        if self._align is not None:
            frames = self._align.process(frames)
        color = frames.get_color_frame()
        if not color:
            return None
        depth_array = None
        if self.config.depth:
            depth = frames.get_depth_frame()
            if not depth:
                return None
            depth_array = self._np.array(depth.get_data(), copy=True)
        self.timestamp_domain = str(color.get_frame_timestamp_domain())
        return CameraFrame(
            # Copy so librealsense can recycle its frame buffer immediately.
            color=self._np.array(color.get_data(), copy=True),
            depth=depth_array,
            arrival_mono_ns=arrival_mono_ns,
            arrival_wall_ns=arrival_wall_ns,
            device_ts_ms=float(color.get_timestamp()),
            frame_number=int(color.get_frame_number()),
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None


class FakeCameraSource:
    """Synthetic paced frames so dry-run tests exercise the full pipeline."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.timestamp_domain = "fake_monotonic"
        self._np: Any = None
        self._next_ns = 0
        self._frame_number = 0
        self._started = False

    def start(self) -> dict[str, Any]:
        import numpy

        self._np = numpy
        self._next_ns = time.monotonic_ns()
        self._frame_number = 0
        self._started = True
        return {"model": "fake", "serial": "fake", "librealsense": None}

    def read(self, timeout_s: float) -> CameraFrame | None:
        if not self._started:
            raise RuntimeError("fake camera is not started")
        period_ns = 1_000_000_000 // self.config.fps
        self._next_ns += period_ns
        delay_s = (self._next_ns - time.monotonic_ns()) / 1e9
        if delay_s > timeout_s:
            return None
        if delay_s > 0:
            time.sleep(delay_s)
        else:
            self._next_ns = time.monotonic_ns()
        self._frame_number += 1
        width, height = self.config.width, self.config.height
        color = self._np.zeros((height, width, 3), dtype=self._np.uint8)
        bar = (self._frame_number * 8) % width
        color[:, bar:bar + 8, :] = 255
        color[:, :, 1] = self._frame_number % 256
        depth = None
        if self.config.depth:
            depth = self._np.full(
                (height, width), 500 + self._frame_number, dtype=self._np.uint16
            )
        return CameraFrame(
            color=color,
            depth=depth,
            arrival_mono_ns=time.monotonic_ns(),
            arrival_wall_ns=time.time_ns(),
            device_ts_ms=time.monotonic_ns() / 1e6,
            frame_number=self._frame_number,
        )

    def stop(self) -> None:
        self._started = False


def make_frame_source(config: CameraConfig) -> FrameSource:
    if config.source == "fake":
        return FakeCameraSource(config)
    return RealSenseSource(config)


class EpisodeVideoWriter:
    """Encodes one episode's frames on its own thread."""

    def __init__(
        self,
        config: CameraConfig,
        directory: Path,
        start_ns: int,
    ) -> None:
        import av

        self.config = config
        self.start_ns = start_ns
        self.stop_ns: int | None = None
        self.frames = 0
        self.dropped = 0
        self.first_arrival_ns: int | None = None
        self.last_arrival_ns: int | None = None
        self.error: str | None = None
        self._queue: queue.Queue[CameraFrame | None] = queue.Queue(
            maxsize=ENCODE_QUEUE_FRAMES
        )
        rate = Fraction(config.fps, 1)
        self.color_path = directory / f"camera_{config.name}.mp4"
        self.frames_path = directory / f"camera_{config.name}_frames.bin"
        self.depth_path = (
            directory / f"camera_{config.name}_depth.mkv" if config.depth else None
        )
        self._color = av.open(str(self.color_path), mode="w")
        self._color_stream = self._color.add_stream("libx264", rate=rate)
        self._color_stream.width = config.width
        self._color_stream.height = config.height
        self._color_stream.pix_fmt = "yuv420p"
        self._color_stream.codec_context.thread_count = ENCODER_THREADS
        # One keyframe per second keeps random access cheap for conversion.
        self._color_stream.options = {
            "crf": COLOR_CRF,
            "preset": "veryfast",
            "g": str(config.fps),
        }
        self._depth: Any = None
        self._depth_stream: Any = None
        if self.depth_path is not None:
            self._depth = av.open(str(self.depth_path), mode="w")
            self._depth_stream = self._depth.add_stream("ffv1", rate=rate)
            self._depth_stream.width = config.width
            self._depth_stream.height = config.height
            self._depth_stream.pix_fmt = "gray16le"
            self._depth_stream.codec_context.thread_count = ENCODER_THREADS
        self._frames_file: BinaryIO = self.frames_path.open("wb")
        self._av = av
        self._thread = threading.Thread(
            target=self._run, name=f"camera-{config.name}-encoder", daemon=True
        )
        self._thread.start()

    def offer(self, frame: CameraFrame) -> None:
        """Called from the capture thread; never blocks."""
        if frame.arrival_mono_ns < self.start_ns:
            return
        if self.stop_ns is not None and frame.arrival_mono_ns > self.stop_ns:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            if self.error is not None:
                continue
            try:
                self._encode(frame)
            except Exception as exc:  # encoder boundary: keep robot data
                self.error = f"{type(exc).__name__}: {exc}"
                log.error(
                    "camera encoding failed camera=%s frame=%d error=%s",
                    self.config.name,
                    self.frames,
                    self.error,
                )

    def _encode(self, frame: CameraFrame) -> None:
        index = self.frames
        time_base = Fraction(1, self.config.fps)
        video_frame = self._av.VideoFrame.from_ndarray(frame.color, format="rgb24")
        video_frame.pts = index
        video_frame.time_base = time_base
        for packet in self._color_stream.encode(video_frame):
            self._color.mux(packet)
        if self._depth is not None and frame.depth is not None:
            depth_frame = self._av.VideoFrame.from_ndarray(
                frame.depth, format="gray16le"
            )
            depth_frame.pts = index
            depth_frame.time_base = time_base
            for packet in self._depth_stream.encode(depth_frame):
                self._depth.mux(packet)
        self._frames_file.write(
            CAMERA_FRAME_LAYOUT.pack(
                (
                    index,
                    frame.arrival_mono_ns,
                    frame.arrival_wall_ns,
                    frame.device_ts_ms,
                    frame.frame_number,
                )
            )
        )
        if self.first_arrival_ns is None:
            self.first_arrival_ns = frame.arrival_mono_ns
        self.last_arrival_ns = frame.arrival_mono_ns
        self.frames += 1

    def close(self) -> dict[str, Any]:
        """Flush queued frames and close all files; returns episode metadata."""
        self._queue.put(None)
        self._thread.join()
        try:
            for packet in self._color_stream.encode(None):
                self._color.mux(packet)
            if self._depth is not None:
                for packet in self._depth_stream.encode(None):
                    self._depth.mux(packet)
        except Exception as exc:  # encoder boundary
            self.error = self.error or f"{type(exc).__name__}: {exc}"
        finally:
            self._color.close()
            if self._depth is not None:
                self._depth.close()
            self._frames_file.flush()
            self._frames_file.close()
        return {
            "video": self.color_path.name,
            "depth_video": self.depth_path.name if self.depth_path else None,
            "frames_file": self.frames_path.name,
            "frames": self.frames,
            "dropped_frames": self.dropped,
            "first_arrival_ns": self.first_arrival_ns,
            "last_arrival_ns": self.last_arrival_ns,
            "error": self.error,
        }


class CameraRecorder:
    """Owns the capture thread and the active episode writer."""

    def __init__(self, config: CameraConfig, source: FrameSource) -> None:
        self.config = config
        self.source = source
        self.device_info: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.last_frame_ns = 0
        self.frames_seen = 0
        self.fps_measured: float | None = None
        self._writer: EpisodeVideoWriter | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-{config.name}-capture",
            daemon=True,
        )

    @property
    def streaming(self) -> bool:
        return (
            self.last_frame_ns > 0
            and time.monotonic_ns() - self.last_frame_ns < STREAM_STALE_S * 1e9
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _capture_loop(self) -> None:
        started = False
        previous_ns = 0
        while not self._stop.is_set():
            if not started:
                try:
                    self.device_info = self.source.start()
                except Exception as exc:  # device boundary: retry while idle
                    self.last_error = f"camera start failed: {exc}"
                    log.warning("%s; retrying in %.0f s", self.last_error, RESTART_DELAY_S)
                    self._stop.wait(RESTART_DELAY_S)
                    continue
                started = True
                self.last_error = None
                log.info("camera streaming name=%s info=%s", self.config.name, {
                    key: self.device_info.get(key)
                    for key in ("model", "serial", "firmware")
                })
            try:
                frame = self.source.read(timeout_s=1.0)
            except Exception as exc:  # device boundary: restart the stream
                self.last_error = f"camera read failed: {exc}"
                log.warning("%s; restarting stream", self.last_error)
                started = False
                try:
                    self.source.stop()
                except Exception:  # already failing; restart handles it
                    pass
                self._stop.wait(RESTART_DELAY_S)
                continue
            if frame is None:
                continue
            if previous_ns:
                interval_s = (frame.arrival_mono_ns - previous_ns) / 1e9
                if interval_s > 0:
                    instant = 1.0 / interval_s
                    self.fps_measured = (
                        instant
                        if self.fps_measured is None
                        else 0.9 * self.fps_measured + 0.1 * instant
                    )
            previous_ns = frame.arrival_mono_ns
            self.last_frame_ns = frame.arrival_mono_ns
            self.frames_seen += 1
            with self._lock:
                writer = self._writer
            if writer is not None:
                writer.offer(frame)
        if started:
            try:
                self.source.stop()
            except Exception as exc:  # shutting down anyway
                log.warning("camera stop failed: %s", exc)

    def begin_episode(self, directory: Path, start_ns: int) -> None:
        if not self.streaming:
            raise CameraUnavailable(self.last_error or "camera is not streaming")
        try:
            writer = EpisodeVideoWriter(self.config, directory, start_ns)
        except Exception as exc:  # encoder boundary
            raise CameraUnavailable(f"camera encoder init failed: {exc}") from exc
        with self._lock:
            self._writer = writer

    def end_episode(self, stop_ns: int) -> None:
        with self._lock:
            if self._writer is not None:
                self._writer.stop_ns = stop_ns

    def caught_up(self, stop_ns: int) -> bool:
        return self.last_frame_ns >= stop_ns

    def finish_episode(self) -> dict[str, Any] | None:
        with self._lock:
            writer = self._writer
            self._writer = None
        if writer is None:
            return None
        metadata = writer.close()
        metadata.update(
            {
                "name": self.config.name,
                "width": self.config.width,
                "height": self.config.height,
                "fps": self.config.fps,
                "codec": "h264 (libx264 crf " + COLOR_CRF + ", yuv420p)",
                "depth_codec": "ffv1 gray16le" if self.config.depth else None,
                "timestamp_domain": getattr(self.source, "timestamp_domain", None),
                "frame_layout": CAMERA_FRAME_LAYOUT.metadata(),
                "device": self.device_info,
            }
        )
        return metadata

    def status(self) -> dict[str, Any]:
        with self._lock:
            writer = self._writer
        return {
            "name": self.config.name,
            "streaming": self.streaming,
            "fps": self.fps_measured,
            "episode_frames": writer.frames if writer else None,
            "episode_dropped": writer.dropped if writer else None,
            "last_error": self.last_error,
            "serial": (self.device_info or {}).get("serial"),
            "last_frame_ns": self.last_frame_ns,
        }


# ── Camera process ────────────────────────────────────────────────

CAMERA_STATUS_PERIOD_S = 0.05
BEGIN_TIMEOUT_S = 3.0
# Flushing up to ENCODE_QUEUE_FRAMES queued frames plus closing the files.
FINISH_TIMEOUT_S = 5.0


def run_camera_process(
    config: CameraConfig,
    command_queue: Any,
    event_queue: Any,
    log_level: str,
) -> None:
    """Entry point of the Camera process (child of the recorder)."""
    import os
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=(
            "%(asctime)s [pid=%(process)d] [%(processName)s] "
            "[%(levelname)s] %(message)s"
        ),
    )
    event_queue.cancel_join_thread()
    parent_pid = os.getppid()
    camera = CameraRecorder(config, make_frame_source(config))
    camera.start()
    last_status_ns = 0
    try:
        while os.getppid() == parent_pid:
            try:
                command = command_queue.get(timeout=CAMERA_STATUS_PERIOD_S)
            except queue.Empty:
                command = None
            if command is not None:
                kind = command[0]
                if kind == "stop":
                    break
                if kind == "begin":
                    _, request_id, directory, start_ns = command
                    try:
                        camera.begin_episode(Path(directory), start_ns)
                        reply: Any = None
                    except CameraUnavailable as exc:
                        reply = str(exc)
                    event_queue.put(("reply", request_id, reply))
                elif kind == "end":
                    camera.end_episode(command[1])
                elif kind == "finish":
                    event_queue.put(("reply", command[1], camera.finish_episode()))
            now_ns = time.monotonic_ns()
            if now_ns - last_status_ns >= CAMERA_STATUS_PERIOD_S * 1e9:
                try:
                    event_queue.put_nowait(("status", camera.status()))
                    last_status_ns = now_ns
                except queue.Full:
                    pass
    finally:
        camera.finish_episode()
        camera.stop()


class CameraProcessHandle:
    """Recorder-side proxy with the same interface as ``CameraRecorder``."""

    def __init__(self, config: CameraConfig, log_level: str = "INFO") -> None:
        import multiprocessing

        self.config = config
        self.log_level = log_level
        self._context = multiprocessing.get_context("spawn")
        self.process: Any = None
        self._commands: Any = None
        self._events: Any = None
        self._status: dict[str, Any] = {}
        self._replies: dict[int, Any] = {}
        self._next_request = 0

    def start(self) -> None:
        self._commands = self._context.Queue(maxsize=64)
        self._events = self._context.Queue(maxsize=256)
        self.process = self._context.Process(
            target=run_camera_process,
            args=(self.config, self._commands, self._events, self.log_level),
            name=f"Camera-{self.config.name}",
            daemon=True,
        )
        self.process.start()
        log.info("camera process started pid=%s", self.process.pid)

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    @property
    def last_error(self) -> str | None:
        self._drain()
        if not self.alive and self.process is not None:
            return f"camera process exited (code {self.process.exitcode})"
        return self._status.get("last_error")

    @property
    def streaming(self) -> bool:
        self._drain()
        return self.alive and bool(self._status.get("streaming"))

    def _drain(self, timeout_s: float = 0.0) -> None:
        if self._events is None:
            return
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            try:
                if remaining > 0:
                    event = self._events.get(timeout=remaining)
                else:
                    event = self._events.get_nowait()
            except queue.Empty:
                return
            if event[0] == "status":
                self._status = event[1]
            elif event[0] == "reply":
                self._replies[event[1]] = event[2]
                return

    def _request(self, message: tuple[Any, ...], timeout_s: float) -> tuple[bool, Any]:
        if not self.alive:
            return False, None
        request_id = self._next_request
        self._next_request += 1
        try:
            self._commands.put((message[0], request_id, *message[1:]), timeout=1.0)
        except queue.Full:
            return False, None
        deadline = time.monotonic() + timeout_s
        while request_id not in self._replies:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.alive:
                return False, None
            self._drain(min(remaining, 0.1))
        return True, self._replies.pop(request_id)

    def begin_episode(self, directory: Path, start_ns: int) -> None:
        if not self.streaming:
            raise CameraUnavailable(self.last_error or "camera is not streaming")
        ok, error = self._request(("begin", str(directory), start_ns), BEGIN_TIMEOUT_S)
        if not ok:
            raise CameraUnavailable("camera process did not answer")
        if error is not None:
            raise CameraUnavailable(error)

    def end_episode(self, stop_ns: int) -> None:
        if self.alive:
            try:
                self._commands.put(("end", stop_ns), timeout=1.0)
            except queue.Full:
                log.error("camera command queue full; end_episode lost")

    def caught_up(self, stop_ns: int) -> bool:
        self._drain()
        return int(self._status.get("last_frame_ns") or 0) >= stop_ns

    def finish_episode(self) -> dict[str, Any] | None:
        ok, metadata = self._request(("finish",), FINISH_TIMEOUT_S)
        if not ok:
            log.error("camera process did not finish the episode video")
            return {
                "name": self.config.name,
                "frames": 0,
                "dropped_frames": 0,
                "error": "camera process did not finish the episode",
                "video": f"camera_{self.config.name}.mp4",
                "frames_file": f"camera_{self.config.name}_frames.bin",
            }
        return metadata

    def status(self) -> dict[str, Any]:
        self._drain()
        status = dict(self._status) if self._status else {
            "name": self.config.name,
            "streaming": False,
            "fps": None,
            "episode_frames": None,
            "episode_dropped": None,
            "last_error": None,
            "serial": None,
        }
        status["streaming"] = self.streaming
        if not self.alive and self.process is not None:
            status["last_error"] = self.last_error
        status.pop("last_frame_ns", None)
        return status

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.is_alive():
            try:
                self._commands.put(("stop",), timeout=1.0)
            except queue.Full:
                pass
            process.join(timeout=2.0)
        if process.is_alive():
            log.error("camera process did not stop; terminating")
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        for camera_queue in (self._commands, self._events):
            try:
                camera_queue.close()
                camera_queue.cancel_join_thread()
            except (OSError, ValueError):
                pass
        self.process = None
