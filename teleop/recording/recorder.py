"""Raw episode recorder process and its parent-side handle.

Data path::

    RobotWorker --sample batches--> parent (asyncio) --commands+samples--> Recorder

Only the parent writes to the recorder queue, so a killed worker can never
corrupt it. Recording is observation-only: nothing here sends robot commands,
and recorder failures only mark the episode, never the robot pipeline.

Episode layout (``root_dir/episode_000123/``)::

    meta.json          task, outcome, timing, per-hand counts, sample schema
    robot_<hand>.bin   packed samples, see ``teleop.recording.schema``
    events.jsonl       start/stop/worker-generation events
    camera_<name>*     optional camera video + frame timestamps, see ``camera``

An episode is written to ``episode_NNNNNN.tmp`` and renamed when finalized.
Leftover ``.tmp`` directories from a crash are renamed to ``.incomplete``.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import queue
import re
import shutil
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from teleop import __version__
from teleop.config import PROJECT_ROOT, CameraConfig, TeleopConfig
from teleop.recording.camera import (
    CameraProcessHandle,
    CameraRecorder,
    CameraUnavailable,
)
from teleop.recording.schema import (
    RAW_FORMAT,
    RAW_FORMAT_VERSION,
    pack_sample,
    schema_metadata,
)


log = logging.getLogger(__name__)

EPISODE_PATTERN = re.compile(r"^episode_(\d{6})(?:\.[a-z]+)?$")
TMP_SUFFIX = ".tmp"
INCOMPLETE_SUFFIX = ".incomplete"
# How long to wait after stop for in-flight worker samples to arrive.
TAIL_WAIT_S = 1.0
STATUS_PERIOD_S = 0.2
COMMAND_QUEUE_SIZE = 1024


class RecorderState(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    FINALIZING = "FINALIZING"


class EpisodeOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    # Recorder or server shut down while the episode was still open.
    INTERRUPTED = "interrupted"


class RecorderUnavailable(RuntimeError):
    """The recorder process cannot accept a command right now."""


@dataclass(frozen=True)
class RecorderSpec:
    root_dir: Path
    hands: tuple[str, ...]
    session_metadata: dict[str, Any]
    log_level: str = "INFO"
    cameras: tuple[CameraConfig, ...] = ()


@dataclass
class _HandStream:
    file: BinaryIO
    path: Path
    samples: int = 0
    first_t_ns: int | None = None
    last_t_ns: int | None = None
    worker_dropped_batches: int = 0
    generations: list[int] = field(default_factory=list)


@dataclass
class _Episode:
    index: int
    task: str
    tmp_dir: Path
    start_ns: int
    started_at: str
    events: TextIO
    streams: dict[str, _HandStream] = field(default_factory=dict)
    forward_dropped_batches: int = 0
    stop_ns: int | None = None
    outcome: EpisodeOutcome | None = None
    finalize_deadline_ns: int = 0


def build_session_metadata(
    config: TeleopConfig,
    *,
    robot_kind: str,
    hands: tuple[str, ...],
) -> dict[str, Any]:
    """Static per-server-run facts copied into every episode's meta.json."""
    arms: dict[str, dict[str, Any]] = {}
    if config.arms:
        for arm in config.arms:
            arms[arm.hand] = {
                "robot_ip": arm.robot_ip,
                "sdk_path": str(arm.sdk_path or config.sdk_path),
                "gripper_open_position": arm.gripper.open_position,
                "gripper_closed_position": arm.gripper.closed_position,
                "gripper_enabled": arm.gripper.enabled,
            }
    else:
        arms[hands[0]] = {
            "robot_ip": config.robot_ip,
            "sdk_path": str(config.sdk_path),
            "gripper_open_position": config.gripper.open_position,
            "gripper_closed_position": config.gripper.closed_position,
            "gripper_enabled": config.gripper.enabled,
        }
    return {
        "teleop_version": __version__,
        "robot_kind": robot_kind,
        "dry_run": config.dry_run,
        "hands": list(hands),
        "arms": arms,
        "servo_period_s": config.servo_period_s,
        "pose_timeout_s": config.pose_timeout_s,
        "position_scale": config.position_scale,
        "ema_alpha": config.ema_alpha,
        "max_velocity_mm_s": config.max_velocity_mm_s,
        "max_step_mm": config.max_step_mm,
        "orientation_enabled": config.orientation.enabled,
        "tcp_units": "mm, deg (Fairino rx,ry,rz)",
        "joint_units": "deg",
        "clock": "CLOCK_MONOTONIC ns, shared by worker and recorder",
    }


class RecorderRuntime:
    """Runs inside the recorder process and owns all episode files."""

    def __init__(
        self,
        spec: RecorderSpec,
        command_queue: Any,
        status_queue: Any,
    ) -> None:
        self.spec = spec
        self.command_queue = command_queue
        self.status_queue = status_queue
        self.parent_pid = os.getppid()
        self.state = RecorderState.IDLE
        self.episode: _Episode | None = None
        self.next_index = 0
        self.saved_count = 0
        self.last_task: str | None = None
        self.last_saved: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.error_seq = 0
        self.session_metadata = dict(spec.session_metadata)
        self._latest_seen_t_ns: dict[str, int] = {}
        self._worker_dropped: dict[tuple[str, int], int] = {}
        self._forward_dropped = 0
        self._last_status_ns = 0
        # One CameraProcessHandle per enabled camera, keyed by camera name.
        # Tests may inject in-process CameraRecorders (same interface).
        self.cameras: dict[str, CameraProcessHandle | CameraRecorder] = {}

    # ── lifecycle ─────────────────────────────────────────

    def start_cameras(self) -> None:
        # A separate process per camera: a blocking librealsense call on one
        # device must not stall the others.
        for camera_config in self.spec.cameras:
            if not camera_config.enabled:
                continue
            camera = CameraProcessHandle(camera_config, self.spec.log_level)
            camera.start()
            self.cameras[camera_config.name] = camera

    def run(self) -> None:
        self._prepare_root()
        self.session_metadata["git_commit"] = _git_commit()
        self.start_cameras()
        self._publish_status(force=True)
        try:
            self._run_loop()
        finally:
            self._for_each_camera(lambda camera: camera.stop())

    def _run_loop(self) -> None:
        running = True
        while running:
            for message in self._next_messages():
                if message[0] == "shutdown":
                    running = False
                    break
                self._handle_safely(message)
            if os.getppid() != self.parent_pid:
                log.error("recorder parent exited; closing open episode")
                running = False
            self._handle_safely(("tick",))
            self._publish_status()
        if self.episode is not None:
            self._handle_safely(("shutdown_close",))
        self._publish_status(force=True)

    def _next_messages(self) -> list[tuple[Any, ...]]:
        try:
            messages = [self.command_queue.get(timeout=0.05)]
        except queue.Empty:
            return []
        while len(messages) < 256:
            try:
                messages.append(self.command_queue.get_nowait())
            except queue.Empty:
                break
        return messages

    def _prepare_root(self) -> None:
        root = self.spec.root_dir
        root.mkdir(parents=True, exist_ok=True)
        highest = -1
        latest_saved: Path | None = None
        for entry in root.iterdir():
            match = EPISODE_PATTERN.match(entry.name)
            if match is None or not entry.is_dir():
                continue
            highest = max(highest, int(match.group(1)))
            # Zero-padded names sort numerically; suffixed dirs are unfinished.
            if "." not in entry.name and (
                latest_saved is None or entry.name > latest_saved.name
            ):
                latest_saved = entry
            if entry.name.endswith(TMP_SUFFIX):
                target = entry.with_name(
                    entry.name[: -len(TMP_SUFFIX)] + INCOMPLETE_SUFFIX
                )
                log.warning(
                    "recorder found unfinished episode %s; renamed to %s",
                    entry.name,
                    target.name,
                )
                entry.rename(target)
        self.next_index = highest + 1
        if latest_saved is not None:
            # Restore the task so Quest A works right after a restart.
            try:
                meta = json.loads(
                    (latest_saved / "meta.json").read_text(encoding="utf-8")
                )
                task = meta.get("task")
                if isinstance(task, str) and task.strip():
                    self.last_task = task.strip()
            except (OSError, ValueError) as exc:
                log.warning("could not read task from %s: %s", latest_saved.name, exc)
        log.info(
            "recorder ready root=%s next_episode=%d hands=%s",
            root,
            self.next_index,
            ",".join(self.spec.hands),
        )

    # ── message handling ──────────────────────────────────

    def _handle_safely(self, message: tuple[Any, ...]) -> None:
        try:
            self._handle(message)
        except OSError as exc:
            log.error(
                "recorder I/O failure during %s: %s",
                message[0],
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            self._abort_episode(f"I/O error: {exc}")

    def _handle(self, message: tuple[Any, ...]) -> None:
        kind = message[0]
        if kind == "samples":
            _, hand, generation, worker_dropped, forward_dropped, samples = message
            self._on_samples(hand, generation, worker_dropped, forward_dropped, samples)
        elif kind == "start":
            _, start_ns, task = message
            self._on_start(start_ns, task)
        elif kind == "stop":
            _, stop_ns, outcome = message
            self._on_stop(stop_ns, EpisodeOutcome(outcome))
        elif kind == "discard":
            self._on_discard()
        elif kind == "set_task":
            self._on_set_task(message[1])
        elif kind == "tick":
            self._maybe_finalize(time.monotonic_ns())
        elif kind == "shutdown_close":
            episode = self.episode
            if episode is not None:
                if episode.stop_ns is None:
                    episode.stop_ns = time.monotonic_ns()
                    episode.outcome = EpisodeOutcome.INTERRUPTED
                    self._event("stop", outcome=episode.outcome.value)
                    for camera in self.cameras.values():
                        camera.end_episode(episode.stop_ns)
                self._finalize()
        else:
            log.warning("recorder ignored unknown message kind=%r", kind)

    def _on_start(self, start_ns: int, task: str | None) -> None:
        if self.episode is not None:
            self._set_error("episode already in progress")
            return
        task = (task or self.last_task or "").strip()
        if not task:
            self._set_error("a language task is required to start recording")
            return
        index = self.next_index
        while any(
            (self.spec.root_dir / f"episode_{index:06d}{suffix}").exists()
            for suffix in ("", TMP_SUFFIX, INCOMPLETE_SUFFIX)
        ):
            index += 1
        tmp_dir = self.spec.root_dir / f"episode_{index:06d}{TMP_SUFFIX}"
        tmp_dir.mkdir(parents=False, exist_ok=False)
        begun: list[CameraProcessHandle | CameraRecorder] = []
        for name, camera in self.cameras.items():
            try:
                camera.begin_episode(tmp_dir, start_ns)
            except CameraUnavailable as exc:
                # All cameras or none: close the writers already opened.
                for started in begun:
                    started.finish_episode()
                shutil.rmtree(tmp_dir, ignore_errors=True)
                log.warning("recording not started: camera %s: %s", name, exc)
                self._set_error(f"recording not started: camera {name}: {exc}")
                return
            begun.append(camera)
        self.next_index = index + 1
        self.episode = _Episode(
            index=index,
            task=task,
            tmp_dir=tmp_dir,
            start_ns=start_ns,
            started_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
            events=(tmp_dir / "events.jsonl").open("w", encoding="utf-8"),
        )
        self.last_task = task
        self.last_error = None
        self.state = RecorderState.RECORDING
        self._event("start", t_ns=start_ns, task=task)
        log.info("recording started episode=%d task=%r", index, task)
        self._publish_status(force=True)

    def _set_error(self, message: str) -> None:
        """Record an error; error_seq lets clients detect repeats of the same text."""
        self.last_error = message
        self.error_seq += 1
        self._publish_status(force=True)

    def _on_set_task(self, task: str) -> None:
        """Set the task for the next start without a task (e.g. Quest A)."""
        task = task.strip()
        if not task:
            self._set_error("task must not be empty")
            return
        self.last_task = task
        self.last_error = None
        self._publish_status(force=True)

    def _on_stop(self, stop_ns: int, outcome: EpisodeOutcome) -> None:
        episode = self.episode
        if episode is None or episode.stop_ns is not None:
            self._set_error("no episode is recording")
            return
        episode.stop_ns = stop_ns
        episode.outcome = outcome
        episode.finalize_deadline_ns = time.monotonic_ns() + int(TAIL_WAIT_S * 1e9)
        for camera in self.cameras.values():
            camera.end_episode(stop_ns)
        self.state = RecorderState.FINALIZING
        self._event("stop", t_ns=stop_ns, outcome=outcome.value)
        self._publish_status(force=True)

    def _on_discard(self) -> None:
        episode = self.episode
        if episode is None:
            self._set_error("no episode to discard")
            return
        self._close_files(episode, sync=False)
        self._finish_cameras()
        shutil.rmtree(episode.tmp_dir, ignore_errors=True)
        log.info("recording discarded episode=%d", episode.index)
        # Reuse the index so discarded takes leave no numbering gap.
        self.next_index = episode.index
        self.episode = None
        self.state = RecorderState.IDLE
        self.last_error = None
        self._publish_status(force=True)

    def _on_samples(
        self,
        hand: str,
        generation: int,
        worker_dropped: int,
        forward_dropped: int,
        samples: list[tuple[Any, ...]],
    ) -> None:
        if samples:
            self._latest_seen_t_ns[hand] = max(
                self._latest_seen_t_ns.get(hand, 0), samples[-1][0]
            )
        key = (hand, generation)
        worker_delta = worker_dropped - self._worker_dropped.get(key, 0)
        self._worker_dropped[key] = worker_dropped
        forward_delta = forward_dropped - self._forward_dropped
        self._forward_dropped = forward_dropped

        episode = self.episode
        if episode is None:
            return
        episode.forward_dropped_batches += max(0, forward_delta)
        stop_ns = episode.stop_ns
        kept = [
            sample
            for sample in samples
            if sample[0] >= episode.start_ns
            and (stop_ns is None or sample[0] <= stop_ns)
        ]
        if not kept:
            return
        stream = episode.streams.get(hand)
        if stream is None:
            path = episode.tmp_dir / f"robot_{hand}.bin"
            stream = _HandStream(file=path.open("wb"), path=path)
            episode.streams[hand] = stream
        if generation not in stream.generations:
            stream.generations.append(generation)
            if len(stream.generations) > 1:
                self._event(
                    "worker_generation_changed",
                    hand=hand,
                    generation=generation,
                    t_ns=kept[0][0],
                )
        stream.worker_dropped_batches += max(0, worker_delta)
        stream.file.write(b"".join(pack_sample(sample) for sample in kept))
        stream.samples += len(kept)
        if stream.first_t_ns is None:
            stream.first_t_ns = kept[0][0]
        stream.last_t_ns = kept[-1][0]

    # ── finalize / abort ──────────────────────────────────

    def _maybe_finalize(self, now_ns: int) -> None:
        episode = self.episode
        if episode is None or episode.stop_ns is None:
            return
        caught_up = all(
            self._latest_seen_t_ns.get(hand, 0) >= episode.stop_ns
            for hand in self.spec.hands
        ) and all(
            camera.caught_up(episode.stop_ns) for camera in self.cameras.values()
        )
        if caught_up or now_ns >= episode.finalize_deadline_ns:
            self._finalize()

    def _finalize(self) -> None:
        episode = self.episode
        assert episode is not None and episode.stop_ns is not None
        assert episode.outcome is not None
        self._close_files(episode, sync=True)
        cameras_meta: dict[str, Any] = {}
        for camera_meta in self._finish_cameras():
            if camera_meta is not None:
                cameras_meta[camera_meta["name"]] = camera_meta
                for name in (camera_meta["video"], camera_meta["frames_file"]):
                    _fsync_file(episode.tmp_dir / name)
        camera_ok = (
            set(cameras_meta) == set(self.cameras)
            and all(
                meta["frames"] > 0
                and meta["dropped_frames"] == 0
                and meta["error"] is None
                for meta in cameras_meta.values()
            )
        )
        hands_meta = {
            hand: {
                "file": stream.path.name,
                "samples": stream.samples,
                "first_t_ns": stream.first_t_ns,
                "last_t_ns": stream.last_t_ns,
                "worker_dropped_batches": stream.worker_dropped_batches,
                "generations": stream.generations,
            }
            for hand, stream in episode.streams.items()
        }
        missing_hands = [h for h in self.spec.hands if h not in episode.streams]
        complete = (
            not missing_hands
            and camera_ok
            and episode.forward_dropped_batches == 0
            and all(
                stream.worker_dropped_batches == 0 and len(stream.generations) == 1
                for stream in episode.streams.values()
            )
        )
        meta = {
            "format": RAW_FORMAT,
            "format_version": RAW_FORMAT_VERSION,
            "episode_index": episode.index,
            "task": episode.task,
            "outcome": episode.outcome.value,
            "complete": complete,
            "missing_hands": missing_hands,
            "started_at": episode.started_at,
            "start_ns": episode.start_ns,
            "stop_ns": episode.stop_ns,
            "duration_s": (episode.stop_ns - episode.start_ns) / 1e9,
            "hands": hands_meta,
            "forward_dropped_batches": episode.forward_dropped_batches,
            "cameras": cameras_meta,
            "sample_schema": schema_metadata(),
            "session": self.session_metadata,
        }
        _write_json_atomic(episode.tmp_dir / "meta.json", meta)
        final_dir = episode.tmp_dir.with_name(f"episode_{episode.index:06d}")
        episode.tmp_dir.rename(final_dir)
        _fsync_dir(self.spec.root_dir)
        self.saved_count += 1
        self.last_saved = {
            "episode_index": episode.index,
            "outcome": episode.outcome.value,
            "complete": complete,
            "duration_s": meta["duration_s"],
            "samples": {hand: s.samples for hand, s in episode.streams.items()},
            "camera_frames": {
                name: meta["frames"] for name, meta in cameras_meta.items()
            },
            "path": str(final_dir),
        }
        log.info(
            "recording saved episode=%d outcome=%s complete=%s duration_s=%.2f "
            "samples=%s",
            episode.index,
            episode.outcome.value,
            complete,
            meta["duration_s"],
            self.last_saved["samples"],
        )
        self.episode = None
        self.state = RecorderState.IDLE
        self._publish_status(force=True)

    def _abort_episode(self, reason: str) -> None:
        episode = self.episode
        self.episode = None
        self.state = RecorderState.IDLE
        self._set_error(reason)
        if episode is None:
            return
        try:
            self._close_files(episode, sync=False)
        except OSError:
            pass
        self._finish_cameras()
        target = episode.tmp_dir.with_name(
            f"episode_{episode.index:06d}{INCOMPLETE_SUFFIX}"
        )
        try:
            episode.tmp_dir.rename(target)
        except OSError:
            pass
        log.error("recording aborted episode=%d reason=%s", episode.index, reason)

    def _for_each_camera(self, action: Any) -> list[Any]:
        """Run ``action`` on every camera at once, results in camera order.

        Each call waits on its own Camera process (finish flushes the encoder,
        stop joins the process), so waiting in turn would add the times up and
        could overrun the shutdown budget in RecorderHandle.shutdown.
        """
        cameras = list(self.cameras.values())
        if len(cameras) <= 1:
            return [action(camera) for camera in cameras]
        with ThreadPoolExecutor(max_workers=len(cameras)) as pool:
            return list(pool.map(action, cameras))

    def _finish_cameras(self) -> list[dict[str, Any] | None]:
        return self._for_each_camera(lambda camera: camera.finish_episode())

    def _close_files(self, episode: _Episode, *, sync: bool) -> None:
        for stream in episode.streams.values():
            if stream.file.closed:
                continue
            stream.file.flush()
            if sync:
                os.fsync(stream.file.fileno())
            stream.file.close()
        if not episode.events.closed:
            episode.events.close()

    def _event(self, kind: str, **fields: Any) -> None:
        if self.episode is None:
            return
        fields.setdefault("t_ns", time.monotonic_ns())
        self.episode.events.write(json.dumps({"event": kind, **fields}) + "\n")

    # ── status ────────────────────────────────────────────

    def _publish_status(self, *, force: bool = False) -> None:
        now_ns = time.monotonic_ns()
        if not force and now_ns - self._last_status_ns < STATUS_PERIOD_S * 1e9:
            return
        episode = self.episode
        status = {
            "state": self.state.value,
            "episode_index": episode.index if episode else None,
            "task": episode.task if episode else self.last_task,
            # Used by a start without a task (Quest A); set by set_task.
            "next_task": self.last_task,
            "elapsed_s": (
                ((episode.stop_ns or now_ns) - episode.start_ns) / 1e9
                if episode
                else None
            ),
            "samples": (
                {hand: s.samples for hand, s in episode.streams.items()}
                if episode
                else {}
            ),
            "dropped_batches": (
                episode.forward_dropped_batches
                + sum(s.worker_dropped_batches for s in episode.streams.values())
                if episode
                else 0
            ),
            "next_index": self.next_index,
            "saved_count": self.saved_count,
            "last_saved": self.last_saved,
            "last_error": self.last_error,
            "error_seq": self.error_seq,
            "root_dir": str(self.spec.root_dir),
            # One status per enabled camera, in config order; [] when none.
            "cameras": [camera.status() for camera in self.cameras.values()],
        }
        try:
            self.status_queue.put_nowait(status)
            self._last_status_ns = now_ns
        except queue.Full:
            pass


def run_recorder(spec: RecorderSpec, command_queue: Any, status_queue: Any) -> None:
    # The parent converts terminal/service signals into an ordered shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(
        level=getattr(logging, spec.log_level),
        format=(
            "%(asctime)s [pid=%(process)d] [%(processName)s] "
            "[%(levelname)s] %(message)s"
        ),
    )
    status_queue.cancel_join_thread()
    RecorderRuntime(spec, command_queue, status_queue).run()


class RecorderHandle:
    """Parent-side owner of the recorder process. Never blocks the event loop
    except in ``shutdown``, which callers run in a thread."""

    def __init__(
        self,
        config: TeleopConfig,
        *,
        hands: tuple[str, ...],
        robot_kind: str,
    ) -> None:
        self.spec = RecorderSpec(
            root_dir=config.recording.root_dir,
            hands=hands,
            session_metadata=build_session_metadata(
                config, robot_kind=robot_kind, hands=hands
            ),
            log_level=config.log_level,
            cameras=config.recording.enabled_cameras,
        )
        self._context = multiprocessing.get_context("spawn")
        self.process: Any = None
        self.command_queue: Any = None
        self.status_queue: Any = None
        self.latest_status: dict[str, Any] | None = None
        self.forward_dropped_batches = 0

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def start(self) -> None:
        self.command_queue = self._context.Queue(maxsize=COMMAND_QUEUE_SIZE)
        self.status_queue = self._context.Queue(maxsize=16)
        self.process = self._context.Process(
            target=run_recorder,
            args=(self.spec, self.command_queue, self.status_queue),
            name="Recorder",
            daemon=False,
        )
        self.process.start()
        log.info(
            "recorder started pid=%s root=%s", self.process.pid, self.spec.root_dir
        )

    def _put(self, message: tuple[Any, ...]) -> bool:
        if self.command_queue is None or not self.alive:
            return False
        try:
            self.command_queue.put_nowait(message)
            return True
        except queue.Full:
            return False

    def submit_samples(
        self,
        hand: str,
        batches: list[tuple[int, int, list[tuple[Any, ...]]]],
    ) -> None:
        for generation, worker_dropped, samples in batches:
            if not self._put(
                (
                    "samples",
                    hand,
                    generation,
                    worker_dropped,
                    self.forward_dropped_batches,
                    samples,
                )
            ):
                self.forward_dropped_batches += 1

    def start_episode(self, task: str | None) -> None:
        self._command(("start", time.monotonic_ns(), task))

    def stop_episode(self, outcome: EpisodeOutcome) -> None:
        self._command(("stop", time.monotonic_ns(), outcome.value))

    def discard_episode(self) -> None:
        self._command(("discard",))

    def set_task(self, task: str) -> None:
        self._command(("set_task", task))

    def _command(self, message: tuple[Any, ...]) -> None:
        if not self._put(message):
            raise RecorderUnavailable("recorder process is not accepting commands")

    def drain_status(self) -> dict[str, Any] | None:
        if self.status_queue is None:
            return self.latest_status
        while True:
            try:
                self.latest_status = self.status_queue.get_nowait()
            except queue.Empty:
                break
        return self.latest_status

    def status_payload(self) -> dict[str, Any]:
        payload = dict(self.latest_status or {"state": "STARTING"})
        payload["alive"] = self.alive
        payload["forward_dropped_batches"] = self.forward_dropped_batches
        return payload

    def shutdown(self) -> None:
        process = self.process
        if process is None:
            return
        if process.is_alive() and self.command_queue is not None:
            try:
                self.command_queue.put(("shutdown",), timeout=1.0)
            except queue.Full:
                log.error("recorder command queue full during shutdown")
        # Workers are already stopped, so only the open episode's finalize
        # (including the camera video flush) and the camera process stop
        # remain. Keep the total within systemd's TimeoutStopSec.
        process.join(timeout=TAIL_WAIT_S + 4.0)
        if process.is_alive():
            log.error("recorder did not stop; terminating (open episode may be lost)")
            process.terminate()
            process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        self.drain_status()
        for recorder_queue in (self.command_queue, self.status_queue):
            if recorder_queue is None:
                continue
            try:
                recorder_queue.close()
                recorder_queue.cancel_join_thread()
            except (OSError, ValueError):
                pass
        log.info("recorder stopped exit_code=%s", process.exitcode)
        self.process = None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".part")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else None
