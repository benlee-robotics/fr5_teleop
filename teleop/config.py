from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class WorkspaceBounds:
    x: tuple[float, float] = (-600.0, 600.0)
    y: tuple[float, float] = (-600.0, 600.0)
    z: tuple[float, float] = (50.0, 700.0)

    def validate(self) -> None:
        for axis, bounds in (("x", self.x), ("y", self.y), ("z", self.z)):
            if (
                len(bounds) != 2
                or not all(isfinite(value) for value in bounds)
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"workspace {axis} bounds must be increasing")


@dataclass(frozen=True)
class GripperConfig:
    enabled: bool = False
    index: int = 1
    activate_on_start: bool = False
    initially_closed: bool = False
    open_position: int = 0
    closed_position: int = 100
    velocity: int = 50
    force: int = 50
    command_max_time_ms: int = 3000
    action_timeout_s: float = 5.0
    poll_period_s: float = 0.050

    def validate(self) -> None:
        if self.index < 1:
            raise ValueError("gripper.index must be positive")
        for name, value in (
            ("open_position", self.open_position),
            ("closed_position", self.closed_position),
            ("velocity", self.velocity),
            ("force", self.force),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"gripper.{name} must be in [0, 100]")
        if self.open_position == self.closed_position:
            raise ValueError("gripper open and closed positions must differ")
        if not 1 <= self.command_max_time_ms <= 30_000:
            raise ValueError("gripper.command_max_time_ms must be in [1, 30000]")
        if not isfinite(self.poll_period_s) or self.poll_period_s <= 0:
            raise ValueError("gripper.poll_period_s must be positive")
        if (
            not isfinite(self.action_timeout_s)
            or self.action_timeout_s < self.command_max_time_ms / 1000
            or self.action_timeout_s < self.poll_period_s
        ):
            raise ValueError(
                "gripper.action_timeout_s must cover command_max_time_ms "
                "and at least one poll period"
            )


@dataclass(frozen=True)
class OrientationConfig:
    """Disabled by default in code. The vendor SDK does not document the
    Fairino rx,ry,rz Euler convention; fixed-angle XYZ is assumed and its
    roll/pitch/yaw axis and sign were verified on a v3.9.8 robot."""

    enabled: bool = False
    scale: float = 1.0
    ema_alpha: float = 0.2
    max_angular_velocity_deg_s: float = 15.0
    max_step_deg: float = 1.0
    max_deviation_deg: float = 20.0

    def validate(self) -> None:
        if not isfinite(self.scale) or self.scale <= 0:
            raise ValueError("orientation.scale must be positive")
        if not isfinite(self.ema_alpha) or not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("orientation.ema_alpha must be in (0, 1]")
        if (
            not isfinite(self.max_angular_velocity_deg_s)
            or self.max_angular_velocity_deg_s <= 0
        ):
            raise ValueError("orientation.max_angular_velocity_deg_s must be positive")
        if not isfinite(self.max_step_deg) or self.max_step_deg <= 0:
            raise ValueError("orientation.max_step_deg must be positive")
        if not isfinite(self.max_deviation_deg) or not (
            0.0 < self.max_deviation_deg <= 90.0
        ):
            raise ValueError("orientation.max_deviation_deg must be in (0, 90]")


@dataclass(frozen=True)
class HomeConfig:
    """Servo-streamed return to a stored pose (Quest X or /monitor).

    The worker interpolates from the current TCP to ``tcp`` (straight line,
    quaternion slerp, minimum-jerk timing) and streams it with ServoCart, so
    the motion is recorded with valid commands like teleop. ``joints_deg``
    only checks that start and end stay in the home joint configuration.
    """

    enabled: bool = False
    tcp: tuple[float, float, float, float, float, float] | None = None  # mm, deg
    joints_deg: tuple[float, float, float, float, float, float] | None = None
    # Peak speeds of the minimum-jerk profile.
    max_velocity_mm_s: float = 100.0
    max_angular_velocity_deg_s: float = 30.0
    # Keep streaming the home target this long after arrival, so recordings
    # end with the robot settled at home.
    hold_s: float = 1.0
    # Refuse to start if any joint is farther than this from joints_deg: the
    # straight-line path could then flip the IK solution.
    max_start_joint_delta_deg: float = 120.0
    # Warn after arrival if a joint ended farther than this from joints_deg.
    arrive_joint_tolerance_deg: float = 5.0

    def validate(
        self,
        where: str,
        *,
        workspace: WorkspaceBounds,
        max_velocity_mm_s: float,
        max_angular_velocity_deg_s: float,
    ) -> None:
        for name, value in (
            ("max_velocity_mm_s", self.max_velocity_mm_s),
            ("max_angular_velocity_deg_s", self.max_angular_velocity_deg_s),
            ("max_start_joint_delta_deg", self.max_start_joint_delta_deg),
            ("arrive_joint_tolerance_deg", self.arrive_joint_tolerance_deg),
        ):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{where}.{name} must be positive")
        if not isfinite(self.hold_s) or not (0.0 <= self.hold_s <= 10.0):
            raise ValueError(f"{where}.hold_s must be in [0, 10]")
        if not self.enabled:
            return
        if self.max_velocity_mm_s > max_velocity_mm_s:
            raise ValueError(
                f"{where}.max_velocity_mm_s must not exceed motion.max_velocity_mm_s"
            )
        if self.max_angular_velocity_deg_s > max_angular_velocity_deg_s:
            raise ValueError(
                f"{where}.max_angular_velocity_deg_s must not exceed "
                "orientation.max_step_deg per servo period"
            )
        if self.tcp is None or self.joints_deg is None:
            raise ValueError(f"{where}.tcp and {where}.joints_deg are required")
        if not all(isfinite(value) for value in (*self.tcp, *self.joints_deg)):
            raise ValueError(f"{where} values must be finite")
        for axis, value, bounds in (
            ("x", self.tcp[0], workspace.x),
            ("y", self.tcp[1], workspace.y),
            ("z", self.tcp[2], workspace.z),
        ):
            if not bounds[0] <= value <= bounds[1]:
                raise ValueError(
                    f"{where}.tcp {axis}={value} is outside the workspace {bounds}"
                )


# RealSense D435/D435i RGB stream modes (both use the same color sensor).
# Depth is always captured at 640x480 and aligned to the color stream.
D435_COLOR_RESOLUTIONS = frozenset(
    {
        (320, 180),
        (320, 240),
        (424, 240),
        (640, 360),
        (640, 480),
        (848, 480),
        (960, 540),
        (1280, 720),
        (1920, 1080),
    }
)
CAMERA_SOURCES = frozenset({"realsense", "fake"})


@dataclass(frozen=True)
class CameraConfig:
    """One camera recorded alongside robot samples (RealSense D435/D435i)."""

    enabled: bool = False
    # "fake" produces synthetic frames for dry-run tests without a camera.
    source: str = "realsense"
    name: str = "wrist"
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    depth: bool = False

    def validate(self, where: str = "recording.cameras[]") -> None:
        if self.source not in CAMERA_SOURCES:
            raise ValueError(
                f"{where}.source must be one of "
                + ", ".join(sorted(CAMERA_SOURCES))
            )
        if not self.name or not all(
            char.isascii() and (char.isalnum() or char == "_") for char in self.name
        ):
            raise ValueError(f"{where}.name must be ASCII letters, digits or _")
        if self.fps not in {6, 15, 30, 60}:
            raise ValueError(f"{where}.fps must be 6, 15, 30, or 60")
        if (
            self.source == "realsense"
            and (self.width, self.height) not in D435_COLOR_RESOLUTIONS
        ):
            raise ValueError(
                f"{where} {self.width}x{self.height} is not a D435 color resolution"
            )
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError(f"{where} width and height must be even")
        if not self.enabled:
            return
        modules = ["av", "numpy"]
        if self.source == "realsense":
            modules.append("pyrealsense2")
        missing = [name for name in modules if importlib.util.find_spec(name) is None]
        if missing:
            raise ValueError(
                f"{where} requires "
                + ", ".join(missing)
                + "; install requirements-camera.txt"
            )


@dataclass(frozen=True)
class RecordingConfig:
    """Raw demonstration recording. Never affects robot motion."""

    enabled: bool = False
    root_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "recordings")
    # Each enabled camera gets its own Camera process and video files.
    cameras: tuple[CameraConfig, ...] = ()

    @property
    def enabled_cameras(self) -> tuple[CameraConfig, ...]:
        return tuple(camera for camera in self.cameras if camera.enabled)

    def validate(self) -> None:
        if self.root_dir.exists() and not self.root_dir.is_dir():
            raise ValueError(
                f"recording.root_dir is not a directory: {self.root_dir}"
            )
        names = [camera.name for camera in self.cameras]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"recording.cameras names must be unique: {', '.join(duplicates)}"
            )
        if not self.enabled:
            return
        for index, camera in enumerate(self.cameras):
            camera.validate(f"recording.cameras[{index}]")
        realsense = [
            camera for camera in self.enabled_cameras if camera.source == "realsense"
        ]
        if len(realsense) > 1:
            # Without a serial, librealsense opens whichever device it finds
            # first, so two unpinned cameras could swap or collide.
            unpinned = [camera.name for camera in realsense if not camera.serial]
            if unpinned:
                raise ValueError(
                    "recording.cameras need a serial when more than one "
                    f"RealSense is enabled: {', '.join(unpinned)}"
                )
            serials = [camera.serial for camera in realsense]
            if len(set(serials)) != len(serials):
                raise ValueError("recording.cameras serials must be unique")


@dataclass(frozen=True)
class ArmConfig:
    """Per-arm configuration for bimanual teleoperation."""
    hand: str
    robot_ip: str | None = None
    sdk_path: Path | None = None
    gripper: GripperConfig = field(default_factory=GripperConfig)
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    exaxis_default: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    home: HomeConfig = field(default_factory=HomeConfig)

    def validate(self, *, dry_run: bool) -> None:
        if self.hand not in {"left", "right"}:
            raise ValueError(f"arm hand must be 'left' or 'right', got '{self.hand}'")
        if not dry_run and not self.robot_ip:
            raise ValueError(f"arms.{self.hand}.ip is required outside dry-run")
        if self.sdk_path is not None and not self.sdk_path.exists():
            raise ValueError(f"arms.{self.hand}.sdk_path does not exist: {self.sdk_path}")
        self.workspace.validate()
        self.gripper.validate()


@dataclass(frozen=True)
class TeleopConfig:
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8443
    dry_run: bool = True
    robot_ip: str | None = None
    arms: tuple[ArmConfig, ...] = ()
    sdk_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "fairino-python-sdk-2.2.8_robot_v3.9.8" / "linux"
    )
    web_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "web")
    tls_cert_path: Path | None = field(
        default_factory=lambda: PROJECT_ROOT / "certs" / "cert.pem"
    )
    tls_key_path: Path | None = field(
        default_factory=lambda: PROJECT_ROOT / "certs" / "key.pem"
    )
    servo_period_s: float = 0.008
    servo_transition_window_s: float = 1.0
    servo_transition_limit: int = 4
    pose_timeout_s: float = 0.200
    worker_watchdog_s: float = 0.500
    worker_startup_timeout_s: float = 10.0
    graceful_shutdown_s: float = 2.0
    status_hz: float = 10.0
    position_scale: float = 500.0
    ema_alpha: float = 0.25
    max_velocity_mm_s: float = 50.0
    max_step_mm: float = 0.75
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    home: HomeConfig = field(default_factory=HomeConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    exaxis_default: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    max_ws_message_bytes: int = 4096
    allowed_origins: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level must be DEBUG, INFO, WARNING, or ERROR")
        if not self.host:
            raise ValueError("host must not be empty")
        if not (1 <= self.port <= 65535):
            raise ValueError("port must be in [1, 65535]")
        if self.arms:
            hands = tuple(arm.hand for arm in self.arms)
            if self.robot_ip is not None:
                raise ValueError("robot.ip cannot be combined with robot.arms")
            if len(hands) != 2 or set(hands) != {"left", "right"}:
                raise ValueError(
                    "robot.arms must configure exactly one left and one right arm"
                )
            robot_ips = [arm.robot_ip for arm in self.arms if arm.robot_ip]
            if len(robot_ips) != len(set(robot_ips)):
                raise ValueError("robot.arms must use a different IP for each arm")
        if not self.dry_run and not self.robot_ip and not self.arms:
            raise ValueError("robot_ip or arms configuration is required outside dry-run")
        for arm in self.arms:
            arm.validate(dry_run=self.dry_run)
        if not isfinite(self.servo_period_s) or self.servo_period_s <= 0:
            raise ValueError("servo_period_s must be positive")
        if (
            not isfinite(self.servo_transition_window_s)
            or self.servo_transition_window_s <= 0
        ):
            raise ValueError("servo_transition_window_s must be positive")
        if self.servo_transition_limit < 2:
            raise ValueError("servo_transition_limit must be at least 2")
        if (
            not isfinite(self.pose_timeout_s)
            or self.pose_timeout_s < self.servo_period_s
        ):
            raise ValueError("pose_timeout_s must be at least one servo period")
        if (
            not isfinite(self.worker_watchdog_s)
            or self.worker_watchdog_s <= self.pose_timeout_s
        ):
            raise ValueError("worker_watchdog_s must exceed pose_timeout_s")
        if (
            not isfinite(self.worker_startup_timeout_s)
            or self.worker_startup_timeout_s <= self.worker_watchdog_s
        ):
            raise ValueError("worker_startup_timeout_s must exceed worker_watchdog_s")
        if (
            not isfinite(self.graceful_shutdown_s)
            or self.graceful_shutdown_s <= 0
        ):
            raise ValueError("graceful_shutdown_s must be positive")
        if not isfinite(self.status_hz) or not (1.0 <= self.status_hz <= 60.0):
            raise ValueError("status_hz must be in [1, 60]")
        if not isfinite(self.position_scale) or self.position_scale <= 0:
            raise ValueError("position_scale must be positive")
        if not isfinite(self.ema_alpha) or not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")
        if (
            not isfinite(self.max_velocity_mm_s)
            or self.max_velocity_mm_s <= 0
        ):
            raise ValueError("max_velocity_mm_s must be positive")
        if not isfinite(self.max_step_mm) or self.max_step_mm <= 0:
            raise ValueError("max_step_mm must be positive")
        if len(self.exaxis_default) != 4:
            raise ValueError("exaxis_default must contain four values")
        if not all(isfinite(value) for value in self.exaxis_default):
            raise ValueError("exaxis_default values must be finite")
        if self.max_ws_message_bytes < 256:
            raise ValueError("max_ws_message_bytes is too small")
        if not self.web_dir.joinpath("index.html").is_file():
            raise ValueError(f"WebXR UI not found: {self.web_dir / 'index.html'}")
        if not self.web_dir.joinpath("monitor.html").is_file():
            raise ValueError(
                f"operator monitor not found: {self.web_dir / 'monitor.html'}"
            )
        if not self.dry_run and not self.sdk_path.joinpath("fairino", "Robot.py").is_file():
            raise ValueError(f"Fairino Linux SDK not found: {self.sdk_path}")
        if (self.tls_cert_path is None) != (self.tls_key_path is None):
            raise ValueError("TLS certificate and key must be configured together")
        if self.tls_cert_path is not None:
            if not self.tls_cert_path.is_file():
                raise ValueError(f"TLS certificate not found: {self.tls_cert_path}")
            if not self.tls_key_path or not self.tls_key_path.is_file():
                raise ValueError(f"TLS key not found: {self.tls_key_path}")
        self.workspace.validate()
        self.gripper.validate()
        self.orientation.validate()
        self.recording.validate()
        if self.arms and self.home.enabled:
            raise ValueError(
                "home is per arm in bimanual mode; use robot.arms.<hand>.home"
            )
        homes = [("home", self.home, self.workspace)] + [
            (f"robot.arms.{arm.hand}.home", arm.home, arm.workspace)
            for arm in self.arms
        ]
        for where, home, workspace in homes:
            home.validate(
                where,
                workspace=workspace,
                max_velocity_mm_s=self.max_velocity_mm_s,
                max_angular_velocity_deg_s=(
                    self.orientation.max_step_deg / self.servo_period_s
                ),
            )

    @classmethod
    def from_yaml(cls, path: Path, *, validate: bool = True) -> TeleopConfig:
        config_path = path.expanduser().resolve()
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"unable to read config file {config_path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc

        root = _mapping(raw, "config")
        _reject_unknown(
            root,
            {
                "runtime",
                "server",
                "tls",
                "robot",
                "timing",
                "motion",
                "gripper",
                "orientation",
                "home",
                "recording",
            },
            "config",
        )
        runtime = _section(root, "runtime")
        server = _section(root, "server")
        tls = _section(root, "tls")
        robot = _section(root, "robot")
        timing = _section(root, "timing")
        motion = _section(root, "motion")
        workspace = _section(motion, "workspace")
        gripper = _section(root, "gripper")
        orientation = _section(root, "orientation")
        recording = _section(root, "recording")

        _reject_unknown(runtime, {"dry_run", "log_level", "status_hz"}, "runtime")
        if "camera" in recording:
            raise ValueError(
                "recording.camera was replaced by the recording.cameras list; "
                "put the camera mapping in a list under recording.cameras"
            )
        _reject_unknown(recording, {"enabled", "root_dir", "cameras"}, "recording")
        cameras = tuple(
            _parse_camera(item, f"recording.cameras[{index}]")
            for index, item in enumerate(
                _sequence(recording.get("cameras", []), "recording.cameras")
            )
        )
        _reject_unknown(
            server,
            {"host", "port", "web_dir", "max_ws_message_bytes", "allowed_origins"},
            "server",
        )
        _reject_unknown(tls, {"enabled", "cert_path", "key_path"}, "tls")
        _reject_unknown(robot, {"ip", "sdk_path", "exaxis_default", "arms"}, "robot")
        _reject_unknown(
            timing,
            {
                "servo_period_s",
                "servo_transition_window_s",
                "servo_transition_limit",
                "pose_timeout_s",
                "worker_watchdog_s",
                "worker_startup_timeout_s",
                "graceful_shutdown_s",
            },
            "timing",
        )
        _reject_unknown(
            motion,
            {
                "position_scale",
                "ema_alpha",
                "max_velocity_mm_s",
                "max_step_mm",
                "workspace",
            },
            "motion",
        )
        _reject_unknown(workspace, {"x", "y", "z"}, "motion.workspace")
        _reject_unknown(
            gripper,
            {
                "enabled",
                "index",
                "activate_on_start",
                "initially_closed",
                "open_position",
                "closed_position",
                "velocity",
                "force",
                "command_max_time_ms",
                "action_timeout_s",
                "poll_period_s",
            },
            "gripper",
        )
        _reject_unknown(
            orientation,
            {
                "enabled",
                "scale",
                "ema_alpha",
                "max_angular_velocity_deg_s",
                "max_step_deg",
                "max_deviation_deg",
            },
            "orientation",
        )

        defaults = cls()
        base_dir = config_path.parent
        tls_enabled = _boolean(tls.get("enabled", True), "tls.enabled")
        cert_path = (
            _path(tls.get("cert_path", defaults.tls_cert_path), "tls.cert_path", base_dir)
            if tls_enabled
            else None
        )
        key_path = (
            _path(tls.get("key_path", defaults.tls_key_path), "tls.key_path", base_dir)
            if tls_enabled
            else None
        )
        robot_ip = robot.get("ip", defaults.robot_ip)
        if robot_ip is not None:
            robot_ip = _string(robot_ip, "robot.ip")

        config = cls(
            log_level=_choice(
                runtime.get("log_level", defaults.log_level),
                "runtime.log_level",
                {"DEBUG", "INFO", "WARNING", "ERROR"},
            ),
            host=_string(server.get("host", defaults.host), "server.host"),
            port=_integer(server.get("port", defaults.port), "server.port"),
            dry_run=_boolean(
                runtime.get("dry_run", defaults.dry_run), "runtime.dry_run"
            ),
            robot_ip=robot_ip,
            sdk_path=_path(
                robot.get("sdk_path", defaults.sdk_path), "robot.sdk_path", base_dir
            ),
            web_dir=_path(
                server.get("web_dir", defaults.web_dir), "server.web_dir", base_dir
            ),
            tls_cert_path=cert_path,
            tls_key_path=key_path,
            servo_period_s=_number(
                timing.get("servo_period_s", defaults.servo_period_s),
                "timing.servo_period_s",
            ),
            servo_transition_window_s=_number(
                timing.get(
                    "servo_transition_window_s",
                    defaults.servo_transition_window_s,
                ),
                "timing.servo_transition_window_s",
            ),
            servo_transition_limit=_integer(
                timing.get(
                    "servo_transition_limit", defaults.servo_transition_limit
                ),
                "timing.servo_transition_limit",
            ),
            pose_timeout_s=_number(
                timing.get("pose_timeout_s", defaults.pose_timeout_s),
                "timing.pose_timeout_s",
            ),
            worker_watchdog_s=_number(
                timing.get("worker_watchdog_s", defaults.worker_watchdog_s),
                "timing.worker_watchdog_s",
            ),
            worker_startup_timeout_s=_number(
                timing.get(
                    "worker_startup_timeout_s", defaults.worker_startup_timeout_s
                ),
                "timing.worker_startup_timeout_s",
            ),
            graceful_shutdown_s=_number(
                timing.get("graceful_shutdown_s", defaults.graceful_shutdown_s),
                "timing.graceful_shutdown_s",
            ),
            status_hz=_number(
                runtime.get("status_hz", defaults.status_hz), "runtime.status_hz"
            ),
            position_scale=_number(
                motion.get("position_scale", defaults.position_scale),
                "motion.position_scale",
            ),
            ema_alpha=_number(
                motion.get("ema_alpha", defaults.ema_alpha), "motion.ema_alpha"
            ),
            max_velocity_mm_s=_number(
                motion.get(
                    "max_velocity_mm_s",
                    defaults.max_velocity_mm_s,
                ),
                "motion.max_velocity_mm_s",
            ),
            max_step_mm=_number(
                motion.get("max_step_mm", defaults.max_step_mm),
                "motion.max_step_mm",
            ),
            workspace=WorkspaceBounds(
                x=_bounds(workspace.get("x", defaults.workspace.x), "motion.workspace.x"),
                y=_bounds(workspace.get("y", defaults.workspace.y), "motion.workspace.y"),
                z=_bounds(workspace.get("z", defaults.workspace.z), "motion.workspace.z"),
            ),
            gripper=GripperConfig(
                enabled=_boolean(
                    gripper.get("enabled", defaults.gripper.enabled),
                    "gripper.enabled",
                ),
                index=_integer(
                    gripper.get("index", defaults.gripper.index),
                    "gripper.index",
                ),
                activate_on_start=_boolean(
                    gripper.get(
                        "activate_on_start",
                        defaults.gripper.activate_on_start,
                    ),
                    "gripper.activate_on_start",
                ),
                initially_closed=_boolean(
                    gripper.get(
                        "initially_closed",
                        defaults.gripper.initially_closed,
                    ),
                    "gripper.initially_closed",
                ),
                open_position=_integer(
                    gripper.get(
                        "open_position",
                        defaults.gripper.open_position,
                    ),
                    "gripper.open_position",
                ),
                closed_position=_integer(
                    gripper.get(
                        "closed_position",
                        defaults.gripper.closed_position,
                    ),
                    "gripper.closed_position",
                ),
                velocity=_integer(
                    gripper.get("velocity", defaults.gripper.velocity),
                    "gripper.velocity",
                ),
                force=_integer(
                    gripper.get("force", defaults.gripper.force),
                    "gripper.force",
                ),
                command_max_time_ms=_integer(
                    gripper.get(
                        "command_max_time_ms",
                        defaults.gripper.command_max_time_ms,
                    ),
                    "gripper.command_max_time_ms",
                ),
                action_timeout_s=_number(
                    gripper.get(
                        "action_timeout_s",
                        defaults.gripper.action_timeout_s,
                    ),
                    "gripper.action_timeout_s",
                ),
                poll_period_s=_number(
                    gripper.get(
                        "poll_period_s",
                        defaults.gripper.poll_period_s,
                    ),
                    "gripper.poll_period_s",
                ),
            ),
            orientation=OrientationConfig(
                enabled=_boolean(
                    orientation.get("enabled", defaults.orientation.enabled),
                    "orientation.enabled",
                ),
                scale=_number(
                    orientation.get("scale", defaults.orientation.scale),
                    "orientation.scale",
                ),
                ema_alpha=_number(
                    orientation.get("ema_alpha", defaults.orientation.ema_alpha),
                    "orientation.ema_alpha",
                ),
                max_angular_velocity_deg_s=_number(
                    orientation.get(
                        "max_angular_velocity_deg_s",
                        defaults.orientation.max_angular_velocity_deg_s,
                    ),
                    "orientation.max_angular_velocity_deg_s",
                ),
                max_step_deg=_number(
                    orientation.get(
                        "max_step_deg", defaults.orientation.max_step_deg
                    ),
                    "orientation.max_step_deg",
                ),
                max_deviation_deg=_number(
                    orientation.get(
                        "max_deviation_deg", defaults.orientation.max_deviation_deg
                    ),
                    "orientation.max_deviation_deg",
                ),
            ),
            home=_parse_home(root.get("home", {}), "home"),
            recording=RecordingConfig(
                enabled=_boolean(
                    recording.get("enabled", defaults.recording.enabled),
                    "recording.enabled",
                ),
                root_dir=_path(
                    recording.get("root_dir", defaults.recording.root_dir),
                    "recording.root_dir",
                    base_dir,
                ),
                cameras=cameras,
            ),
            exaxis_default=_four_numbers(
                robot.get("exaxis_default", defaults.exaxis_default),
                "robot.exaxis_default",
            ),
            max_ws_message_bytes=_integer(
                server.get(
                    "max_ws_message_bytes", defaults.max_ws_message_bytes
                ),
                "server.max_ws_message_bytes",
            ),
            allowed_origins=_strings(
                server.get("allowed_origins", defaults.allowed_origins),
                "server.allowed_origins",
            ),
            arms=_parse_arms(
                robot.get("arms", {}),
                gripper_defaults=GripperConfig(
                    enabled=_boolean(
                        gripper.get("enabled", defaults.gripper.enabled),
                        "gripper.enabled",
                    ),
                    index=_integer(
                        gripper.get("index", defaults.gripper.index),
                        "gripper.index",
                    ),
                    activate_on_start=_boolean(
                        gripper.get("activate_on_start", defaults.gripper.activate_on_start),
                        "gripper.activate_on_start",
                    ),
                    initially_closed=_boolean(
                        gripper.get("initially_closed", defaults.gripper.initially_closed),
                        "gripper.initially_closed",
                    ),
                    open_position=_integer(
                        gripper.get("open_position", defaults.gripper.open_position),
                        "gripper.open_position",
                    ),
                    closed_position=_integer(
                        gripper.get("closed_position", defaults.gripper.closed_position),
                        "gripper.closed_position",
                    ),
                    velocity=_integer(
                        gripper.get("velocity", defaults.gripper.velocity),
                        "gripper.velocity",
                    ),
                    force=_integer(
                        gripper.get("force", defaults.gripper.force),
                        "gripper.force",
                    ),
                    command_max_time_ms=_integer(
                        gripper.get("command_max_time_ms", defaults.gripper.command_max_time_ms),
                        "gripper.command_max_time_ms",
                    ),
                    action_timeout_s=_number(
                        gripper.get("action_timeout_s", defaults.gripper.action_timeout_s),
                        "gripper.action_timeout_s",
                    ),
                    poll_period_s=_number(
                        gripper.get("poll_period_s", defaults.gripper.poll_period_s),
                        "gripper.poll_period_s",
                    ),
                ),
                workspace_defaults=WorkspaceBounds(
                    x=_bounds(workspace.get("x", defaults.workspace.x), "motion.workspace.x"),
                    y=_bounds(workspace.get("y", defaults.workspace.y), "motion.workspace.y"),
                    z=_bounds(workspace.get("z", defaults.workspace.z), "motion.workspace.z"),
                ),
                exaxis_defaults=_four_numbers(
                    robot.get("exaxis_default", defaults.exaxis_default),
                    "robot.exaxis_default",
                ),
                config_dir=base_dir,
            ),
        )
        if validate:
            config.validate()
        return config


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _section(parent: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = parent.get(name, {})
    return _mapping(value, name)


def _reject_unknown(
    value: Mapping[str, Any], allowed: set[str], name: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} key(s): {', '.join(unknown)}")


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _choice(value: Any, name: str, choices: set[str]) -> str:
    result = _string(value, name).upper()
    if result not in choices:
        raise ValueError(f"{name} must be one of {', '.join(sorted(choices))}")
    return result


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _path(value: Any, name: str, base_dir: Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a path string")
    path = Path(_string(str(value), name)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _sequence(value: Any, name: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a YAML sequence")
    return value


def _bounds(value: Any, name: str) -> tuple[float, float]:
    values = _sequence(value, name)
    if len(values) != 2:
        raise ValueError(f"{name} must contain two numbers")
    return (_number(values[0], name), _number(values[1], name))


def _four_numbers(
    value: Any, name: str
) -> tuple[float, float, float, float]:
    values = _sequence(value, name)
    if len(values) != 4:
        raise ValueError(f"{name} must contain four numbers")
    return tuple(_number(item, name) for item in values)  # type: ignore[return-value]


def _strings(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_string(item, name) for item in _sequence(value, name))


def _six_numbers(
    value: Any, name: str
) -> tuple[float, float, float, float, float, float]:
    values = _sequence(value, name)
    if len(values) != 6:
        raise ValueError(f"{name} must contain six numbers")
    return tuple(_number(item, name) for item in values)  # type: ignore[return-value]


def _parse_home(value: Any, name: str) -> HomeConfig:
    home = _mapping(value, name)
    _reject_unknown(
        home,
        {
            "enabled", "tcp", "joints_deg", "max_velocity_mm_s",
            "max_angular_velocity_deg_s", "hold_s", "max_start_joint_delta_deg",
            "arrive_joint_tolerance_deg",
        },
        name,
    )
    defaults = HomeConfig()
    return HomeConfig(
        enabled=_boolean(home.get("enabled", defaults.enabled), f"{name}.enabled"),
        tcp=(
            _six_numbers(home["tcp"], f"{name}.tcp")
            if home.get("tcp") is not None
            else None
        ),
        joints_deg=(
            _six_numbers(home["joints_deg"], f"{name}.joints_deg")
            if home.get("joints_deg") is not None
            else None
        ),
        max_velocity_mm_s=_number(
            home.get("max_velocity_mm_s", defaults.max_velocity_mm_s),
            f"{name}.max_velocity_mm_s",
        ),
        max_angular_velocity_deg_s=_number(
            home.get("max_angular_velocity_deg_s", defaults.max_angular_velocity_deg_s),
            f"{name}.max_angular_velocity_deg_s",
        ),
        hold_s=_number(home.get("hold_s", defaults.hold_s), f"{name}.hold_s"),
        max_start_joint_delta_deg=_number(
            home.get("max_start_joint_delta_deg", defaults.max_start_joint_delta_deg),
            f"{name}.max_start_joint_delta_deg",
        ),
        arrive_joint_tolerance_deg=_number(
            home.get("arrive_joint_tolerance_deg", defaults.arrive_joint_tolerance_deg),
            f"{name}.arrive_joint_tolerance_deg",
        ),
    )


def _parse_camera(value: Any, name: str) -> CameraConfig:
    camera = _mapping(value, name)
    _reject_unknown(
        camera,
        {"enabled", "source", "name", "serial", "width", "height", "fps", "depth"},
        name,
    )
    defaults = CameraConfig()
    serial = camera.get("serial")
    if serial is not None:
        # YAML reads an unquoted serial such as 123456789012 as an int.
        if isinstance(serial, bool) or not isinstance(serial, (str, int)):
            raise ValueError(f"{name}.serial must be a string")
        serial = _string(str(serial), f"{name}.serial")
    return CameraConfig(
        enabled=_boolean(camera.get("enabled", defaults.enabled), f"{name}.enabled"),
        source=_string(camera.get("source", defaults.source), f"{name}.source"),
        name=_string(camera.get("name", defaults.name), f"{name}.name"),
        serial=serial,
        width=_integer(camera.get("width", defaults.width), f"{name}.width"),
        height=_integer(camera.get("height", defaults.height), f"{name}.height"),
        fps=_integer(camera.get("fps", defaults.fps), f"{name}.fps"),
        depth=_boolean(camera.get("depth", defaults.depth), f"{name}.depth"),
    )


def _parse_arms(
    raw_arms: Any,
    *,
    gripper_defaults: GripperConfig,
    workspace_defaults: WorkspaceBounds,
    exaxis_defaults: tuple[float, float, float, float],
    config_dir: Path,
) -> tuple[ArmConfig, ...]:
    if not raw_arms:
        return ()
    arms_map = _mapping(raw_arms, "robot.arms")
    allowed_hands = {"left", "right"}
    unknown = sorted(set(arms_map) - allowed_hands)
    if unknown:
        raise ValueError(f"unknown arm(s): {', '.join(unknown)}; must be 'left' or 'right'")
    result: list[ArmConfig] = []
    for hand in ("left", "right"):
        if hand not in arms_map:
            continue
        arm_section = _mapping(arms_map[hand], f"robot.arms.{hand}")
        _reject_unknown(
            arm_section,
            {"ip", "sdk_path", "gripper", "workspace", "exaxis_default", "home"},
            f"robot.arms.{hand}",
        )
        arm_ip_raw = arm_section.get("ip")
        arm_ip = _string(arm_ip_raw, f"robot.arms.{hand}.ip") if arm_ip_raw is not None else None
        arm_sdk_raw = arm_section.get("sdk_path")
        arm_sdk = (
            _path(arm_sdk_raw, f"robot.arms.{hand}.sdk_path", config_dir)
            if arm_sdk_raw is not None
            else None
        )
        arm_gripper_section = _section(arm_section, "gripper")
        if arm_gripper_section:
            _reject_unknown(
                arm_gripper_section,
                {
                    "enabled", "index", "activate_on_start", "initially_closed",
                    "open_position", "closed_position", "velocity", "force",
                    "command_max_time_ms", "action_timeout_s", "poll_period_s",
                },
                f"robot.arms.{hand}.gripper",
            )
        arm_gripper = GripperConfig(
            enabled=_boolean(arm_gripper_section.get("enabled", gripper_defaults.enabled), f"arms.{hand}.gripper.enabled"),
            index=_integer(arm_gripper_section.get("index", gripper_defaults.index), f"arms.{hand}.gripper.index"),
            activate_on_start=_boolean(arm_gripper_section.get("activate_on_start", gripper_defaults.activate_on_start), f"arms.{hand}.gripper.activate_on_start"),
            initially_closed=_boolean(arm_gripper_section.get("initially_closed", gripper_defaults.initially_closed), f"arms.{hand}.gripper.initially_closed"),
            open_position=_integer(arm_gripper_section.get("open_position", gripper_defaults.open_position), f"arms.{hand}.gripper.open_position"),
            closed_position=_integer(arm_gripper_section.get("closed_position", gripper_defaults.closed_position), f"arms.{hand}.gripper.closed_position"),
            velocity=_integer(arm_gripper_section.get("velocity", gripper_defaults.velocity), f"arms.{hand}.gripper.velocity"),
            force=_integer(arm_gripper_section.get("force", gripper_defaults.force), f"arms.{hand}.gripper.force"),
            command_max_time_ms=_integer(arm_gripper_section.get("command_max_time_ms", gripper_defaults.command_max_time_ms), f"arms.{hand}.gripper.command_max_time_ms"),
            action_timeout_s=_number(arm_gripper_section.get("action_timeout_s", gripper_defaults.action_timeout_s), f"arms.{hand}.gripper.action_timeout_s"),
            poll_period_s=_number(arm_gripper_section.get("poll_period_s", gripper_defaults.poll_period_s), f"arms.{hand}.gripper.poll_period_s"),
        )
        arm_ws_section = _section(arm_section, "workspace")
        if arm_ws_section:
            _reject_unknown(arm_ws_section, {"x", "y", "z"}, f"robot.arms.{hand}.workspace")
        arm_workspace = WorkspaceBounds(
            x=_bounds(arm_ws_section.get("x", workspace_defaults.x), f"arms.{hand}.workspace.x"),
            y=_bounds(arm_ws_section.get("y", workspace_defaults.y), f"arms.{hand}.workspace.y"),
            z=_bounds(arm_ws_section.get("z", workspace_defaults.z), f"arms.{hand}.workspace.z"),
        )
        arm_exaxis = _four_numbers(
            arm_section.get("exaxis_default", list(exaxis_defaults)),
            f"arms.{hand}.exaxis_default",
        )
        result.append(ArmConfig(
            hand=hand,
            robot_ip=arm_ip,
            sdk_path=arm_sdk,
            gripper=arm_gripper,
            workspace=arm_workspace,
            exaxis_default=arm_exaxis,
            home=_parse_home(arm_section.get("home", {}), f"robot.arms.{hand}.home"),
        ))
    return tuple(result)
