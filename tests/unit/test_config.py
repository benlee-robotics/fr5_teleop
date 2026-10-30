from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from teleop.config import (
    ArmConfig,
    GripperConfig,
    HomeConfig,
    OrientationConfig,
    TeleopConfig,
)
from teleop.__main__ import build_parser, resolve_config


def write_config(path: Path, extra: str = "") -> Path:
    web_dir = path.parent / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("<!doctype html>", encoding="utf-8")
    (web_dir / "monitor.html").write_text("<!doctype html>", encoding="utf-8")
    path.write_text(
        """
runtime:
  dry_run: true
  log_level: warning
server:
  host: 127.0.0.1
  port: 9000
  web_dir: web
  allowed_origins:
    - https://quest.local
tls:
  enabled: false
robot:
  ip: 192.168.58.2
  sdk_path: sdk/linux
timing:
  servo_transition_window_s: 2.0
  servo_transition_limit: 6
motion:
  position_scale: 250
  max_velocity_mm_s: 40
  workspace:
    x: [-100, 100]
    y: [-200, 200]
    z: [100, 500]
gripper:
  enabled: true
  index: 1
  open_position: 5
  closed_position: 95
  velocity: 30
  force: 40
  command_max_time_ms: 2000
  action_timeout_s: 3.0
  poll_period_s: 0.1
"""
        + extra,
        encoding="utf-8",
    )
    return path


def test_yaml_loads_values_and_resolves_relative_paths(tmp_path: Path) -> None:
    config = TeleopConfig.from_yaml(write_config(tmp_path / "config.yaml"))

    assert config.dry_run is True
    assert config.log_level == "WARNING"
    assert config.host == "127.0.0.1"
    assert config.port == 9000
    assert config.web_dir == (tmp_path / "web").resolve()
    assert config.sdk_path == (tmp_path / "sdk/linux").resolve()
    assert config.tls_cert_path is None
    assert config.tls_key_path is None
    assert config.servo_transition_window_s == 2.0
    assert config.servo_transition_limit == 6
    assert config.position_scale == 250.0
    assert config.max_velocity_mm_s == 40.0
    assert config.workspace.z == (100.0, 500.0)
    assert config.allowed_origins == ("https://quest.local",)
    assert config.gripper.enabled is True
    assert config.gripper.closed_position == 95
    assert config.gripper.poll_period_s == 0.1


def test_yaml_rejects_unknown_keys(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml")
    path.write_text(
        path.read_text(encoding="utf-8") + "\nunknown_section: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown config key"):
        TeleopConfig.from_yaml(path)


def test_yaml_requires_real_boolean_values(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml")
    text = path.read_text(encoding="utf-8").replace(
        "dry_run: true", 'dry_run: "true"'
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="runtime.dry_run must be a boolean"):
        TeleopConfig.from_yaml(path)


def test_yaml_rejects_non_finite_motion_limits(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml")
    text = path.read_text(encoding="utf-8").replace(
        "position_scale: 250", "position_scale: .nan"
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="position_scale must be positive"):
        TeleopConfig.from_yaml(path)


def test_cli_no_tls_override_is_applied_before_path_validation(
    tmp_path: Path,
) -> None:
    path = write_config(tmp_path / "config.yaml")
    text = path.read_text(encoding="utf-8").replace(
        "enabled: false",
        "enabled: true\n  cert_path: missing/cert.pem\n  key_path: missing/key.pem",
    )
    path.write_text(text, encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(["--config", str(path), "--no-tls"])

    config = resolve_config(parser, args)

    assert config.tls_cert_path is None
    assert config.tls_key_path is None


def test_cli_hardware_mode_requires_explicit_confirmation(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml")
    parser = build_parser()
    args = parser.parse_args(
        ["--config", str(path), "--robot", "192.168.58.2", "--no-tls"]
    )

    with pytest.raises(SystemExit):
        resolve_config(parser, args)


def test_gripper_timeout_must_cover_controller_command() -> None:
    config = TeleopConfig(
        tls_cert_path=None,
        tls_key_path=None,
        gripper=GripperConfig(
            command_max_time_ms=3000,
            action_timeout_s=2.0,
        ),
    )

    with pytest.raises(ValueError, match="must cover command_max_time_ms"):
        config.validate()


def test_bimanual_yaml_loads_per_arm_overrides(tmp_path: Path) -> None:
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("", encoding="utf-8")
    (web_dir / "monitor.html").write_text("", encoding="utf-8")
    (tmp_path / "sdk/linux").mkdir(parents=True)
    (tmp_path / "sdk/left/linux").mkdir(parents=True)
    (tmp_path / "sdk/right/linux").mkdir(parents=True)
    path = tmp_path / "bimanual.yaml"
    path.write_text(
        """
runtime:
  dry_run: true
server:
  web_dir: web
tls:
  enabled: false
robot:
  sdk_path: sdk/linux
  arms:
    left:
      ip: 192.168.58.2
      sdk_path: sdk/left/linux
      workspace:
        x: [-300, 0]
    right:
      ip: 192.168.58.3
      sdk_path: sdk/right/linux
      gripper:
        enabled: true
motion:
  workspace:
    x: [-600, 600]
    y: [-600, 600]
    z: [50, 700]
gripper:
  enabled: false
""",
        encoding="utf-8",
    )

    config = TeleopConfig.from_yaml(path)

    assert tuple(arm.hand for arm in config.arms) == ("left", "right")
    assert config.arms[0].workspace.x == (-300.0, 0.0)
    assert config.arms[0].sdk_path == (tmp_path / "sdk/left/linux").resolve()
    assert config.arms[0].gripper.enabled is False
    assert config.arms[1].workspace.x == (-600.0, 600.0)
    assert config.arms[1].sdk_path == (tmp_path / "sdk/right/linux").resolve()
    assert config.arms[1].gripper.enabled is True


def test_bimanual_config_requires_exactly_left_and_right() -> None:
    with pytest.raises(ValueError, match="exactly one left and one right"):
        TeleopConfig(
            tls_cert_path=None,
            tls_key_path=None,
            arms=(ArmConfig(hand="left"),),
        ).validate()


def test_bimanual_config_rejects_ambiguous_single_robot_ip() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        TeleopConfig(
            tls_cert_path=None,
            tls_key_path=None,
            robot_ip="192.168.58.10",
            arms=(ArmConfig(hand="left"), ArmConfig(hand="right")),
        ).validate()


def test_bimanual_config_rejects_duplicate_robot_ips() -> None:
    with pytest.raises(ValueError, match="different IP"):
        TeleopConfig(
            tls_cert_path=None,
            tls_key_path=None,
            arms=(
                ArmConfig(hand="left", robot_ip="192.168.58.2"),
                ArmConfig(hand="right", robot_ip="192.168.58.2"),
            ),
        ).validate()


def test_repository_configs_keep_safe_defaults() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("config.yaml", "config_bimanual.yaml"):
        config = TeleopConfig.from_yaml(root / name, validate=False)
        assert config.dry_run is True
        assert isinstance(config.gripper.enabled, bool)
        assert isinstance(config.gripper.activate_on_start, bool)
        assert config.max_velocity_mm_s <= 250.0
        assert config.max_step_mm <= 5.0
        assert config.orientation.max_step_deg <= 2.0
        assert config.orientation.max_deviation_deg <= 30.0


def test_orientation_disabled_by_default() -> None:
    assert TeleopConfig().orientation.enabled is False
    assert TeleopConfig().orientation == OrientationConfig()


def test_yaml_loads_orientation_section(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml")
    path.write_text(
        path.read_text(encoding="utf-8")
        + """
orientation:
  enabled: true
  scale: 0.5
  ema_alpha: 0.3
  max_angular_velocity_deg_s: 10.0
  max_step_deg: 2.0
  max_deviation_deg: 15.0
""",
        encoding="utf-8",
    )

    config = TeleopConfig.from_yaml(path)

    assert config.orientation.enabled is True
    assert config.orientation.scale == 0.5
    assert config.orientation.ema_alpha == 0.3
    assert config.orientation.max_angular_velocity_deg_s == 10.0
    assert config.orientation.max_step_deg == 2.0
    assert config.orientation.max_deviation_deg == 15.0


def test_yaml_rejects_unknown_orientation_keys(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml")
    path.write_text(
        path.read_text(encoding="utf-8") + "\norientation:\n  bogus: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown orientation key"):
        TeleopConfig.from_yaml(path)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"scale": 0.0}, "orientation.scale must be positive"),
        ({"ema_alpha": 0.0}, "orientation.ema_alpha must be in"),
        ({"ema_alpha": 1.5}, "orientation.ema_alpha must be in"),
        (
            {"max_angular_velocity_deg_s": -1.0},
            "orientation.max_angular_velocity_deg_s must be positive",
        ),
        ({"max_step_deg": 0.0}, "orientation.max_step_deg must be positive"),
        ({"max_deviation_deg": 0.0}, "orientation.max_deviation_deg must be in"),
        ({"max_deviation_deg": 91.0}, "orientation.max_deviation_deg must be in"),
    ],
)
def test_orientation_config_range_validation(overrides, message) -> None:
    config = TeleopConfig(
        tls_cert_path=None,
        tls_key_path=None,
        orientation=OrientationConfig(**overrides),
    )
    with pytest.raises(ValueError, match=message):
        config.validate()


HOME_YAML = """
home:
  enabled: true
  tcp: [50, 0, 300, 180, 0, 0]
  joints_deg: [0, -90, 90, -90, 90, 0]
  max_velocity_mm_s: 30
  max_angular_velocity_deg_s: 20
  hold_s: 0.5
"""


def test_yaml_loads_home_section(tmp_path: Path) -> None:
    config = TeleopConfig.from_yaml(
        write_config(tmp_path / "config.yaml", HOME_YAML), validate=False
    )
    config.validate()
    home = config.home
    assert home.enabled and home.tcp == (50.0, 0.0, 300.0, 180.0, 0.0, 0.0)
    assert home.joints_deg == (0.0, -90.0, 90.0, -90.0, 90.0, 0.0)
    assert (home.max_velocity_mm_s, home.max_angular_velocity_deg_s) == (30.0, 20.0)
    assert home.hold_s == 0.5


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "  tcp: [50, 0, 300, 180, 0, 0]\n",
            "",
            "home.tcp and home.joints_deg are required",
        ),
        # x is limited to [-100, 100] by write_config's workspace.
        (
            "  tcp: [50, 0, 300, 180, 0, 0]\n",
            "  tcp: [500, 0, 300, 180, 0, 0]\n",
            "outside the workspace",
        ),
        # motion.max_velocity_mm_s is 40 in write_config.
        (
            "  max_velocity_mm_s: 30\n",
            "  max_velocity_mm_s: 80\n",
            "must not exceed motion.max_velocity_mm_s",
        ),
    ],
)
def test_home_validation(tmp_path: Path, old: str, new: str, message: str) -> None:
    config = TeleopConfig.from_yaml(
        write_config(tmp_path / "config.yaml", HOME_YAML.replace(old, new)),
        validate=False,
    )
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_disabled_home_needs_no_pose(tmp_path: Path) -> None:
    config = TeleopConfig.from_yaml(
        write_config(tmp_path / "config.yaml", "home: {enabled: false}\n"),
        validate=False,
    )
    config.validate()
    assert not config.home.enabled


def test_bimanual_home_is_per_arm() -> None:
    arm_home = HomeConfig(
        enabled=True,
        tcp=(300.0, 0.0, 400.0, 180.0, 0.0, 0.0),
        joints_deg=(0.0,) * 6,
        max_velocity_mm_s=40.0,
    )
    base = TeleopConfig(
        web_dir=Path(__file__).resolve().parents[2] / "web",
        tls_cert_path=None,
        tls_key_path=None,
        arms=(ArmConfig(hand="left", home=arm_home), ArmConfig(hand="right")),
    )
    base.validate()
    with pytest.raises(ValueError, match="home is per arm"):
        replace(base, home=arm_home).validate()
