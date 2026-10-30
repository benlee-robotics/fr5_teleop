from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from teleop.control_math import euler_deg_to_quat
from teleop.recording.lerobot_convert import (
    ConvertOptions,
    SkipEpisode,
    build_episode,
    dataset_features,
    euler_deg_to_matrix,
    find_episodes,
    global_vector_stats,
    iter_video_frames,
    matrix_to_quat_xyzw,
    matrix_to_rot6d,
    parse_camera_latency,
)
from teleop.recording.schema import (
    CAMERA_FRAME_LAYOUT,
    build_sample,
    pack_sample,
    schema_metadata,
)
from teleop.robot.state import WorkerState


ROOT = Path(__file__).resolve().parents[2]
PERIOD_NS = 8_000_000
T0 = 10_000_000_000


def robot_sample(
    i: int,
    *,
    active: bool,
    homing: bool = False,
    gripper_closed: bool | None = None,
    t_ns: int | None = None,
    state_valid: bool = True,
) -> tuple:
    x = 300.0 + i  # 1 mm per 8 ms tick
    # A home return streams the servo like teleop, in its own worker state.
    servo = active or homing
    if homing:
        state = WorkerState.HOMING
    else:
        state = WorkerState.ACTIVE if active else WorkerState.SLEEPING
    return build_sample(
        t_ns=T0 + i * PERIOD_NS if t_ns is None else t_ns,
        tick=i,
        generation=1,
        state=state,
        fresh=True,
        servo_active=servo,
        clutch_id=1,
        cmd_tcp=(x + 0.5, 0.0, 400.0, 180.0, 0.0, 90.0) if servo else None,
        gripper_target=100.0 if (i >= 50 if gripper_closed is None else gripper_closed) else 0.0,
        gripper_busy=False,
        vr_valid=True,
        vr_seq=i,
        vr_received_ns=T0 + i * PERIOD_NS,
        vr_position_m=(0.0, 1.0, 0.0),
        vr_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        grip=active,
        trigger=False,
        joint_pos_deg=(0.0, -90.0, 90.0, -90.0, 90.0, float(i)) if state_valid else None,
        joint_vel_deg_s=(0.0,) * 6 if state_valid else None,
        tcp=(x, 0.0, 400.0, 180.0, 0.0, 90.0) if state_valid else None,
        gripper_position=50.0 if state_valid else None,
        robot_frame_cnt=i % 256,
        robot_time_ms=i * 8,
    )


def write_episode(
    root: Path,
    index: int,
    *,
    samples: int = 125,
    active: range | None = None,
    cameras: dict[str, list[int]] | None = None,
    frame_numbers: list[int] | None = None,
    outcome: str = "success",
    complete: bool = True,
    invalid_states: range = range(0),
    homing: range = range(0),
    gripper_closed: range | None = None,
) -> Path:
    episode = root / f"episode_{index:06d}"
    episode.mkdir(parents=True)
    active = active if active is not None else range(samples)
    data = b"".join(
        pack_sample(
            robot_sample(
                i,
                active=i in active,
                homing=i in homing,
                gripper_closed=None if gripper_closed is None else i in gripper_closed,
                state_valid=i not in invalid_states,
            )
        )
        for i in range(samples)
    )
    (episode / "robot_right.bin").write_bytes(data)
    cameras_meta = {}
    for name, camera_times_ns in (cameras or {}).items():
        import av

        numbers = frame_numbers or list(range(len(camera_times_ns)))
        records = b"".join(
            CAMERA_FRAME_LAYOUT.pack((k, t, t, t / 1e6, numbers[k]))
            for k, t in enumerate(camera_times_ns)
        )
        (episode / f"camera_{name}_frames.bin").write_bytes(records)
        with av.open(str(episode / f"camera_{name}.mp4"), mode="w") as container:
            stream = container.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
            stream.options = {"crf": "0"}
            for k in range(len(camera_times_ns)):
                # Frame k is a flat gray of level 4 * k for identification.
                image = np.full((24, 32, 3), (4 * k) % 256, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                frame.pts = k
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        cameras_meta[name] = {
            "name": name,
            "width": 32,
            "height": 24,
            "video": f"camera_{name}.mp4",
            "frames_file": f"camera_{name}_frames.bin",
            "frames": len(camera_times_ns),
            "frame_layout": CAMERA_FRAME_LAYOUT.metadata(),
        }
    meta = {
        "format": "fr5_teleop_raw",
        "format_version": 1,
        "episode_index": index,
        "task": f"task {index}",
        "outcome": outcome,
        "complete": complete,
        "start_ns": T0,
        "stop_ns": T0 + samples * PERIOD_NS,
        "hands": {"right": {"file": "robot_right.bin", "samples": samples}},
        "cameras": cameras_meta,
        "sample_schema": schema_metadata(),
        "session": {
            "arms": {
                "right": {"gripper_open_position": 0, "gripper_closed_position": 100}
            }
        },
    }
    (episode / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return episode


def test_rotation_matches_teleop_euler_convention() -> None:
    rng = np.random.default_rng(0)
    for euler in rng.uniform(-179, 179, size=(50, 3)):
        matrix = euler_deg_to_matrix(euler[None, :])
        quat = matrix_to_quat_xyzw(matrix)[0]
        expected = np.array(euler_deg_to_quat(*euler))
        assert min(np.abs(quat - expected).max(), np.abs(quat + expected).max()) < 1e-9
        # rot6d is the first two columns of an orthonormal matrix.
        rot6d = matrix_to_rot6d(matrix)[0]
        assert abs(np.dot(rot6d[:3], rot6d[3:])) < 1e-9
        assert np.allclose(np.linalg.norm(rot6d[:3]), 1.0)


def test_quaternions_are_sign_continuous() -> None:
    # Rotating through 180 deg flips a canonical quaternion's sign.
    euler = np.array([[0.0, 0.0, angle] for angle in np.linspace(170, 190, 21)])
    quat = matrix_to_quat_xyzw(euler_deg_to_matrix(euler))
    assert (np.sum(quat[1:] * quat[:-1], axis=1) > 0).all()


def test_robot_only_episode_uses_zero_order_hold(tmp_path: Path) -> None:
    episode = write_episode(tmp_path, 0, samples=125)
    data = build_episode(episode, ConvertOptions(trim_idle=False))
    plan = data.plan
    # 124 ticks * 8 ms = 0.992 s → 30 Hz grid of 30 frames.
    assert len(plan.grid_ns) == 30
    assert plan.frame_index == {} and data.video_paths == {}
    assert "cameras" not in plan.report
    # Latest sample at or before each grid time.
    robot_t = T0 + np.arange(125) * PERIOD_NS
    for k, t in enumerate(plan.grid_ns):
        i = plan.robot_index[k]
        assert robot_t[i] <= t < robot_t[i] + PERIOD_NS
    k = 10
    i = plan.robot_index[k]
    # state = [x m, y, z, rot6d, gripper]; rx=180, rz=90.
    assert data.state[k][0] == pytest.approx((300.0 + i) / 1000.0)
    assert data.state[k][2] == pytest.approx(0.4)
    assert np.allclose(data.state[k][3:9], [0, 1, 0, 1, 0, 0], atol=1e-6)
    assert data.state[k][9] == pytest.approx(0.5)
    # Command action is the sent ServoCart target, 0.5 mm ahead.
    assert data.action[k][0] == pytest.approx((300.5 + i) / 1000.0)
    assert data.joint_pos[k][5] == pytest.approx(math.radians(i))
    assert plan.report["robot_stale_frames"] == 0


def test_idle_edges_are_trimmed_and_idle_action_holds(tmp_path: Path) -> None:
    episode = write_episode(tmp_path, 0, samples=250, active=range(60, 200))
    data = build_episode(episode, ConvertOptions(trim_pad_s=0.0))
    robot_index = data.plan.robot_index
    assert 59 <= robot_index[0] <= 60
    assert 196 <= robot_index[-1] <= 199
    untrimmed = build_episode(episode, ConvertOptions(trim_idle=False))
    idle = untrimmed.plan.robot_index < 60
    # Without a command the action holds the measured pose.
    assert np.allclose(untrimmed.action[idle, :9], untrimmed.state[idle, :9])
    # The gripper action is the commanded target, normalized to [0, 1].
    assert set(np.unique(untrimmed.action[:, 9])) == {0.0, 1.0}


def test_trim_keeps_the_gripper_release_after_the_last_servo(tmp_path: Path) -> None:
    # Field pattern: move over the box, release grip (servo stops), open the
    # gripper, wait, then start the home return.
    episode = write_episode(
        tmp_path,
        0,
        samples=300,
        active=range(20, 150),
        gripper_closed=range(50, 180),
        homing=range(240, 280),
    )
    # Without a pad the grid ends within one frame (4 ticks) of the command.
    unpadded = build_episode(episode, ConvertOptions(trim_pad_s=0.0))
    assert unpadded.plan.robot_index[-1] >= 180 - 4
    # The 0.25 s pad follows the release; the home return still cuts.
    data = build_episode(episode, ConvertOptions())
    assert 180 + 27 <= data.plan.robot_index[-1] < 240
    assert data.action[-1, 9] == 0.0


def test_trim_waits_for_the_gripper_to_finish_moving(tmp_path: Path) -> None:
    episode = write_episode(
        tmp_path, 0, samples=300, active=range(20, 150), gripper_closed=range(50, 180)
    )
    path = episode / "robot_right.bin"
    from teleop.recording.schema import ROBOT_LAYOUT

    records = np.frombuffer(path.read_bytes(), dtype=np.dtype(ROBOT_LAYOUT.numpy_dtype_descr())).copy()
    # The gripper opens over 0.8 s after the command at sample 180: the
    # position changes every sample up to 280, then stays at 0.
    records["gripper_position"][180:] = np.maximum(50.0 - np.arange(120) * 0.5, 0.0)
    path.write_bytes(records.tobytes())
    data = build_episode(episode, ConvertOptions(trim_pad_s=0.0))
    assert 276 <= data.plan.robot_index[-1] <= 280


def test_global_stats_span_every_episode(tmp_path: Path) -> None:
    near = build_episode(write_episode(tmp_path, 0, samples=125), ConvertOptions(trim_idle=False))
    far = build_episode(write_episode(tmp_path, 1, samples=250), ConvertOptions(trim_idle=False))
    stats = global_vector_stats([near, far])
    both = np.concatenate([near.action, far.action]).astype(np.float64)
    assert set(stats["action"]) == {"min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"}
    assert stats["action"]["count"].tolist() == [len(both)]
    assert np.allclose(stats["action"]["q99"], np.quantile(both, 0.99, axis=0))
    assert np.allclose(stats["action"]["q01"], np.quantile(both, 0.01, axis=0))
    # Wider than LeRobot's frame-weighted average of per-episode quantiles.
    averaged = sum(
        np.quantile(d.action[:, 0], 0.99) * len(d.action) for d in (near, far)
    ) / len(both)
    assert stats["action"]["q99"][0] > averaged + 0.01
    assert stats["observation.joint_pos"]["mean"].shape == (6,)


def test_next_state_action_mode(tmp_path: Path) -> None:
    episode = write_episode(tmp_path, 0)
    data = build_episode(
        episode, ConvertOptions(trim_idle=False, action_mode="next_state")
    )
    assert np.array_equal(data.action[:-1], data.state[1:])
    assert np.array_equal(data.action[-1], data.state[-1])


@pytest.mark.parametrize(
    ("rotation", "dims"), [("rot6d", 10), ("quat", 8), ("euler", 7)]
)
def test_rotation_options_set_vector_size(
    tmp_path: Path, rotation: str, dims: int
) -> None:
    episode = write_episode(tmp_path, 0)
    data = build_episode(episode, ConvertOptions(trim_idle=False, rotation=rotation))
    assert data.state.shape[1] == dims == data.action.shape[1]
    features = dataset_features(ConvertOptions(rotation=rotation), [])
    assert features["observation.state"]["shape"] == (dims,)
    assert len(features["action"]["names"]) == dims


def test_camera_frames_are_matched_to_nearest_capture_time(tmp_path: Path) -> None:
    # 30 fps camera with jitter, starting 20 ms after the robot.
    rng = np.random.default_rng(1)
    times = [
        int(T0 + 20e6 + k * 1e9 / 30 + rng.uniform(-3e6, 3e6)) for k in range(28)
    ]
    episode = write_episode(tmp_path, 0, cameras={"wrist": times})
    data = build_episode(episode, ConvertOptions(trim_idle=False))
    plan = data.plan
    assert plan.grid_ns[0] == times[0]
    capture = np.array(times)
    for k, t in enumerate(plan.grid_ns):
        assert np.abs(capture - t).argmin() == plan.frame_index["wrist"][k]
    assert plan.report["cameras"]["wrist"]["gap_frames"] == 0
    # Decoding yields exactly the planned frames (gray level 4 * index).
    levels = [
        int(round(image.mean() / 4))
        for image in iter_video_frames(
            data.video_paths["wrist"], plan.frame_index["wrist"]
        )
    ]
    assert levels == plan.frame_index["wrist"].tolist()


def test_two_cameras_are_matched_independently(tmp_path: Path) -> None:
    # Unsynchronized cameras: top starts 50 ms after wrist and ends earlier.
    wrist = [int(T0 + k * 1e9 / 30) for k in range(30)]
    top = [int(T0 + 50e6 + k * 1e9 / 30 + 7e6) for k in range(25)]
    episode = write_episode(tmp_path, 0, cameras={"wrist": wrist, "top": top})
    data = build_episode(episode, ConvertOptions(trim_idle=False))
    plan = data.plan
    # The grid covers only the time both cameras recorded.
    assert plan.grid_ns[0] == top[0]
    assert plan.grid_ns[-1] <= top[-1]
    assert sorted(plan.frame_index) == ["top", "wrist"]
    assert sorted(data.video_paths) == ["top", "wrist"]
    for name, times in (("wrist", wrist), ("top", top)):
        capture = np.array(times)
        for k, t in enumerate(plan.grid_ns):
            assert np.abs(capture - t).argmin() == plan.frame_index[name][k]
        assert plan.report["cameras"][name]["gap_frames"] == 0

    # A per-camera latency moves only that camera's matching.
    shifted = build_episode(
        episode,
        ConvertOptions(trim_idle=False, camera_latency_ms_by_name={"top": 40.0}),
    )
    assert shifted.plan.grid_ns[0] == top[0] - 40_000_000


def test_parse_camera_latency() -> None:
    assert parse_camera_latency([]) == (0.0, {})
    assert parse_camera_latency(["12.5", "top=40"]) == (12.5, {"top": 40.0})
    options = ConvertOptions(
        camera_latency_ms=12.5, camera_latency_ms_by_name={"top": 40.0}
    )
    assert options.latency_ns("wrist") == 12_500_000
    assert options.latency_ns("top") == 40_000_000
    with pytest.raises(SystemExit, match="not a number"):
        parse_camera_latency(["top=fast"])


def test_camera_latency_shifts_matching(tmp_path: Path) -> None:
    times = [int(T0 + 100e6 + k * 1e9 / 30) for k in range(20)]
    episode = write_episode(tmp_path, 0, cameras={"wrist": times})
    plain = build_episode(episode, ConvertOptions(trim_idle=False))
    shifted = build_episode(
        episode, ConvertOptions(trim_idle=False, camera_latency_ms=40.0)
    )
    # Frames now count as captured 40 ms earlier: robot samples start 40 ms
    # earlier for the same frames.
    assert shifted.plan.grid_ns[0] == plain.plan.grid_ns[0] - 40_000_000
    assert shifted.plan.robot_index[0] == plain.plan.robot_index[0] - 5


def test_camera_gaps_skip_episode_and_count_internal_drops(tmp_path: Path) -> None:
    times = [int(T0 + k * 1e9 / 30) for k in range(10)]
    times += [int(T0 + k * 1e9 / 30) for k in range(20, 30)]  # 0.33 s hole
    numbers = list(range(10)) + list(range(20, 30))
    episode = write_episode(
        tmp_path, 0, cameras={"wrist": times}, frame_numbers=numbers
    )
    with pytest.raises(SkipEpisode, match="no camera wrist frame"):
        build_episode(episode, ConvertOptions(trim_idle=False))
    data = build_episode(
        episode, ConvertOptions(trim_idle=False, max_camera_gap_ratio=1.0)
    )
    assert data.plan.report["cameras"]["wrist"]["gap_frames"] > 0
    assert data.plan.report["cameras"]["wrist"]["internal_drops"] == 10


def test_episode_filters(tmp_path: Path) -> None:
    failure = write_episode(tmp_path, 0, outcome="failure")
    with pytest.raises(SkipEpisode, match="outcome is failure"):
        build_episode(failure, ConvertOptions())
    assert build_episode(failure, ConvertOptions(include_failures=True))

    incomplete = write_episode(tmp_path, 1, complete=False)
    with pytest.raises(SkipEpisode, match="incomplete"):
        build_episode(incomplete, ConvertOptions())
    assert build_episode(incomplete, ConvertOptions(allow_incomplete=True))

    idle = write_episode(tmp_path, 2, active=range(0))
    with pytest.raises(SkipEpisode, match="servo never active"):
        build_episode(idle, ConvertOptions())

    # Longer than one 33 ms grid step so a dataset frame must use it.
    broken = write_episode(tmp_path, 3, invalid_states=range(40, 46))
    with pytest.raises(SkipEpisode, match="non-finite"):
        build_episode(broken, ConvertOptions(trim_idle=False))

    (tmp_path / "episode_000004.incomplete").mkdir()
    (tmp_path / "episode_000005.tmp").mkdir()
    assert [p.name for p in find_episodes(tmp_path)] == [
        f"episode_{i:06d}" for i in range(4)
    ]


@pytest.mark.skipif(
    not os.environ.get("LEROBOT_PYTHON"),
    reason="set LEROBOT_PYTHON to a Python with lerobot 0.4.x installed",
)
def test_convert_and_load_with_lerobot(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    times = [int(T0 + k * 1e9 / 30) for k in range(28)]
    cameras = {"wrist": times, "top": times}
    write_episode(raw, 0, cameras=cameras)
    write_episode(raw, 1, cameras=cameras, outcome="failure")
    # Longer, so it reaches further in x: per-episode quantiles differ.
    long_times = [int(T0 + k * 1e9 / 30) for k in range(60)]
    write_episode(raw, 2, samples=250, cameras={"wrist": long_times, "top": long_times})
    out = tmp_path / "dataset"
    script = f"""
import json, sys
from pathlib import Path
from teleop.recording.lerobot_convert import main
main(["--raw", {str(raw)!r}, "--out", {str(out)!r}, "--repo-id", "local/test",
      "--no-trim", "--vcodec", "h264"])
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/test", root={str(out)!r}, video_backend="pyav")
sample = ds[3]
action = np.stack(ds.hf_dataset["action"]).astype(np.float64)
print("RESULT" + json.dumps({{
    "episodes": ds.num_episodes,
    "frames": ds.num_frames,
    "tasks": sorted(set(ds.meta.tasks.index)),
    "image": list(sample["observation.images.wrist"].shape),
    "top": list(sample["observation.images.top"].shape),
    "state": len(sample["observation.state"]),
    "stats_q99_is_global": bool(np.allclose(
        ds.meta.stats["action"]["q99"], np.quantile(action, 0.99, axis=0))),
    "stats_count": int(ds.meta.stats["action"]["count"][0]),
}}))
"""
    result = subprocess.run(
        [os.environ["LEROBOT_PYTHON"], "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("RESULT")]
    assert lines, result.stdout[-2000:] + result.stderr[-2000:]
    summary = json.loads(lines[0][len("RESULT"):])
    assert summary == {
        "episodes": 2,
        "frames": 87,
        "tasks": ["task 0", "task 2"],
        "image": [3, 24, 32],
        "top": [3, 24, 32],
        "state": 10,
        "stats_q99_is_global": True,
        "stats_count": 87,
    }
    report = json.loads((out / "fr5_conversion.json").read_text())
    assert report["skipped"] == {"episode_000001": "outcome is failure"}


def test_euler_output_is_unwrapped_across_180(tmp_path: Path) -> None:
    # rx oscillates around 180 deg as reported by the controller: +179.5,
    # -179.5, ... (field episodes had 10+ such jumps).
    episode = write_episode(tmp_path, 0, samples=125)
    path = episode / "robot_right.bin"
    from teleop.recording.schema import ROBOT_LAYOUT

    records = np.frombuffer(path.read_bytes(), dtype=np.dtype(ROBOT_LAYOUT.numpy_dtype_descr())).copy()
    rx = np.where(np.arange(125) % 20 < 10, 179.5, -179.5)
    records["tcp"][:, 3] = rx
    records["cmd_tcp"][:, 3] = -rx  # command on the other side of the wrap
    path.write_bytes(records.tobytes())

    euler = build_episode(episode, ConvertOptions(trim_idle=False, rotation="euler"))
    state_rx = euler.state[:, 3]
    assert np.abs(np.diff(state_rx)).max() < math.radians(2)
    assert np.abs(euler.action[:, 3] - state_rx).max() < math.radians(2)
    # rot6d is continuous without any unwrapping.
    rot6d = build_episode(episode, ConvertOptions(trim_idle=False))
    assert np.abs(np.diff(rot6d.state[:, 3:9], axis=0)).max() < 0.05


def test_home_return_after_demo_is_excluded_by_default(tmp_path: Path) -> None:
    episode = write_episode(
        tmp_path, 0, samples=250, active=range(20, 150), homing=range(160, 220)
    )
    data = build_episode(episode, ConvertOptions())
    # Cut just before the first HOMING sample, even inside the 0.25 s pad.
    assert data.plan.robot_index[-1] < 160
    assert data.plan.report["homing"] == {
        "mode": "excluded",
        "resumed_after_return": False,
        "homing_s": pytest.approx(59 * PERIOD_NS / 1e9),
    }


def test_home_return_can_be_included(tmp_path: Path) -> None:
    episode = write_episode(
        tmp_path, 0, samples=250, active=range(20, 150), homing=range(160, 220)
    )
    data = build_episode(episode, ConvertOptions(homing="include"))
    plan = data.plan
    assert plan.robot_index[-1] >= 219
    assert plan.report["homing"]["mode"] == "included"
    homing_frames = (plan.robot_index >= 160) & (plan.robot_index < 220)
    assert homing_frames.any()
    # The streamed home targets are valid command actions.
    x = 300.5 + plan.robot_index[homing_frames]
    assert np.allclose(data.action[homing_frames, 0], x / 1000.0)


@pytest.mark.parametrize("homing_mode", ["exclude", "include"])
def test_teleop_after_a_return_cuts_the_episode_at_the_return(
    tmp_path: Path, homing_mode: str
) -> None:
    episode = write_episode(
        tmp_path,
        0,
        samples=250,
        active=list(range(20, 100)) + list(range(160, 200)),
        homing=range(110, 150),
    )
    data = build_episode(episode, ConvertOptions(homing=homing_mode))
    assert data.plan.robot_index[-1] < 110
    assert data.plan.report["homing"]["mode"] == "excluded"
    assert data.plan.report["homing"]["resumed_after_return"] is True


def test_home_return_before_the_demo_is_dropped(tmp_path: Path) -> None:
    episode = write_episode(
        tmp_path, 0, samples=250, active=range(40, 200), homing=range(0, 30)
    )
    for options in (ConvertOptions(), ConvertOptions(trim_idle=False)):
        data = build_episode(episode, options)
        assert data.plan.robot_index[0] >= 30
