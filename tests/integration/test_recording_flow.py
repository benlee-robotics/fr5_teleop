from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

from teleop.app import create_app
from teleop.config import CameraConfig, RecordingConfig, TeleopConfig
from teleop.recording.schema import STATE_NAMES, iter_robot_samples

from tests.integration.test_web_app import receive_state, receive_status, receive_type


ROOT = Path(__file__).resolve().parents[2]


def make_config(**overrides) -> TeleopConfig:
    return TeleopConfig(
        web_dir=ROOT / "web",
        tls_cert_path=None,
        tls_key_path=None,
        status_hz=20.0,
        **overrides,
    )


@pytest.mark.parametrize("camera_names", [(), ("wrist",), ("wrist", "top")])
async def test_operator_records_episode_while_quest_teleoperates(
    tmp_path: Path,
    camera_names: tuple[str, ...],
) -> None:
    if camera_names:
        pytest.importorskip("av")
        pytest.importorskip("numpy")
    cameras = tuple(
        CameraConfig(
            enabled=True, source="fake", name=name, width=64, height=48, fps=30
        )
        for name in camera_names
    )
    config = make_config(
        recording=RecordingConfig(enabled=True, root_dir=tmp_path, cameras=cameras)
    )
    async with TestServer(create_app(config)) as server:
        async with TestClient(server) as client:
            quest = await client.ws_connect("/ws")
            quest_id = (await receive_type(quest, "hello"))["session_id"]
            monitor = await client.ws_connect("/ws")
            monitor_id = (await receive_type(monitor, "hello"))["session_id"]

            await quest.send_json(
                {
                    "version": 1,
                    "type": "event",
                    "session_id": quest_id,
                    "event": "claim_control",
                }
            )
            assert (await receive_type(quest, "control"))["granted"] is True
            await receive_state(quest, "IDLE")

            def record(action: str, **fields) -> dict:
                return {
                    "version": 1,
                    "type": "record",
                    "session_id": monitor_id,
                    "action": action,
                    **fields,
                }

            if camera_names:
                streaming = await receive_status(
                    monitor,
                    lambda s: (
                        len(cameras := (s.get("recording") or {}).get("cameras") or [])
                        == len(camera_names)
                        and all(camera["streaming"] for camera in cameras)
                    ),
                )
                assert [
                    camera["name"] for camera in streaming["recording"]["cameras"]
                ] == list(camera_names)

            # The read-only monitor controls recording without the lease.
            await monitor.send_json(record("start", task="push the block left"))
            recording = await receive_status(
                monitor,
                lambda s: (s.get("recording") or {}).get("state") == "RECORDING",
            )
            assert recording["recording"]["task"] == "push the block left"

            seq = 0

            async def send_pose(grip: bool, x: float) -> None:
                nonlocal seq
                await quest.send_json(
                    {
                        "version": 1,
                        "type": "pose",
                        "session_id": quest_id,
                        "seq": seq,
                        "client_time_ms": float(seq),
                        "hand": "right",
                        "position_m": [x, 1.0, 0.0],
                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        "grip": grip,
                        "trigger": False,
                    }
                )
                seq += 1

            await send_pose(False, 0.0)
            for step in range(25):
                await send_pose(True, step * 0.002)
                await asyncio.sleep(0.02)
            for _ in range(5):
                await send_pose(False, 0.05)
                await asyncio.sleep(0.02)
            await receive_state(quest, "SLEEPING")

            await monitor.send_json(record("stop", outcome="success"))
            saved = await receive_status(
                monitor,
                lambda s: (
                    (s.get("recording") or {}).get("state") == "IDLE"
                    and (s["recording"].get("last_saved") or {}).get(
                        "episode_index"
                    )
                    == 0
                ),
            )
            assert saved["recording"]["alive"] is True
            await quest.close()
            await monitor.close()

    episode_dir = tmp_path / "episode_000000"
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["task"] == "push the block left"
    assert meta["outcome"] == "success"
    assert meta["complete"] is True
    assert meta["session"]["robot_kind"] == "fake"
    assert meta["session"]["hands"] == ["right"]
    assert sorted(meta["cameras"]) == sorted(camera_names)
    for camera_meta in meta["cameras"].values():
        # ~0.75 s of recording at 30 fps.
        assert camera_meta["frames"] >= 10
        assert (episode_dir / camera_meta["video"]).stat().st_size > 0

    samples = list(iter_robot_samples(episode_dir / "robot_right.bin"))
    assert meta["hands"]["right"]["samples"] == len(samples)
    # ~0.6 s of teleop at an 8 ms servo period.
    assert len(samples) > 40
    times = [s["t_ns"] for s in samples]
    assert times == sorted(times)
    assert meta["start_ns"] <= times[0] and times[-1] <= meta["stop_ns"]
    assert all(s["state_valid"] == 1 for s in samples)

    states = {STATE_NAMES[s["state"]] for s in samples}
    assert {"ACTIVE", "SLEEPING"} <= states
    commanded = [s for s in samples if s["cmd_valid"]]
    assert commanded and all(STATE_NAMES[s["state"]] == "ACTIVE" for s in commanded)
    assert {s["clutch_id"] for s in commanded} == {1}
    # The fake robot reaches each target, so the next observation equals the
    # previous tick's command: observation is read before the command.
    for previous, current in zip(samples, samples[1:]):
        if previous["cmd_valid"] and current["tick"] == previous["tick"] + 1:
            assert current["tcp"] == previous["cmd_tcp"]
            break
    else:
        raise AssertionError("no consecutive commanded ticks found")


async def test_record_command_is_rejected_when_recording_disabled() -> None:
    async with TestServer(create_app(make_config())) as server:
        async with TestClient(server) as client:
            ws = await client.ws_connect("/ws")
            session_id = (await receive_type(ws, "hello"))["session_id"]
            status = await receive_type(ws, "status")
            assert status["recording"] is None
            await ws.send_json(
                {
                    "version": 1,
                    "type": "record",
                    "session_id": session_id,
                    "action": "start",
                    "task": "task",
                }
            )
            error = await receive_type(ws, "error")
            assert error["code"] == "recording_unavailable"
            await ws.close()


async def test_quest_buttons_use_task_set_by_monitor(tmp_path: Path) -> None:
    config = make_config(recording=RecordingConfig(enabled=True, root_dir=tmp_path))
    async with TestServer(create_app(config)) as server:
        async with TestClient(server) as client:
            quest = await client.ws_connect("/ws")
            quest_id = (await receive_type(quest, "hello"))["session_id"]
            monitor = await client.ws_connect("/ws")
            monitor_id = (await receive_type(monitor, "hello"))["session_id"]

            def record(session_id: str, action: str, **fields) -> dict:
                return {
                    "version": 1,
                    "type": "record",
                    "session_id": session_id,
                    "action": action,
                    **fields,
                }

            # Quest A with no task anywhere is refused and bumps error_seq.
            await quest.send_json(record(quest_id, "start"))
            refused = await receive_status(
                quest,
                lambda s: ((s.get("recording") or {}).get("error_seq") or 0) >= 1,
            )
            assert "task is required" in refused["recording"]["last_error"]

            await monitor.send_json(record(monitor_id, "set_task", task="sort the screws"))
            await receive_status(
                quest,
                lambda s: (s.get("recording") or {}).get("next_task")
                == "sort the screws",
            )
            # What RecordButtons sends for A in IDLE, then A in RECORDING.
            await quest.send_json(record(quest_id, "start"))
            recording = await receive_status(
                quest,
                lambda s: (s.get("recording") or {}).get("state") == "RECORDING",
            )
            assert recording["recording"]["task"] == "sort the screws"
            await asyncio.sleep(0.2)
            await quest.send_json(record(quest_id, "stop", outcome="success"))
            await receive_status(
                quest,
                lambda s: (
                    ((s.get("recording") or {}).get("last_saved") or {}).get(
                        "episode_index"
                    )
                    == 0
                ),
            )
            await quest.close()
            await monitor.close()

    meta = json.loads((tmp_path / "episode_000000" / "meta.json").read_text())
    assert meta["task"] == "sort the screws"
    assert meta["outcome"] == "success"
