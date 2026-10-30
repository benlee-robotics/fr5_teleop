from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from teleop.app import create_app
from teleop.config import ArmConfig, HomeConfig, TeleopConfig
from teleop.robot.fake_client import FAKE_JOINTS


async def receive_type(ws, message_type: str, timeout: float = 3.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        message = await asyncio.wait_for(ws.receive_json(), timeout=timeout)
        if message.get("type") == message_type:
            return message
    raise AssertionError(f"did not receive message type {message_type}")


async def receive_state(ws, state: str, timeout: float = 3.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        remaining = deadline - asyncio.get_running_loop().time()
        message = await asyncio.wait_for(ws.receive_json(), timeout=remaining)
        if message.get("type") == "status" and message.get("state") == state:
            return message
    raise AssertionError(f"did not receive worker state {state}")


async def receive_status(ws, predicate, timeout: float = 3.0) -> dict:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        remaining = deadline - asyncio.get_running_loop().time()
        message = await asyncio.wait_for(ws.receive_json(), timeout=remaining)
        if message.get("type") == "status" and predicate(message):
            return message
    raise AssertionError("did not receive expected worker status")


async def test_websocket_claim_pose_and_release_flow() -> None:
    root = Path(__file__).resolve().parents[2]
    config = TeleopConfig(
        web_dir=root / "web",
        tls_cert_path=None,
        tls_key_path=None,
        status_hz=20.0,
    )
    async with TestServer(create_app(config)) as server:
        async with TestClient(server) as client:
            live = await client.get("/health/live")
            assert live.status == 200
            monitor = await client.get("/monitor")
            assert monitor.status == 200
            assert "VR Teleop 현장 모니터" in await monitor.text()

            ws = await client.ws_connect("/ws")
            hello = await receive_type(ws, "hello")
            session_id = hello["session_id"]
            assert hello["control_hands"] == ["right"]
            await ws.send_json(
                {
                    "version": 1,
                    "type": "event",
                    "session_id": session_id,
                    "event": "claim_control",
                }
            )
            control = await receive_type(ws, "control")
            assert control["granted"] is True
            idle = await receive_state(ws, "IDLE")
            assert idle["gripper_enabled"] is False
            assert idle["gripper_busy"] is False
            assert idle["gripper_position"] == 0

            await ws.send_json(
                {
                    "version": 1,
                    "type": "pose",
                    "session_id": session_id,
                    "seq": 0,
                    "client_time_ms": 0.0,
                    "hand": "left",
                    "position_m": [0.0, 1.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "grip": False,
                    "trigger": False,
                }
            )
            wrong_hand = await receive_type(ws, "error")
            assert "no robot is configured" in wrong_hand["message"]

            await ws.send_json(
                {
                    "version": 1,
                    "type": "telemetry",
                    "session_id": session_id,
                    "seq": 0,
                    "client_time_ms": 1000.0,
                    "xr_frame_count": 90,
                    "valid_pose_count": 89,
                    "pose_send_count": 89,
                    "pose_drop_count": 0,
                    "tracking_loss_count": 1,
                    "max_xr_frame_gap_ms": 14.0,
                    "max_pose_gap_ms": 28.0,
                    "ws_buffered_amount": 0,
                    "last_rtt_ms": 5.0,
                    "max_rtt_ms": 7.5,
                }
            )
            telemetry_ack = await receive_type(ws, "telemetry_ack")
            assert telemetry_ack["seq"] == 0
            telemetry_status = await receive_status(
                ws,
                lambda status: (
                    status.get("client_telemetry", {}).get("seq") == 0
                ),
            )
            assert telemetry_status["client_telemetry"]["max_rtt_ms"] == 7.5

            def pose(seq: int, grip: bool, x: float = 0.0) -> dict:
                return {
                    "version": 1,
                    "type": "pose",
                    "session_id": session_id,
                    "seq": seq,
                    "client_time_ms": float(seq),
                    "hand": "right",
                    "position_m": [x, 1.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "grip": grip,
                    "trigger": False,
                }

            await ws.send_json(pose(0, False))
            await ws.send_json(pose(1, True, 0.01))
            active = await receive_state(ws, "ACTIVE")
            assert active["tracking"] is True
            assert active["rearm_required"] is False

            await ws.send_json(pose(2, False, 0.01))
            await ws.send_json(pose(3, True, 0.02))
            resumed = await receive_status(
                ws,
                lambda status: (
                    status.get("state") == "ACTIVE"
                    and status.get("counters", {}).get("servo_start_count") == 2
                ),
            )
            assert resumed["tracking"] is True
            assert resumed["rearm_required"] is False
            assert resumed["controller_id"] == session_id
            assert resumed["counters"]["servo_start_count"] == 2
            assert resumed["counters"]["servo_end_count"] == 1

            await ws.send_json(
                {
                    "version": 1,
                    "type": "event",
                    "session_id": session_id,
                    "event": "webxr_ended",
                }
            )
            idle = await receive_state(ws, "IDLE")
            assert idle["tracking"] is False
            await ws.close()


async def test_bimanual_websocket_routes_both_hands_and_reports_ack() -> None:
    root = Path(__file__).resolve().parents[2]
    config = TeleopConfig(
        web_dir=root / "web",
        tls_cert_path=None,
        tls_key_path=None,
        status_hz=30.0,
        pose_timeout_s=0.500,
        worker_watchdog_s=1.000,
        arms=(ArmConfig(hand="left"), ArmConfig(hand="right")),
    )
    async with TestServer(create_app(config)) as server:
        async with TestClient(server) as client:
            ws = await client.ws_connect("/ws")
            hello = await receive_type(ws, "hello")
            session_id = hello["session_id"]
            assert hello["control_hands"] == ["left", "right"]
            await ws.send_json(
                {
                    "version": 1,
                    "type": "event",
                    "session_id": session_id,
                    "event": "claim_control",
                }
            )
            assert (await receive_type(ws, "control"))["granted"] is True
            await receive_status(
                ws,
                lambda status: all(
                    status.get("arms", {}).get(hand, {}).get("state") == "IDLE"
                    for hand in ("left", "right")
                ),
            )

            def hand_pose(hand: str, seq: int, grip: bool) -> dict:
                return {
                    "version": 1,
                    "type": "pose",
                    "session_id": session_id,
                    "seq": seq,
                    "client_time_ms": float(seq),
                    "hand": hand,
                    "position_m": [0.01 * seq, 1.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "grip": grip,
                    "trigger": False,
                }

            for hand in ("left", "right"):
                await ws.send_json(hand_pose(hand, 0, False))
                await ws.send_json(hand_pose(hand, 1, True))

            active = await receive_status(
                ws,
                lambda status: all(
                    status.get("arms", {}).get(hand, {}).get("state") == "ACTIVE"
                    for hand in ("left", "right")
                ),
            )
            assert active["ack_seq"] == 1
            assert active["ack_seq_by_hand"] == {"left": 1, "right": 1}

            await ws.send_json(
                {
                    "version": 1,
                    "type": "pose",
                    "session_id": session_id,
                    "seq": 0,
                    "client_time_ms": 0.0,
                    "hand": "none",
                    "position_m": [0.0, 1.0, 0.0],
                    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "grip": False,
                    "trigger": False,
                }
            )
            error = await receive_type(ws, "error")
            assert "no robot is configured" in error["message"]
            await ws.close()


async def test_home_return_lease_rules() -> None:
    root = Path(__file__).resolve().parents[2]
    config = TeleopConfig(
        web_dir=root / "web",
        tls_cert_path=None,
        tls_key_path=None,
        status_hz=20.0,
        max_velocity_mm_s=50.0,
        max_step_mm=5.0,
        # 20 mm/s over ~57 mm: several seconds, so it can be stopped midway.
        home=HomeConfig(
            enabled=True,
            tcp=(350.0, 20.0, 420.0, 180.0, 0.0, 10.0),
            joints_deg=FAKE_JOINTS,
            max_velocity_mm_s=20.0,
        ),
    )

    def event(session_id: str, name: str) -> dict:
        return {"version": 1, "type": "event", "session_id": session_id, "event": name}

    async with TestServer(create_app(config)) as server:
        async with TestClient(server) as client:
            quest = await client.ws_connect("/ws")
            quest_id = (await receive_type(quest, "hello"))["session_id"]
            monitor = await client.ws_connect("/ws")
            monitor_id = (await receive_type(monitor, "hello"))["session_id"]
            await receive_state(monitor, "IDLE")

            # No Quest holds the lease: /monitor may start and stop a return.
            await monitor.send_json(event(monitor_id, "go_home"))
            await receive_state(monitor, "HOMING")
            await monitor.send_json(event(monitor_id, "stop_home"))
            stopped = await receive_status(
                monitor,
                lambda s: s["state"] == "IDLE" and s.get("reason") == "home_stopped",
            )
            assert stopped["home_enabled"] is True

            # Once the Quest holds the lease, only the Quest may start one.
            await quest.send_json(event(quest_id, "claim_control"))
            assert (await receive_type(quest, "control"))["granted"] is True
            await monitor.send_json(event(monitor_id, "go_home"))
            error = await receive_type(monitor, "error")
            assert "another session owns" in error["message"]

            await quest.send_json(event(quest_id, "go_home"))
            await receive_state(monitor, "HOMING")
            # Stopping is allowed from any session.
            await monitor.send_json(event(monitor_id, "stop_home"))
            await receive_status(
                monitor,
                lambda s: s["state"] == "IDLE" and s.get("reason") == "home_stopped",
            )
            await quest.close()
            await monitor.close()
