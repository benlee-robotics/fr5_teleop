"""Ctrl+C shutdown of a real ``python -m teleop`` process.

Ctrl+C on a TTY (terminal, ``docker compose run``) sends SIGINT to the whole
foreground process group, often several times. With a monitor WebSocket open
and an episode recording, shutdown must still stop the robot worker and
finalize the episode promptly instead of waiting on the open socket or
aborting cleanup on the repeated signal.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import pytest

from tests.integration.test_web_app import receive_type


ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="process-group signals are Linux-only here"
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_config(tmp_path: Path, port: int) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "runtime:\n"
        "  dry_run: true\n"
        "server:\n"
        "  host: 127.0.0.1\n"
        f"  port: {port}\n"
        f"  web_dir: {ROOT / 'web'}\n"
        "tls:\n"
        "  enabled: false\n"
        "recording:\n"
        "  enabled: true\n"
        "  root_dir: recordings\n",
        encoding="utf-8",
    )
    return path


async def _wait_ready(session: aiohttp.ClientSession, base: str, proc) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        assert proc.poll() is None, "server exited during startup"
        try:
            async with session.get(f"{base}/health/ready") as response:
                if response.status == 200 and (await response.json())["ready"]:
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(0.1)
    raise AssertionError("server did not become ready")


@pytest.mark.parametrize("presses", [1, 10])
async def test_ctrl_c_with_open_socket_finalizes_recording(
    tmp_path: Path, presses: int
) -> None:
    port = _free_port()
    config = _write_config(tmp_path, port)
    log_path = tmp_path / "server.log"
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "teleop", "--config", str(config)],
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        base = f"http://127.0.0.1:{port}"
        async with aiohttp.ClientSession() as session:
            await _wait_ready(session, base, proc)
            monitor = await session.ws_connect(f"{base}/ws")
            session_id = (await receive_type(monitor, "hello"))["session_id"]
            await monitor.send_json(
                {
                    "version": 1,
                    "type": "record",
                    "session_id": session_id,
                    "action": "start",
                    "task": "shutdown test",
                }
            )
            await asyncio.sleep(1.0)

            started = time.monotonic()
            for _ in range(presses):
                os.killpg(proc.pid, signal.SIGINT)
                await asyncio.sleep(0.05)
            while proc.poll() is None and time.monotonic() - started < 10.0:
                await asyncio.sleep(0.05)
            elapsed = time.monotonic() - started
            await monitor.close()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

    log = log_path.read_text(encoding="utf-8", errors="replace")
    assert proc.returncode == 0, log
    # Well inside compose.yaml's stop_grace_period (15 s).
    assert elapsed < 5.0, log
    assert "robot worker shutdown" in log and "graceful=True" in log, log

    root = tmp_path / "recordings"
    assert sorted(path.name for path in root.iterdir()) == ["episode_000000"]
    meta = json.loads((root / "episode_000000" / "meta.json").read_text())
    assert meta["outcome"] == "interrupted"
    assert meta["task"] == "shutdown test"
