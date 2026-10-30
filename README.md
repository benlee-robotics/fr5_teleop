# VR Teleop Bridge for Fairino FR5 — User Guide

This guide covers running the teleop system on a field PC: you control a Fairino
FR5 arm with a Meta Quest headset and record demonstrations. It covers usage
only. Development and release are handled separately.

The program is distributed as a Docker image (`ghcr.io/benlee-robotics/fr5_teleop`).
You do not need to install Python, the Fairino SDK, or any other dependency on
the field PC.

All examples below use version **v0.4.3** (image tag `0.4.3`).

> Note: the Quest web page and the `/monitor` page are in Korean. Button
> labels are shown below in Korean with their English meanings.

---

## Field PC setup

The field PC needs two things:

| What | Contents | How to get it |
| --- | --- | --- |
| The program | Python, dependencies, code, web UI, Fairino SDK | Docker image, pulled from GHCR with `docker compose pull` |
| Run files | `compose.yaml`, `config.yaml`, `config_bimanual.yaml`, `scripts/make_cert.sh`. For the RealSense cameras you also need `scripts/install_realsense_udev.sh` and `scripts/realsense/99-realsense-libusb.rules` | **Option A** (git clone) or **Option B** (copy files) below |

**Always use the same version for the run files and the image.** The program
rejects unknown config keys, so a newer `config.yaml` with an older image can
fail at startup.

### One-time preparation

1. Install Docker Engine (from Docker's Ubuntu apt repository):

   ```bash
   sudo apt-get update
   sudo apt-get install -y ca-certificates curl
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
   https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
     | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
   ```

2. Add your account to the `docker` group, then **log out and log back in**.
   Terminals that were already open (including VS Code) will still get
   `permission denied` until you log in again.

   ```bash
   sudo usermod -aG docker "$USER"
   # after logging in again:
   docker run --rm hello-world
   id -u        # must be 1000 (the container runs as UID 1000)
   ```

3. The image is public, so `docker login` is not needed.

### Option A: git clone (recommended)

```bash
sudo apt-get install -y git          # only if git is missing
git clone https://github.com/benlee-robotics/fr5_teleop.git ~/fr5_teleop
cd ~/fr5_teleop
git checkout v0.4.3                  # match the image tag 0.4.3
```

- The repository also contains a `Dockerfile`. Do not build on the field PC;
  pull the image instead (see below).
- After you edit `config.yaml`, `git status` shows it as modified. That is
  expected.

### Option B: copy only the needed files

Use this if git is not available. Only the directory layout matters.

```bash
mkdir -p ~/fr5_teleop/scripts/realsense
cd ~/fr5_teleop
V=v0.4.3
for f in compose.yaml config.yaml config_bimanual.yaml scripts/make_cert.sh \
         scripts/install_realsense_udev.sh scripts/realsense/99-realsense-libusb.rules; do
    curl -fsSL -o "$f" "https://raw.githubusercontent.com/benlee-robotics/fr5_teleop/$V/$f"
done
```

You can also copy the same files by scp or USB from another PC. Keep
`make_cert.sh` under `scripts/`. Always run it as `bash scripts/make_cert.sh`,
because copying can drop the execute permission.

### Finish setup (both options)

Run these in `~/fr5_teleop`.

1. Create the recordings folder. If it is missing, Docker creates it as root
   and the container cannot write to it.

   ```bash
   mkdir -p recordings
   ```

2. Create the TLS certificate (`certs/cert.pem`, `certs/key.pem`):

   ```bash
   ip -4 -br addr                                   # find the PC address the Quest will use
   TELEOP_HOST_IP=192.168.1.50 bash scripts/make_cert.sh
   ```

   - `TELEOP_HOST_IP` is **the PC address that the Quest connects to**, not
     the robot LAN address. Always set it on PCs with more than one network.
   - If a certificate already exists, the script keeps it and only warns when
     the address differs. If the PC address changed or the certificate is
     older than one year, delete `certs/*.pem` and run the script again.
   - The certificate is self-signed. The first time you open the page in the
     Quest browser, accept the security warning, or WebXR will not work. You
     must accept it again after you create a new certificate.

3. Pin the image version and pull it:

   ```bash
   echo "TELEOP_TAG=0.4.3" > .env
   docker compose pull
   ```

4. Edit `config.yaml` for your robot (`robot.ip`, `motion.workspace`,
   `motion.position_scale`, ...). See [Configuration](#configuration). For
   two arms, edit `config_bimanual.yaml`.

5. Check with a dry run (no robot connection):

   ```bash
   docker compose run --rm teleop
   ```

   - The log shows `starting teleop mode=DRY-RUN url=https://0.0.0.0:8443`.
   - After about 20 seconds, `docker ps` in another terminal shows `(healthy)`.
   - `https://<PC address>:8443` opens in the Quest browser, and
     `https://<PC address>:8443/monitor` opens in a PC browser.
   - Stop with Ctrl+C.

The finished directory looks like this (Option A also contains the rest of the
repository):

```text
~/fr5_teleop/
├── compose.yaml
├── config.yaml            # edit on site
├── config_bimanual.yaml   # edit on site
├── .env                   # TELEOP_TAG=0.4.3
├── scripts/make_cert.sh
├── scripts/install_realsense_udev.sh, realsense/99-realsense-libusb.rules   # for cameras
├── certs/cert.pem, key.pem
└── recordings/            # recorded episodes (stay on the host)
```

---

## Running

Start the program yourself for each session and stop it with Ctrl+C. The first
Ctrl+C stops the robot worker and finalizes any recording in order. This can
take up to about 12 seconds, and further Ctrl+C presses are ignored during
that time. Use `docker kill` only if shutdown hangs.

```bash
docker compose run --rm teleop                                   # dry run (no robot)
docker compose run --rm teleop --confirm-hardware                # single real robot
docker compose run --rm teleop --config /app/config_bimanual.yaml --confirm-hardware   # two arms
```

- Real robot mode always needs `--confirm-hardware`. Hardware mode never
  starts automatically.
- The config files are mounted read-only into the container. To change
  settings, edit `config.yaml` on the host and run the program again.

**Check before running on the real robot:**

- The physical E-stop is within reach.
- The scale is low and the workspace is conservative.
- The area around the robot is clear, and an observer is present.

### Typical session

1. Run the program on the PC.
2. On the PC, open `https://<PC address>:8443/monitor`. This page only
   observes and does not take control.
3. In the Quest browser, open `https://<PC address>:8443` and press
   **🥽 VR 세션 시작** (Start VR session).
4. Hold the grip and move the controller to move the robot (see
   [Quest controller operation](#quest-controller-operation)).
5. When finished, exit VR and press Ctrl+C on the PC.

---

## Quest controller operation

In single-arm mode, the **right** controller moves the robot. In two-arm mode,
each controller moves the arm on its side.

| Input | Hand | Action |
| --- | --- | --- |
| **Grip** (side button), hold | control hand | Dead-man switch. While held, the robot follows the controller's motion (position, and rotation if `orientation.enabled`). Release to stop. |
| **Trigger** (index finger) | control hand | Opens/closes the gripper. Only works while the grip is **released** (only when `gripper.enabled: true`). |
| **X** | left | Return to the home pose. Press again during the return to stop it. Only works while the grip is released. |
| **A** | right | Recording: start / save as success |
| **B** short press | right | Recording: save as failure (fires on release) |
| **B** hold ≥ 1 s | right | Recording: discard the episode |

### Grip (dead-man switch)

- Each time you press the grip, the current controller position and current
  robot pose become the new starting point. You can therefore release the
  grip, move your hand to a comfortable position, and press it again to
  continue ("clutching").
- Releasing the grip stops the robot (state `SLEEPING`), but the VR session
  stays connected. Press the grip again to continue.
- If the pose input stops for too long (e.g. tracking loss), the robot stops.
  The Quest page then shows "재무장 대기" (waiting to re-arm). **Fully release
  the grip and press it again** to continue.
- Exiting VR or losing the connection releases control (state `IDLE`).

### Trigger (gripper)

- The controller box returns an error if a gripper command arrives while the
  robot is following the controller. Therefore **release the grip first, then
  pull the trigger.** A trigger pull while the grip is held is ignored.
- After the gripper finishes, the robot does not resume by itself. Press the
  grip again to continue.

### X (home return)

- The robot moves smoothly from its current pose to the saved home pose
  (`home.tcp`). The gripper does not move.
- During the return (state `HOMING`), grip and trigger are ignored. When it
  finishes, release and press the grip again to resume teleop.
- The return is refused if the robot is currently following the controller,
  if the current joints are too far from the home joints, or if the current
  TCP is outside the workspace. A refused or blocked press gives a strong
  vibration.
- You can also use **🏠 홈으로** (Go home) and **■ 중단** (Stop) on
  `/monitor`. "Go home" works there only while the Quest does not hold
  control. "Stop" works from either screen.

### A / B (recording)

Type the task on `/monitor` before you put on the headset (see
[Demonstration recording](#demonstration-recording)). The A button always
starts an episode with that task.

| Button | Recorder state | Action | Vibration | Status light (top of view) |
| --- | --- | --- | --- | --- |
| A | idle | Start recording | 1 short | Red, blinking while recording |
| A | recording | Save as **success** | 2 short | Green 1.5 s |
| B short (on release) | recording | Save as **failure** | 1 long | Orange 1.5 s |
| B hold ≥ 1 s | recording | **Discard** | 3 short | Gray 1.5 s |
| anything else | recording off, finalizing, disconnected, ... | Refused | Strong 0.5 s | Purple, blinking |

- To record many episodes with the same task, repeat: A (start) → demonstrate
  → A (save).
- Grip and trigger work normally during recording. Recording never moves the
  robot.
- If you press A by mistake during a recording, the episode up to that point
  is saved as success. Note it, and exclude it later if needed.

---

## Cameras (wrist D435i + top-view D435)

The Docker image already contains the RealSense libraries. The container uses
the cameras on the host directly.

1. Install the udev rules on the host and re-plug the cameras (once):

   ```bash
   sudo bash scripts/install_realsense_udev.sh
   lsusb -d 8086:0b3a        # D435i (wrist) is connected
   lsusb -d 8086:0b07        # D435 (top view) is connected
   lsusb -t                  # both cameras at 5000M (USB 3) or faster
   ls -l /dev/video*         # RealSense nodes show crw-rw-rw-
   ```

   Always use **USB 3 ports**. If you enable depth, connect the two cameras to
   different USB controllers if possible.

2. Check that the container can see the cameras:

   ```bash
   docker compose run --rm --entrypoint python teleop -c \
     "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) + ' ' + d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()])"
   ```

   It should print both cameras, e.g.
   `['Intel RealSense D435I <serial>', 'Intel RealSense D435 <serial>']`.
   If the list is empty, go back to step 1.

3. In `config.yaml` under `recording.cameras`, put the D435i serial in `wrist`
   and the D435 serial in `top` (in quotes), set `enabled: true`, and run the
   program again:

   ```yaml
   recording:
     cameras:
       - name: wrist          # dataset key observation.images.wrist
         enabled: true
         source: realsense    # "fake" = synthetic frames, for dry-run tests
         serial: "123456789012"   # required when more than one camera is enabled
         width: 640
         height: 480
         fps: 30
         depth: false         # true: also save 16-bit depth aligned to color
       - name: top
         enabled: true
         source: realsense
         serial: "234567890123"
         width: 640
         height: 480
         fps: 30
         depth: false
   ```

4. On `/monitor`, `Camera wrist` and `Camera top` should show `30.0 fps` in
   green. Then you can record.

If any enabled camera is not streaming, recording refuses to start and
`/monitor` shows the reason. The camera process tries to reconnect every 2
seconds. Robot teleop itself works with or without cameras.

---

## Architecture

```text
Quest browser (WebXR page)
    │  HTTPS / WebSocket (port 8443)
    ▼
Teleop server (Python, in Docker)
    ├─ Robot worker process   → Fairino SDK → FR5 controller (8 ms servo)
    ├─ Recorder process       → recordings/
    └─ Camera process × N     → RealSense D435i / D435 (USB 3)

PC browser → https://<PC address>:8443/monitor   (status, recording, home)
```

- The Quest sends the controller pose and buttons to the server. The robot
  worker turns them into Cartesian servo commands for the FR5.
- The robot, recorder, and each camera run in separate processes, so a
  recording or camera failure does not affect robot control.
- The container uses the host network directly (robot TCP 20003/20005,
  UDP 20007, Quest 8443).

Useful URLs:

```text
https://<PC address>:8443           Quest control page
https://<PC address>:8443/monitor   Operator monitor (observe only)
https://<PC address>:8443/health/live, /health/ready
```

---

## Configuration

Settings are in `config.yaml` (single arm) and `config_bimanual.yaml` (two
arms). The program rejects unknown keys and wrong types at startup. After
editing, run the program again.

The settings you are most likely to change:

| Key | Meaning |
| --- | --- |
| `robot.ip` | Robot controller IP (two arms: `robot.arms.left.ip`, `robot.arms.right.ip`) |
| `motion.position_scale` | Controller-to-robot motion scale. Lower = robot moves less per hand movement |
| `motion.max_velocity_mm_s` | Robot TCP speed limit |
| `motion.max_step_mm` | Maximum TCP step per 8 ms servo tick |
| `motion.workspace` | Allowed TCP box in mm (`x`, `y`, `z` as `[min, max]`); commands are clipped to it |
| `orientation.enabled` | Also follow controller rotation (roll/pitch/yaw) |
| `orientation.scale`, `max_angular_velocity_deg_s`, `max_step_deg`, `max_deviation_deg` | Rotation scale and safety limits |
| `gripper.enabled` | Enable the trigger → gripper. Off by default; enable it only after the gripper is set up on the controller and tested |
| `home.*` | Home pose and home-return speed (see below) |
| `recording.enabled`, `recording.root_dir` | Demonstration recording on/off and output folder |
| `recording.cameras` | Camera list (see [Cameras](#cameras-wrist-d435i--top-view-d435)) |

Example (excerpt):

```yaml
robot:
  ip: 192.168.58.2
motion:
  position_scale: 1500.0
  max_velocity_mm_s: 250.0
  max_step_mm: 5.0
  workspace:
    x: [-100.0, 800.0]
    y: [-600.0, 600.0]
    z: [-500.0, 700.0]
gripper:
  enabled: false
orientation:
  enabled: true
home:
  enabled: true
  tcp: [393.416, -122.637, 503.735, -179.441, 1.038, -3.243]   # mm, deg
  joints_deg: [-3.138, -77.234, -97.088, -96.715, 90.561, -89.895]
  max_velocity_mm_s: 100.0          # <= motion.max_velocity_mm_s
  max_angular_velocity_deg_s: 30.0
recording:
  enabled: true
  root_dir: recordings
```

**Changing the home pose:** move the robot to the pose you want, press
**현재 자세 → home YAML** (Current pose → home YAML) on `/monitor`, and paste
the result into `config.yaml`. With two arms, put it under
`robot.arms.<left|right>.home`. The home button returns every arm that has a
home pose.

Do not change the `timing`, `server`, `tls`, or `sdk_path` settings unless
you are asked to.

---

## Demonstration recording

When `recording.enabled: true`, the program saves demonstrations as raw
episodes in `recordings/`. Recording only observes. It never sends robot
commands, and a recording failure does not affect robot control.

### From the monitor (PC)

On `/monitor`, in the **Demonstration recording** card:

1. Type the language instruction (task) in the text box, e.g.
   `pick up the red cube and place it in the box`.
2. Press **● 녹화 시작** (Start recording) and demonstrate with the Quest.
3. End the episode with one of these buttons:
   - **✔ 성공 저장** (Save as success)
   - **✘ 실패 저장** (Save as failure)
   - **폐기** (Discard)

### From the headset (working alone)

1. Before you put on the headset, type the task on `/monitor`. It is sent to
   the server after 0.5 s. When `Next task (Quest A)` turns green, the task
   is set. The Quest start page also shows the next task.
2. In VR, use the right-hand A/B buttons (see
   [A / B (recording)](#a--b-recording)).

### Notes

- Periods with the grip released (`SLEEPING`) are recorded too.
- If you press Ctrl+C during a recording, wait for shutdown to finish (up to
  about 12 seconds) so the episode is saved completely.
- If the server stops abnormally, the unfinished episode is renamed to
  `.incomplete` at the next start.

### Output

```text
recordings/episode_000000/
  meta.json                  task, outcome (success/failure), complete flag, timing, config
  robot_right.bin            one sample per 8 ms servo tick (joints, TCP, commands, Quest input, gripper)
  events.jsonl               start/stop/worker-restart events
  camera_<name>.mp4          RGB video, H.264 (per enabled camera)
  camera_<name>_frames.bin   per-frame timestamps (same clock as the robot samples)
  camera_<name>_depth.mkv    16-bit depth, only when depth: true
```

`"complete": false` in `meta.json` means something was missing (dropped
samples or camera frames, a worker restart, or a missing arm). Tell the
maintainer which episodes are affected.
