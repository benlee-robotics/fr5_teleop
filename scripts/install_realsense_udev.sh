#!/usr/bin/env bash
# Install the librealsense udev rules on the HOST (not inside the container).
# The container runs as UID 1000 and opens the D435i through the host /dev,
# so the camera's USB and /dev/video* nodes must be accessible (mode 0666).
# Usage: sudo bash scripts/install_realsense_udev.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULES_SRC="${SCRIPT_DIR}/realsense/99-realsense-libusb.rules"
RULES_DST="/etc/udev/rules.d/99-realsense-libusb.rules"

if [[ "${EUID}" -ne 0 ]]; then
    echo "Run with sudo: sudo bash $0" >&2
    exit 1
fi
if [[ ! -f "${RULES_SRC}" ]]; then
    echo "Rules file not found: ${RULES_SRC}" >&2
    exit 1
fi

install -m 0644 "${RULES_SRC}" "${RULES_DST}"
udevadm control --reload-rules
udevadm trigger
echo "Installed ${RULES_DST}. Unplug and replug the camera, then check:"
echo "  lsusb -d 8086:0b3a      # D435i on the USB bus"
echo "  ls -l /dev/video*       # crw-rw-rw- for the RealSense nodes"
