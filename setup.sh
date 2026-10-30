#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv-linux"
CERT_DIR="${SCRIPT_DIR}/certs"

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This setup script targets Ubuntu 22.04." >&2
    exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 10))'; then
    echo "Python 3.10 is required for the bundled Fairino SDK." >&2
    exit 1
fi

echo "[1/4] Creating Python 3.10 virtual environment"
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements-dev.txt"
"${VENV_DIR}/bin/python" -m pip install -r "${SCRIPT_DIR}/requirements-camera.txt"
"${VENV_DIR}/bin/python" -m pip install --no-deps -e "${SCRIPT_DIR}"

echo "[2/4] Preparing local TLS certificate"
LOCAL_IP="${TELEOP_HOST_IP:-$(hostname -I | awk '{print $1}')}"
TELEOP_HOST_IP="${LOCAL_IP}" bash "${SCRIPT_DIR}/scripts/make_cert.sh" "${CERT_DIR}"

echo "[3/4] Running non-hardware tests"
"${VENV_DIR}/bin/python" -m pytest

echo "[4/4] Running dry-run server smoke test"
TELEOP_PYTHON="${VENV_DIR}/bin/python" \
    bash "${SCRIPT_DIR}/tests/smoke/run_server_smoke.sh"

echo
echo "Setup complete."
echo "Dry-run: ${VENV_DIR}/bin/python -m teleop"
echo "Config: ${SCRIPT_DIR}/config.yaml"
echo "Quest URL: https://${LOCAL_IP}:8443"
echo "Set TELEOP_HOST_IP before setup when Quest uses a different network interface."
echo "Hardware mode always requires --confirm-hardware, including when enabled in YAML."
