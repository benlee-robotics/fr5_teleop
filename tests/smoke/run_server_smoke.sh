#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PORT="${TELEOP_SMOKE_PORT:-18443}"
LOG_FILE="${TELEOP_SMOKE_LOG:-/var/tmp/vr-teleop-server-smoke.log}"
READY_FILE="/var/tmp/vr-teleop-ready-$$.json"
if [[ -n "${TELEOP_PYTHON:-}" ]]; then
    PYTHON_BIN="${TELEOP_PYTHON}"
elif [[ -x "${PROJECT_DIR}/.venv-linux/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/.venv-linux/bin/python"
else
    PYTHON_BIN="python3"
fi
CONFIG_ARGS=()
if [[ -n "${TELEOP_CONFIG:-}" ]]; then
    CONFIG_ARGS=(--config "${TELEOP_CONFIG}")
fi

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -m teleop "${CONFIG_ARGS[@]}" \
    --no-tls --host 127.0.0.1 --port "${PORT}" >"${LOG_FILE}" 2>&1 &
SERVER_PID=$!

cleanup() {
    rm -f "${READY_FILE}"
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}"
        wait "${SERVER_PID}" || true
    fi
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 100); do
    if curl --fail --silent --output "${READY_FILE}" \
        "http://127.0.0.1:${PORT}/health/ready" \
        && "${PYTHON_BIN}" -c 'import json,sys; raise SystemExit(not json.load(open(sys.argv[1], encoding="utf-8"))["ready"])' "${READY_FILE}"
    then
        READY=1
        break
    fi
    sleep 0.05
done

if [[ "${READY}" != "1" ]]; then
    echo "Server did not become ready. Log follows:" >&2
    sed -n '1,200p' "${LOG_FILE}" >&2
    exit 1
fi

curl --fail --silent "http://127.0.0.1:${PORT}/health/live" \
    | "${PYTHON_BIN}" -c 'import json,sys; raise SystemExit(not json.load(sys.stdin)["live"])'
curl --fail --silent "http://127.0.0.1:${PORT}/" \
    | grep --quiet "VR Teleop"

kill -TERM "${SERVER_PID}"
wait "${SERVER_PID}"
rm -f "${READY_FILE}"
trap - EXIT

grep --quiet "graceful=True" "${LOG_FILE}"
echo "server smoke: OK"
