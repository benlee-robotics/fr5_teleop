#!/usr/bin/env bash
# Create the self-signed TLS certificate Quest WebXR needs (HTTPS).
# Usage: [TELEOP_HOST_IP=192.168.1.50] scripts/make_cert.sh [cert_dir]
# cert_dir defaults to ./certs next to this script's parent directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="${1:-$(dirname "${SCRIPT_DIR}")/certs}"
LOCAL_IP="${TELEOP_HOST_IP:-$(hostname -I | awk '{print $1}')}"

mkdir -p "${CERT_DIR}"
if [[ ! -f "${CERT_DIR}/cert.pem" || ! -f "${CERT_DIR}/key.pem" ]]; then
    if [[ -z "${LOCAL_IP}" ]]; then
        echo "Unable to determine a local IPv4 address." >&2
        exit 1
    fi
    openssl req -x509 -newkey rsa:2048 \
        -keyout "${CERT_DIR}/key.pem" \
        -out "${CERT_DIR}/cert.pem" \
        -days 365 -nodes \
        -subj "/CN=${LOCAL_IP}" \
        -addext "subjectAltName=IP:${LOCAL_IP},IP:127.0.0.1"
    # The container user is UID 1000, so the host owner (also 1000) suffices.
    chmod 600 "${CERT_DIR}/key.pem"
    echo "Created ${CERT_DIR}/cert.pem for ${LOCAL_IP}"
elif ! openssl x509 -in "${CERT_DIR}/cert.pem" -noout -ext subjectAltName \
    | grep -Fq "IP Address:${LOCAL_IP}"; then
    echo "WARNING: existing certificate does not cover ${LOCAL_IP}." >&2
    echo "Remove ${CERT_DIR}/cert.pem and key.pem to regenerate it for this host." >&2
fi
