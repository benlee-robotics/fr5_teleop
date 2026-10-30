# syntax=docker/dockerfile:1
#
# Targets:
#   docker build --target test .   # pytest + server smoke inside the image
#   docker build .                 # runtime image (default, last stage)
#
# The app runs from /app without installing the package: PROJECT_ROOT in
# teleop/config.py resolves web/, certs/ and config.yaml relative to the
# source tree, so a site-packages install would break those defaults.

FROM python:3.10-slim-bookworm AS deps

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# pyrealsense2 links against the system libusb (the wheel does not bundle it).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lock.txt requirements-camera.txt /tmp/req/
RUN python -m pip install -r /tmp/req/requirements-lock.txt \
        -r /tmp/req/requirements-camera.txt \
    && rm -rf /tmp/req

WORKDIR /app


FROM deps AS test

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-dev.txt /tmp/req/
RUN python -m pip install -r /tmp/req/requirements-dev.txt && rm -rf /tmp/req

COPY . /app
RUN python -m pytest -p no:cacheprovider \
    && bash tests/smoke/run_server_smoke.sh


FROM deps AS runtime

ARG TELEOP_UID=1000
ARG TELEOP_GID=1000
RUN groupadd --gid "${TELEOP_GID}" teleop \
    && useradd --uid "${TELEOP_UID}" --gid "${TELEOP_GID}" --no-create-home \
        --shell /usr/sbin/nologin teleop

COPY teleop/ /app/teleop/
COPY web/ /app/web/
COPY config.yaml config_bimanual.yaml /app/
COPY fairino-python-sdk-2.2.8_robot_v3.9.8/linux/fairino/Robot.py \
    /app/fairino-python-sdk-2.2.8_robot_v3.9.8/linux/fairino/Robot.py
RUN mkdir -p /app/certs /app/recordings \
    && chown teleop:teleop /app/recordings

USER teleop

LABEL org.opencontainers.image.source="https://github.com/benlee-robotics/fr5_teleop" \
      org.opencontainers.image.description="Meta Quest WebXR to Fairino FR5 teleoperation bridge"

# Liveness only; assumes the default port 8443 with TLS.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import ssl, urllib.request; urllib.request.urlopen('https://127.0.0.1:8443/health/live', context=ssl._create_unverified_context(), timeout=2)"]

ENTRYPOINT ["python", "-m", "teleop"]
CMD ["--config", "/app/config.yaml"]
