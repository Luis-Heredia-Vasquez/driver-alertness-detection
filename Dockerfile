# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build for Driver Alertness Detection API
#
# Stage 1 (builder)  – install Python packages with build tools available
# Stage 2 (runtime)  – minimal runtime image; no compiler, non-root user
# ─────────────────────────────────────────────────────────────────────────────

# ─── Stage 1: dependency builder ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages required to compile opencv-python and mediapipe wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into an isolated prefix so only this prefix is copied to runtime
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Stage 2: runtime image ──────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Minimal runtime system libraries (OpenCV needs libGL / libGLib at run time)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root system user — no home directory, no shell
RUN groupadd --system appgroup \
    && useradd  --system \
                --gid appgroup \
                --no-create-home \
                --shell /usr/sbin/nologin \
                appuser

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source (chowned to appuser at build time)
COPY --chown=appuser:appgroup . .

# Create output directories that the app writes to at runtime
RUN mkdir -p outputs/models outputs/plots outputs/logs \
    && chown -R appuser:appgroup outputs/

# Drop to non-root for all subsequent commands
USER appuser

EXPOSE 7860

# Docker health check — polls the liveness endpoint every 30 s
HEALTHCHECK \
    --interval=30s \
    --timeout=10s  \
    --start-period=15s \
    --retries=3 \
    CMD curl -fsS http://localhost:7860/health || exit 1

# gunicorn: 2 worker processes × 2 threads; logs to stdout/stderr for `docker logs`
CMD ["gunicorn", \
     "--bind",            "0.0.0.0:7860", \
     "--workers",         "2",            \
     "--threads",         "2",            \
     "--timeout",         "60",           \
     "--access-logfile",  "-",            \
     "--error-logfile",   "-",            \
     "src.api.app:app"]
