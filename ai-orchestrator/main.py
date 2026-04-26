import logging
import os
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import docker
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==============================
# LOGGING
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ai-orchestrator")

# ==============================
# CONFIG (from environment variables)
# ==============================

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "5"))
TRAINING_DURATION_SECONDS = int(os.getenv("TRAINING_DURATION_SECONDS", "120"))
CONTAMINATION = float(os.getenv("CONTAMINATION", "0.1"))
STABILIZATION_SECONDS = int(os.getenv("STABILIZATION_SECONDS", "20"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "30"))
MAX_RESTARTS_PER_WINDOW = int(os.getenv("MAX_RESTARTS_PER_WINDOW", "3"))
WINDOW_RESET_SECONDS = int(os.getenv("WINDOW_RESET_SECONDS", "120"))
MAX_TRAINING_SAMPLES = int(os.getenv("MAX_TRAINING_SAMPLES", "1000"))
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", "200"))

AUTH_QUERY = """
sum(rate(auth_request_latency_seconds_sum[30s]))
/
sum(rate(auth_request_latency_seconds_count[30s]))
"""

ORDER_QUERY = """
sum(rate(order_request_latency_seconds_sum[30s]))
/
sum(rate(order_request_latency_seconds_count[30s]))
"""

# ==============================
# SERVICE DEPENDENCIES
# ==============================

SERVICE_DEPENDENCIES: dict = {
    "order-service": ["auth-service"],
    "auth-service": [],
}

# ==============================
# PROMETHEUS METRICS
# ==============================

healing_attempts_total = Counter("healing_attempts_total", "Total healing attempts")
healing_success_total = Counter("healing_success_total", "Successful healings")
healing_failures_total = Counter("healing_failures_total", "Escalation events")
healing_mttr_seconds = Histogram("healing_mttr_seconds", "Mean time to recovery")
active_incident = Gauge("active_incident", "1 if incident active")
incident_escalated = Gauge("incident_escalated", "1 if escalation occurred")
model_ready_metric = Gauge("model_ready", "1 if ML model trained")

# ==============================
# STATE + LOCK
# ==============================

_state_lock = threading.Lock()

try:
    docker_client = docker.from_env()
except Exception as e:
    logger.error("Failed to connect to Docker daemon: %s", e)
    docker_client = None

# Bounded deque prevents unbounded memory growth in long-running deployments
training_data: deque = deque(maxlen=MAX_TRAINING_SAMPLES)
model: Optional[IsolationForest] = None
scaler = StandardScaler()

model_ready: bool = False
training_start_time: datetime = datetime.now()

last_restart_time: Optional[datetime] = None
restart_count: int = 0
window_start_time: datetime = datetime.now()
stabilizing_until: Optional[datetime] = None

incident_active: bool = False
escalated: bool = False
incident_start_time: Optional[datetime] = None

# ==============================
# FASTAPI
# ==============================

app = FastAPI(title="AI Orchestrator", version="1.0.0")


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type="text/plain")


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    with _state_lock:
        ready = model_ready
    return {"status": "healthy", "model_ready": ready}


# ==============================
# PROMETHEUS METRIC FETCH
# ==============================

def get_metric(query: str) -> Optional[float]:
    """
    Query Prometheus for a single scalar metric value.

    Args:
        query: PromQL query string.

    Returns:
        Float value if successful, None otherwise.
    """
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            logger.warning("Prometheus query returned non-success status: %s", data.get("status"))
            return None

        results = data["data"]["result"]
        if not results:
            return None

        value = float(results[0]["value"][1])
        # Prometheus returns "NaN" for 0/0 queries (e.g. no traffic yet).
        # math.isfinite rejects both NaN and Inf so they become None → 0.0.
        import math
        if not math.isfinite(value):
            return None
        return value

    except requests.RequestException as e:
        logger.warning("Prometheus request failed: %s", e)
        return None
    except (KeyError, ValueError, IndexError) as e:
        logger.warning("Failed to parse Prometheus response: %s", e)
        return None


# ==============================
# ROOT CAUSE LOGIC
# ==============================

# How many standard deviations above the training mean before a service
# is considered "elevated". Tune via environment variable if needed.
Z_THRESHOLD = float(os.getenv("Z_THRESHOLD", "2.0"))
# Z-score threshold for the secondary (bypass) anomaly detector.
# Lower than Z_THRESHOLD so it catches cases IsolationForest misses.
DIRECT_ANOMALY_THRESHOLD = float(os.getenv("DIRECT_ANOMALY_THRESHOLD", "3.0"))


def determine_root_cause(
    auth_latency: float,
    order_latency: float,
    fitted_scaler: StandardScaler,
) -> Optional[str]:
    """
    Determine the root cause service using the known dependency model:

        order-service → auth-service   (order calls auth on every request)
        auth-service  → (independent)

    Failure propagation rules:
      - auth slow/down  → both auth AND order latency rise (upstream failure)
      - order slow/down → only order latency rises

    Root cause decision:
      - auth elevated (alone or together with order) → restart auth-service first,
        because fixing auth will also fix order's elevated latency.
      - only order elevated → restart order-service (independent problem).
      - neither clearly elevated → return None (caller falls back to auth-service).

    "Elevated" is defined as more than Z_THRESHOLD standard deviations above
    the mean latency observed during the training baseline, which avoids any
    hard-coded latency thresholds.

    Args:
        auth_latency: Current auth service avg latency in seconds.
        order_latency: Current order service avg latency in seconds.
        fitted_scaler: The StandardScaler fitted on training data. Provides
                       per-feature mean and std for z-score calculation.

    Returns:
        Service name string, or None if root cause cannot be determined.
    """
    # Extract training baseline stats (safe against zero-std features)
    auth_mean  = fitted_scaler.mean_[0]
    order_mean = fitted_scaler.mean_[1]
    auth_std   = max(float(fitted_scaler.scale_[0]), 1e-6)
    order_std  = max(float(fitted_scaler.scale_[1]), 1e-6)

    auth_z  = (auth_latency  - auth_mean)  / auth_std
    order_z = (order_latency - order_mean) / order_std

    auth_elevated  = auth_z  > Z_THRESHOLD
    order_elevated = order_z > Z_THRESHOLD

    logger.debug(
        "Z-scores — auth: %.2f (elevated=%s)  order: %.2f (elevated=%s)",
        auth_z, auth_elevated, order_z, order_elevated,
    )

    if auth_elevated:
        # Auth is the root cause regardless of whether order is also elevated.
        # Restarting auth will resolve both: this is upstream failure propagation.
        return "auth-service"

    if order_elevated:
        # Only order is elevated — order-service has an independent problem.
        return "order-service"

    # Neither service is clearly elevated above baseline.
    return None


# ==============================
# RESTART LOGIC
# ==============================

def restart_service(service_name: str) -> None:
    """
    Attempt to restart a named Docker container.

    Respects cooldown period, escalation state, and restart window limits.
    All shared state mutations are performed under _state_lock.

    Args:
        service_name: The name (or partial name) of the Docker container to restart.
    """
    global last_restart_time, restart_count, window_start_time
    global stabilizing_until, escalated, incident_active, incident_start_time

    if docker_client is None:
        logger.error("Docker client unavailable — cannot restart %s", service_name)
        return

    with _state_lock:
        if escalated:
            return

        now = datetime.now()

        # Reset restart counter if the tracking window has elapsed
        if (now - window_start_time).total_seconds() > WINDOW_RESET_SECONDS:
            restart_count = 0
            window_start_time = now

        if restart_count >= MAX_RESTARTS_PER_WINDOW:
            incident_escalated.set(1)
            healing_failures_total.inc()
            escalated = True
            logger.warning(
                "Escalation triggered — max restarts (%d) exceeded in window",
                MAX_RESTARTS_PER_WINDOW,
            )
            return

        if last_restart_time and (now - last_restart_time).total_seconds() < COOLDOWN_SECONDS:
            logger.debug("Skipping restart — still in cooldown period")
            return

        healing_attempts_total.inc()
        active_incident.set(1)

        if not incident_active:
            incident_active = True
            incident_start_time = now

        try:
            containers = docker_client.containers.list(all=True)
        except Exception as e:
            logger.error("Failed to list Docker containers: %s", e)
            return

        for container in containers:
            if service_name in container.name:
                try:
                    logger.info("Restarting container: %s", container.name)
                    container.restart()
                    restart_count += 1
                    last_restart_time = now
                    stabilizing_until = now + timedelta(seconds=STABILIZATION_SECONDS)
                    logger.info(
                        "Restart #%d complete for %s. Stabilizing for %ds.",
                        restart_count,
                        service_name,
                        STABILIZATION_SECONDS,
                    )
                except Exception as e:
                    logger.error("Failed to restart container %s: %s", container.name, e)
                return

        logger.warning("No container found matching service name: %s", service_name)


# ==============================
# MONITORING LOOP
# ==============================

def monitoring_loop() -> None:
    """
    Main monitoring loop running in a background thread.

    Phases:
      1. Training: collect baseline latency samples for TRAINING_DURATION_SECONDS.
      2. Detection: run Isolation Forest on each new sample and trigger recovery.
      3. Recovery: record MTTR when the system returns to normal prediction.
    """
    global model, model_ready, training_start_time
    global incident_active, escalated, restart_count

    logger.info("AI Orchestrator started. Training phase begins (%ds).", TRAINING_DURATION_SECONDS)

    while True:
        try:
            with _state_lock:
                now = datetime.now()
                is_stabilizing = bool(stabilizing_until and now < stabilizing_until)
                is_model_ready = model_ready

            if is_stabilizing:
                time.sleep(CHECK_INTERVAL)
                continue

            _auth_raw = get_metric(AUTH_QUERY)
            _order_raw = get_metric(ORDER_QUERY)
            auth_latency = _auth_raw if _auth_raw is not None else 0.0
            order_latency = _order_raw if _order_raw is not None else 0.0
            sample = np.array([auth_latency, order_latency])

            # ================= TRAINING =================

            if not is_model_ready:
                with _state_lock:
                    if np.isfinite(sample).all():
                        training_data.append(sample)
                    else:
                        logger.debug("Skipping non-finite training sample: %s", sample)
                    elapsed = (now - training_start_time).total_seconds()

                if elapsed >= TRAINING_DURATION_SECONDS:
                    with _state_lock:
                        snapshot = list(training_data)

                    if len(snapshot) < 10:
                        logger.warning(
                            "Only %d valid training samples collected — waiting for traffic before training.",
                            len(snapshot),
                        )
                        with _state_lock:
                            training_start_time = datetime.now()
                        time.sleep(CHECK_INTERVAL)
                        continue

                    with _state_lock:
                        X = np.array(snapshot)

                    X_scaled = scaler.fit_transform(X)
                    new_model = IsolationForest(
                        n_estimators=N_ESTIMATORS,
                        contamination=CONTAMINATION,
                        random_state=42,
                    )
                    new_model.fit(X_scaled)

                    with _state_lock:
                        model = new_model
                        model_ready = True

                    model_ready_metric.set(1)
                    logger.info("ML model trained on %d samples.", len(X))

                time.sleep(CHECK_INTERVAL)
                continue

            # ================= DETECTION =================

            with _state_lock:
                current_model = model
                current_scaler = scaler

            sample_scaled = current_scaler.transform([sample])
            prediction = current_model.predict(sample_scaled)[0]
            score = current_model.decision_function(sample_scaled)[0]

            # Compute z-scores for direct anomaly detection.
            # IsolationForest can miss single-dimension anomalies when one feature
            # was constant (e.g. all zeros) during training. Z-scores are computed
            # per-feature and provide a reliable fallback.
            _auth_scale = max(float(current_scaler.scale_[0]), 1e-6)
            _order_scale = max(float(current_scaler.scale_[1]), 1e-6)
            auth_z  = (auth_latency  - current_scaler.mean_[0]) / _auth_scale
            order_z = (order_latency - current_scaler.mean_[1]) / _order_scale

            # Primary detection: IsolationForest flags the combined sample.
            # Secondary detection: either z-score exceeds the direct threshold.
            # Using OR so that a single-service anomaly is never missed.
            zscore_anomaly = (auth_z > DIRECT_ANOMALY_THRESHOLD) or (order_z > DIRECT_ANOMALY_THRESHOLD)
            is_anomaly = (prediction == -1) or zscore_anomaly

            logger.info(
                "auth=%.4fs order=%.4fs | score=%.3f | auth_z=%.1f order_z=%.1f | %s%s",
                auth_latency, order_latency, score,
                auth_z, order_z,
                "ANOMALY" if is_anomaly else "normal",
                " (z-score trigger)" if zscore_anomaly and prediction != -1 else "",
            )

            with _state_lock:
                is_escalated = escalated

            # ── Auto-reset escalation when system normalises ──────────────
            # Fixes the case where escalation state leaks between test runs:
            # if latency is toggled off manually (no container restart), the
            # system returns to normal but escalated was never cleared.
            if not is_anomaly and is_escalated:
                with _state_lock:
                    escalated = False
                    restart_count = 0
                incident_escalated.set(0)
                logger.info("System normalised — escalation state reset automatically.")

            if is_anomaly and not is_escalated:
                logger.warning(
                    "Anomaly detected (IF score=%.3f, auth_z=%.1f, order_z=%.1f)",
                    score, auth_z, order_z,
                )
                root = determine_root_cause(auth_latency, order_latency, current_scaler)

                if root:
                    logger.info("Root cause inferred: %s", root)
                    restart_service(root)
                else:
                    # Neither z-score was high enough for a clear attribution,
                    # but the combined sample is anomalous. Default to auth-service
                    # since it is the upstream dependency.
                    logger.warning(
                        "Root cause unclear — defaulting to auth-service (upstream dependency)"
                    )
                    restart_service("auth-service")

            # ================= RECOVERY =================

            with _state_lock:
                is_incident = incident_active
                inc_start = incident_start_time

            if is_incident and not is_anomaly:
                duration = (datetime.now() - inc_start).total_seconds()
                healing_success_total.inc()
                healing_mttr_seconds.observe(duration)
                active_incident.set(0)
                incident_escalated.set(0)

                with _state_lock:
                    incident_active = False
                    escalated = False
                    restart_count = 0

                logger.info("System recovered in %.2fs.", duration)

        except Exception as e:
            logger.error("Unexpected error in monitoring loop: %s", e, exc_info=True)

        time.sleep(CHECK_INTERVAL)


# ==============================
# START BACKGROUND THREAD
# ==============================

_monitor_thread = threading.Thread(
    target=monitoring_loop,
    daemon=True,
    name="monitoring-loop",
)
_monitor_thread.start()
