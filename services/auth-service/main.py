import logging
import random
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest

# ==============================
# LOGGING
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("auth-service")

# ==============================
# METRICS
# ==============================

REQUEST_COUNT = Counter("auth_requests_total", "Total auth requests")
REQUEST_LATENCY = Histogram("auth_request_latency_seconds", "Auth request latency")

# ==============================
# STATE
# ==============================

LATENCY_MODE: bool = False
CRASH_MODE: bool = False
state_lock = Lock()

# ==============================
# LIFESPAN
# ==============================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Handle application startup and graceful shutdown."""
    logger.info("Auth service starting up.")
    yield
    logger.info("Auth service shutting down gracefully.")


app = FastAPI(title="Auth Service", version="1.0.0", lifespan=lifespan)

# ==============================
# ENDPOINTS
# ==============================

@app.get("/login")
def login() -> dict:
    """
    Simulate a login request.

    Behaviour is controlled by runtime toggle endpoints:
    - LATENCY_MODE: injects a 6-10s sleep to simulate a degraded service.
    - CRASH_MODE: raises HTTP 503 to simulate a crashed service (allows the
      orchestrator to detect and restart the container).
    """
    start = time.time()

    with state_lock:
        crash = CRASH_MODE
        latency = LATENCY_MODE

    if crash:
        logger.warning("Crash mode active — returning 503.")
        raise HTTPException(status_code=503, detail="Service crash simulated")

    if latency:
        delay = random.uniform(6, 10)
        logger.debug("Latency mode active — sleeping %.2fs.", delay)
        time.sleep(delay)

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(time.time() - start)

    return {"status": "logged in"}


@app.post("/toggle-latency")
def toggle_latency(enabled: bool = Query(...)) -> dict:
    """Enable or disable artificial latency on the /login endpoint."""
    global LATENCY_MODE
    with state_lock:
        LATENCY_MODE = enabled
    logger.info("Latency mode set to %s.", enabled)
    return {"latency_mode": LATENCY_MODE}


@app.post("/toggle-crash")
def toggle_crash(enabled: bool = Query(...)) -> dict:
    """
    Enable or disable crash simulation on the /login endpoint.

    When enabled, /login returns HTTP 503 instead of exiting the process,
    allowing the container to remain running and restartable by the orchestrator.
    """
    global CRASH_MODE
    with state_lock:
        CRASH_MODE = enabled
    logger.info("Crash mode set to %s.", enabled)
    return {"crash_mode": CRASH_MODE}


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type="text/plain")
