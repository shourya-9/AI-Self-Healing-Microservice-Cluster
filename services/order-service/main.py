import logging
import os
import random
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import AsyncGenerator

import requests
from fastapi import FastAPI, Query
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
logger = logging.getLogger("order-service")

# ==============================
# CONFIG
# ==============================

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://auth-service:8000")

# ==============================
# METRICS
# ==============================

REQUEST_COUNT = Counter("order_requests_total", "Total order requests")
REQUEST_LATENCY = Histogram("order_request_latency_seconds", "Order request latency")

# ==============================
# STATE
# ==============================

LATENCY_MODE: bool = False
state_lock = Lock()

# ==============================
# LIFESPAN
# ==============================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Handle application startup and graceful shutdown."""
    logger.info("Order service starting up.")
    yield
    logger.info("Order service shutting down gracefully.")


app = FastAPI(title="Order Service", version="1.0.0", lifespan=lifespan)

# ==============================
# ENDPOINTS
# ==============================

@app.post("/create-order")
def create_order() -> dict:
    """
    Simulate an order creation request.

    Calls auth-service as an upstream dependency. If auth is unavailable,
    the request continues anyway so that order-service metrics are still
    recorded — useful for testing partial degradation scenarios.
    Supports LATENCY_MODE toggle to simulate independent slowdowns.
    """
    start = time.time()

    try:
        requests.get(f"{AUTH_SERVICE_URL}/login", timeout=2)
    except requests.RequestException as e:
        logger.warning("Auth service call failed (continuing): %s", e)

    with state_lock:
        latency = LATENCY_MODE

    if latency:
        delay = random.uniform(1.5, 3.0)
        logger.debug("Latency mode active — sleeping %.2fs.", delay)
        time.sleep(delay)

    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(time.time() - start)

    return {"status": "order created"}


@app.post("/toggle-latency")
def toggle_latency(enabled: bool = Query(...)) -> dict:
    """Enable or disable artificial latency on the /create-order endpoint."""
    global LATENCY_MODE
    with state_lock:
        LATENCY_MODE = enabled
    logger.info("Latency mode set to %s.", enabled)
    return {"latency_mode": LATENCY_MODE}


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type="text/plain")
