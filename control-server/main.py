import logging
import os
import subprocess
import threading

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ==============================
# LOGGING
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("control-server")

# ==============================
# CONFIG
# ==============================

# Comma-separated list of allowed CORS origins.
# Override via environment variable for production deployments.
# "null" is required for browsers opening the dashboard as a local file:// URL
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://localhost:3000,http://127.0.0.1,null")
ALLOWED_ORIGINS: list = [o.strip() for o in _raw_origins.split(",") if o.strip()]

AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
ORDER_SERVICE_URL = os.getenv("ORDER_SERVICE_URL", "http://localhost:8002")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "5"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ==============================
# APP
# ==============================

app = FastAPI(title="Control Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ==============================
# STATE
# ==============================

_traffic_lock = threading.Lock()
order_traffic: bool = False
auth_traffic: bool = False


# ==============================
# TRAFFIC GENERATORS
# ==============================

def order_generator() -> None:
    """Continuously send POST requests to the order service while traffic is enabled."""
    logger.info("Order traffic generator started.")
    while True:
        with _traffic_lock:
            if not order_traffic:
                break
        try:
            requests.post(f"{ORDER_SERVICE_URL}/create-order", timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            logger.debug("Order request failed (expected during outages): %s", e)
    logger.info("Order traffic generator stopped.")


def auth_generator() -> None:
    """Continuously send GET requests to the auth service while traffic is enabled."""
    logger.info("Auth traffic generator started.")
    while True:
        with _traffic_lock:
            if not auth_traffic:
                break
        try:
            requests.get(f"{AUTH_SERVICE_URL}/login", timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            logger.debug("Auth request failed (expected during outages): %s", e)
    logger.info("Auth traffic generator stopped.")


# ==============================
# TRAFFIC CONTROL ENDPOINTS
# ==============================

@app.post("/start-order-traffic")
def start_order_traffic() -> dict:
    """Start the order traffic generator in a background thread."""
    global order_traffic
    with _traffic_lock:
        order_traffic = True
    threading.Thread(target=order_generator, daemon=True, name="order-traffic").start()
    logger.info("Order traffic started.")
    return {"status": "order traffic started"}


@app.post("/stop-order-traffic")
def stop_order_traffic() -> dict:
    """Stop the order traffic generator."""
    global order_traffic
    with _traffic_lock:
        order_traffic = False
    logger.info("Order traffic stopped.")
    return {"status": "order traffic stopped"}


@app.post("/start-auth-traffic")
def start_auth_traffic() -> dict:
    """Start the auth traffic generator in a background thread."""
    global auth_traffic
    with _traffic_lock:
        auth_traffic = True
    threading.Thread(target=auth_generator, daemon=True, name="auth-traffic").start()
    logger.info("Auth traffic started.")
    return {"status": "auth traffic started"}


@app.post("/stop-auth-traffic")
def stop_auth_traffic() -> dict:
    """Stop the auth traffic generator."""
    global auth_traffic
    with _traffic_lock:
        auth_traffic = False
    logger.info("Auth traffic stopped.")
    return {"status": "auth traffic stopped"}


# ==============================
# SYSTEM CONTROL ENDPOINTS
# ==============================

@app.post("/start-system")
def start_system() -> dict:
    """Start the full microservice stack via docker compose."""
    try:
        result = subprocess.run(
            ["docker", "compose", "up", "--build", "-d"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error("docker compose up failed: %s", result.stderr)
            raise HTTPException(status_code=500, detail=f"docker compose up failed: {result.stderr}")
        logger.info("System started successfully.")
        return {"status": "system started"}
    except subprocess.TimeoutExpired:
        logger.error("docker compose up timed out.")
        raise HTTPException(status_code=504, detail="docker compose timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error starting system: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stop-system")
def stop_system() -> dict:
    """Stop the full microservice stack via docker compose."""
    try:
        result = subprocess.run(
            ["docker", "compose", "down"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logger.error("docker compose down failed: %s", result.stderr)
            raise HTTPException(status_code=500, detail=f"docker compose down failed: {result.stderr}")
        logger.info("System stopped successfully.")
        return {"status": "system stopped"}
    except subprocess.TimeoutExpired:
        logger.error("docker compose down timed out.")
        raise HTTPException(status_code=504, detail="docker compose timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error stopping system: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system-status")
def system_status() -> dict:
    """Check whether the docker compose stack has any running services."""
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--services", "--filter", "status=running"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        running = bool(result.stdout.strip())
        return {"running": running}
    except subprocess.TimeoutExpired:
        logger.error("docker compose ps timed out.")
        raise HTTPException(status_code=504, detail="docker compose ps timed out")
    except Exception as e:
        logger.error("Unexpected error checking system status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
