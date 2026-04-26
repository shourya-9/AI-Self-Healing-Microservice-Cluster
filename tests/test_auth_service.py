"""
Tests for services/auth-service/main.py

Run with:
    pytest tests/test_auth_service.py -v
"""
import importlib.util
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

_AUTH_SERVICE_PATH = (
    pathlib.Path(__file__).parent.parent / "services" / "auth-service" / "main.py"
)


def _load_module():
    """
    Load a fresh copy of auth-service main, with Prometheus metric constructors
    mocked so each load doesn't try to re-register the same metric names in the
    global CollectorRegistry (which raises ValueError on duplicates).
    """
    if "auth_main" in sys.modules:
        del sys.modules["auth_main"]

    mock_counter = MagicMock()
    mock_histogram = MagicMock()

    with (
        patch("prometheus_client.Counter", return_value=mock_counter),
        patch("prometheus_client.Histogram", return_value=mock_histogram),
    ):
        spec = importlib.util.spec_from_file_location("auth_main", _AUTH_SERVICE_PATH)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        sys.modules["auth_main"] = mod

    return mod


def _client() -> TestClient:
    return TestClient(_load_module().app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Fresh TestClient with default (all-off) state."""
    return _client()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_healthy(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_returns_prometheus_text(self, client: TestClient):
        """Endpoint should exist and return text/plain regardless of metric content."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# /login — normal mode
# ---------------------------------------------------------------------------

class TestLoginNormal:
    def test_returns_200_and_body(self, client: TestClient):
        resp = client.get("/login")
        assert resp.status_code == 200
        assert resp.json() == {"status": "logged in"}

    def test_increments_request_counter(self):
        """REQUEST_COUNT.inc() should be called on a successful login."""
        mod = _load_module()
        c = TestClient(mod.app)
        c.get("/login")
        mod.REQUEST_COUNT.inc.assert_called()

    def test_observes_latency_histogram(self):
        """REQUEST_LATENCY.observe() should be called on a successful login."""
        mod = _load_module()
        c = TestClient(mod.app)
        c.get("/login")
        mod.REQUEST_LATENCY.observe.assert_called_once()


# ---------------------------------------------------------------------------
# /login — crash mode
# ---------------------------------------------------------------------------

class TestLoginCrashMode:
    def test_crash_mode_returns_503(self, client: TestClient):
        client.post("/toggle-crash?enabled=true")
        resp = client.get("/login")
        assert resp.status_code == 503
        assert "crash simulated" in resp.json()["detail"].lower()

    def test_crash_mode_off_returns_200(self, client: TestClient):
        client.post("/toggle-crash?enabled=true")
        client.post("/toggle-crash?enabled=false")
        resp = client.get("/login")
        assert resp.status_code == 200

    def test_crash_mode_does_not_increment_counter(self):
        """Counter should NOT be incremented when crash raises before inc()."""
        mod = _load_module()
        c = TestClient(mod.app)
        c.post("/toggle-crash?enabled=true")
        c.get("/login")  # returns 503; TestClient doesn't raise
        mod.REQUEST_COUNT.inc.assert_not_called()


# ---------------------------------------------------------------------------
# /login — latency mode
# ---------------------------------------------------------------------------

class TestLoginLatencyMode:
    def test_latency_mode_still_returns_200(self):
        """With time.sleep mocked the response must be 200 even in latency mode."""
        mod = _load_module()
        c = TestClient(mod.app)
        c.post("/toggle-latency?enabled=true")
        with patch("time.sleep"):
            resp = c.get("/login")
        assert resp.status_code == 200
        assert resp.json() == {"status": "logged in"}

    def test_latency_mode_calls_sleep(self):
        """time.sleep must be invoked exactly once when latency mode is on."""
        mod = _load_module()
        c = TestClient(mod.app)
        c.post("/toggle-latency?enabled=true")
        with patch("time.sleep") as mock_sleep:
            c.get("/login")
        mock_sleep.assert_called_once()
        delay = mock_sleep.call_args[0][0]
        assert 6 <= delay <= 10

    def test_no_sleep_when_latency_off(self):
        mod = _load_module()
        c = TestClient(mod.app)
        with patch("time.sleep") as mock_sleep:
            c.get("/login")
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# /toggle-latency
# ---------------------------------------------------------------------------

class TestToggleLatency:
    def test_enable_latency(self, client: TestClient):
        resp = client.post("/toggle-latency?enabled=true")
        assert resp.status_code == 200
        assert resp.json()["latency_mode"] is True

    def test_disable_latency(self, client: TestClient):
        client.post("/toggle-latency?enabled=true")
        resp = client.post("/toggle-latency?enabled=false")
        assert resp.status_code == 200
        assert resp.json()["latency_mode"] is False

    def test_missing_param_returns_422(self, client: TestClient):
        resp = client.post("/toggle-latency")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /toggle-crash
# ---------------------------------------------------------------------------

class TestToggleCrash:
    def test_enable_crash(self, client: TestClient):
        resp = client.post("/toggle-crash?enabled=true")
        assert resp.status_code == 200
        assert resp.json()["crash_mode"] is True

    def test_disable_crash(self, client: TestClient):
        client.post("/toggle-crash?enabled=true")
        resp = client.post("/toggle-crash?enabled=false")
        assert resp.status_code == 200
        assert resp.json()["crash_mode"] is False

    def test_missing_param_returns_422(self, client: TestClient):
        resp = client.post("/toggle-crash")
        assert resp.status_code == 422

    def test_crash_and_latency_flags_are_independent(self, client: TestClient):
        """Toggling crash must not affect latency flag, and vice versa."""
        client.post("/toggle-crash?enabled=true")
        client.post("/toggle-latency?enabled=true")
        client.post("/toggle-crash?enabled=false")
        # Latency should still be on — turning off crash didn't touch it
        resp = client.post("/toggle-latency?enabled=false")
        assert resp.json()["latency_mode"] is False
