"""
Tests for services/order-service/main.py

Run with:
    pytest tests/test_order_service.py -v
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

_ORDER_SERVICE_PATH = (
    pathlib.Path(__file__).parent.parent / "services" / "order-service" / "main.py"
)


def _load_module(auth_url: str = "http://auth-service:8000"):
    """
    Load a fresh copy of order-service main.

    - Mocks Prometheus Counter/Histogram so repeated loads don't hit
      the "duplicate timeseries" ValueError in the global CollectorRegistry.
    - Sets AUTH_SERVICE_URL via env so tests can inject a custom URL.
    """
    if "order_main" in sys.modules:
        del sys.modules["order_main"]

    mock_counter = MagicMock()
    mock_histogram = MagicMock()

    with (
        patch("prometheus_client.Counter", return_value=mock_counter),
        patch("prometheus_client.Histogram", return_value=mock_histogram),
        patch.dict("os.environ", {"AUTH_SERVICE_URL": auth_url}),
    ):
        spec = importlib.util.spec_from_file_location("order_main", _ORDER_SERVICE_PATH)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        sys.modules["order_main"] = mod

    return mod


def _client(auth_url: str = "http://auth-service:8000") -> TestClient:
    return TestClient(_load_module(auth_url).app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
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
        """Endpoint must return text/plain."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# /create-order — auth reachable
# ---------------------------------------------------------------------------

class TestCreateOrderAuthReachable:
    def test_returns_200_when_auth_ok(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp):
            c = _client()
            resp = c.post("/create-order")

        assert resp.status_code == 200
        assert resp.json() == {"status": "order created"}

    def test_increments_counter_on_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp):
            mod = _load_module()
            c = TestClient(mod.app)
            c.post("/create-order")

        mod.REQUEST_COUNT.inc.assert_called_once()

    def test_observes_latency_on_success(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp):
            mod = _load_module()
            c = TestClient(mod.app)
            c.post("/create-order")

        mod.REQUEST_LATENCY.observe.assert_called_once()


# ---------------------------------------------------------------------------
# /create-order — auth unreachable (graceful degradation)
# ---------------------------------------------------------------------------

class TestCreateOrderAuthUnreachable:
    def test_returns_200_despite_connection_error(self):
        """
        Order service must continue and return 200 even when auth is down.
        Validates the graceful-degradation design.
        """
        import requests as _requests

        with patch("requests.get", side_effect=_requests.ConnectionError("auth down")):
            c = _client()
            resp = c.post("/create-order")

        assert resp.status_code == 200
        assert resp.json() == {"status": "order created"}

    def test_returns_200_on_timeout(self):
        import requests as _requests

        with patch("requests.get", side_effect=_requests.Timeout("auth timeout")):
            c = _client()
            resp = c.post("/create-order")

        assert resp.status_code == 200

    def test_counter_still_increments_when_auth_down(self):
        import requests as _requests

        with patch("requests.get", side_effect=_requests.ConnectionError("auth down")):
            mod = _load_module()
            c = TestClient(mod.app)
            c.post("/create-order")

        mod.REQUEST_COUNT.inc.assert_called_once()

    def test_auth_url_comes_from_env(self):
        """AUTH_SERVICE_URL env var must be picked up at import time."""
        import requests as _requests

        captured_urls: list = []

        def fake_get(url, **kwargs):
            captured_urls.append(url)
            raise _requests.ConnectionError("not real")

        with patch("requests.get", side_effect=fake_get):
            c = _client(auth_url="http://my-custom-auth:9999")
            c.post("/create-order")

        assert any("my-custom-auth:9999" in u for u in captured_urls)


# ---------------------------------------------------------------------------
# /create-order — latency mode
# ---------------------------------------------------------------------------

class TestCreateOrderLatencyMode:
    def test_latency_mode_still_returns_200(self):
        import requests as _requests

        with patch("requests.get", side_effect=_requests.ConnectionError()):
            with patch("time.sleep"):
                c = _client()
                c.post("/toggle-latency?enabled=true")
                resp = c.post("/create-order")

        assert resp.status_code == 200

    def test_latency_mode_calls_sleep(self):
        """time.sleep must be called with a value in [1.5, 3.0] when on."""
        import requests as _requests

        mod = _load_module()
        c = TestClient(mod.app)
        c.post("/toggle-latency?enabled=true")

        with patch("requests.get", side_effect=_requests.ConnectionError()):
            with patch("time.sleep") as mock_sleep:
                c.post("/create-order")

        mock_sleep.assert_called_once()
        delay = mock_sleep.call_args[0][0]
        assert 1.5 <= delay <= 3.0

    def test_no_sleep_when_latency_off(self):
        import requests as _requests

        mod = _load_module()
        c = TestClient(mod.app)
        # latency mode is off by default

        with patch("requests.get", side_effect=_requests.ConnectionError()):
            with patch("time.sleep") as mock_sleep:
                c.post("/create-order")

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
