"""tests/unit/test_metrics_server.py — MetricsServer endpoint testleri."""
from __future__ import annotations
import pytest
from aiohttp.test_utils import TestClient, TestServer
import aiohttp
from wallet_manager.metrics_server import MetricsServer


@pytest.fixture
async def metrics_client():
    server = MetricsServer(port=0)  # port=0 → OS random port
    app = server.build_app()
    async with TestClient(TestServer(app)) as client:
        yield client


class TestMetricsEndpoint:
    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, metrics_client):
        resp = await metrics_client.get("/metrics")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_metrics_content_type(self, metrics_client):
        resp = await metrics_client.get("/metrics")
        ct = resp.headers.get("Content-Type", "")
        assert "text/plain" in ct

    @pytest.mark.asyncio
    async def test_metrics_contains_wallet_prefix(self, metrics_client):
        resp = await metrics_client.get("/metrics")
        text = await resp.text()
        assert "wallet_" in text

    @pytest.mark.asyncio
    async def test_health_returns_200(self, metrics_client):
        resp = await metrics_client.get("/health")
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_health_response_json(self, metrics_client):
        resp = await metrics_client.get("/health")
        data = await resp.json()
        assert data["status"] == "ok"
        assert "ts" in data


class TestMetricUpdates:
    def test_update_metric(self):
        server = MetricsServer(port=0)
        server.update_metric("wallet_cycle_total", 42.0)
        assert server._metrics.get("wallet_cycle_total") == 42.0

    def test_update_labeled_metric(self):
        server = MetricsServer(port=0)
        server.update_labeled_metric("wallet_pillar_value_usd", "ALGO", 500.0)
        key = ("wallet_pillar_value_usd", "ALGO")
        assert server._labeled_metrics.get(key) == 500.0
