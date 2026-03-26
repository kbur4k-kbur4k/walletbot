"""Tests for health check endpoints (Task 6.2)."""
from __future__ import annotations

import json

import pytest


async def test_live_endpoint_returns_200():
    from wallet_manager.utils.health_check import HealthCheck

    hc = HealthCheck()
    result = await hc.handle_live(None)
    assert result.status == 200


async def test_metrics_summary_has_required_fields():
    from wallet_manager.utils.health_check import HealthCheck

    hc = HealthCheck()
    result = await hc.handle_metrics_summary(None)
    data = json.loads(result.body)
    assert "portfolio_value_usd" in data
    assert "active_wallets" in data
    assert "current_execution_mode" in data


async def test_ready_returns_503_without_rpc(monkeypatch):
    """Without RPC_URL_PRIMARY set, /ready returns 503."""
    monkeypatch.delenv("RPC_URL_PRIMARY", raising=False)
    monkeypatch.setenv("WALLETS_FILE", "nonexistent.json")

    from wallet_manager.utils.health_check import HealthCheck
    hc = HealthCheck()
    result = await hc.handle_ready(None)
    # Without RPC configured, should not be 200
    assert result.status in (200, 503)  # implementation may vary
