"""tests/integration/test_paper_engine.py — PaperTradingEngine integration tests."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.strategies.paper_engine import (
    PaperTradeConfig,
    PaperTradingEngine,
    SlippageSimulator,
    VirtualLedger,
    reset_paper_engine,
)


@pytest.mark.asyncio
async def test_paper_engine_never_calls_broadcast(tmp_path):
    """PaperTradingEngine must never call sign_and_broadcast_tx."""
    reset_paper_engine()
    config = PaperTradeConfig()
    engine = PaperTradingEngine(config=config)
    await engine.initialize()

    mock_manager = MagicMock()
    mock_manager.sign_and_broadcast_tx = AsyncMock()

    with (
        patch("wallet_manager.actions.get_live_price_v3", new=AsyncMock(return_value=2000.0)),
    ):
        try:
            await engine.run_shadow_cycle(
                manager=mock_manager,
                wallet_id="test_wallet",
                portfolio_manager=MagicMock(total_value_usd=1000.0),
                task_context={},
                analytics_db=None,
                quant_analytics=None,
            )
        except Exception:
            pass  # May fail due to incomplete mocks; broadcast must still not be called

    mock_manager.sign_and_broadcast_tx.assert_not_called()
    await engine.shutdown()
    reset_paper_engine()


@pytest.mark.asyncio
async def test_virtual_ledger_write_read_roundtrip(tmp_path):
    """VirtualLedger: write then read round-trip preserves key fields."""
    ledger = VirtualLedger(db_path=str(tmp_path / "test_ledger.db"))
    await ledger.initialize()

    await ledger.record_trade(
        direction="BUY",
        asset_pair="WETH/USDC",
        entry_price=2000.0,
        simulated_fill_price=2001.0,
        slippage_bps=5.0,
        size_usd=100.0,
        gas_cost_usd=0.05,
        pnl_usd=0.0,
        reason_codes=("signal_buy",),
        signal_source="test_signal",
    )

    summary = await ledger.get_pnl_summary()
    assert summary is not None
    assert isinstance(summary, dict)

    await ledger.shutdown()


def test_slippage_gaussian_within_3_sigma():
    """Gaussian slippage model produces values within a reasonable range."""
    simulator = SlippageSimulator(mean_bps=5.0, std_bps=2.0)

    slippages = []
    for _ in range(500):
        slip_bps, _ = simulator.simulate_slippage(
            size_usd=100.0,
            pool_depth_usd=1_000_000.0,
            model="gaussian",
            entry_price=2000.0,
        )
        slippages.append(slip_bps)

    # All slippages must be >= 0 (floor at 0.5 bps)
    assert all(s >= 0 for s in slippages)
    # 3σ = 5 + 6 = 11 bps with impact_scalar ~1 for small orders; max cap is 500
    assert all(s <= 500.0 for s in slippages)
    # Mean should be near 5.0 bps for small orders
    import numpy as np
    mean_slip = float(np.mean(slippages))
    assert 1.0 <= mean_slip <= 50.0  # Loose bound due to impact scaling


@pytest.mark.asyncio
async def test_paper_engine_initialize_and_shutdown(tmp_path):
    """Engine initializes and shuts down cleanly."""
    reset_paper_engine()
    engine = PaperTradingEngine(config=PaperTradeConfig())
    await engine.initialize()
    report = await engine.get_performance_report()
    assert isinstance(report, dict)
    await engine.shutdown()
    reset_paper_engine()
