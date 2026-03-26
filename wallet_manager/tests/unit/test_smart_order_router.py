"""tests/unit/test_smart_order_router.py — SmartOrderRouter testleri."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.strategies.smart_order_router import RoutePlan, SmartOrderRouter


class TestSmartOrderRouter:
    def _router(self):
        return SmartOrderRouter()

    @pytest.mark.asyncio
    async def test_get_best_route_returns_plan(self):
        router = self._router()
        with patch.object(router, "_quote_uniswap_v3", new=AsyncMock(return_value=int(1800e6))):
            with patch.object(router, "_quote_sushiswap", new=AsyncMock(return_value=int(1790e6))):
                plan = await router.get_best_route(
                    from_token="ETH",
                    to_token="USDC",
                    amount_in_wei=int(1e18),
                    rpc_url="https://mainnet.base.org",
                )
        assert isinstance(plan, RoutePlan)
        assert plan.expected_out > 0

    @pytest.mark.asyncio
    async def test_router_selects_highest_output(self):
        """Router en yüksek çıktıyı veren DEX'i seçmeli."""
        router = self._router()
        with patch.object(router, "_quote_uniswap_v3", new=AsyncMock(return_value=int(1850e6))):
            with patch.object(router, "_quote_sushiswap", new=AsyncMock(return_value=int(1700e6))):
                plan = await router.get_best_route("ETH", "USDC", int(1e18), "https://rpc")
        assert plan.dex == "uniswap_v3"

    @pytest.mark.asyncio
    async def test_all_quotes_fail_raises(self):
        router = self._router()
        with patch.object(router, "_quote_uniswap_v3", new=AsyncMock(side_effect=Exception("RPC fail"))):
            with patch.object(router, "_quote_sushiswap", new=AsyncMock(side_effect=Exception("RPC fail"))):
                with pytest.raises(Exception):
                    await router.get_best_route("ETH", "USDC", int(1e18), "https://rpc")
