"""tests/unit/test_actions.py — actions.py pure fonksiyon testleri."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wallet_manager.actions import (
    _checksum,
    eth_balance_usd_value,
    get_chain_addresses,
    get_rotated_fee_tier,
    is_swap_eligible,
    is_usdc_swap_eligible,
    usdc_balance_usd_value,
)


class TestBalanceCalculations:
    def test_eth_balance_usd_value(self):
        """1 ETH = 2000 USD."""
        result = eth_balance_usd_value(int(1e18), eth_price_usd=2000.0)
        assert result == pytest.approx(2000.0)

    def test_eth_balance_usd_value_fractional(self):
        """0.5 ETH = 1000 USD."""
        result = eth_balance_usd_value(int(0.5e18), eth_price_usd=2000.0)
        assert result == pytest.approx(1000.0)

    def test_eth_balance_zero_price(self):
        result = eth_balance_usd_value(int(1e18), eth_price_usd=0.0)
        assert result == pytest.approx(0.0)

    def test_usdc_balance_usd_value(self):
        """100 USDC = $100.00."""
        result = usdc_balance_usd_value(int(100e6))
        assert result == pytest.approx(100.0)

    def test_usdc_balance_zero(self):
        result = usdc_balance_usd_value(0)
        assert result == pytest.approx(0.0)


class TestSwapEligibility:
    def test_eligible_swap(self):
        """Yeterli bakiye ve düşük gas ile swap eligible."""
        result = is_swap_eligible(
            eth_balance_wei=int(0.1e18),
            eth_price_usd=2000.0,
            gas_price_gwei=1.0,
            min_trade_usd=5.0,
            max_gas_gwei=50.0,
        )
        assert result is True

    def test_high_gas_prevents_swap(self):
        result = is_swap_eligible(
            eth_balance_wei=int(0.1e18),
            eth_price_usd=2000.0,
            gas_price_gwei=100.0,
            min_trade_usd=5.0,
            max_gas_gwei=50.0,
        )
        assert result is False

    def test_insufficient_balance_prevents_swap(self):
        result = is_swap_eligible(
            eth_balance_wei=int(0.001e18),  # $2 — altında min_trade
            eth_price_usd=2000.0,
            gas_price_gwei=1.0,
            min_trade_usd=5.0,
            max_gas_gwei=50.0,
        )
        assert result is False


class TestFeeTierRotation:
    def test_fee_tier_cycles(self):
        """Fee tier döngüsel olarak değişmeli."""
        tiers = {get_rotated_fee_tier(i) for i in range(10)}
        assert len(tiers) > 1

    def test_fee_tier_valid_values(self):
        """Fee tier Uniswap V3 geçerli değerlerinden biri olmalı."""
        valid_tiers = {100, 500, 3000, 10000}
        for i in range(20):
            assert get_rotated_fee_tier(i) in valid_tiers


class TestChainAddresses:
    def test_base_chain_addresses(self):
        addrs = get_chain_addresses(8453)
        assert hasattr(addrs, "usdc")
        assert hasattr(addrs, "weth")
        assert addrs.usdc.startswith("0x")
        assert addrs.weth.startswith("0x")

    def test_unsupported_chain_raises(self):
        with pytest.raises(Exception):
            get_chain_addresses(99999)  # desteklenmeyen chain


class TestChecksumAddress:
    def test_lowercase_to_checksum(self):
        addr = "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
        result = _checksum(addr)
        assert result.startswith("0x")
        assert result != addr  # checksum farklı

    def test_already_checksum_unchanged(self):
        addr = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
        result = _checksum(addr)
        assert result == addr
