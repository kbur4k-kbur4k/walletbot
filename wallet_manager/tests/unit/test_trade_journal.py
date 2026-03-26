"""tests/unit/test_trade_journal.py — TradeJournal testleri."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wallet_manager.trade_journal import TradeJournal


class TestTradeJournal:
    @pytest.fixture
    def journal(self, tmp_path):
        return TradeJournal(journal_path=str(tmp_path / "trades.jsonl"), max_mb=1.0)

    def test_record_trade_writes_line(self, journal, tmp_path):
        journal.record_trade(
            wallet_id="w1",
            strategy="ALGO",
            direction="BUY",
            amount_usd=100.0,
            tx_hash="0xabc",
        )
        lines = Path(tmp_path / "trades.jsonl").read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["wallet_id"] == "w1"
        assert data["strategy"] == "ALGO"

    def test_private_key_masked(self, journal, tmp_path):
        """Private key asla log'a yazılmamalı."""
        journal.record_trade(
            wallet_id="w1",
            strategy="ALGO",
            direction="BUY",
            amount_usd=50.0,
            tx_hash="0xdef",
            extra={"private_key": "0xsecretkey"},
        )
        content = Path(tmp_path / "trades.jsonl").read_text()
        assert "0xsecretkey" not in content

    def test_get_recent_trades_count(self, journal):
        for i in range(10):
            journal.record_trade(f"w{i}", "ALGO", "BUY", 10.0, f"0x{i:04x}")
        recent = journal.get_recent_trades(n=5)
        assert len(recent) == 5

    def test_get_recent_trades_order(self, journal):
        """En son trade ilk sırada olmalı."""
        for i in range(5):
            journal.record_trade("w1", "ALGO", "BUY", float(i) * 10, f"0x{i:04x}")
        recent = journal.get_recent_trades(n=3)
        assert recent[0]["amount_usd"] == pytest.approx(40.0)

    def test_rotation_on_size_limit(self, journal, tmp_path):
        """Dosya max_mb aşınca rotate edilmeli."""
        journal._max_bytes = 100  # çok küçük limit
        for i in range(20):
            journal.record_trade("w1", "ALGO", "BUY", 100.0, f"0x{i:08x}")
        current_size = Path(tmp_path / "trades.jsonl").stat().st_size
        assert current_size <= 100 * 5  # küçük dosya
