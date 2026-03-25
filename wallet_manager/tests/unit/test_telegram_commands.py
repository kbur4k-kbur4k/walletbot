"""tests/unit/test_telegram_commands.py — Telegram komut testleri."""
from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestTelegramCommands:
    def _make_notifier(self, mocker, chat_id="123456789"):
        from wallet_manager.security.telegram_manager import TelegramManager
        cfg = MagicMock()
        cfg.telegram_bot_token = "test_token"
        cfg.telegram_chat_id = chat_id
        mgr = TelegramManager(settings=cfg)
        mgr._send = AsyncMock(return_value=True)
        return mgr

    def test_wrong_chat_id_rejected(self, mocker):
        from wallet_manager.security.telegram_manager import TelegramManager
        cfg = MagicMock()
        cfg.telegram_bot_token = "test_token"
        cfg.telegram_chat_id = "ALLOWED_CHAT"
        mgr = TelegramManager(settings=cfg)
        update = MagicMock()
        update.message.chat_id = "EVIL_CHAT_ID"
        result = mgr.is_authorized(update)
        assert result is False

    def test_correct_chat_id_authorized(self, mocker):
        from wallet_manager.security.telegram_manager import TelegramManager
        cfg = MagicMock()
        cfg.telegram_bot_token = "test_token"
        cfg.telegram_chat_id = "123456789"
        mgr = TelegramManager(settings=cfg)
        update = MagicMock()
        update.message.chat_id = "123456789"
        result = mgr.is_authorized(update)
        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excess_commands(self, mocker):
        from wallet_manager.security.telegram_manager import TelegramManager
        cfg = MagicMock()
        cfg.telegram_bot_token = "test_token"
        cfg.telegram_chat_id = "123456789"
        mgr = TelegramManager(settings=cfg)
        for _ in range(3):
            mgr._record_confirm_attempt("123456789")
        assert mgr._is_confirm_rate_limited("123456789") is True


# ---------------------------------------------------------------------------
# New tests for CommandCenter commands (ADIM 4.2)
# ---------------------------------------------------------------------------


@pytest.fixture
def command_state():
    from wallet_manager.telegram_notifier import CommandState
    return CommandState()


@pytest.fixture
def credentials():
    from wallet_manager.telegram_notifier import _TelegramCredentials
    return _TelegramCredentials(_bot_token="test_token", _chat_id="123456789")


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.send_message = AsyncMock()
    return client


@pytest.fixture
def command_center(credentials, command_state, mock_client):
    from wallet_manager.telegram_notifier import CommandCenter
    return CommandCenter(
        credentials=credentials,
        command_state=command_state,
        client=mock_client,
    )


@pytest.mark.asyncio
async def test_setmode_command_validates_mode(command_center, mock_client):
    """_handle_setmode must reject unknown modes."""
    await command_center._handle_setmode("123456789", "invalid_mode_xyz")
    call_args = mock_client.send_message.call_args[0][0]
    assert "Unknown mode" in call_args or "invalid_mode" in call_args.lower()


@pytest.mark.asyncio
async def test_setmode_ml_quant_requires_confirm(command_center, mock_client):
    """Setting ml_quant mode without 'confirm' must ask for confirmation."""
    await command_center._handle_setmode("123456789", "ml_quant")
    call_args = mock_client.send_message.call_args[0][0]
    assert "confirm" in call_args.lower() or "production" in call_args.lower()


@pytest.mark.asyncio
async def test_setmode_ml_quant_with_confirm_succeeds(command_center, command_state, mock_client):
    """Setting ml_quant with 'confirm' must request mode change."""
    await command_center._handle_setmode("123456789", "ml_quant confirm")
    assert command_state.pop_mode_change() == "ml_quant"


@pytest.mark.asyncio
async def test_setmode_legacy_no_confirm_needed(command_center, command_state, mock_client):
    """Setting legacy mode must NOT require confirm."""
    await command_center._handle_setmode("123456789", "legacy")
    assert command_state.pop_mode_change() == "legacy"


@pytest.mark.asyncio
async def test_wallets_command_masks_addresses(command_center, mock_client):
    """_handle_wallets must mask wallet addresses."""
    mock_wallet = MagicMock()
    mock_wallet.wallet_id = "wallet_01"
    mock_wallet.address = "0xAbCd1234567890AbCd1234567890AbCd12345678"
    mock_wallet.max_gas_price_gwei = 50.0

    with patch("wallet_manager.orchestrator._get_wallet_configs", return_value=[mock_wallet]), \
         patch("wallet_manager.orchestrator._get_circuit_breakers", return_value={}):
        await command_center._handle_wallets("123456789")

    call_args = mock_client.send_message.call_args[0][0]
    # Full address must NOT appear
    assert "0xAbCd1234567890AbCd1234567890AbCd12345678" not in call_args


@pytest.mark.asyncio
async def test_allocation_command_shows_drift(command_center, mock_client):
    """_handle_allocation must display allocation info."""
    from wallet_manager.portfolio_manager import PortfolioManager, StrategyType
    pm = PortfolioManager()
    pm.update_pillar_value(StrategyType.YIELD, 500.0)
    pm.update_pillar_value(StrategyType.ALGO, 300.0)
    pm.update_pillar_value(StrategyType.SHORT, 100.0)
    pm.update_pillar_value(StrategyType.AIRDROP, 100.0)

    with patch("wallet_manager.orchestrator._get_portfolio_manager", return_value=pm):
        await command_center._handle_allocation("123456789")

    call_args = mock_client.send_message.call_args[0][0]
    assert "YIELD" in call_args or "allocation" in call_args.lower()


@pytest.mark.asyncio
async def test_reset_circuit_requires_confirm(command_center, mock_client):
    """_handle_reset_circuit without 'confirm' must ask for confirmation."""
    await command_center._handle_reset_circuit("123456789", "wallet_01")
    call_args = mock_client.send_message.call_args[0][0]
    assert "confirm" in call_args.lower()


@pytest.mark.asyncio
async def test_reset_circuit_with_confirm_calls_force_close(command_center, mock_client):
    """_handle_reset_circuit with 'confirm' must call force_close on the CB."""
    mock_cb = MagicMock()
    mock_cb.force_close = MagicMock()
    mock_cb.state = MagicMock()
    mock_cb.state.name = "CLOSED"

    with patch("wallet_manager.orchestrator._get_circuit_breakers", return_value={"wallet_01": mock_cb}):
        await command_center._handle_reset_circuit("123456789", "wallet_01 confirm")

    mock_cb.force_close.assert_called_once()


def test_rate_limiter_blocks_flood(command_center):
    """After 10 commands in 60s, the rate limiter must return False."""
    from wallet_manager.telegram_notifier import _RATE_LIMIT_MAX
    chat_id = "flood_test_unique_42"

    for _ in range(_RATE_LIMIT_MAX):
        result = command_center._check_rate_limit(chat_id, "/status")
        assert result is True

    result = command_center._check_rate_limit(chat_id, "/status")
    assert result is False


def test_rate_limiter_allows_after_window_expires(command_center):
    """Commands after the rate limit window expires should be allowed again."""
    import time as _time
    from wallet_manager.telegram_notifier import _RATE_LIMIT_MAX, _RATE_LIMIT_WINDOW
    chat_id = "window_test_unique_42"

    for _ in range(_RATE_LIMIT_MAX):
        command_center._check_rate_limit(chat_id, "/status")

    old_time = _time.monotonic() - _RATE_LIMIT_WINDOW - 1
    command_center._cmd_history[chat_id] = [old_time] * _RATE_LIMIT_MAX

    result = command_center._check_rate_limit(chat_id, "/status")
    assert result is True
