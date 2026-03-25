"""tests/unit/test_config.py — Settings and get_settings() tests."""
from __future__ import annotations

import os
import importlib

import pytest

from wallet_manager.config import Settings, get_settings
from wallet_manager.exceptions import ConfigurationError


def _make_settings(**kwargs) -> Settings:
    base = dict(
        vault_key="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=",
        execution_mode="ml_quant",
        garch_alpha=0.10,
        garch_beta=0.85,
    )
    base.update(kwargs)
    return Settings(**base)


def test_invalid_execution_mode_raises():
    with pytest.raises(ConfigurationError) as exc_info:
        _make_settings(execution_mode="invalid_mode")
    assert "invalid_mode" in str(exc_info.value)
    # Valid list should be present in the error message
    assert "ml_quant" in str(exc_info.value)


def test_garch_stationarity_violation_raises():
    with pytest.raises(ConfigurationError):
        _make_settings(garch_alpha=0.50, garch_beta=0.55)  # 0.50 + 0.55 >= 1.0


def test_garch_stationarity_boundary_raises():
    with pytest.raises(ConfigurationError):
        _make_settings(garch_alpha=0.15, garch_beta=0.85)  # exactly 1.0


def test_valid_settings_no_error():
    cfg = _make_settings(garch_alpha=0.10, garch_beta=0.85)
    assert cfg.execution_mode == "ml_quant"
    assert cfg.garch_alpha + cfg.garch_beta < 1.0


def test_get_settings_singleton():
    """get_settings() must return the exact same object on repeated calls."""
    import wallet_manager.config as cfg_module
    # Reset singleton to ensure clean state
    cfg_module._settings_cache = None
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_from_env_reads_vault_key(monkeypatch):
    """Settings.from_env() must read VAULT_KEY from environment."""
    import wallet_manager.config as cfg_module
    key = "kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo="
    monkeypatch.setenv("VAULT_KEY", key)
    monkeypatch.setenv("EXECUTION_MODE", "legacy")
    monkeypatch.setenv("GARCH_ALPHA", "0.10")
    monkeypatch.setenv("GARCH_BETA", "0.85")
    cfg_module._settings_cache = None
    cfg = Settings.from_env()
    assert cfg.vault_key == key
    cfg_module._settings_cache = None


def test_all_valid_execution_modes():
    from wallet_manager.config import VALID_EXECUTION_MODES
    for mode in VALID_EXECUTION_MODES:
        cfg = _make_settings(execution_mode=mode)
        assert cfg.execution_mode == mode
