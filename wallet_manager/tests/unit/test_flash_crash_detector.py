"""tests/unit/test_flash_crash_detector.py — FlashCrashDetector unit tests."""
from __future__ import annotations

import time

import pytest

from wallet_manager.strategies.flash_crash_detector import (
    CRASH_THRESHOLD_PCT,
    KILL_SWITCH_COOLDOWN_SECONDS,
    FlashCrashDetector,
)


def _make_detector(**kwargs) -> FlashCrashDetector:
    return FlashCrashDetector(**kwargs)


def test_no_trigger_small_drop():
    """A drop just below the threshold (e.g. 7.9% < 8%) must NOT trigger."""
    detector = _make_detector()
    base = 1000.0
    # Ingest a starting price
    detector.ingest_price(base)
    # Drop by 7.9% — below CRASH_THRESHOLD_PCT (8%)
    low = base * (1.0 - 0.079)
    alert = detector.ingest_price(low)
    assert alert is None, f"Expected no alert, got {alert}"
    assert not detector.is_kill_switch_active()


def test_trigger_on_large_drop():
    """A rapid drop >= 8% MUST trigger a CrashAlert and activate kill-switch."""
    detector = _make_detector()
    base = 1000.0
    detector.ingest_price(base)
    # Drop by exactly CRASH_THRESHOLD_PCT + a bit
    low = base * (1.0 - CRASH_THRESHOLD_PCT - 0.005)
    alert = detector.ingest_price(low)
    assert alert is not None, "Expected CrashAlert on large drop"
    assert detector.is_kill_switch_active()
    assert alert.drawdown_pct >= CRASH_THRESHOLD_PCT


def test_cooldown_prevents_retriggering():
    """After a trigger, a subsequent drop within cooldown must NOT retrigger."""
    detector = _make_detector(cooldown_seconds=KILL_SWITCH_COOLDOWN_SECONDS)
    base = 1000.0
    detector.ingest_price(base)
    # First crash
    low = base * (1.0 - CRASH_THRESHOLD_PCT - 0.01)
    first = detector.ingest_price(low)
    assert first is not None, "First crash should trigger"

    # Another large drop immediately — cooldown should prevent re-triggering
    even_lower = low * (1.0 - CRASH_THRESHOLD_PCT - 0.01)
    second = detector.ingest_price(even_lower)
    assert second is None, "Cooldown should prevent re-triggering"
