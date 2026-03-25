"""
wallet_manager/orchestrator.py

WalletOrchestrator — top-level runtime coordinator.

Responsibility
--------------
Orchestrate the full execution lifecycle across all configured wallets:

1. Boot notification
2. Concurrent wallet execution (bounded by a semaphore)
3. Per-wallet gas monitoring with pause/resume
4. Per-wallet success / failure notifications
5. Shutdown notification with execution summary

This module sits above all other modules in the dependency graph.  It knows
about ``WalletManager``, ``GasMonitor``, and ``TelegramNotifier`` — but each
of those depends only on its own narrow interface.

Concurrency model
-----------------
All 20 wallets run concurrently under an ``asyncio.Semaphore`` that caps
simultaneous in-flight operations.  The default concurrency cap is
``MAX_CONCURRENT_WALLETS`` (env var, default 5).  Each wallet coroutine is
wrapped in a ``try/except`` so a single wallet failure never cancels the
others — the orchestrator continues and records the failure.

Gas monitoring
--------------
``GasMonitor`` is polled once before each wallet's task batch.  If the
current gas price exceeds ``max_gas_price_gwei`` (from ``WalletConfig``),
the orchestrator suspends that wallet's coroutine and retries on a
configurable back-off schedule until gas normalises or a timeout expires.
A ``GAS_PAUSE`` notification is sent on first suspension; ``GAS_RESUME`` is
sent when execution proceeds.

Error handling contract
-----------------------
* Wallet-level exceptions (``BroadcastError``, ``CircuitOpenError``, etc.)
  are caught, logged, and recorded as ``WALLET_FAILURE`` events.
* Notification errors are caught inside ``_safe_notify()`` and never
  propagate upward — a Telegram outage must not stop the wallets.
* ``KeyboardInterrupt`` / ``asyncio.CancelledError`` trigger a graceful
  shutdown path that still sends a ``SYSTEM_SHUTDOWN`` notification.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from wallet_manager import actions
from wallet_manager.config import Settings, get_settings
from wallet_manager.core.interfaces import INotifier
from wallet_manager.core.manager import WalletManager
from wallet_manager.core.models import (
    EventKind,
    NotificationEvent,
    TxReceipt,
    WalletConfig,
)
from wallet_manager.exceptions import (
    GasPriceTooHighError,
    WalletManagerError,
)
from wallet_manager.portfolio_manager import (
    PortfolioManager,
    StrategyType,
)
from wallet_manager.scheduler import SchedulerManager, should_run_swap_this_cycle
from wallet_manager.telegram_notifier import StatusProvider
from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Environment-driven tuning constants (loaded from centralized config)
# ---------------------------------------------------------------------------


def _cfg() -> Settings:
    """Shorthand for the global settings singleton."""
    return get_settings()


_last_heartbeat_ts: float = 0.0

# Module-level constants derived from config (resolved once at import time)
_cfg_snapshot = get_settings()
_HEARTBEAT_INTERVAL_SECONDS: float = _cfg_snapshot.heartbeat_interval_seconds
_TASK_LOOP_CYCLES: int = _cfg_snapshot.default_task_loop_cycles
_GAS_POLL_INTERVAL: float = _cfg_snapshot.gas_poll_interval
_AAVE_APY_ALERT_THRESHOLD: float = _cfg_snapshot.aave_apy_alert_threshold_pct / 100.0
_ALGO_LOOP_INTERVAL_SECONDS: float = _cfg_snapshot.algo_loop_interval_seconds
del _cfg_snapshot  # avoid polluting module namespace with mutable reference

# ConceptDriftDetector singleton — created once at module load
try:
    from wallet_manager.strategies.wf_validator import ConceptDriftDetector as _ConceptDriftDetector
    _drift_detector = _ConceptDriftDetector()
    log.info("ConceptDriftDetector: initialized.")
except Exception as _cdd_exc:
    _drift_detector = None
    log.warning("ConceptDriftDetector: init failed: %s", _cdd_exc)

# ProactiveAlerter singleton — created once at module load
try:
    from wallet_manager.telegram_notifier import ProactiveAlerter as _ProactiveAlerter
    _proactive_alerter: "_ProactiveAlerter | None" = _ProactiveAlerter()
    log.info("ProactiveAlerter: initialized.")
except Exception as _pa_exc:
    _proactive_alerter = None
    log.warning("ProactiveAlerter: init failed: %s", _pa_exc)


# Module-level references for metrics server access
_runtime_portfolio_manager = None
_runtime_rpc_router = None


def _get_portfolio_manager():
    """Return the current runtime PortfolioManager (set during run())."""
    return _runtime_portfolio_manager


def _get_rpc_router():
    """Return the current RPCRouter instance (set during run())."""
    return _runtime_rpc_router


# Module-level wallet configs and circuit breakers for command handler access
_runtime_wallet_configs: list = []
_runtime_circuit_breakers: dict = {}


def _get_wallet_configs() -> list:
    """Return the current runtime wallet configs."""
    return _runtime_wallet_configs


def _get_circuit_breakers() -> dict:
    """Return the current runtime circuit breakers dict (wallet_id -> CB)."""
    return _runtime_circuit_breakers


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------


async def _validate_startup_config() -> None:
    """Run before the orchestrator begins any wallet work.

    Checklist:
    - Is VAULT_KEY set? (if vault is enabled)
    - Are TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID set?
    - Is at least one of INFURA_KEY or ALCHEMY_KEY set?
    - Does WALLET_CONFIG_PATH exist and parse successfully?
    - Is EXECUTION_MODE a valid value?
    - Does the ml_models/ directory exist? (in ml_quant mode)
    - Are model files present in ml_models/? (warning, not error)

    Raises:
        ConfigurationError: On critical failures.
    """
    from wallet_manager.exceptions import ConfigurationError

    cfg = _cfg()
    warnings: list[str] = []

    # Telegram credentials
    if not cfg.telegram_bot_token or not cfg.telegram_chat_id:
        warnings.append(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — notifications disabled."
        )

    # RPC keys
    if not cfg.infura_key and not cfg.alchemy_key:
        raise ConfigurationError(
            "At least one of INFURA_KEY or ALCHEMY_KEY must be set."
        )

    # Wallet config file
    wallet_path = cfg.wallet_config_path
    if not wallet_path.is_file():
        raise ConfigurationError(
            f"WALLET_CONFIG_PATH={wallet_path} does not exist or is not a file."
        )

    # EXECUTION_MODE is already validated in Settings.__post_init__

    # ML models check (warning only)
    if cfg.execution_mode == "ml_quant":
        ml_dir = Path(__file__).resolve().parent / "ml_models"
        if not ml_dir.is_dir():
            warnings.append(
                "ml_models/ directory not found. ML engine will use statistical fallback."
            )
        else:
            model_files = list(ml_dir.glob("*.json")) + list(ml_dir.glob("*.pt"))
            if not model_files:
                warnings.append(
                    "No model files in ml_models/. Run scripts/train_models.py first. "
                    "ML engine will use statistical fallback."
                )

    # Paper trading gate: require minimum paper trades before going live
    _exec_mode = cfg.execution_mode
    if _exec_mode in ("ml_quant", "robust", "robust_live"):
        try:
            from wallet_manager.exceptions import ConfigurationError as _CE
            from wallet_manager.strategies.paper_engine import get_paper_engine
            import asyncio as _asyncio
            _pe = await get_paper_engine()
            _pnl = await _pe._ledger.get_pnl_summary()
            _paper_trades = _pnl.get("trade_count", 0)
            _required = cfg.min_paper_trades_before_live
            if _paper_trades < _required:
                raise _CE(
                    f"EXECUTION_MODE={_exec_mode} requires at least {_required} "
                    f"completed paper trades (found: {_paper_trades}). "
                    f"Run with EXECUTION_MODE=robust_paper first."
                )
        except ConfigurationError:
            raise
        except Exception as _pe_exc:
            log.warning(
                "Paper trade gate check failed (%s) — proceeding. "
                "Ensure paper engine has sufficient trade history before going live.",
                _pe_exc,
            )

    # RPC + Aave connectivity checks
    try:
        from wallet_manager.utils.health_check import run_startup_health_checks
        health_problems = await run_startup_health_checks(cfg)
        for problem in health_problems:
            warnings.append(f"Health check: {problem}")
    except Exception as _hc_exc:
        log.warning("Startup health checks could not run: %s", _hc_exc)

    for w in warnings:
        log.warning(f"Startup check: {w}")

    # Send warnings via Telegram if possible
    if warnings and cfg.telegram_bot_token and cfg.telegram_chat_id:
        try:
            from wallet_manager.telegram_notifier import TelegramNotifier

            async with TelegramNotifier() as notifier:
                await notifier.send(
                    NotificationEvent(
                        kind=EventKind.SYSTEM_BOOT,
                        detail="⚠️ Startup warnings:\n" + "\n".join(f"• {w}" for w in warnings),
                    )
                )
        except Exception:
            pass  # Best-effort


# ---------------------------------------------------------------------------
# Vault migration
# ---------------------------------------------------------------------------


async def _migrate_wallets_to_vault(
    vault,
    config_path: Path,
) -> None:
    """Migrate plaintext private keys in wallets.json to Vault encryption.

    For every private_key field that is a plaintext hex string:
      1. Write backup: wallets.json.bak (before modifying anything)
      2. Encrypt the key with vault.encrypt()
      3. Replace the field value with the encrypted token (base64 str)
      4. Add field: "vault_encrypted": true
      5. Write the modified wallets.json atomically (write to .tmp, rename)
    Idempotent: skips any wallet where vault_encrypted is already true.
    Logs each migrated wallet_id (masked, never the key itself).
    """
    if not config_path.is_file():
        log.warning(f"_migrate_wallets_to_vault: {config_path} not found, skipping.")
        return

    raw = config_path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error(f"_migrate_wallets_to_vault: failed to parse {config_path}: {exc}")
        return

    wallets = data if isinstance(data, list) else [data]
    migrated = 0

    for wallet in wallets:
        if wallet.get("vault_encrypted"):
            continue
        key = wallet.get("private_key", "")
        if not key:
            continue
        # Write backup before first migration
        if migrated == 0:
            bak = config_path.with_suffix(".json.bak")
            bak.write_text(raw, encoding="utf-8")
            log.info(f"Vault migration: backup written to {bak}.")

        token_bytes = vault.encrypt(key)
        token_b64 = base64.b64encode(token_bytes).decode("ascii")
        wallet["private_key"] = token_b64
        wallet["vault_encrypted"] = True

        wallet_id = wallet.get("wallet_id", "<unknown>")
        log.info(f"Vault migration: encrypted key for wallet '{wallet_id}'.")
        migrated += 1

    if migrated == 0:
        log.debug("Vault migration: all wallets already encrypted, nothing to do.")
        return

    # Atomic write: .tmp then rename
    updated = json.dumps(data if not isinstance(data, list) else wallets, indent=2)
    tmp_path = config_path.with_suffix(".json.tmp")
    tmp_path.write_text(updated, encoding="utf-8")
    tmp_path.replace(config_path)
    log.info(f"Vault migration: {migrated} wallet(s) migrated and written atomically.")


# ---------------------------------------------------------------------------
# Gas monitor
# ---------------------------------------------------------------------------


class GasMonitor:
    """Async gas price monitor using the RPC endpoint of a reference wallet.

    Polls ``eth_gasPrice`` via web3.py and compares against a configurable
    ceiling.  Stateless between calls — each :meth:`current_gwei` invocation
    fires a fresh RPC request.

    Args:
        rpc_url: JSON-RPC endpoint used for gas price queries.
    """

    def __init__(self, rpc_url: str) -> None:
        self._rpc_url = rpc_url

    async def current_gwei(self) -> float:
        """Fetch the current network gas price in Gwei.

        Returns:
            Gas price as a float in Gwei.

        Raises:
            Exception: Propagates web3 / network errors to the caller.
        """
        from web3 import AsyncWeb3

        w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(self._rpc_url))
        wei: int = await w3.eth.gas_price
        return wei / 1e9

    async def is_safe(self, max_gwei: float) -> tuple[bool, float]:
        """Check whether the current gas price is within the safe ceiling.

        Args:
            max_gwei: Maximum acceptable gas price in Gwei.

        Returns:
            A ``(safe, current_gwei)`` tuple where ``safe`` is ``True``
            if the current price is ≤ *max_gwei*.
        """
        current = await self.current_gwei()
        return current <= max_gwei, current


# ---------------------------------------------------------------------------
# LiveStatusProvider — /status dashboard for Telegram
# ---------------------------------------------------------------------------


class LiveStatusProvider(StatusProvider):
    """Real-time status for /status command: full on-chain portfolio snapshot."""

    def __init__(
        self,
        configs: list[WalletConfig],
        scheduler: SchedulerManager,
        portfolio_manager: PortfolioManager | None = None,
    ) -> None:
        self._configs = configs
        self._scheduler = scheduler
        self._portfolio_manager = portfolio_manager

    async def build_status_message(self) -> str:
        """Build Markdown status from real on-chain balances via portfolio_manager."""
        if not self._configs:
            return "ℹ️ No wallets configured."

        config = self._configs[0]
        address = _get_wallet_address(config)
        chain_id = int(config.chain_id) if config.chain_id is not None else 0

        # --- Fetch all on-chain balances ---
        try:
            balance_wei = await actions.fetch_native_balance(config.rpc_url, address)
        except Exception:
            balance_wei = 0
        try:
            balance_usdc = await actions.fetch_usdc_balance(
                config.rpc_url, address, chain_id
            )
        except Exception:
            balance_usdc = 0
        try:
            eth_price = await actions.fetch_eth_price_usd()
        except Exception:
            eth_price = 3000.0

        # --- Fetch Aave deposited USDC ---
        try:
            from wallet_manager.strategies import yield_strategy
            aave_raw = await yield_strategy.get_aave_usdc_balance(config.rpc_url, address)
            aave_usdc = aave_raw / 1e6
        except Exception:
            aave_usdc = 0.0

        # --- Fetch short position value ---
        try:
            from wallet_manager.strategies import short_strategy
            short_state = short_strategy.get_short_position_state()
            short_usd = float(getattr(short_state, "position_value_usd", 0.0))
        except Exception:
            short_usd = 0.0

        # --- Sync portfolio manager and get snapshot ---
        if self._portfolio_manager is not None:
            await self._portfolio_manager.sync_from_chain(
                wallet_address=address,
                rpc_url=config.rpc_url,
                chain_id=chain_id,
                eth_price_usd=eth_price,
            )
            snapshot = self._portfolio_manager.get_snapshot()
            total_usd = snapshot.total_value_usd
        else:
            eth_usd = actions.eth_balance_usd_value(balance_wei, eth_price)
            total_usd = eth_usd + (balance_usdc / 1e6) + aave_usdc + short_usd

        eth_val = balance_wei / 1e18
        eth_usd = actions.eth_balance_usd_value(balance_wei, eth_price)
        usdc_val = balance_usdc / 1e6

        secs = self._scheduler.seconds_until_due(config.wallet_id)
        if secs >= 3600:
            next_str = f"{secs / 3600:.1f} hours later"
        elif secs >= 60:
            next_str = f"{secs / 60:.0f} mins later"
        elif secs > 0:
            next_str = f"{secs:.0f} secs later"
        else:
            next_str = "due now"

        return (
            "📊 *Portfolio Live Status*\n"
            f"💳 Wallet: {address}\n"
            f"🔹 ETH: {eth_val:.4f} ETH (~${eth_usd:.2f})\n"
            f"💵 Spot USDC: {usdc_val:.2f} USDC\n"
            f"🏦 Aave USDC: {aave_usdc:.2f} USDC\n"
            f"📉 Short Pos: ${short_usd:.2f}\n"
            f"💼 *Total Portfolio: ${total_usd:.2f}*\n"
            f"⏱️ Next Swap: {next_str}"
        )


# ---------------------------------------------------------------------------
# Per-wallet execution result
# ---------------------------------------------------------------------------


@dataclass
class WalletResult:
    """Summary of a single wallet's execution attempt.

    Attributes:
        wallet_id: Identifier of the wallet.
        success: ``True`` if all tasks completed without error.
        receipts: List of transaction receipts produced during the run.
        error: Exception caught during execution, if any.
        duration_seconds: Wall-clock time for this wallet's run.
    """

    wallet_id: str
    success: bool
    receipts: list[TxReceipt] = field(default_factory=list)
    error: Exception | None = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class WalletOrchestrator:
    """Top-level runtime coordinator for the 20-wallet asset management system.

    Wires together ``WalletManager``, ``GasMonitor``, and ``TelegramNotifier``
    (all injected) and drives the full execution lifecycle.

    Args:
        wallet_configs: Ordered list of wallets to process.
        task_factory: Async callable ``(manager, wallet_id) → list[TxReceipt]``
            that defines the work performed for each wallet.  Inject this to
            keep the orchestrator decoupled from business logic.
        notifier: Optional :class:`~wallet_manager.core.interfaces.INotifier`.
            When ``None``, a :class:`~wallet_manager.telegram_notifier.TelegramNotifier`
            is constructed from environment variables.
        settings: Optional :class:`~wallet_manager.config.Settings` override.
        max_concurrent: Maximum wallets executing simultaneously.
        gas_poll_interval: Seconds between gas re-checks while paused.
        gas_wait_timeout: Max seconds to wait for gas before skipping a wallet.

    Example:
        .. code-block:: python

            async def my_tasks(mgr, wallet_id):
                balance = await mgr.get_safe_balance(wallet_id)
                # ... build and send transactions ...
                return receipts

            orchestrator = WalletOrchestrator(configs, task_factory=my_tasks)
            results = await orchestrator.run()
    """

    def __init__(
        self,
        wallet_configs: list[WalletConfig],
        task_factory,  # Callable[[WalletManager, str, dict], Awaitable[list[TxReceipt]]]
        *,
        notifier: INotifier | None = None,
        settings: Settings | None = None,
        max_concurrent: int = 0,
        gas_poll_interval: float = 0.0,
        gas_wait_timeout: float = 0.0,
        command_state=None,
        confirmation_gate=None,
    ) -> None:
        self._configs = wallet_configs
        self._task_factory = task_factory
        self._command_state = command_state
        self._confirmation_gate = confirmation_gate
        self._scheduler = SchedulerManager()
        self._settings = settings or get_settings()
        self._max_concurrent = max_concurrent or self._settings.max_concurrent_wallets
        self._gas_poll_interval = gas_poll_interval or self._settings.gas_poll_interval
        self._gas_wait_timeout = gas_wait_timeout or self._settings.gas_wait_timeout

        # Notifier resolved lazily in run() so the event loop is already running
        self._notifier_override: INotifier | None = notifier

        # Use the first wallet's RPC as the gas reference endpoint
        if not wallet_configs:
            raise ValueError("wallet_configs must contain at least one entry.")
        self._gas_monitor = GasMonitor(rpc_url=wallet_configs[0].rpc_url)

        log.debug(
            f"WalletOrchestrator created: {len(wallet_configs)} wallets, concurrency={max_concurrent}.",
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> list[WalletResult]:
        """Execute all configured wallets and return per-wallet results.

        Lifecycle
        ---------
        1. Resolve and start the notifier.
        2. Open the ``WalletManager`` async context.
        3. Send ``SYSTEM_BOOT``.
        4. Run all wallets concurrently under the semaphore.
        5. Send ``SYSTEM_SHUTDOWN`` with a summary.
        6. Stop the notifier (drains queue).

        Returns:
            List of :class:`WalletResult` objects, one per wallet, in the
            order they were provided to the constructor.

        Raises:
            Exception: Only if an unrecoverable error occurs at the system
                level (not wallet-level; those are captured in results).
        """
        notifier = self._resolve_notifier()
        command_center = None

        async with notifier:
            # Start CommandCenter if notifier supports it and we have command_state
            if self._command_state is not None and hasattr(notifier, "build_command_center"):
                status_provider = LiveStatusProvider(
                    configs=self._configs,
                    scheduler=self._scheduler,
                    portfolio_manager=None,  # updated after portfolio_manager is created below
                )
                command_center = notifier.build_command_center(
                    self._command_state,
                    confirmation_gate=self._confirmation_gate,
                    status_provider=status_provider,
                )
                await command_center.start()
            await self._safe_notify(
                notifier,
                NotificationEvent(kind=EventKind.SYSTEM_BOOT),
            )

            results: list[WalletResult] = []
            portfolio_manager = PortfolioManager()
            analytics_db = None
            try:
                from wallet_manager.resilience.analytics import AnalyticsDB

                _adb = AnalyticsDB()
                await _adb.open()
                analytics_db = _adb
            except Exception as exc:  # noqa: BLE001
                log.info(f"AnalyticsDB unavailable ({type(exc).__name__}), continuing without analytics.")

            try:
                async with WalletManager(
                    self._configs, settings=self._settings
                ) as manager:
                    # Sync real on-chain balances before any strategy decisions
                    if self._configs:
                        try:
                            _first = self._configs[0]
                            _addr = _get_wallet_address(_first)
                            _chain_id = int(_first.chain_id) if _first.chain_id is not None else 0
                            try:
                                _eth_price = await actions.fetch_eth_price_usd()
                            except Exception:
                                _eth_price = 3000.0
                            try:
                                await portfolio_manager.sync_from_chain(
                                    wallet_address=_addr,
                                    rpc_url=_first.rpc_url,
                                    chain_id=_chain_id,
                                    eth_price_usd=_eth_price,
                                )
                            except Exception as _exc:
                                log.warning(f"sync_from_chain failed: {_exc}; continuing with zero balances.")
                        except Exception as _addr_exc:
                            log.warning(f"Could not resolve wallet address: {_addr_exc}; skipping balance sync.")

                    # Expose runtime references for metrics server
                    global _runtime_portfolio_manager
                    _runtime_portfolio_manager = portfolio_manager

                    # Launch metrics server background task
                    try:
                        from wallet_manager.metrics_server import start_metrics_server
                        asyncio.create_task(
                            start_metrics_server(),
                            name="metrics_server",
                        )
                    except Exception as _ms_exc:
                        log.warning("MetricsServer: startup failed: %s", _ms_exc)

                    # Launch AaveAPYMonitor background task
                    if self._configs:
                        try:
                            from wallet_manager.strategies.yield_strategy import get_apy_monitor
                            _apy_rpc = self._configs[0].rpc_url
                            asyncio.create_task(
                                get_apy_monitor().poll_forever(_apy_rpc),
                                name="aave_apy_monitor",
                            )
                            log.info("AaveAPYMonitor: background poll task started.")
                        except Exception as _apy_exc:
                            log.warning("AaveAPYMonitor: failed to start: %s", _apy_exc)

                    # ── ProactiveAlerter 5-minute background loop ─────────────────────────
                    if _proactive_alerter is not None and notifier is not None:
                        async def _proactive_alert_loop() -> None:
                            _loop_cfg = get_settings()
                            _interval = getattr(_loop_cfg, "proactive_alert_interval_seconds", 300)
                            while True:
                                await asyncio.sleep(_interval)
                                try:
                                    _snapshot = None
                                    try:
                                        _snapshot = portfolio_manager.get_snapshot()
                                    except Exception:
                                        pass
                                    _pa_inputs = await _collect_alerter_inputs(_snapshot, get_settings())
                                    await _proactive_alerter.check_and_alert(
                                        notifier=notifier,
                                        **_pa_inputs,
                                    )
                                except Exception as _loop_exc:
                                    log.debug("ProactiveAlerter loop iteration failed: %s", _loop_exc)

                        asyncio.create_task(        # ← create_task, NOT ensure_future
                            _proactive_alert_loop(),
                            name="proactive_alerter_loop",
                        )
                        log.info("ProactiveAlerter: 5-minute background loop started.")

                    semaphore = asyncio.Semaphore(self._max_concurrent)
                    task_context_base = {
                        "notifier": notifier,
                        "confirmation_gate": self._confirmation_gate,
                        "command_state": self._command_state,
                        "scheduler": self._scheduler,
                        "portfolio_manager": portfolio_manager,
                        "analytics_db": analytics_db,
                        "trade_notifications": [],
                    }
                    tasks = [
                        self._run_wallet(manager, cfg, semaphore, notifier, task_context_base)
                        for cfg in self._configs
                    ]
                    # gather preserves order; return_exceptions ensures one
                    # wallet failure never cancels the rest.
                    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

                    for i, outcome in enumerate(raw_results):
                        if isinstance(outcome, BaseException):
                            # Unhandled exception from _run_wallet itself
                            # (should be rare — _run_wallet catches internally)
                            results.append(WalletResult(
                                wallet_id=self._configs[i].wallet_id,
                                success=False,
                                error=outcome,
                            ))
                        else:
                            results.append(outcome)

            except (KeyboardInterrupt, asyncio.CancelledError):
                log.info("Orchestrator interrupted — sending shutdown notification.")
                await self._safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.SYSTEM_SHUTDOWN,
                        detail="Interrupted by operator signal.",
                    ),
                )
                raise

            if analytics_db is not None:
                try:
                    sc = sum(1 for r in results if r.success)
                    await analytics_db.close_session(
                        total_wallets=len(results),
                        success_count=sc,
                        fail_count=len(results) - sc,
                    )
                    await analytics_db.close()
                except Exception as exc:  # noqa: BLE001
                    log.info(f"AnalyticsDB close: {type(exc).__name__}: {exc}")

            # --- Heartbeat every 6 hours (MasterOrchestrator handles) ---
            if hasattr(self, "_maybe_send_heartbeat"):
                await self._maybe_send_heartbeat(
                    portfolio_manager, analytics_db, notifier
                )

            # --- Build shutdown summary ---
            success_count = sum(1 for r in results if r.success)
            fail_count = len(results) - success_count
            summary = (
                f"{success_count}/{len(results)} wallets succeeded, "
                f"{fail_count} failed."
            )
            log.info(f"Orchestrator run complete. {summary}")

            await self._safe_notify(
                notifier,
                NotificationEvent(
                    kind=EventKind.SYSTEM_SHUTDOWN,
                    detail=summary,
                ),
            )

        if command_center is not None:
            await command_center.stop()

        return results

    # ------------------------------------------------------------------
    # Per-wallet coroutine
    # ------------------------------------------------------------------

    async def _run_wallet(
        self,
        manager: WalletManager,
        config: WalletConfig,
        semaphore: asyncio.Semaphore,
        notifier: INotifier,
        task_context: dict | None = None,
    ) -> WalletResult:
        """Execute one wallet's task batch, with gas gating and notifications.

        This coroutine is designed to be *error-isolated*: any exception
        is caught, recorded, and returned as a ``WalletResult`` with
        ``success=False``.

        Args:
            manager: Active ``WalletManager`` context.
            config: Configuration for the wallet being processed.
            semaphore: Shared concurrency limiter.
            notifier: Active notification channel.

        Returns:
            :class:`WalletResult` summarising the outcome.
        """
        async with semaphore:
            wallet_id = config.wallet_id
            start_time = time.monotonic()

            # CommandState: stop or pause
            if self._command_state is not None:
                if self._command_state.is_stopped:
                    log.info("CommandState: STOP requested, skipping remaining wallets.")
                    return WalletResult(
                        wallet_id=wallet_id,
                        success=False,
                        error=Exception("Stopped by operator"),
                    )
                await self._command_state.wait_for_resume()

            log.info(f"Starting wallet '{wallet_id}'.")

            try:
                # --- Gas gate: wait until gas is safe ---
                await self._wait_for_safe_gas(config, notifier)

                # --- Execute the injected task batch ---
                ctx = task_context or {}
                _portfolio_mgr = ctx.get("portfolio_manager")
                # Build kwargs without conflicting positional arg names
                _extra = {k: v for k, v in ctx.items()
                          if k not in ("portfolio_manager",)}
                receipts: list[TxReceipt] = await self._task_factory(
                    config, manager, self._settings, _portfolio_mgr, **_extra
                )

                duration = time.monotonic() - start_time
                log.info(
                    f"Wallet '{wallet_id}' completed in {duration:.2f}s ({len(receipts)} TX).",
                )

                # Notification: success
                tx_hash = receipts[-1].tx_hash if receipts else None
                await self._safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.WALLET_SUCCESS,
                        wallet_id=wallet_id,
                        tx_hash=tx_hash,
                    ),
                )

                return WalletResult(
                    wallet_id=wallet_id,
                    success=True,
                    receipts=receipts,
                    duration_seconds=duration,
                )

            except GasPriceTooHighError as exc:
                # Gas too high — smart deferral: reschedule 2-4h, do not treat as failure
                self._scheduler.schedule_gas_deferral(wallet_id)
                log.info(
                    f"Wallet '{wallet_id}': gas {exc.requested_gwei:.2f} Gwei > {exc.max_gwei:.2f}. "
                    "Rescheduled for 2-4h.",
                )
                await self._safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.GAS_DEFERRED,
                        wallet_id=wallet_id,
                        gas_price_gwei=exc.requested_gwei,
                    ),
                )
                return WalletResult(
                    wallet_id=wallet_id,
                    success=False,
                    error=exc,
                    duration_seconds=time.monotonic() - start_time,
                )

            except WalletManagerError as exc:
                reason = f"{type(exc).__name__}: {exc.message}"
                log.error(f"Wallet '{wallet_id}' failed: {reason}")
                await self._safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.WALLET_FAILURE,
                        wallet_id=wallet_id,
                        detail=reason,
                    ),
                )
                return WalletResult(
                    wallet_id=wallet_id,
                    success=False,
                    error=exc,
                    duration_seconds=time.monotonic() - start_time,
                )

            except Exception as exc:  # noqa: BLE001 — broad catch intentional for isolation
                reason = f"Unexpected error: {type(exc).__name__}: {exc}"
                log.exception(f"Wallet '{wallet_id}' raised unhandled exception.")
                await self._safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.WALLET_FAILURE,
                        wallet_id=wallet_id,
                        detail=reason,
                    ),
                )
                return WalletResult(
                    wallet_id=wallet_id,
                    success=False,
                    error=exc,
                    duration_seconds=time.monotonic() - start_time,
                )

    # ------------------------------------------------------------------
    # Gas gating
    # ------------------------------------------------------------------

    async def _wait_for_safe_gas(
        self,
        config: WalletConfig,
        notifier: INotifier,
    ) -> None:
        """Suspend the current wallet coroutine until gas is within limits.

        Sends a ``GAS_PAUSE`` notification on first suspension and a
        ``GAS_RESUME`` notification when execution proceeds.

        Args:
            config: Wallet config carrying the ``max_gas_price_gwei`` ceiling.
            notifier: Active notification channel.

        Raises:
            GasPriceTooHighError: If gas has not normalised within
                ``_gas_wait_timeout`` seconds.
        """
        try:
            safe, current = await self._gas_monitor.is_safe(float(config.max_gas_price_gwei))
        except Exception as _gas_exc:
            log.debug("Gas monitor unavailable (%s); proceeding as safe.", _gas_exc)
            return  # default to safe when monitor is unreachable
        if safe:
            return

        # --- First check failed: enter polling loop ---
        log.info(
            f"Gas too high for wallet '{config.wallet_id}': {current:.2f} Gwei (max {config.max_gas_price_gwei:.2f}). Pausing.",
        )
        await self._safe_notify(
            notifier,
            NotificationEvent(
                kind=EventKind.GAS_PAUSE,
                wallet_id=config.wallet_id,
                gas_price_gwei=float(current),
                detail=(
                    f"Gas {float(current):.2f} Gwei exceeds ceiling "
                    f"{float(config.max_gas_price_gwei):.2f} Gwei."
                ),
            ),
        )

        deadline = time.monotonic() + self._gas_wait_timeout

        while time.monotonic() < deadline:
            await asyncio.sleep(self._gas_poll_interval)
            safe, current = await self._gas_monitor.is_safe(float(config.max_gas_price_gwei))

            if safe:
                log.info(
                    f"Gas normalised for wallet '{config.wallet_id}': {current:.2f} Gwei. Resuming.",
                )
                await self._safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.GAS_RESUME,
                        wallet_id=config.wallet_id,
                        gas_price_gwei=float(current),
                    ),
                )
                return

            log.debug(
                f"Gas still high for wallet '{config.wallet_id}': {current:.2f} Gwei. Retrying in {self._gas_poll_interval:.0f}s.",
            )

        # --- Timeout exceeded ---
        raise GasPriceTooHighError(
            f"Gas wait timeout ({self._gas_wait_timeout:.0f}s) exceeded "
            f"for wallet '{config.wallet_id}'.",
            wallet_id=config.wallet_id,
            requested_gwei=float(current),
            max_gwei=float(config.max_gas_price_gwei),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_notifier(self) -> INotifier:
        """Return the injected notifier or construct the default.

        Returns:
            An :class:`INotifier` instance ready to be used as an async
            context manager.
        """
        if self._notifier_override is not None:
            return self._notifier_override

        from wallet_manager.telegram_notifier import TelegramNotifier

        return TelegramNotifier()

    @staticmethod
    async def _safe_notify(notifier: INotifier, event: NotificationEvent) -> None:
        """Fire-and-forget wrapper that swallows all notifier exceptions.

        A Telegram outage or misconfiguration must never interrupt wallet
        execution.  All errors are logged at WARNING level.

        Args:
            notifier: The active notifier instance.
            event: Event to deliver.
        """
        try:
            await notifier.send(event)
        except Exception as exc:  # noqa: BLE001
            log.info(
                f"Notification for {event.kind.name} skipped: {type(exc).__name__}: {exc}",
            )


# ---------------------------------------------------------------------------
# 6-Hour Heartbeat Telemetry
# ---------------------------------------------------------------------------


async def _collect_alerter_inputs(
    snapshot,           # PortfolioSnapshot — may be None
    cfg,                # Settings instance
) -> dict:
    """Collect all data ProactiveAlerter.check_and_alert() needs."""
    import time as _time

    # 1. Drift level
    _drift_name = "NOMINAL"
    try:
        if _drift_detector is not None and hasattr(
            _drift_detector, "_last_drift_level"
        ):
            _drift_name = _drift_detector._last_drift_level.name
    except Exception:
        pass

    # 2. Aave APY
    _apy = 0.0
    try:
        from wallet_manager.strategies.yield_strategy import get_apy_monitor
        _apy = get_apy_monitor().last_apy
    except Exception:
        pass

    # 3. Model staleness (days since ppo_model_path was last written)
    _staleness = 0.0
    try:
        import os as _os
        _model_path = cfg.ppo_model_path
        if _os.path.exists(_model_path):
            _staleness = (_time.time() - _os.path.getmtime(_model_path)) / 86400.0
    except Exception:
        pass

    # 4. RPC latency — worst-case across all tracked endpoints
    _rpc_latency = 0.0
    try:
        from wallet_manager.network.router import get_router
        _latency_tracker = getattr(get_router(), "_latency_tracker", {})
        _all = [
            max(v) for v in _latency_tracker.values()
            if isinstance(v, (list, tuple)) and v
        ]
        if _all:
            _rpc_latency = max(_all)
    except Exception:
        pass

    # 5. Current drawdown vs peak allocation
    _drawdown = 0.0
    try:
        if snapshot is not None and snapshot.total_value_usd > 0:
            _peak = max(
                (a.current_usd for a in snapshot.allocations), default=0.0
            )
            _drawdown = max(0.0, 1.0 - snapshot.total_value_usd / max(_peak, 1.0))
    except Exception:
        pass

    return dict(
        drift_level=_drift_name,
        apy=_apy,
        model_staleness_days=_staleness,
        rpc_latency_ms=_rpc_latency,
        drawdown=_drawdown,
    )


async def send_heartbeat_telemetry(
    portfolio_manager: PortfolioManager,
    rpc_url: str,
    address: str,
    task_context: dict,
    analytics_db=None,
) -> None:
    """Send 6-hour heartbeat telemetry via Telegram.

    Reports: Total Equity, Aave Yield, Realized P&L, Gas Consumption,
    Paper trade stats, and portfolio allocation breakdown.
    """
    global _last_heartbeat_ts
    now = time.time()
    if now - _last_heartbeat_ts < _HEARTBEAT_INTERVAL_SECONDS:
        return
    _last_heartbeat_ts = now

    try:
        snapshot = portfolio_manager.get_snapshot()

        # Aave stats — graceful fallback
        aave_apy = 0.0
        aave_balance = 0
        try:
            from wallet_manager.strategies import yield_strategy
            aave_apy = await yield_strategy.get_apy(rpc_url)
            aave_balance = await yield_strategy.get_aave_usdc_balance(rpc_url, address)
        except Exception:
            pass

        # Short position state — graceful fallback
        short_pnl = 0.0
        short_active = False
        with contextlib.suppress(Exception):
            from wallet_manager.strategies.short_strategy import get_short_position_state
            short_state = get_short_position_state()
            short_pnl = short_state.total_short_pnl_usd
            short_active = short_state.position.is_active

        # Gas consumption from analytics
        gas_consumed_usd = 0.0
        if analytics_db and hasattr(analytics_db, "get_total_gas_usd"):
            with contextlib.suppress(Exception):
                gas_consumed_usd = await analytics_db.get_total_gas_usd()

        # Paper trade stats
        paper_lines: list[str] = []
        _exec_mode = _cfg().execution_mode
        if _exec_mode == "robust_paper":
            try:
                from wallet_manager.strategies.paper_engine import get_paper_engine
                paper_engine = await get_paper_engine()
                eth_price = await actions.fetch_eth_price_usd()
                paper_equity = await paper_engine._ledger.get_total_equity_usd(eth_price=eth_price)
                pnl_summary = await paper_engine._ledger.get_pnl_summary()
                starting = paper_engine._config.initial_balance_usd
                change_pct = ((paper_equity - starting) / starting * 100) if starting > 0 else 0
                paper_lines = [
                    "",
                    "\U0001f4b0 *Paper Trading:*",
                    f"  Balance: ${paper_equity:.2f} ({change_pct:+.1f}% vs ${starting:.0f})",
                    f"  Trades: {pnl_summary.get('trade_count', 0)} "
                    f"(Win: {pnl_summary.get('win_rate', 0):.0f}%)",
                    f"  Realized PnL: ${pnl_summary.get('realized_pnl', 0):+.2f}",
                    f"  Max DD: {pnl_summary.get('max_drawdown', 0):.2%}",
                ]
            except Exception:
                pass

        # Build heartbeat message
        lines = [
            "\U0001f493 *6h Heartbeat Telemetry*",
            f"\U0001f4ca Total Equity: ${snapshot.total_value_usd:.2f}",
            "",
            "\U0001f4cb *Pillar Breakdown:*",
        ]
        for alloc in snapshot.allocations:
            pct = (alloc.current_usd / snapshot.total_value_usd * 100) if snapshot.total_value_usd > 0 else 0
            lines.append(
                f"  {alloc.strategy.value:>7}: ${alloc.current_usd:.2f} ({pct:.1f}%)"
            )
        lines.extend([
            "",
            f"\U0001f3e6 Aave Yield: ${aave_balance / 1e6:.2f} USDC @ {aave_apy:.2f}% APY",
            f"\U0001f4c8 Short P&L: ${short_pnl:+.2f} "
            f"({'ACTIVE' if short_active else 'no position'})",
            f"\u26fd Gas Consumed: ${gas_consumed_usd:.2f}",
            f"\U0001f504 Rebalance needed: {'Yes' if snapshot.needs_rebalance else 'No'}",
        ])
        lines.extend(paper_lines)

        # Include paper_report if provided
        paper_report = task_context.get("paper_report")
        if paper_report:
            lines.extend(["", "\U0001f4dd *Paper Report:*", paper_report])

        message = "\n".join(lines)

        # Send via TelegramManager
        tg = task_context.get("telegram_manager")
        if tg and hasattr(tg, "_send"):
            await tg._send(message, parse_mode="Markdown")
            log.info("Heartbeat telemetry sent.")
        else:
            # Fallback via basic notifier
            from wallet_manager.core.models import EventKind, NotificationEvent
            notifier = task_context.get("notifier")
            if notifier:
                await notifier.send(
                    NotificationEvent(
                        kind=EventKind.SYSTEM_BOOT,
                        detail=message,
                    )
                )

        # ── ProactiveAlerter: evaluate all alert conditions ──────────────────
        if _proactive_alerter is not None:
            try:
                _pa_notifier = task_context.get("notifier")
                if _pa_notifier is not None:
                    _pa_inputs = await _collect_alerter_inputs(snapshot, _cfg())
                    await _proactive_alerter.check_and_alert(
                        notifier=_pa_notifier,
                        **_pa_inputs,
                    )
            except Exception as _alert_exc:
                log.debug("ProactiveAlerter heartbeat check failed: %s", _alert_exc)

    except Exception as exc:
        log.info(f"Heartbeat telemetry skipped: {type(exc).__name__}: {exc}")


class _SystemAlertNotifierWrapper(INotifier):
    """Wraps INotifier and forwards system alerts to TelegramManager."""

    def __init__(self, inner: INotifier, telegram_manager: object | None) -> None:
        self._inner = inner
        self._tg = telegram_manager

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    async def __aenter__(self) -> _SystemAlertNotifierWrapper:
        await self._inner.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._inner.stop()

    async def send(self, event: NotificationEvent) -> None:
        await self._inner.send(event)
        if self._tg and event.kind in (
            EventKind.GAS_PAUSE,
            EventKind.GAS_DEFERRED,
            EventKind.WALLET_FAILURE,
        ):
            try:
                from wallet_manager.security.telegram_manager import (
                    SystemAlertPayload,
                )
                alert_type = event.kind.name
                detail = event.detail or ""
                if event.gas_price_gwei is not None:
                    detail = f"Gas: {event.gas_price_gwei:.2f} Gwei. {detail}"
                await self._tg.send_system_alert(
                    SystemAlertPayload(
                        alert_type=alert_type,
                        detail=detail.strip(),
                        severity="critical" if event.kind == EventKind.WALLET_FAILURE else "warning",
                    )
                )
            except Exception as exc:
                log.info(f"System alert forward skipped: {type(exc).__name__}: {exc}")

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self) -> None:
        await self._inner.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def validate_wallets_schema(data: list[dict]) -> None:
    """Validate each wallet entry in a parsed wallets.json data list.

    Required fields per wallet entry:
      wallet_id:              str (non-empty, ^[a-zA-Z0-9_-]+$)
      address:                str (valid checksummed Ethereum address)
      private_key OR
        encrypted_private_key: str (non-empty)
      max_gas_price_gwei:     float (0 < value < 10000)
      enabled:                bool (optional, default True)

    All errors are collected before raising so the caller sees every problem
    at once rather than having to fix one at a time.

    On validation failure:
      Raise ConfigurationError with all accumulated errors.
    """
    import re as _re
    import base64 as _base64
    from wallet_manager.exceptions import ConfigurationError

    _WALLET_ID_RE = _re.compile(r"^[a-zA-Z0-9_-]+$")
    all_errors: list[str] = []

    for idx, item in enumerate(data):
        # Skip non-dict entries (e.g., _vault_meta)
        if not isinstance(item, dict) or "_vault_meta" in item:
            continue

        errors: list[str] = []

        # wallet_id
        wid = str(item.get("wallet_id", "")).strip()
        masked_wid = f"{wid[:6]}...{wid[-4:]}" if len(wid) >= 10 else wid or f"#{idx}"
        if not wid:
            errors.append(f"entry #{idx}: 'wallet_id' is missing or empty")
        elif not _WALLET_ID_RE.match(wid):
            errors.append(f"wallet '{masked_wid}': 'wallet_id' must match ^[a-zA-Z0-9_-]+$")

        # address (required, checksummed)
        address = item.get("address")
        if address is None:
            errors.append(f"wallet '{masked_wid}': 'address' is required")
        else:
            addr_str = str(address).strip()
            valid_addr = False
            if addr_str.startswith("0x") and len(addr_str) == 42:
                try:
                    import contextlib as _cl
                    # Try web3 checksum validation if available
                    from web3 import Web3
                    if Web3.is_checksum_address(addr_str):
                        valid_addr = True
                    else:
                        errors.append(
                            f"wallet '{masked_wid}': 'address' is not a valid checksummed Ethereum address"
                        )
                except ImportError:
                    valid_addr = True  # web3 not available, accept any 0x-prefixed 42-char address
            if not valid_addr and not errors or (errors and "checksummed" not in errors[-1] if errors else True):
                if not (addr_str.startswith("0x") and len(addr_str) == 42):
                    errors.append(
                        f"wallet '{masked_wid}': 'address' must be a 42-char 0x-prefixed hex string"
                    )

        # private_key or encrypted_private_key
        pk = item.get("private_key") or item.get("encrypted_private_key")
        if not pk or not str(pk).strip():
            errors.append(f"wallet '{masked_wid}': 'private_key' or 'encrypted_private_key' is required and non-empty")
        else:
            # Validate base64 if it looks like an encrypted key (not a raw hex key)
            pk_str = str(pk).strip()
            if not all(c in "0123456789abcdefABCDEF" for c in pk_str.lstrip("0x")):
                try:
                    _base64.b64decode(pk_str, validate=True)
                except Exception:
                    errors.append(f"wallet '{masked_wid}': 'encrypted_private_key' is not valid base64")

        # max_gas_price_gwei
        max_gas = item.get("max_gas_price_gwei")
        if max_gas is not None:
            try:
                max_gas_f = float(max_gas)
                if not (0 < max_gas_f < 10000):
                    errors.append(
                        f"wallet '{masked_wid}': 'max_gas_price_gwei' must be in (0, 10000); got {max_gas_f}"
                    )
            except (ValueError, TypeError):
                errors.append(f"wallet '{masked_wid}': 'max_gas_price_gwei' must be a number")

        # enabled (optional bool)
        enabled = item.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            errors.append(f"wallet '{masked_wid}': 'enabled' must be a boolean")

        all_errors.extend(errors)

    if all_errors:
        raise ConfigurationError(
            "wallets.json validation failed:\n  - " + "\n  - ".join(all_errors)
        )


def _load_wallet_configs() -> list[WalletConfig]:
    """Load wallet configurations from JSON file at WALLET_CONFIG_PATH.

    JSON format: list of objects with wallet_id, private_key (plain hex),
    rpc_url, chain_id; optional: max_gas_price_gwei, address.
    Private keys are read as plain text (no encryption).

    Returns:
        List of :class:`WalletConfig` instances.

    Raises:
        ConfigurationError: If WALLET_CONFIG_PATH is unset or file invalid.
    """
    import json

    from wallet_manager.exceptions import ConfigurationError

    path = str(_cfg().wallet_config_path)
    if not path or not os.path.isfile(path):
        raise ConfigurationError(
            "WALLET_CONFIG_PATH must point to a JSON file with wallet configs. "
            "See .env.example for format."
        )

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    validate_wallets_schema(raw)

    configs: list[WalletConfig] = []
    for item in raw:
        wid = item.get("wallet_id", "?")
        pk = item.get("private_key")
        if "encrypted_key" in item:
            raise ConfigurationError(
                f"Wallet '{wid}' has 'encrypted_key' but vault is disabled. "
                "Use 'private_key' with plain text hex key instead."
            )
        if not pk or not str(pk).strip():
            raise ConfigurationError(
                f"Wallet '{wid}' must have 'private_key' (plain text hex)."
            )
        configs.append(
            WalletConfig(
                wallet_id=item["wallet_id"],
                private_key=str(pk).strip(),
                rpc_url=item["rpc_url"],
                chain_id=int(item["chain_id"]),
                max_gas_price_gwei=float(item.get("max_gas_price_gwei", 100.0)),
                address=item.get("address"),
            )
        )
    return configs


async def _default_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """Task batch with active gas monitoring loop.

    Runs multiple cycles per wallet: each cycle waits for safe gas,
    fetches balance, then sleeps. Configurable via DEFAULT_TASK_LOOP_CYCLES.
    Replace with your own task_factory for production transactions.
    """
    config = manager.get_config(wallet_id)
    gas_monitor = GasMonitor(rpc_url=config.rpc_url)
    max_gwei = float(config.max_gas_price_gwei)
    receipts: list[TxReceipt] = []

    for cycle in range(_TASK_LOOP_CYCLES):
        safe, current = await gas_monitor.is_safe(max_gwei)
        if not safe:
            log.info(
                f"Wallet '{wallet_id}' cycle {cycle + 1}/{_TASK_LOOP_CYCLES}: gas {current:.2f} Gwei > max {max_gwei:.2f}, waiting.",
            )
            await asyncio.sleep(_GAS_POLL_INTERVAL)
            continue

        try:
            await manager.get_safe_balance(wallet_id)
        except Exception as exc:  # noqa: BLE001
            log.info(f"Wallet '{wallet_id}' cycle {cycle + 1}: balance fetch skipped ({type(exc).__name__})")
            continue

        if cycle < _TASK_LOOP_CYCLES - 1:
            await asyncio.sleep(_GAS_POLL_INTERVAL)

    return receipts


def _get_wallet_address(config: WalletConfig) -> str:
    """Return the Ethereum address for the wallet."""
    if config.address:
        return config.address
    from eth_account import Account
    return Account.from_key(config.private_key).address


async def _robust_paper_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """Paper Trade task factory — high-fidelity simulation with full Robust pipeline.

    Runs the entire Robust execution pipeline (HMM regime filter, fracdiff alpha,
    walk-forward validation, Monte Carlo stress test, liquidity profiling, latency
    shield) but stops BEFORE send_transaction. Simulates slippage and gas costs.
    Records to a virtual ledger (paper_trades.db).

    EXECUTION_MODE=robust_paper activates this factory.
    """
    from wallet_manager.strategies.paper_engine import get_paper_engine
    from wallet_manager.strategies.stochastic_stress_test import get_stress_tester
    from wallet_manager.strategies.latency_shield import get_latency_shield
    from wallet_manager.strategies.regime_filter import get_regime_filter
    from wallet_manager.strategies.robust_alpha import get_robust_alpha_engine

    ctx = task_context or {}
    portfolio_manager = ctx.get("portfolio_manager")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if portfolio_manager is None:
        log.info("PaperTrade: No portfolio_manager in context — falling back to airdrop only.")
        return await _airdrop_task_factory(manager, wallet_id, task_context)

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    address = _get_wallet_address(config)
    rpc_url = config.rpc_url

    if chain_id != 8453:
        log.info("PaperTrade: Skipping — Base Mainnet only.")
        return receipts

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gas = config.max_gas_price_gwei if config.max_gas_price_gwei is not None else 100.0
    max_gwei = min(25.0, float(max_gas))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)

    try:
        # Initialize engines
        paper_engine = await get_paper_engine()
        stress_tester = get_stress_tester()
        latency_shield = get_latency_shield(rpc_url)
        regime_engine = get_regime_filter()
        robust_engine = get_robust_alpha_engine()

        # 1. Latency shield — validate quote freshness
        eth_price = await actions.fetch_eth_price_usd()
        quote_validation, latency_profile = await latency_shield.apply_latency_shield(
            price=eth_price,
            quote_timestamp=time.time(),
        )

        if quote_validation.should_invalidate_signals:
            log.info(
                f"PAPER: Stale quote ({quote_validation.quote_age_s:.1f}s) — "
                f"signals invalidated. Reason: {quote_validation.reason_code}"
            )
            return receipts

        # 2. Regime filter
        regime_engine.ingest_market_data(
            price=eth_price, volume=0.0, timestamp=time.time(),
        )
        override_to_safety, safety_reason = regime_engine.should_override_to_safety()
        if override_to_safety:
            log.info(
                f"PAPER REGIME OVERRIDE: {safety_reason} — "
                f"Would move to 100% USDC + Aave. Skipping alpha."
            )
            return receipts

        # 3. Robust alpha signal
        robust_signal = robust_engine.ingest_price(
            price=eth_price, volume=0.0, timestamp=time.time(),
        )

        # 4. Monte Carlo stress test — chaos validation
        if robust_signal is not None and (
            robust_signal.should_enter_long or robust_signal.should_enter_short
        ):
            price_history = list(robust_engine._price_history)
            direction = 1.0 if robust_signal.should_enter_long else -1.0
            verdict = stress_tester.validate_signal(
                current_price=eth_price,
                signal_alpha=robust_signal.robust_alpha,
                signal_direction=direction,
                price_history=price_history,
            )
            if not verdict.approved:
                log.info(
                    f"PAPER STRESS TEST REJECTED: "
                    f"MC win_rate={verdict.monte_carlo.win_rate:.1%} "
                    f"reasons={verdict.rejection_reasons}"
                )
                return receipts
            log.info(
                f"PAPER STRESS TEST PASSED: "
                f"MC win_rate={verdict.monte_carlo.win_rate:.1%} "
                f"confidence={verdict.confidence_score:.2f}"
            )

        # 5. Shadow execution via PaperTradingEngine
        paper_result = await paper_engine.run_shadow_cycle(
            manager=manager,
            wallet_id=wallet_id,
            portfolio_manager=portfolio_manager,
            task_context=ctx,
            analytics_db=analytics_db,
            quant_analytics=None,
        )

        # 6. Get performance report for logging
        await paper_engine.get_performance_report()
        telegram_report = await paper_engine.get_telegram_report()

        log.info(
            f"PAPER TRADE: action={paper_result.action} "
            f"alpha={paper_result.signal_alpha:+.3f} "
            f"fill={paper_result.simulated_fill_price:.2f} "
            f"slip={paper_result.slippage_bps:.1f}bps "
            f"gas=${paper_result.gas_cost_usd:.4f} "
            f"vPnL=${paper_result.virtual_pnl_usd:+.2f} "
            f"latency={latency_profile.total_latency_ms:.0f}ms "
            f"regime={regime_engine.get_current_regime().regime.value} "
            f"duration={paper_result.cycle_duration_ms:.0f}ms"
        )

        # 7. Report via Telegram (virtual P&L vs gas cost estimate)
        with contextlib.suppress(Exception):
            await send_heartbeat_telemetry(
                portfolio_manager=portfolio_manager,
                rpc_url=rpc_url,
                address=address,
                task_context={**ctx, "paper_report": telegram_report},
            )

    except Exception as exc:
        log.info(
            f"PaperTrade engine cycle skipped: {type(exc).__name__}: {exc}. "
            f"Will retry next cycle."
        )

    return receipts


async def _robust_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """Ultra-Robust Anti-Overfit task factory — Phase 4+5 combined.

    Layers on top of the ML-Quant pipeline with institutional defences:
    1. LatencyShield — Stale quote protection + RPC health monitoring
    2. RegimeFilter (HMM) — Market regime detection + economic calendar blackout
    3. RobustAlphaEngine — Fractional Differentiation + Permutation Importance
    4. StochasticStressTester — Monte Carlo chaos validation + RFE
    5. WalkForwardValidator — Purged K-Fold cross-validation + retrain control
    6. LiquidityAlpha — V3 tick-level profiling + split-order execution
    7. MLQuantExecutionEngine — Full ML pipeline (GARCH + Kalman + XGBoost/LSTM)

    If the HMM detects HIGH_VOL or NEWS_BLACKOUT, ALL alpha signals are overridden
    and capital moves to 100% USDC + Aave Supply.
    YIELD and AIRDROP pillars execute alongside.

    EXECUTION_MODE=robust or robust_live activates this factory.
    """
    from wallet_manager.strategies.robust_alpha import get_robust_alpha_engine
    from wallet_manager.strategies.regime_filter import get_regime_filter
    from wallet_manager.strategies.liquidity_alpha import get_liquidity_alpha
    from wallet_manager.strategies.wf_validator import get_wf_validator
    from wallet_manager.strategies.execution_logic import get_ml_execution_engine
    from wallet_manager.strategies.quant_analytics import get_quant_analytics
    from wallet_manager.strategies.stochastic_stress_test import get_stress_tester
    from wallet_manager.strategies.latency_shield import get_latency_shield

    ctx = task_context or {}
    portfolio_manager = ctx.get("portfolio_manager")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if portfolio_manager is None:
        log.info("Robust: No portfolio_manager in context — falling back to airdrop only.")
        return await _airdrop_task_factory(manager, wallet_id, task_context)

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    address = _get_wallet_address(config)
    rpc_url = config.rpc_url

    # Rebalance check
    scheduler = ctx.get("scheduler")
    if scheduler is not None and scheduler.is_monthly_rebalance_due():
        is_initial = scheduler.is_initial_rebalance()
        injection = (
            portfolio_manager._initial_budget
            if is_initial
            else portfolio_manager._monthly_injection
        )
        splits = portfolio_manager.rebalance_portfolio(injection)
        rebal_type = "Initial" if is_initial else "Monthly"
        log.info(
            f"Robust {rebal_type} rebalance (${injection:.0f}): "
            f"YIELD=${splits[StrategyType.YIELD]:.2f}, "
            f"ALGO=${splits[StrategyType.ALGO]:.2f}, "
            f"AIRDROP=${splits[StrategyType.AIRDROP]:.2f}.",
        )
        scheduler.mark_monthly_rebalance_done()

    # Update pillar values from on-chain
    try:
        from wallet_manager.strategies import yield_strategy

        yield_val = await yield_strategy.get_yield_pillar_value_usd(rpc_url, address)
        portfolio_manager.update_pillar_value(StrategyType.YIELD, yield_val)

        balance_wei = await manager.get_safe_balance(wallet_id)
        balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
        eth_price = await actions.fetch_eth_price_usd()
        eth_val = actions.eth_balance_usd_value(balance_wei, eth_price)
        usdc_val = actions.usdc_balance_usd_value(balance_usdc)
        liquid_val = eth_val + usdc_val

        short_val = 0.0
        try:
            from wallet_manager.strategies.short_strategy import get_short_position_state
            short_state = get_short_position_state()
            if short_state.position.is_active:
                entry_cost = (short_state.position.borrowed_weth_wei / 1e18) * short_state.position.entry_price
                current_cost = (short_state.position.borrowed_weth_wei / 1e18) * eth_price
                short_val = entry_cost - current_cost
            short_val += short_state.total_short_pnl_usd
        except Exception:
            pass
        portfolio_manager.update_pillar_value(StrategyType.SHORT, max(0.0, short_val))

        liquid_weight_total = 0.35 + 0.15
        algo_val = liquid_val * (0.35 / liquid_weight_total) if liquid_weight_total > 0 else 0
        airdrop_val = liquid_val * (0.15 / liquid_weight_total) if liquid_weight_total > 0 else 0
        portfolio_manager.update_pillar_value(StrategyType.ALGO, algo_val)
        portfolio_manager.update_pillar_value(StrategyType.AIRDROP, airdrop_val)
    except Exception as exc:
        log.info(f"Robust pillar value update skipped: {type(exc).__name__}: {exc}")

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        return receipts

    # 1. YIELD (Base only)
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import yield_strategy

            snapshot = portfolio_manager.get_snapshot()
            yield_alloc = next(a for a in snapshot.allocations if a.strategy == StrategyType.YIELD)
            if yield_alloc.delta_usd > 10:
                free_usdc = await yield_strategy.get_usdc_balance(rpc_url, address)
                supply_amt = min(int(yield_alloc.delta_usd * 1e6), free_usdc)
                if supply_amt >= int(5 * 1e6):
                    recs = await yield_strategy.execute_supply_usdc(
                        manager, wallet_id, rpc_url, address,
                        supply_amt, current_gwei, analytics_db=analytics_db,
                    )
                    receipts.extend(recs)
        except Exception as exc:
            log.debug(f"Robust YIELD skip: {exc}")

    # 2. LATENCY SHIELD — Stale quote protection
    if chain_id == 8453:
        try:
            latency_shield = get_latency_shield(rpc_url)
            eth_price = await actions.fetch_eth_price_usd()
            quote_val, lat_prof = await latency_shield.apply_latency_shield(
                price=eth_price, quote_timestamp=time.time(),
            )
            if quote_val.should_invalidate_signals:
                log.info(
                    f"LATENCY SHIELD: Stale quote ({quote_val.quote_age_s:.1f}s) — "
                    f"ALL signals invalidated. Reason: {quote_val.reason_code}"
                )
                return receipts
        except Exception as exc:
            log.debug(f"Latency shield skip: {exc}")
            eth_price = await actions.fetch_eth_price_usd()

    # 3. REGIME FILTER — HMM + Economic Calendar guardrail
    if chain_id == 8453:
        regime_engine = get_regime_filter()
        try:
            regime_engine.ingest_market_data(
                price=eth_price, volume=0.0, timestamp=time.time(),
            )
        except Exception as exc:
            log.debug(f"Robust regime ingest skip: {exc}")

        override_to_safety, safety_reason = regime_engine.should_override_to_safety()
        if override_to_safety:
            log.info(
                f"REGIME OVERRIDE: {safety_reason} — "
                f"Moving to 100% USDC + Aave Supply. Skipping all alpha."
            )
            # Execute safety: supply all idle USDC to Aave
            try:
                from wallet_manager.strategies import yield_strategy
                free_usdc = await yield_strategy.get_usdc_balance(rpc_url, address)
                if free_usdc >= int(5 * 1e6):
                    recs = await yield_strategy.execute_supply_usdc(
                        manager, wallet_id, rpc_url, address,
                        free_usdc, current_gwei, analytics_db=analytics_db,
                    )
                    receipts.extend(recs)
                    log.info(f"REGIME OVERRIDE: Supplied {free_usdc / 1e6:.2f} USDC to Aave")
            except Exception as exc:
                log.info(f"Robust safety supply skipped: {type(exc).__name__}: {exc}")

            # Skip all alpha — go straight to heartbeat
            with contextlib.suppress(Exception):
                await send_heartbeat_telemetry(
                    portfolio_manager=portfolio_manager,
                    rpc_url=rpc_url,
                    address=address,
                    task_context=ctx,
                )
            return receipts

    # 4. ROBUST ALPHA + STRESS TEST + WALK-FORWARD + LIQUIDITY + ML-QUANT ENGINE
    if chain_id == 8453:
        try:
            robust_engine = get_robust_alpha_engine()
            liquidity_engine = get_liquidity_alpha()
            wf_validator = await get_wf_validator()
            stress_tester = get_stress_tester()

            # Walk-forward retrain check
            try:
                if wf_validator.should_retrain(wf_validator._controller._trade_count):
                    log.info("WALK-FORWARD: Retrain triggered — validating model OOS performance")
                    decision = await wf_validator.run_retrain_cycle()
                    log.info(
                        f"WALK-FORWARD: {decision.action} "
                        f"(mean_oos_sharpe={decision.mean_oos_sharpe:.3f})"
                    )
            except Exception as exc:
                log.debug(f"Walk-forward retrain skip: {exc}")

            # Ingest latest price into robust alpha
            try:
                robust_signal = robust_engine.ingest_price(
                    price=eth_price, volume=0.0, timestamp=time.time(),
                )
            except Exception as exc:
                log.debug(f"Robust alpha ingest skip: {exc}")
                robust_signal = None

            # Monte Carlo stress test — chaos gate
            if robust_signal is not None and (
                robust_signal.should_enter_long or robust_signal.should_enter_short
            ):
                try:
                    price_history = list(robust_engine._price_history)
                    direction = 1.0 if robust_signal.should_enter_long else -1.0
                    verdict = stress_tester.validate_signal(
                        current_price=eth_price,
                        signal_alpha=robust_signal.robust_alpha,
                        signal_direction=direction,
                        price_history=price_history,
                    )
                    if not verdict.approved:
                        log.info(
                            f"STRESS TEST REJECTED: MC win_rate={verdict.monte_carlo.win_rate:.1%} "
                            f"reasons={verdict.rejection_reasons} — trade discarded"
                        )
                        robust_signal = None
                    else:
                        log.info(
                            f"STRESS TEST PASSED: MC win_rate={verdict.monte_carlo.win_rate:.1%} "
                            f"confidence={verdict.confidence_score:.2f}"
                        )
                except Exception as exc:
                    log.debug(f"Stress test skip: {exc}")

            # Liquidity profiling — reduce size if thin
            liquidity_multiplier = 1.0
            try:
                should_reduce, liq_mult, reason = await liquidity_engine.should_reduce_size()
                if should_reduce:
                    liquidity_multiplier = liq_mult
                    log.info(
                        f"LIQUIDITY: Size reduced to {liq_mult:.0%} — {reason}"
                    )
            except Exception as exc:
                log.debug(f"Liquidity profiling skip: {exc}")

            # ML-Quant engine with robust overlay
            engine = get_ml_execution_engine()
            quant_analytics = None
            with contextlib.suppress(Exception):
                quant_analytics = await get_quant_analytics()

            cycle_result = await engine.run_cycle(
                manager=manager,
                wallet_id=wallet_id,
                portfolio_manager=portfolio_manager,
                task_context=ctx,
                analytics_db=analytics_db,
                quant_analytics=quant_analytics,
            )
            receipts.extend(cycle_result.receipts)
            receipts.extend(cycle_result.sweep_receipts)

            # Record trade for walk-forward counter
            with contextlib.suppress(Exception):
                wf_validator.record_trade()

            log.info(
                f"Robust cycle: action={cycle_result.action_taken} "
                f"risk_src={cycle_result.risk_source} "
                f"drawdown={cycle_result.drawdown_state} "
                f"VaR99={cycle_result.var_99:.2f} "
                f"liq_mult={liquidity_multiplier:.0%} "
                f"robust_d={robust_engine._current_d:.2f} "
                f"stationary={robust_engine._is_stationary} "
                f"pnl=${cycle_result.pnl_usd:+.2f} "
                f"duration={cycle_result.cycle_duration_ms:.0f}ms"
            )
        except Exception as exc:
            log.info(f"Robust engine cycle skipped: {type(exc).__name__}: {exc}")

    # 4. AIRDROP (ping-pong, 15% cap)
    airdrop_budget = portfolio_manager.get_pillar_budget_usd(StrategyType.AIRDROP)
    if airdrop_budget >= 10:
        ctx_with_cap = {**ctx, "airdrop_budget_usd": airdrop_budget}
        from wallet_manager.strategies import airdrop_strategy

        recs = await airdrop_strategy.run_airdrop_strategy(manager, wallet_id, ctx_with_cap)
        receipts.extend(recs)

    # 5. HEARTBEAT
    try:
        await send_heartbeat_telemetry(
            portfolio_manager=portfolio_manager,
            rpc_url=rpc_url,
            address=address,
            task_context=ctx,
        )
    except Exception as exc:
        log.debug(f"Robust heartbeat skip: {exc}")

    return receipts


async def _ml_quant_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """ML-Quant Alpha Engine task factory — GARCH/Kalman/XGBoost/LSTM + PPO Kelly.

    Replaces QUANT-X with the full ML-integrated pipeline:
    1. Stochastic ML Alpha Engine (GARCH + Kalman + XGBoost/LSTM ensemble)
    2. Deep Kelly Risk Engine (PPO RL + VaR + Drawdown control)
    3. Smart Order Router (Multi-DEX aggregation)
    4. MEV Shield (Private RPC + Gaussian Jitter + Dynamic RBF)
    5. Flash Crash Circuit Breaker (8% / 5min)
    6. Quant Analytics (TCA + Implementation Shortfall)

    YIELD and AIRDROP pillars still execute alongside.
    """
    from wallet_manager.strategies.execution_logic import get_ml_execution_engine
    from wallet_manager.strategies.quant_analytics import get_quant_analytics

    ctx = task_context or {}
    portfolio_manager = ctx.get("portfolio_manager")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if portfolio_manager is None:
        log.info("ML-Quant: No portfolio_manager in context — falling back to airdrop only.")
        return await _airdrop_task_factory(manager, wallet_id, task_context)

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    address = _get_wallet_address(config)
    rpc_url = config.rpc_url

    # Rebalance check
    scheduler = ctx.get("scheduler")
    if scheduler is not None and scheduler.is_monthly_rebalance_due():
        is_initial = scheduler.is_initial_rebalance()
        injection = (
            portfolio_manager._initial_budget
            if is_initial
            else portfolio_manager._monthly_injection
        )
        splits = portfolio_manager.rebalance_portfolio(injection)
        rebal_type = "Initial" if is_initial else "Monthly"
        log.info(
            f"ML-Quant {rebal_type} rebalance (${injection:.0f}): "
            f"YIELD=${splits[StrategyType.YIELD]:.2f}, "
            f"ALGO=${splits[StrategyType.ALGO]:.2f}, "
            f"AIRDROP=${splits[StrategyType.AIRDROP]:.2f}.",
        )
        scheduler.mark_monthly_rebalance_done()

    # Update pillar values from on-chain
    try:
        from wallet_manager.strategies import yield_strategy

        yield_val = await yield_strategy.get_yield_pillar_value_usd(rpc_url, address)
        portfolio_manager.update_pillar_value(StrategyType.YIELD, yield_val)

        balance_wei = await manager.get_safe_balance(wallet_id)
        balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
        eth_price = await actions.fetch_eth_price_usd()
        eth_val = actions.eth_balance_usd_value(balance_wei, eth_price)
        usdc_val = actions.usdc_balance_usd_value(balance_usdc)
        liquid_val = eth_val + usdc_val

        short_val = 0.0
        try:
            from wallet_manager.strategies.short_strategy import get_short_position_state
            short_state = get_short_position_state()
            if short_state.position.is_active:
                entry_cost = (short_state.position.borrowed_weth_wei / 1e18) * short_state.position.entry_price
                current_cost = (short_state.position.borrowed_weth_wei / 1e18) * eth_price
                short_val = entry_cost - current_cost
            short_val += short_state.total_short_pnl_usd
        except Exception:
            pass
        portfolio_manager.update_pillar_value(StrategyType.SHORT, max(0.0, short_val))

        liquid_weight_total = 0.35 + 0.15
        algo_val = liquid_val * (0.35 / liquid_weight_total) if liquid_weight_total > 0 else 0
        airdrop_val = liquid_val * (0.15 / liquid_weight_total) if liquid_weight_total > 0 else 0
        portfolio_manager.update_pillar_value(StrategyType.ALGO, algo_val)
        portfolio_manager.update_pillar_value(StrategyType.AIRDROP, airdrop_val)
    except Exception as exc:
        log.info(f"ML-Quant pillar value update skipped: {type(exc).__name__}: {exc}")

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        return receipts

    # 1. YIELD (Base only)
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import yield_strategy

            snapshot = portfolio_manager.get_snapshot()
            yield_alloc = next(a for a in snapshot.allocations if a.strategy == StrategyType.YIELD)
            if yield_alloc.delta_usd > 10:
                free_usdc = await yield_strategy.get_usdc_balance(rpc_url, address)
                supply_amt = min(int(yield_alloc.delta_usd * 1e6), free_usdc)
                if supply_amt >= int(5 * 1e6):
                    recs = await yield_strategy.execute_supply_usdc(
                        manager, wallet_id, rpc_url, address,
                        supply_amt, current_gwei, analytics_db=analytics_db,
                    )
                    receipts.extend(recs)
        except Exception as exc:
            log.debug(f"ML-Quant YIELD skip: {exc}")

    # 2. ML-QUANT ALPHA ENGINE (replaces QUANT-X + legacy algo)
    _severe_drift = False
    if chain_id == 8453:
        try:
            engine = get_ml_execution_engine()
            quant_analytics = None
            try:
                quant_analytics = await get_quant_analytics()
            except Exception as exc:
                log.debug(f"ML-Quant analytics init skip: {exc}")

            cycle_result = await engine.run_cycle(
                manager=manager,
                wallet_id=wallet_id,
                portfolio_manager=portfolio_manager,
                task_context=ctx,
                analytics_db=analytics_db,
                quant_analytics=quant_analytics,
            )
            receipts.extend(cycle_result.receipts)
            receipts.extend(cycle_result.sweep_receipts)

            log.info(
                f"ML-Quant cycle: action={cycle_result.action_taken} "
                f"risk_src={cycle_result.risk_source} "
                f"drawdown={cycle_result.drawdown_state} "
                f"VaR99={cycle_result.var_99:.2f} "
                f"dex={cycle_result.dex_venue or 'N/A'} "
                f"pnl=${cycle_result.pnl_usd:+.2f} "
                f"duration={cycle_result.cycle_duration_ms:.0f}ms"
            )

            # ConceptDrift check — after run_cycle() the feature vector is available
            if _drift_detector is not None:
                import time as _time
                try:
                    _fv = engine.alpha.get_last_feature_vector()
                    if _fv is not None:
                        _drift_level = _drift_detector.update(
                            new_feature_vector=_fv,
                            timestamp=_time.time(),
                        )
                        if _drift_level.value >= 2:  # MODERATE or SEVERE
                            log.warning(
                                "ConceptDriftDetector: drift_level=%s",
                                _drift_level.name,
                            )
                            if _drift_level.name == "SEVERE":
                                _severe_drift = True
                except Exception as _drift_exc:
                    log.debug("ConceptDriftDetector update skipped: %s", _drift_exc)

            # ── FeatureDriftMonitor — per-feature KL divergence tracking ──────────
            try:
                from wallet_manager.strategies.robust_alpha import get_feature_drift_monitor
                from wallet_manager.strategies.alpha_engine import FeatureStore
                _fdm = get_feature_drift_monitor()
                _fv2 = engine.alpha.get_last_feature_vector()
                if _fv2 is not None:
                    _fnames = getattr(FeatureStore, "FEATURE_NAMES", None) or [
                        f"feature_{i}" for i in range(len(_fv2))
                    ]
                    _fdm.update(_fv2, _fnames)
                    _report = _fdm.get_drift_report()
                    _drifted = _report.get("drifted_features", [])
                    if _drifted:
                        log.warning(
                            "FeatureDriftMonitor: %d feature(s) drifted | "
                            "worst=%s (KL=%.4f)",
                            len(_drifted),
                            _report.get("worst_feature", "?"),
                            float(_report.get("worst_kl", 0.0)),
                        )
            except Exception as _fdm_exc:
                log.debug("FeatureDriftMonitor update skipped: %s", _fdm_exc)
        except Exception as exc:
            log.info(f"ML-Quant engine cycle skipped: {type(exc).__name__}: {exc}")

    if _severe_drift:
        log.warning("ConceptDriftDetector: SEVERE drift — skipping remainder of wallet cycle.")
        return receipts

    # 3. AIRDROP (ping-pong, 15% cap)
    airdrop_budget = portfolio_manager.get_pillar_budget_usd(StrategyType.AIRDROP)
    if airdrop_budget >= 10:
        ctx_with_cap = {**ctx, "airdrop_budget_usd": airdrop_budget}
        from wallet_manager.strategies import airdrop_strategy

        recs = await airdrop_strategy.run_airdrop_strategy(manager, wallet_id, ctx_with_cap)
        receipts.extend(recs)

    # 4. HEARTBEAT
    try:
        await send_heartbeat_telemetry(
            portfolio_manager=portfolio_manager,
            rpc_url=rpc_url,
            address=address,
            task_context=ctx,
        )
    except Exception as exc:
        log.debug(f"ML-Quant heartbeat skip: {exc}")

    return receipts


async def _quantx_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """QUANT-X Engine task factory — institutional-grade DeFi execution.

    Replaces the legacy ALGO pillar with the QUANT-X pipeline:
    1. Signal Intelligence (Z-Score 2.0 + OFI + VWAP)
    2. Kelly Criterion position sizing
    3. Smart Order Router (Multi-DEX)
    4. MEV Shield (Private RPC + Gaussian Jitter)
    5. Flash Crash Circuit Breaker (8% / 5min)

    YIELD, SHORT, and AIRDROP pillars are still executed alongside.
    """
    from wallet_manager.strategies.quantx_engine import get_quantx_engine

    ctx = task_context or {}
    portfolio_manager = ctx.get("portfolio_manager")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if portfolio_manager is None:
        log.info("QUANT-X: No portfolio_manager in context — falling back to airdrop only.")
        return await _airdrop_task_factory(manager, wallet_id, task_context)

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    address = _get_wallet_address(config)
    rpc_url = config.rpc_url

    # Rebalance check (same as legacy)
    scheduler = ctx.get("scheduler")
    if scheduler is not None and scheduler.is_monthly_rebalance_due():
        is_initial = scheduler.is_initial_rebalance()
        injection = (
            portfolio_manager._initial_budget
            if is_initial
            else portfolio_manager._monthly_injection
        )
        splits = portfolio_manager.rebalance_portfolio(injection)
        rebal_type = "Initial" if is_initial else "Monthly"
        log.info(
            f"QUANT-X {rebal_type} rebalance (${injection:.0f}): "
            f"YIELD=${splits[StrategyType.YIELD]:.2f}, "
            f"ALGO=${splits[StrategyType.ALGO]:.2f}, "
            f"AIRDROP=${splits[StrategyType.AIRDROP]:.2f}.",
        )
        scheduler.mark_monthly_rebalance_done()

    # Update pillar values from on-chain
    try:
        from wallet_manager.strategies import yield_strategy

        yield_val = await yield_strategy.get_yield_pillar_value_usd(rpc_url, address)
        portfolio_manager.update_pillar_value(StrategyType.YIELD, yield_val)

        balance_wei = await manager.get_safe_balance(wallet_id)
        balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
        eth_price = await actions.fetch_eth_price_usd()
        eth_val = actions.eth_balance_usd_value(balance_wei, eth_price)
        usdc_val = actions.usdc_balance_usd_value(balance_usdc)
        liquid_val = eth_val + usdc_val

        # SHORT pillar
        short_val = 0.0
        try:
            from wallet_manager.strategies.short_strategy import get_short_position_state
            short_state = get_short_position_state()
            if short_state.position.is_active:
                entry_cost = (short_state.position.borrowed_weth_wei / 1e18) * short_state.position.entry_price
                current_cost = (short_state.position.borrowed_weth_wei / 1e18) * eth_price
                short_val = entry_cost - current_cost
            short_val += short_state.total_short_pnl_usd
        except Exception:
            pass
        portfolio_manager.update_pillar_value(StrategyType.SHORT, max(0.0, short_val))

        liquid_weight_total = 0.35 + 0.15
        algo_val = liquid_val * (0.35 / liquid_weight_total) if liquid_weight_total > 0 else 0
        airdrop_val = liquid_val * (0.15 / liquid_weight_total) if liquid_weight_total > 0 else 0
        portfolio_manager.update_pillar_value(StrategyType.ALGO, algo_val)
        portfolio_manager.update_pillar_value(StrategyType.AIRDROP, airdrop_val)
    except Exception as exc:
        log.info(f"QUANT-X pillar value update skipped: {type(exc).__name__}: {exc}")

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        return receipts

    # 1. YIELD (Base only)
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import yield_strategy

            snapshot = portfolio_manager.get_snapshot()
            yield_alloc = next(a for a in snapshot.allocations if a.strategy == StrategyType.YIELD)
            if yield_alloc.delta_usd > 10:
                free_usdc = await yield_strategy.get_usdc_balance(rpc_url, address)
                supply_amt = min(int(yield_alloc.delta_usd * 1e6), free_usdc)
                if supply_amt >= int(5 * 1e6):
                    recs = await yield_strategy.execute_supply_usdc(
                        manager, wallet_id, rpc_url, address,
                        supply_amt, current_gwei, analytics_db=analytics_db,
                    )
                    receipts.extend(recs)
        except Exception as exc:
            log.debug(f"QUANT-X YIELD skip: {exc}")

    # 2. QUANT-X ALGO ENGINE (replaces legacy algo_strategy + short_strategy)
    if chain_id == 8453:
        try:
            engine = get_quantx_engine()
            cycle_result = await engine.run_cycle(
                manager=manager,
                wallet_id=wallet_id,
                portfolio_manager=portfolio_manager,
                task_context=ctx,
                analytics_db=analytics_db,
            )
            receipts.extend(cycle_result.receipts)
            receipts.extend(cycle_result.sweep_receipts)

            log.info(
                f"QUANT-X cycle: action={cycle_result.action_taken} "
                f"alpha={cycle_result.signal.alpha_score:+.3f if cycle_result.signal else 'N/A'} "
                f"dex={cycle_result.dex_venue or 'N/A'} "
                f"pnl=${cycle_result.pnl_usd:+.2f}"
            )
        except Exception as exc:
            log.info(f"QUANT-X engine cycle skipped: {type(exc).__name__}: {exc}")

    # 3. AIRDROP (ping-pong, 15% cap) — unchanged
    airdrop_budget = portfolio_manager.get_pillar_budget_usd(StrategyType.AIRDROP)
    if airdrop_budget >= 10:
        ctx_with_cap = {**ctx, "airdrop_budget_usd": airdrop_budget}
        from wallet_manager.strategies import airdrop_strategy

        recs = await airdrop_strategy.run_airdrop_strategy(manager, wallet_id, ctx_with_cap)
        receipts.extend(recs)

    # 4. HEARTBEAT
    try:
        await send_heartbeat_telemetry(
            portfolio_manager=portfolio_manager,
            rpc_url=rpc_url,
            address=address,
            task_context=ctx,
        )
    except Exception as exc:
        log.debug(f"QUANT-X heartbeat skip: {exc}")

    return receipts


async def _three_pillar_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """3-Pillar Portfolio Orchestrator: YIELD (50%) + ALGO (35%) + AIRDROP (15%).

    When QUANTX_ENGINE_MODE=quantx (env), delegates the ALGO pillar to the
    QUANT-X engine (signal_intelligence + kelly_criterion + smart_order_router
    + mev_shield). All other pillars (YIELD, SHORT, AIRDROP) remain unchanged.

    Delegates to yield_strategy, algo_strategy, and airdrop (ping-pong) based on
    PortfolioManager state. All txs routed through manager.sign_and_broadcast_tx.
    """
    ctx = task_context or {}
    command_state = ctx.get("command_state")

    # Circuit breaker: halt ALL trading (YIELD, ALGO, AIRDROP) for 24h when triggered
    if command_state and getattr(command_state, "is_circuit_breaker_halted", False) and command_state.is_circuit_breaker_halted:
        log.info(f"Wallet '{wallet_id}': Circuit breaker halted — skip all pillars. Send /resume to override.")
        return []

    # Paper trade mode: Shadow execution with slippage sim + virtual ledger
    _exec_mode = _cfg().execution_mode
    if _exec_mode == "robust_paper":
        return await _robust_paper_task_factory(manager, wallet_id, task_context)

    # Robust live mode: Full anti-overfit pipeline (production)
    if _exec_mode in ("robust", "robust_live"):
        return await _robust_task_factory(manager, wallet_id, task_context)

    # ML-Quant mode: GARCH + Kalman + XGBoost/LSTM + PPO Kelly + TCA
    if _exec_mode == "ml_quant":
        return await _ml_quant_task_factory(manager, wallet_id, task_context)

    # QUANT-X mode: use institutional-grade engine for ALGO pillar
    _quantx_mode = _cfg().quantx_engine_mode
    if _quantx_mode == "quantx":
        return await _quantx_task_factory(manager, wallet_id, task_context)

    portfolio_manager = ctx.get("portfolio_manager")
    analytics_db = ctx.get("analytics_db")
    receipts: list[TxReceipt] = []

    if portfolio_manager is None:
        log.info("No portfolio_manager in context — falling back to airdrop only.")
        return await _airdrop_task_factory(manager, wallet_id, task_context)

    config = manager.get_config(wallet_id)
    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    address = _get_wallet_address(config)
    rpc_url = config.rpc_url

    # Initial $250 or monthly $150 rebalance: distribute according to 50/35/15 when due
    scheduler = task_context.get("scheduler")
    if scheduler is not None and scheduler.is_monthly_rebalance_due():
        is_initial = scheduler.is_initial_rebalance()
        injection = (
            portfolio_manager._initial_budget
            if is_initial
            else portfolio_manager._monthly_injection
        )
        splits = portfolio_manager.rebalance_portfolio(injection)
        rebal_type = "Initial" if is_initial else "Monthly"
        log.info(
            f"{rebal_type} rebalance (${injection:.0f}): "
            f"YIELD=${splits[StrategyType.YIELD]:.2f}, "
            f"ALGO=${splits[StrategyType.ALGO]:.2f}, "
            f"AIRDROP=${splits[StrategyType.AIRDROP]:.2f}.",
        )
        scheduler.mark_monthly_rebalance_done()

    # Update pillar values from on-chain (first wallet as primary)
    try:
        from wallet_manager.strategies import yield_strategy

        yield_val = await yield_strategy.get_yield_pillar_value_usd(rpc_url, address)
        portfolio_manager.update_pillar_value(StrategyType.YIELD, yield_val)

        # Liquid balance: ETH + USDC not in Aave
        balance_wei = await manager.get_safe_balance(wallet_id)
        balance_usdc = await actions.fetch_usdc_balance(rpc_url, address, chain_id)
        eth_price = await actions.fetch_eth_price_usd()
        eth_val = actions.eth_balance_usd_value(balance_wei, eth_price)
        usdc_val = actions.usdc_balance_usd_value(balance_usdc)
        liquid_val = eth_val + usdc_val

        # SHORT pillar: value of active short position (unrealized P&L)
        short_val = 0.0
        try:
            from wallet_manager.strategies.short_strategy import get_short_position_state
            short_state = get_short_position_state()
            if short_state.position.is_active:
                entry_cost = (short_state.position.borrowed_weth_wei / 1e18) * short_state.position.entry_price
                current_cost = (short_state.position.borrowed_weth_wei / 1e18) * eth_price
                short_val = entry_cost - current_cost  # Positive if price dropped
            short_val += short_state.total_short_pnl_usd
        except Exception:
            pass
        portfolio_manager.update_pillar_value(StrategyType.SHORT, max(0.0, short_val))

        # Distribute liquid among ALGO + AIRDROP (proportional to weights)
        liquid_weight_total = 0.35 + 0.15  # ALGO + AIRDROP weights
        algo_val = liquid_val * (0.35 / liquid_weight_total) if liquid_weight_total > 0 else 0
        airdrop_val = liquid_val * (0.15 / liquid_weight_total) if liquid_weight_total > 0 else 0
        portfolio_manager.update_pillar_value(StrategyType.ALGO, algo_val)
        portfolio_manager.update_pillar_value(StrategyType.AIRDROP, airdrop_val)
    except Exception as exc:
        log.info(f"Pillar value update skipped: {type(exc).__name__}: {exc}")

    gas_monitor = GasMonitor(rpc_url=rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))
    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        return receipts

    # 1. YIELD (Base only) — with APY optimization check
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import yield_strategy

            aave_apy = await yield_strategy.get_apy(rpc_url)
            if aave_apy < _AAVE_APY_ALERT_THRESHOLD:
                alt_apy, alt_protocol = await actions.get_alternative_stable_yield_apy(
                    rpc_url, chain_id
                )
                if alt_apy > aave_apy:
                    tg = ctx.get("telegram_manager")
                    if tg and tg._is_configured():
                        try:
                            from wallet_manager.security.telegram_manager import (
                                YieldAlertPayload,
                            )
                            await tg.send_yield_alert(
                                YieldAlertPayload(
                                    aave_apy_pct=aave_apy,
                                    alternative_protocol=alt_protocol,
                                    alternative_apy_pct=alt_apy,
                                    detail=f"Aave at {aave_apy:.2f}% APY. Consider {alt_protocol} ({alt_apy:.2f}%).",
                                )
                            )
                        except Exception as exc:
                            log.info(f"Yield alert skipped: {type(exc).__name__}: {exc}")

            snapshot = portfolio_manager.get_snapshot()
            yield_alloc = next(a for a in snapshot.allocations if a.strategy == StrategyType.YIELD)
            if yield_alloc.delta_usd > 10:  # need to supply
                free_usdc = await yield_strategy.get_usdc_balance(rpc_url, address)
                supply_amt = min(int(yield_alloc.delta_usd * 1e6), free_usdc)
                if supply_amt >= int(5 * 1e6):  # min $5
                    recs = await yield_strategy.execute_supply_usdc(
                        manager, wallet_id, rpc_url, address,
                        supply_amt, current_gwei, analytics_db=analytics_db,
                    )
                    receipts.extend(recs)
        except Exception as exc:
            log.debug(f"YIELD strategy skip: {exc}")

    # --- Flash Crash Detection (before any trading) ---
    from wallet_manager.strategies.flash_crash_detector import (
        get_detector,
        send_kill_switch_alert,
        activate_kill_switch,
    )
    crash_detector = get_detector()
    if crash_detector.is_kill_switch_active:
        log.info("KILL_SWITCH active — all strategies halted.")
        return receipts
    try:
        crash_alert = crash_detector.ingest_price(
            await actions.get_current_price(rpc_url, chain_id)
        )
        if crash_alert is not None:
            activate_kill_switch(ctx, crash_alert)
            await send_kill_switch_alert(ctx, crash_alert)
            log.critical(
                f"🚨 KILL_SWITCH triggered: {crash_alert.drop_pct:.1f}% crash. "
                f"All trading halted."
            )
            return receipts
    except Exception as exc:
        log.debug(f"Flash crash check failed: {exc}")

    # 2. ALGO (Base only) — with Bollinger Bands + Z-Score + RSI
    algo_z_score = 0.0
    algo_rsi = 50.0
    algo_bb_upper = 0.0
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import algo_strategy

            recs = await algo_strategy.run_algo_strategy(
                manager, wallet_id, portfolio_manager, ctx, analytics_db=analytics_db,
            )
            receipts.extend(recs)

            # Capture signals for short strategy reuse
            state = algo_strategy._load_state(algo_strategy._default_history_path())
            if len(state.prices) >= algo_strategy.EMA_PERIOD:
                _, _, algo_z_score = algo_strategy.compute_z_score_ema(
                    state.prices, algo_strategy.EMA_PERIOD
                )
                algo_rsi = algo_strategy.compute_rsi(state.prices, algo_strategy.RSI_PERIOD)
                _, _, algo_bb_upper = algo_strategy.compute_bollinger_bands(
                    state.prices, algo_strategy.EMA_PERIOD
                )
        except Exception as exc:
            log.debug(f"ALGO strategy skip: {exc}")

    # 3. SHORT — Hedged Delta-Neutral (Base only, 10:1 model)
    if chain_id == 8453:
        try:
            from wallet_manager.strategies import short_strategy

            current_price = await actions.get_current_price(rpc_url, chain_id)

            # Pass alpha_signal when not in legacy mode
            _short_alpha = None
            _exec_mode_short = _cfg().execution_mode
            if _exec_mode_short != "legacy":
                try:
                    from wallet_manager.strategies.alpha_engine import get_alpha_engine
                    _ae = get_alpha_engine()
                    _short_alpha = _ae.last_signal
                except Exception:
                    pass

            recs = await short_strategy.run_short_strategy(
                manager=manager,
                wallet_id=wallet_id,
                portfolio_manager=portfolio_manager,
                task_context=ctx,
                z_score=algo_z_score,
                rsi=algo_rsi,
                current_price=current_price,
                bb_upper=algo_bb_upper,
                analytics_db=analytics_db,
                alpha_signal=_short_alpha,
            )
            receipts.extend(recs)
        except Exception as exc:
            log.debug(f"SHORT strategy skip: {exc}")

    # 4. IDLE SWEEP — Zero Lazy Capital (Base only)
    if chain_id == 8453:
        try:
            from wallet_manager.strategies.idle_sweep import run_idle_sweep

            recs = await run_idle_sweep(
                manager=manager,
                wallet_id=wallet_id,
                task_context=ctx,
                analytics_db=analytics_db,
            )
            receipts.extend(recs)
        except Exception as exc:
            log.debug(f"Idle sweep skip: {exc}")

    # 5. AIRDROP (ping-pong, 15% cap)
    airdrop_budget = portfolio_manager.get_pillar_budget_usd(StrategyType.AIRDROP)
    if airdrop_budget >= 10:
        ctx_with_cap = {**ctx, "airdrop_budget_usd": airdrop_budget}
        from wallet_manager.strategies import airdrop_strategy

        recs = await airdrop_strategy.run_airdrop_strategy(manager, wallet_id, ctx_with_cap)
        receipts.extend(recs)

    # 6. HEARTBEAT TELEMETRY — every 6 hours
    try:
        await send_heartbeat_telemetry(
            portfolio_manager=portfolio_manager,
            rpc_url=rpc_url,
            address=address,
            task_context=ctx,
            analytics_db=analytics_db,
        )
    except Exception as exc:
        log.debug(f"Heartbeat telemetry skip: {exc}")

    return receipts


async def _airdrop_task_factory(
    manager: WalletManager,
    wallet_id: str,
    task_context: dict | None = None,
) -> list[TxReceipt]:
    """Ping-Pong airdrop farming: swap ETH<->USDC based on which side is larger.

    If USDC value > $10 and USDC > ETH value: swap 20-50% USDC -> ETH.
    If ETH value > $10: swap 10-20% ETH -> USDC.
    When airdrop_budget_usd is in context, caps swap to stay within 15% allocation.
    """
    import random

    from wallet_manager import actions

    ctx = task_context or {}
    notifier = ctx.get("notifier")
    confirmation_gate = ctx.get("confirmation_gate")
    scheduler = ctx.get("scheduler")
    receipts: list[TxReceipt] = []

    if scheduler is None:
        log.info("No scheduler in task_context — skip airdrop logic.")
        return receipts

    config = manager.get_config(wallet_id)
    gas_monitor = GasMonitor(rpc_url=config.rpc_url)
    max_gwei = min(25.0, float(config.max_gas_price_gwei))

    should_run, seconds_until_due = should_run_swap_this_cycle(scheduler, wallet_id)
    if not should_run:
        if seconds_until_due >= 3600:
            time_str = f"{seconds_until_due / 3600:.1f}h"
        else:
            time_str = f"{seconds_until_due / 60:.1f}m"
        log.info(
            f"Scheduler: next swap in {time_str} for wallet '{wallet_id}'.",
        )
        return receipts

    safe, current_gwei = await gas_monitor.is_safe(max_gwei)
    if not safe:
        log.info(f"Wallet '{wallet_id}': gas {current_gwei:.2f} Gwei > {max_gwei} — skip.")
        return receipts

    chain_id = int(config.chain_id) if config.chain_id is not None else 0
    if chain_id not in (59144, 8453):
        log.info(f"Wallet '{wallet_id}': chain {chain_id} not supported for swap.")
        return receipts

    address = _get_wallet_address(config)

    try:
        balance_wei = await manager.get_safe_balance(wallet_id)
    except Exception as exc:
        log.info(f"Wallet '{wallet_id}': balance fetch unavailable ({type(exc).__name__}: {exc})")
        return receipts

    try:
        balance_usdc = await actions.fetch_usdc_balance(config.rpc_url, address, chain_id)
    except Exception as exc:
        log.info(f"Wallet '{wallet_id}': USDC balance fetch unavailable ({type(exc).__name__})")
        balance_usdc = 0

    eth_price = await actions.fetch_eth_price_usd()
    eth_value_usd = actions.eth_balance_usd_value(balance_wei, eth_price)
    usdc_value_usd = actions.usdc_balance_usd_value(balance_usdc)

    # Ping-Pong: choose direction
    do_usdc_to_eth = (
        actions.is_usdc_swap_eligible(balance_usdc, current_gwei, max_gas_gwei=max_gwei)
        and usdc_value_usd > eth_value_usd
    )
    do_eth_to_usdc = actions.is_swap_eligible(
        balance_wei, eth_price, current_gwei,
        min_eth_value_usd=actions.MIN_ETH_VALUE_USD,
        max_gas_gwei=max_gwei,
    )

    airdrop_budget_usd = (ctx or {}).get("airdrop_budget_usd")
    if airdrop_budget_usd is not None and airdrop_budget_usd > 0:
        max_swap_usd = min(usdc_value_usd + eth_value_usd, airdrop_budget_usd)
        if max_swap_usd < 10:
            log.info(f"Wallet '{wallet_id}': airdrop budget ${max_swap_usd:.0f} < $10.")
            return receipts

    if do_usdc_to_eth:
        # USDC -> ETH: swap 20-50% of USDC (capped by airdrop budget if set)
        pct = random.uniform(0.20, 0.50)  # noqa: S311
        amount_usdc = int(balance_usdc * pct)
        if airdrop_budget_usd is not None and airdrop_budget_usd > 0:
            max_usdc = int(airdrop_budget_usd * 0.5 * 1e6)
            amount_usdc = min(amount_usdc, max_usdc)
        min_usdc = int(10 * 1e6)
        if amount_usdc < min_usdc:
            log.info(f"Wallet '{wallet_id}': USDC amount too small.")
            return receipts

        return await _execute_usdc_to_eth(
            manager, wallet_id, config, address, chain_id,
            amount_usdc, current_gwei,
            notifier, confirmation_gate, scheduler,
            analytics_db=ctx.get("analytics_db"),
        )

    if do_eth_to_usdc:
        # ETH -> USDC: swap 10-20% of ETH (capped by airdrop budget if set)
        pct = random.uniform(0.10, 0.20)  # noqa: S311
        amount_wei = int(balance_wei * pct)
        if airdrop_budget_usd is not None and airdrop_budget_usd > 0:
            max_eth_val_usd = airdrop_budget_usd * 0.5
            max_wei = int(max_eth_val_usd * 1e18 / eth_price)
            amount_wei = min(amount_wei, max_wei)
        if amount_wei < 10**15:
            log.info(f"Wallet '{wallet_id}': amount too small.")
            return receipts

        return await _execute_eth_to_usdc(
            manager, wallet_id, config, address, chain_id,
            amount_wei, current_gwei,
            notifier, confirmation_gate, scheduler,
            analytics_db=ctx.get("analytics_db"),
        )

    log.info(f"Wallet '{wallet_id}': not eligible for ping-pong (balances or gas).")
    return receipts


async def _execute_usdc_to_eth(
    manager: WalletManager,
    wallet_id: str,
    config: WalletConfig,
    address: str,
    chain_id: int,
    amount_usdc: int,
    current_gwei: float,
    notifier,
    confirmation_gate,
    scheduler,
    analytics_db=None,
) -> list[TxReceipt]:
    """Execute USDC -> ETH swap."""
    from wallet_manager import actions

    receipts: list[TxReceipt] = []
    fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)  # Ghost: rotate path
    addrs = actions.get_chain_addresses(chain_id)
    slippage_bps = await actions.get_dynamic_slippage(
        rpc_url=config.rpc_url,
        chain_id=chain_id,
        token_in=addrs.usdc,
        token_out=addrs.weth,
        amount_in=amount_usdc,
        gas_price_gwei=current_gwei,
        fee=fee_tier,
        is_algo=False,
    )
    try:
        plan = await actions.build_usdc_to_eth_plan(
            rpc_url=config.rpc_url,
            chain_id=chain_id,
            from_address=address,
            amount_usdc=amount_usdc,
            slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
        )
    except Exception as exc:
        log.error(f"Wallet '{wallet_id}': USDC->ETH plan build failed: {exc}")
        if notifier and hasattr(notifier, "send"):
            await WalletOrchestrator._safe_notify(
                notifier,
                NotificationEvent(
                    kind=EventKind.WALLET_FAILURE,
                    wallet_id=wallet_id,
                    detail=f"Swap plan build failed: {exc}",
                ),
            )
        return receipts

    amount_str = f"{amount_usdc / 1e6:.2f} USDC"
    if notifier and hasattr(notifier, "send_confirmation_request") and confirmation_gate:
        confirmation_gate.request_confirmation()
        try:
            await notifier.send_confirmation_request(
                wallet_id, amount_str, "USDC -> ETH"
            )
        except Exception as exc:
            log.info(f"Confirmation request skipped: {type(exc).__name__}: {exc}")
        confirmed = await confirmation_gate.wait_for_evet(timeout_seconds=300.0)
        if not confirmed:
            log.info(f"Wallet '{wallet_id}': confirmation timeout or rejected.")
            scheduler.schedule_next(wallet_id)
            return receipts

    for i, step in enumerate(plan.steps):
        try:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            receipts.append(receipt)
            if analytics_db:
                from wallet_manager.core.models import TxRecord, TxStatus

                record = TxRecord(
                    tx_hash=receipt.tx_hash,
                    wallet_id=wallet_id,
                    chain_id=chain_id,
                    nonce=receipt.nonce_used,
                    gas_price_gwei=step.tx_params.gas_price_gwei,
                    gas_limit=step.tx_params.gas,
                    value_wei=step.tx_params.value,
                    status=TxStatus.PENDING,
                    submitted_at=time.time(),
                    strategy_type=StrategyType.AIRDROP.value,
                    strategy_id="airdrop_ping_pong",
                )
                await analytics_db.record_tx(record)
        except Exception as exc:
            err_msg = f"Step {i + 1}/{len(plan.steps)} ({step.description}) failed: {exc}"
            log.error(f"Wallet '{wallet_id}': {err_msg}")
            if notifier and hasattr(notifier, "send"):
                await WalletOrchestrator._safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.WALLET_FAILURE,
                        wallet_id=wallet_id,
                        detail=f"Swap failed: {err_msg}",
                    ),
                )
            scheduler.schedule_next(wallet_id)
            return receipts

    scheduler.schedule_next(wallet_id)
    return receipts


async def _execute_eth_to_usdc(
    manager: WalletManager,
    wallet_id: str,
    config: WalletConfig,
    address: str,
    chain_id: int,
    amount_wei: int,
    current_gwei: float,
    notifier,
    confirmation_gate,
    scheduler,
    analytics_db=None,
) -> list[TxReceipt]:
    """Execute ETH -> USDC swap."""
    from wallet_manager import actions

    receipts: list[TxReceipt] = []
    fee_tier = actions.get_rotated_fee_tier(int(time.time() / 60) % 100)  # Ghost: rotate path
    addrs = actions.get_chain_addresses(chain_id)
    slippage_bps = await actions.get_dynamic_slippage(
        rpc_url=config.rpc_url,
        chain_id=chain_id,
        token_in=addrs.weth,
        token_out=addrs.usdc,
        amount_in=amount_wei,
        gas_price_gwei=current_gwei,
        fee=fee_tier,
        is_algo=False,
    )
    try:
        plan = await actions.build_eth_to_usdc_plan(
            rpc_url=config.rpc_url,
            chain_id=chain_id,
            from_address=address,
            amount_wei=amount_wei,
            slippage_bps=slippage_bps,
            gas_price_gwei=current_gwei,
            fee=fee_tier,
        )
    except Exception as exc:
        log.error(f"Wallet '{wallet_id}': swap plan build failed: {exc}")
        if notifier and hasattr(notifier, "send"):
            await WalletOrchestrator._safe_notify(
                notifier,
                NotificationEvent(
                    kind=EventKind.WALLET_FAILURE,
                    wallet_id=wallet_id,
                    detail=f"Swap plan build failed: {exc}",
                ),
            )
        return receipts

    amount_eth = amount_wei / 1e18
    amount_str = f"{amount_eth:.4f} ETH"
    if notifier and hasattr(notifier, "send_confirmation_request") and confirmation_gate:
        confirmation_gate.request_confirmation()
        try:
            await notifier.send_confirmation_request(
                wallet_id, amount_str, "ETH -> USDC"
            )
        except Exception as exc:
            log.info(f"Confirmation request skipped: {type(exc).__name__}: {exc}")
        confirmed = await confirmation_gate.wait_for_evet(timeout_seconds=300.0)
        if not confirmed:
            log.info(f"Wallet '{wallet_id}': confirmation timeout or rejected.")
            scheduler.schedule_next(wallet_id)
            return receipts

    for i, step in enumerate(plan.steps):
        try:
            receipt = await manager.sign_and_broadcast_tx(wallet_id, step.tx_params)
            receipts.append(receipt)
            if analytics_db:
                from wallet_manager.core.models import TxRecord, TxStatus

                record = TxRecord(
                    tx_hash=receipt.tx_hash,
                    wallet_id=wallet_id,
                    chain_id=chain_id,
                    nonce=receipt.nonce_used,
                    gas_price_gwei=step.tx_params.gas_price_gwei,
                    gas_limit=step.tx_params.gas,
                    value_wei=step.tx_params.value,
                    status=TxStatus.PENDING,
                    submitted_at=time.time(),
                    strategy_type=StrategyType.AIRDROP.value,
                    strategy_id="airdrop_ping_pong",
                )
                await analytics_db.record_tx(record)
        except Exception as exc:
            err_msg = f"Step {i + 1}/{len(plan.steps)} ({step.description}) failed: {exc}"
            log.error(f"Wallet '{wallet_id}': {err_msg}")
            if notifier and hasattr(notifier, "send"):
                await WalletOrchestrator._safe_notify(
                    notifier,
                    NotificationEvent(
                        kind=EventKind.WALLET_FAILURE,
                        wallet_id=wallet_id,
                        detail=f"Swap failed: {err_msg}",
                    ),
                )
            scheduler.schedule_next(wallet_id)
            return receipts

    scheduler.schedule_next(wallet_id)
    return receipts


# ---------------------------------------------------------------------------
# MasterOrchestrator — 5-min Algo loop, $250/$150 rebalance, Telegram integration
# ---------------------------------------------------------------------------

class MasterOrchestrator(WalletOrchestrator):
    """
    Production orchestrator: 5-minute Algo interval, automatic $250/$150 rebalance,
    Telegram trade/summary/alert notifications, strategy_id tagging.

    Uses ALGO_LOOP_INTERVAL_SECONDS (default 300) for main loop.
    All trades pass through GasMonitor; TxRepairService can be wired at broadcast layer.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._telegram_manager = None
        try:
            from wallet_manager.security.telegram_manager import TelegramManager
            self._telegram_manager = TelegramManager()
        except Exception as exc:
            log.debug(f"TelegramManager unavailable: {exc}. Continuing without.")

    def _resolve_notifier(self) -> INotifier:
        inner = super()._resolve_notifier()
        if self._telegram_manager:
            return _SystemAlertNotifierWrapper(inner, self._telegram_manager)
        return inner

    async def _run_wallet(
        self,
        manager: WalletManager,
        config: WalletConfig,
        semaphore: asyncio.Semaphore,
        notifier: INotifier,
        task_context: dict | None = None,
    ) -> WalletResult:
        ctx = task_context or {}
        if self._telegram_manager and "telegram_manager" not in ctx:
            ctx = {**ctx, "telegram_manager": self._telegram_manager}
        result = await super()._run_wallet(
            manager, config, semaphore, notifier, ctx
        )
        if result.success and result.receipts and self._telegram_manager:
            await self._notify_trade_if_algo(ctx, result, config)
        return result

    async def _notify_trade_if_algo(
        self,
        ctx: dict,
        result: WalletResult,
        config: WalletConfig,
    ) -> None:
        """Send trade executed notification from strategy-provided payloads."""
        try:
            from wallet_manager.security.telegram_manager import (
                TradeExecutedPayload,
            )
            mgr = ctx.get("telegram_manager")
            pending = ctx.get("trade_notifications", [])
            if not mgr or not mgr._is_configured() or not pending:
                return
            for payload in pending:
                if isinstance(payload, TradeExecutedPayload):
                    await mgr.send_trade_executed(payload)
            ctx["trade_notifications"] = []
        except Exception as exc:
            log.info(f"Trade notification skipped: {type(exc).__name__}: {exc}")

    async def _maybe_send_heartbeat(
        self,
        portfolio_manager: PortfolioManager,
        analytics_db: object | None,
        notifier: INotifier,
    ) -> None:
        """Send 6-hour heartbeat if due: TVL, positions, unrealized PnL, Aave yield, gas, paper stats."""
        global _last_heartbeat_ts
        if not self._telegram_manager or not self._telegram_manager._is_configured():
            return
        now = time.time()
        if now - _last_heartbeat_ts < _HEARTBEAT_INTERVAL_SECONDS:
            return
        _last_heartbeat_ts = now
        try:
            config = self._configs[0]
            address = _get_wallet_address(config)
            chain_id = int(config.chain_id) if config.chain_id is not None else 8453
            balance_wei = await actions.fetch_native_balance(config.rpc_url, address)
            balance_usdc = await actions.fetch_usdc_balance(
                config.rpc_url, address, chain_id
            )
            eth_price = await actions.fetch_eth_price_usd()
            eth_val = actions.eth_balance_usd_value(balance_wei, eth_price)
            usdc_val = actions.usdc_balance_usd_value(balance_usdc)
            total_tvl = portfolio_manager.total_value_usd or (eth_val + usdc_val)
            try:
                current_gwei = await self._gas_monitor.current_gwei()
            except Exception:
                current_gwei = 0.0
            aave_stats = await actions.get_aave_stats(config.rpc_url, address)
            aave_yield_usd = (
                aave_stats.supplied_usdc * (aave_stats.supply_apy_pct / 100) / 365
                if aave_stats else 0.0
            )  # Daily yield projection
            aave_apy = aave_stats.supply_apy_pct if aave_stats else 0.0
            open_positions = 1 if eth_val >= 10 else 0
            unrealized_pnl = 0.0

            # --- Paper trading stats ---
            paper_balance_usd = 0.0
            paper_starting = 10_000.0
            total_trades = 0
            win_rate_pct = 0.0
            best_trade_pnl = 0.0
            best_trade_pair = ""
            worst_trade_pnl = 0.0
            worst_trade_pair = ""
            last_trade_time = ""
            last_trade_token = ""
            last_trade_pnl = 0.0
            signal_status = ""

            _exec_mode = _cfg().execution_mode
            if _exec_mode == "robust_paper":
                try:
                    from wallet_manager.strategies.paper_engine import get_paper_engine
                    paper_engine = await get_paper_engine()
                    pnl_summary = await paper_engine._ledger.get_pnl_summary()
                    paper_balance_usd = await paper_engine._ledger.get_total_equity_usd(eth_price=eth_price)
                    paper_starting = paper_engine._config.initial_balance_usd
                    total_trades = pnl_summary.get("trade_count", 0)
                    win_rate_pct = pnl_summary.get("win_rate", 0.0)
                    unrealized_pnl = pnl_summary.get("realized_pnl", 0.0)

                    # Best/worst/last trade
                    trade_stats = await paper_engine._ledger.get_best_worst_last_trade()
                    if trade_stats["best_trade"]:
                        best_trade_pnl = trade_stats["best_trade"]["pnl_usd"]
                        best_trade_pair = trade_stats["best_trade"]["pair"]
                    if trade_stats["worst_trade"]:
                        worst_trade_pnl = trade_stats["worst_trade"]["pnl_usd"]
                        worst_trade_pair = trade_stats["worst_trade"]["pair"]
                    if trade_stats["last_trade"]:
                        import datetime
                        ts = trade_stats["last_trade"]["timestamp"]
                        last_trade_time = datetime.datetime.fromtimestamp(ts).strftime("%H:%M")
                        last_trade_token = trade_stats["last_trade"]["pair"]
                        last_trade_pnl = trade_stats["last_trade"]["pnl_usd"]

                    # Signal status — why idle
                    if paper_engine._in_position:
                        signal_status = "In position — waiting for exit signal"
                    elif total_trades == 0:
                        signal_status = "Warming up — collecting price data for alpha signal"
                    else:
                        signal_status = "Monitoring — alpha signal below entry threshold"

                    open_positions = 1 if paper_engine._in_position else 0
                except Exception as exc:
                    log.debug(f"Paper stats for heartbeat failed: {exc}")

            # Gas suitability check
            max_gas = config.max_gas_price_gwei if config.max_gas_price_gwei is not None else 100.0
            gas_suitable = current_gwei <= float(max_gas)

            # Next run seconds
            next_run_seconds = _ALGO_LOOP_INTERVAL_SECONDS

            # Strategy status for heartbeat
            cb_halted = (
                self._command_state.is_circuit_breaker_halted
                if self._command_state and hasattr(self._command_state, "is_circuit_breaker_halted")
                else False
            )
            if cb_halted:
                strategy_status = "CIRCUIT BREAKER HALTED (24h) — /resume to override"
            else:
                aave_supplied = aave_stats.supplied_usdc if aave_stats else 0
                parts = [
                    f"ALGO: {'active' if eth_val >= 10 else 'idle'}",
                    f"YIELD: {'supplied' if aave_supplied > 0 else 'idle'}",
                    "AIRDROP: ready",
                ]
                strategy_status = ", ".join(parts)

            from wallet_manager.security.telegram_manager import (
                HeartbeatPayload,
            )
            await self._telegram_manager.send_heartbeat(
                HeartbeatPayload(
                    total_tvl_usd=total_tvl,
                    open_positions_count=open_positions,
                    unrealized_pnl_usd=unrealized_pnl,
                    aave_yield_earned_usd=aave_yield_usd,
                    aave_apy_pct=aave_apy,
                    current_gas_gwei=current_gwei,
                    strategy_status=strategy_status,
                    paper_balance_usd=paper_balance_usd,
                    paper_starting_balance_usd=paper_starting,
                    total_trades=total_trades,
                    win_rate_pct=win_rate_pct,
                    best_trade_pnl=best_trade_pnl,
                    best_trade_pair=best_trade_pair,
                    worst_trade_pnl=worst_trade_pnl,
                    worst_trade_pair=worst_trade_pair,
                    last_trade_time=last_trade_time,
                    last_trade_token=last_trade_token,
                    last_trade_pnl=last_trade_pnl,
                    signal_status=signal_status,
                    gas_suitable=gas_suitable,
                    next_run_seconds=next_run_seconds,
                )
            )
            log.info("Heartbeat sent.")
        except Exception as exc:
            log.info(f"Heartbeat send skipped: {type(exc).__name__}: {exc}")


async def main() -> None:
    """Entry point: load configs, run MasterOrchestrator with 5-min Algo loop."""
    from wallet_manager.telegram_notifier import CommandState, ConfirmationGate

    configs = _load_wallet_configs()
    command_state = CommandState()
    confirmation_gate = ConfirmationGate()

    cfg = _cfg()
    await _validate_startup_config()

    # Migrate plaintext private keys to vault encryption before loading wallets
    if cfg.vault_key:
        try:
            from wallet_manager.security.vault import Vault
            vault = Vault()
            await _migrate_wallets_to_vault(vault, Path(cfg.wallet_config_path))
        except Exception as _vault_exc:
            log.warning(f"Vault migration skipped: {type(_vault_exc).__name__}: {_vault_exc}")

    use_master = cfg.use_master_orchestrator
    interval = cfg.algo_loop_interval_seconds if use_master else cfg.loop_interval_seconds

    orchestrator_cls = MasterOrchestrator if use_master else WalletOrchestrator
    orchestrator = orchestrator_cls(
        configs,
        task_factory=_three_pillar_task_factory,
        command_state=command_state,
        confirmation_gate=confirmation_gate,
    )
    log.info(
        f"Starting {'Master' if use_master else 'Wallet'}Orchestrator "
        f"(loop interval {interval:.0f}s)."
    )
    while True:
        if command_state.is_stopped:
            log.info("CommandState: STOP — exiting main loop.")
            break
        try:
            await orchestrator.run()
        except (KeyboardInterrupt, asyncio.CancelledError):
            raise
        if command_state.is_stopped:
            break
        log.info(
            f"Sleeping {interval:.0f}s before next run.",
        )
        await asyncio.sleep(interval)


def _setup_signal_handlers(loop: asyncio.AbstractEventLoop, command_state: object) -> None:
    """Register SIGTERM and SIGINT handlers for graceful shutdown.

    Translates OS signals into a :meth:`CommandState.stop` call so the
    orchestrator drains in-flight transactions before exiting.  Docker
    ``stop`` and systemd ``stop`` both use SIGTERM.

    Args:
        loop: The running event loop.
        command_state: Shared :class:`~wallet_manager.telegram_notifier.CommandState`.
    """
    import signal as _signal

    def _handle_signal(sig: _signal.Signals) -> None:
        log.warning(f"Received signal {sig.name} — requesting graceful shutdown.")
        if hasattr(command_state, "stop"):
            command_state.stop()

    for sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s))
        except (NotImplementedError, RuntimeError):
            # Windows does not support loop.add_signal_handler; fall back to
            # the default signal handler (KeyboardInterrupt will still work).
            pass


if __name__ == "__main__":  # pragma: no cover
    import signal as _signal_mod

    async def _main_with_signals() -> None:
        from wallet_manager.telegram_notifier import CommandState, ConfirmationGate

        configs = _load_wallet_configs()
        command_state = CommandState()
        confirmation_gate = ConfirmationGate()

        loop = asyncio.get_running_loop()
        _setup_signal_handlers(loop, command_state)

        cfg = _cfg()
        await _validate_startup_config()

        if cfg.vault_key:
            try:
                from wallet_manager.security.vault import Vault
                vault = Vault()
                await _migrate_wallets_to_vault(vault, Path(cfg.wallet_config_path))
            except Exception as _vault_exc:
                log.warning(f"Vault migration skipped: {type(_vault_exc).__name__}: {_vault_exc}")

        use_master = cfg.use_master_orchestrator
        interval = cfg.algo_loop_interval_seconds if use_master else cfg.loop_interval_seconds

        orchestrator_cls = MasterOrchestrator if use_master else WalletOrchestrator
        orchestrator = orchestrator_cls(
            configs,
            task_factory=_three_pillar_task_factory,
            command_state=command_state,
            confirmation_gate=confirmation_gate,
        )
        log.info(
            f"Starting {'Master' if use_master else 'Wallet'}Orchestrator "
            f"(loop interval {interval:.0f}s)."
        )
        while True:
            if command_state.is_stopped:
                log.info("CommandState: STOP — exiting main loop.")
                break
            try:
                await orchestrator.run()
            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            if command_state.is_stopped:
                break
            log.info(f"Sleeping {interval:.0f}s before next run.")
            await asyncio.sleep(interval)

    asyncio.run(_main_with_signals())
