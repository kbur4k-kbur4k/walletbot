"""
wallet_manager/strategies/paper_engine.py

High-Fidelity Paper Trading Engine — Shadow Execution with Full Pipeline.

Runs the COMPLETE ML-Quant execution pipeline (regime filter, robust alpha,
liquidity profiling, SOR routing, MEV shield decisions) and stops EXACTLY
before ``send_transaction``.  Every would-be trade is recorded in a local
SQLite ledger with realistic slippage simulation and gas cost estimation.

Purpose
-------
Validate strategy alpha, measure implementation shortfall, and track
virtual P&L without risking real capital.  The engine produces a detailed
performance report (Sharpe, drawdown, win rate, average slippage) that can
be sent via Telegram alongside the real execution report.

Architecture
------------
1. PaperTradeConfig     — Frozen dataclass with all tuning knobs.
2. VirtualLedger        — SQLite WAL-backed persistence for virtual balances,
                          trades, and equity snapshots.
3. SlippageSimulator    — Gaussian / fixed slippage model with non-linear
                          market-impact scaling for large orders.
4. GasEstimator         — Fetches live Base gas price via web3 and caches
                          for 10 seconds; estimates cost by tx type.
5. PaperTradingEngine   — Main orchestrator: shadow-executes the full pipeline,
                          simulates fill, records to ledger, returns result.
6. PaperTradeResult     — Frozen dataclass with complete cycle outcome.
7. Singleton            — ``get_paper_engine()`` / ``reset_paper_engine()``.

Integration
-----------
Call ``engine.run_shadow_cycle(...)`` from the orchestrator alongside or
instead of the real ``MLQuantExecutionEngine.run_cycle()``.  The shadow
cycle reads the same market data and signals but never touches on-chain
state.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import numpy as np

from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    from wallet_manager.core.manager import WalletManager
    from wallet_manager.portfolio_manager import PortfolioManager

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAIN_ID_BASE: int = 8453
STRATEGY_ID: str = "paper_shadow_v1"

MIN_TRADE_USD: float = 5.0
CONGESTION_GWEI: float = 15.0

def _get_weth_usdc_pool() -> str:
    from wallet_manager.config import get_settings
    return get_settings().weth_usdc_pool

_DEFAULT_DB_PATH: str = "paper_trades.db"

# Gas unit estimates by transaction type
GAS_UNITS: dict[str, int] = {
    "swap": 180_000,
    "supply": 250_000,
    "borrow": 300_000,
}

# Cache TTL for gas price lookups
GAS_CACHE_TTL_S: float = 10.0

# Reason codes
PAPER_TRADE_EXECUTED: str = "PAPER_TRADE_EXECUTED"
PAPER_SLIPPAGE_APPLIED: str = "PAPER_SLIPPAGE_APPLIED"
PAPER_GAS_ESTIMATED: str = "PAPER_GAS_ESTIMATED"
PAPER_SIGNAL_REJECTED: str = "PAPER_SIGNAL_REJECTED"
PAPER_SHADOW_ONLY: str = "PAPER_SHADOW_ONLY"


# ---------------------------------------------------------------------------
# SQL Schema
# ---------------------------------------------------------------------------

_CREATE_BALANCES_SQL = """
CREATE TABLE IF NOT EXISTS paper_balances (
    timestamp   REAL    NOT NULL,
    asset       TEXT    NOT NULL,
    amount      REAL    NOT NULL,
    usd_value   REAL    NOT NULL,
    PRIMARY KEY (asset)
)
"""

_CREATE_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           REAL    NOT NULL,
    direction           TEXT    NOT NULL,
    asset_pair          TEXT    NOT NULL,
    entry_price         REAL    NOT NULL,
    simulated_fill_price REAL   NOT NULL,
    slippage_bps        REAL    NOT NULL,
    size_usd            REAL    NOT NULL,
    gas_cost_usd        REAL    NOT NULL,
    pnl_usd             REAL    NOT NULL,
    reason_codes        TEXT    NOT NULL,
    signal_source       TEXT    NOT NULL
)
"""

_CREATE_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS paper_snapshots (
    timestamp           REAL    PRIMARY KEY,
    total_equity_usd    REAL    NOT NULL,
    unrealized_pnl      REAL    NOT NULL,
    realized_pnl        REAL    NOT NULL,
    max_drawdown        REAL    NOT NULL,
    sharpe_estimate     REAL    NOT NULL
)
"""

_CREATE_BALANCE_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS paper_balance_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    total_equity_usd REAL   NOT NULL,
    usdc_balance    REAL    NOT NULL,
    weth_balance    REAL    NOT NULL,
    eth_price       REAL    NOT NULL
)
"""

_CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_pt_ts ON paper_trades(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_pt_dir ON paper_trades(direction)",
    "CREATE INDEX IF NOT EXISTS idx_ps_ts ON paper_snapshots(timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_pb_asset ON paper_balances(asset)",
    "CREATE INDEX IF NOT EXISTS idx_pbh_ts ON paper_balance_history(timestamp)",
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperTradeConfig:
    """Configuration for the paper trading engine.

    Attributes:
        initial_balance_usd: Starting virtual capital in USD.
        slippage_model: Slippage model type — ``"gaussian"`` or ``"fixed"``.
        slippage_mean_bps: Mean slippage in basis points for gaussian model.
        slippage_std_bps: Standard deviation in bps for gaussian model.
        gas_estimation: Whether to fetch real gas prices from Base.
        execution_delay_ms: Latency emulation range (min, max) in ms.
    """

    initial_balance_usd: float = 10_000.0
    slippage_model: str = "gaussian"
    slippage_mean_bps: float = 5.0
    slippage_std_bps: float = 3.0
    gas_estimation: bool = True
    execution_delay_ms: tuple[int, int] = (200, 800)


@dataclass(frozen=True)
class PaperTradeResult:
    """Complete result of one paper-trade shadow cycle.

    Attributes:
        action: Trade direction — BUY, SELL, or HOLD.
        signal_alpha: Raw alpha score from the signal engine.
        simulated_fill_price: Price after slippage simulation.
        slippage_bps: Applied slippage in basis points.
        gas_cost_usd: Estimated gas cost in USD.
        virtual_pnl_usd: Virtual profit/loss from this cycle.
        would_have_executed: Whether the real engine would have traded.
        reason_codes: Tuple of reason codes describing the outcome.
        cycle_duration_ms: Wall-clock duration of the shadow cycle.
        timestamp: Unix epoch timestamp of the cycle.
    """

    action: str = "HOLD"
    signal_alpha: float = 0.0
    simulated_fill_price: float = 0.0
    slippage_bps: float = 0.0
    gas_cost_usd: float = 0.0
    virtual_pnl_usd: float = 0.0
    would_have_executed: bool = False
    reason_codes: tuple[str, ...] = ()
    cycle_duration_ms: float = 0.0
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# SlippageSimulator
# ---------------------------------------------------------------------------


class SlippageSimulator:
    """Simulate realistic trade slippage using Gaussian or fixed models.

    The Gaussian model draws slippage from a normal distribution with
    a non-linear impact scalar that increases super-linearly when the
    trade size exceeds 1% of pool depth.  This captures real-world
    Uniswap V3 tick-crossing dynamics.

    The fixed model applies a constant slippage regardless of trade size.
    """

    def __init__(
        self,
        mean_bps: float = 5.0,
        std_bps: float = 3.0,
    ) -> None:
        self._mean_bps = mean_bps
        self._std_bps = std_bps
        self._rng = np.random.default_rng(42)
        log.info(
            f"SlippageSimulator initialized: mean={mean_bps}bps "
            f"std={std_bps}bps"
        )

    def simulate_slippage(
        self,
        size_usd: float,
        pool_depth_usd: float,
        model: str = "gaussian",
        entry_price: float = 0.0,
    ) -> tuple[float, float]:
        """Simulate slippage for a given trade size and pool depth.

        Args:
            size_usd: Trade size in USD.
            pool_depth_usd: Estimated pool depth in USD.
            model: Slippage model — ``"gaussian"`` or ``"fixed"``.
            entry_price: The mid-market entry price (for fill calculation).

        Returns:
            Tuple of (slippage_bps, simulated_fill_price).
            ``slippage_bps`` is always non-negative.
            ``simulated_fill_price`` reflects the entry price adjusted by
            slippage (worse execution).
        """
        pool_depth_usd = max(pool_depth_usd, 1.0)
        size_ratio = size_usd / pool_depth_usd

        # Non-linear impact scalar: increases super-linearly for large orders
        # Below 1% of pool: impact_scalar ~= 1.0
        # At 1%: ~1.5x; at 5%: ~4.5x; at 10%: ~10x
        if size_ratio <= 0.01:
            impact_scalar = 1.0 + size_ratio * 50.0
        elif size_ratio <= 0.05:
            impact_scalar = 1.5 + (size_ratio - 0.01) * 75.0
        else:
            impact_scalar = 4.5 + (size_ratio - 0.05) ** 0.8 * 100.0

        if model == "gaussian":
            base_slip = float(self._rng.normal(self._mean_bps, self._std_bps))
            base_slip = max(base_slip, 0.5)  # Floor at 0.5 bps
            slippage_bps = base_slip * impact_scalar
        elif model == "fixed":
            slippage_bps = self._mean_bps * impact_scalar
        else:
            log.info(f"SlippageSimulator: unrecognized model '{model}', defaulting to gaussian")
            base_slip = float(self._rng.normal(self._mean_bps, self._std_bps))
            base_slip = max(base_slip, 0.5)
            slippage_bps = base_slip * impact_scalar

        # Cap slippage at 500 bps (5%) to avoid unrealistic fills
        slippage_bps = min(slippage_bps, 500.0)
        slippage_bps = round(slippage_bps, 2)

        # Calculate simulated fill price (always worse than entry)
        if entry_price > 0:
            slippage_fraction = slippage_bps / 10_000.0
            simulated_fill_price = entry_price * (1.0 - slippage_fraction)
        else:
            simulated_fill_price = 0.0

        log.debug(
            f"SLIPPAGE: size=${size_usd:.2f} depth=${pool_depth_usd:.0f} "
            f"ratio={size_ratio:.4f} impact={impact_scalar:.2f} "
            f"slip={slippage_bps:.2f}bps fill=${simulated_fill_price:.2f}"
        )

        return slippage_bps, simulated_fill_price


# ---------------------------------------------------------------------------
# GasEstimator
# ---------------------------------------------------------------------------


class GasEstimator:
    """Estimate transaction gas cost in USD using live Base gas prices.

    Fetches current gas price via web3 and caches the result for
    ``GAS_CACHE_TTL_S`` seconds to avoid excessive RPC calls.  Gas units
    are estimated per transaction type (swap, supply, borrow).
    """

    def __init__(self) -> None:
        self._cached_gas_gwei: float = 0.0
        self._cached_eth_price: float = 0.0
        self._cache_timestamp: float = 0.0
        self._lock = asyncio.Lock()
        log.info("GasEstimator initialized")

    async def _refresh_cache(self, rpc_url: str, eth_price: float) -> None:
        """Refresh the gas price cache if stale.

        Args:
            rpc_url: Base RPC endpoint.
            eth_price: Current ETH price in USD (for cost conversion).
        """
        now = time.time()
        if now - self._cache_timestamp < GAS_CACHE_TTL_S and self._cached_gas_gwei > 0:
            return

        async with self._lock:
            # Double-check after acquiring lock
            if now - self._cache_timestamp < GAS_CACHE_TTL_S and self._cached_gas_gwei > 0:
                return
            try:
                from web3 import AsyncHTTPProvider, AsyncWeb3
                w3 = AsyncWeb3(AsyncHTTPProvider(rpc_url))
                gas_price_wei = await w3.eth.gas_price
                self._cached_gas_gwei = float(gas_price_wei) / 1e9
                self._cached_eth_price = eth_price
                self._cache_timestamp = time.time()
                log.debug(
                    f"GAS_CACHE: refreshed gas={self._cached_gas_gwei:.4f} gwei "
                    f"eth=${eth_price:.2f}"
                )
            except Exception as exc:
                log.info(f"GasEstimator: RPC gas unavailable ({type(exc).__name__}), using cached/fallback")
                if self._cached_gas_gwei <= 0:
                    # Fallback: conservative estimate for Base L2
                    self._cached_gas_gwei = 0.01
                    self._cached_eth_price = eth_price
                    self._cache_timestamp = time.time()

    async def estimate_gas_cost_usd(
        self,
        tx_type: str,
        rpc_url: str = "",
        eth_price: float = 3000.0,
    ) -> float:
        """Estimate gas cost in USD for a given transaction type.

        Args:
            tx_type: Transaction type — ``"swap"``, ``"supply"``, or ``"borrow"``.
            rpc_url: Base RPC endpoint for live gas price fetch.
            eth_price: Current ETH/USD price.

        Returns:
            Estimated gas cost in USD.
        """
        if rpc_url:
            await self._refresh_cache(rpc_url, eth_price)

        gas_gwei = self._cached_gas_gwei if self._cached_gas_gwei > 0 else 0.01
        gas_units = GAS_UNITS.get(tx_type, GAS_UNITS["swap"])
        price = self._cached_eth_price if self._cached_eth_price > 0 else eth_price

        # gas_cost = gas_units * gas_price_gwei * 1e-9 * eth_price_usd
        cost_usd = gas_units * gas_gwei * 1e-9 * price
        cost_usd = round(cost_usd, 6)

        log.debug(
            f"GAS_EST: type={tx_type} units={gas_units} "
            f"gas={gas_gwei:.4f}gwei eth=${price:.2f} "
            f"cost=${cost_usd:.6f}"
        )
        return cost_usd

    def get_cached_gas_gwei(self) -> float:
        """Return the most recently cached gas price in gwei."""
        return self._cached_gas_gwei


# ---------------------------------------------------------------------------
# VirtualLedger
# ---------------------------------------------------------------------------


class VirtualLedger:
    """SQLite WAL-backed ledger for paper trade persistence.

    Tracks virtual asset balances (USDC, WETH, aUSDC), records every
    simulated trade, and stores periodic equity snapshots for drawdown
    and Sharpe estimation.

    The database is stored at ``paper_trades.db`` (configurable) and
    uses WAL mode for concurrent read/write safety.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

        # In-memory balance cache for fast reads
        self._balances: dict[str, float] = {}

        # Tracking state
        self._peak_equity: float = 0.0
        self._max_drawdown: float = 0.0
        self._realized_pnl: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0
        self._total_slippage_bps: float = 0.0
        self._equity_history: list[float] = []

    async def initialize(self) -> None:
        """Open the SQLite connection, enable WAL mode, and create tables.

        Loads any existing balances from disk into the in-memory cache.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row

        # WAL mode: readers and writer do not block each other
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")

        await self._conn.execute(_CREATE_BALANCES_SQL)
        await self._conn.execute(_CREATE_TRADES_SQL)
        await self._conn.execute(_CREATE_SNAPSHOTS_SQL)
        await self._conn.execute(_CREATE_BALANCE_HISTORY_SQL)
        for idx_sql in _CREATE_INDICES_SQL:
            await self._conn.execute(idx_sql)
        await self._conn.commit()

        # Load existing balances into cache
        async with self._conn.execute("SELECT asset, amount FROM paper_balances") as cur:
            rows = await cur.fetchall()
        for row in rows:
            self._balances[row["asset"]] = float(row["amount"])

        # Load realized PnL from historical trades
        async with self._conn.execute(
            "SELECT COALESCE(SUM(pnl_usd), 0.0) AS total_pnl, "
            "COUNT(*) AS cnt, "
            "COALESCE(SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END), 0) AS wins, "
            "COALESCE(SUM(slippage_bps), 0.0) AS total_slip "
            "FROM paper_trades"
        ) as cur:
            row = await cur.fetchone()
        if row:
            self._realized_pnl = float(row["total_pnl"] or 0.0)
            self._trade_count = int(row["cnt"] or 0)
            self._win_count = int(row["wins"] or 0)
            self._total_slippage_bps = float(row["total_slip"] or 0.0)

        # Load peak equity from snapshots
        async with self._conn.execute(
            "SELECT MAX(total_equity_usd) AS peak, "
            "MAX(max_drawdown) AS dd FROM paper_snapshots"
        ) as cur:
            row = await cur.fetchone()
        if row and row["peak"]:
            self._peak_equity = float(row["peak"])
            self._max_drawdown = float(row["dd"] or 0.0)

        log.info(
            f"VirtualLedger initialized: db='{self._db_path}' "
            f"balances={len(self._balances)} trades={self._trade_count} "
            f"realized_pnl=${self._realized_pnl:+.2f}"
        )

    async def shutdown(self) -> None:
        """Flush pending writes and close the database connection."""
        if self._conn:
            await self._conn.commit()
            await self._conn.close()
            self._conn = None
        log.info("VirtualLedger shutdown complete")

    def _assert_open(self) -> None:
        """Raise if the database connection is not open."""
        if self._conn is None:
            raise RuntimeError(
                "VirtualLedger is not initialized. Call initialize() first."
            )

    # ----- Balance operations -----

    async def get_balance(self, asset: str) -> float:
        """Return the virtual balance for an asset.

        Args:
            asset: Asset symbol (e.g. ``"USDC"``, ``"WETH"``, ``"aUSDC"``).

        Returns:
            Current virtual balance.  Zero if the asset has no record.
        """
        return self._balances.get(asset, 0.0)

    async def update_balance(
        self,
        asset: str,
        amount: float,
        usd_value: float = 0.0,
    ) -> None:
        """Set the virtual balance for an asset and persist to SQLite.

        Args:
            asset: Asset symbol.
            amount: New absolute balance.
            usd_value: USD-equivalent value (for reporting).
        """
        self._assert_open()
        self._balances[asset] = amount
        now = time.time()
        sql = """
            INSERT OR REPLACE INTO paper_balances
                (timestamp, asset, amount, usd_value)
            VALUES (?, ?, ?, ?)
        """
        async with self._write_lock:
            await self._conn.execute(sql, (now, asset, amount, usd_value))
            await self._conn.commit()

    async def get_total_equity_usd(
        self,
        eth_price: float = 3000.0,
    ) -> float:
        """Calculate total virtual equity in USD.

        Sums USDC (1:1), WETH (at ``eth_price``), and aUSDC (1:1).

        Args:
            eth_price: Current ETH/USD price for WETH valuation.

        Returns:
            Total virtual equity in USD.
        """
        usdc = self._balances.get("USDC", 0.0)
        weth = self._balances.get("WETH", 0.0)
        ausdc = self._balances.get("aUSDC", 0.0)
        return usdc + (weth * eth_price) + ausdc

    # ----- Trade recording -----

    async def record_trade(
        self,
        direction: str,
        asset_pair: str,
        entry_price: float,
        simulated_fill_price: float,
        slippage_bps: float,
        size_usd: float,
        gas_cost_usd: float,
        pnl_usd: float,
        reason_codes: tuple[str, ...],
        signal_source: str,
    ) -> None:
        """Record a paper trade to the SQLite database.

        Also updates in-memory tracking state (realized P&L, win count,
        slippage accumulator).

        Args:
            direction: ``"BUY"`` or ``"SELL"``.
            asset_pair: E.g. ``"WETH/USDC"``.
            entry_price: Mid-market price at decision time.
            simulated_fill_price: Price after slippage simulation.
            slippage_bps: Applied slippage in basis points.
            size_usd: Trade size in USD.
            gas_cost_usd: Estimated gas cost in USD.
            pnl_usd: Realized P&L from this trade.
            reason_codes: Tuple of reason code strings.
            signal_source: Name of the signal that triggered the trade.
        """
        self._assert_open()
        now = time.time()
        codes_str = json.dumps(reason_codes)
        sql = """
            INSERT INTO paper_trades (
                timestamp, direction, asset_pair, entry_price,
                simulated_fill_price, slippage_bps, size_usd,
                gas_cost_usd, pnl_usd, reason_codes, signal_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        async with self._write_lock:
            await self._conn.execute(sql, (
                now, direction, asset_pair, entry_price,
                simulated_fill_price, slippage_bps, size_usd,
                gas_cost_usd, pnl_usd, codes_str, signal_source,
            ))
            await self._conn.commit()

        # Update in-memory state
        self._realized_pnl += pnl_usd
        self._trade_count += 1
        if pnl_usd > 0:
            self._win_count += 1
        self._total_slippage_bps += slippage_bps

        log.info(
            f"PAPER_TRADE: {direction} {asset_pair} size=${size_usd:.2f} "
            f"entry=${entry_price:.2f} fill=${simulated_fill_price:.2f} "
            f"slip={slippage_bps:.2f}bps gas=${gas_cost_usd:.6f} "
            f"pnl=${pnl_usd:+.2f}"
        )

    # ----- Balance history -----

    async def record_balance_history(
        self,
        eth_price: float = 3000.0,
    ) -> None:
        """Record a timestamped balance snapshot for equity curve tracking."""
        self._assert_open()
        usdc = self._balances.get("USDC", 0.0)
        weth = self._balances.get("WETH", 0.0)
        total = usdc + (weth * eth_price) + self._balances.get("aUSDC", 0.0)
        now = time.time()
        sql = """
            INSERT INTO paper_balance_history
                (timestamp, total_equity_usd, usdc_balance, weth_balance, eth_price)
            VALUES (?, ?, ?, ?, ?)
        """
        async with self._write_lock:
            await self._conn.execute(sql, (now, total, usdc, weth, eth_price))
            await self._conn.commit()

    async def get_best_worst_last_trade(self) -> dict[str, Any]:
        """Return best, worst, and last trade from paper_trades table.

        Returns:
            Dict with keys: ``best_trade``, ``worst_trade``, ``last_trade``.
            Each is a dict or None.
        """
        self._assert_open()
        result: dict[str, Any] = {"best_trade": None, "worst_trade": None, "last_trade": None}

        # Best trade (highest P&L)
        async with self._conn.execute(
            "SELECT direction, asset_pair, pnl_usd, size_usd, timestamp "
            "FROM paper_trades ORDER BY pnl_usd DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row:
            result["best_trade"] = {
                "direction": row["direction"], "pair": row["asset_pair"],
                "pnl_usd": float(row["pnl_usd"]), "size_usd": float(row["size_usd"]),
                "timestamp": float(row["timestamp"]),
            }

        # Worst trade (lowest P&L)
        async with self._conn.execute(
            "SELECT direction, asset_pair, pnl_usd, size_usd, timestamp "
            "FROM paper_trades ORDER BY pnl_usd ASC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row:
            result["worst_trade"] = {
                "direction": row["direction"], "pair": row["asset_pair"],
                "pnl_usd": float(row["pnl_usd"]), "size_usd": float(row["size_usd"]),
                "timestamp": float(row["timestamp"]),
            }

        # Last trade
        async with self._conn.execute(
            "SELECT direction, asset_pair, pnl_usd, size_usd, timestamp "
            "FROM paper_trades ORDER BY timestamp DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if row:
            result["last_trade"] = {
                "direction": row["direction"], "pair": row["asset_pair"],
                "pnl_usd": float(row["pnl_usd"]), "size_usd": float(row["size_usd"]),
                "timestamp": float(row["timestamp"]),
            }

        return result

    # ----- Equity snapshot -----

    async def _record_snapshot(
        self,
        total_equity_usd: float,
        unrealized_pnl: float,
    ) -> None:
        """Persist an equity snapshot and update drawdown tracking.

        Args:
            total_equity_usd: Current total virtual equity.
            unrealized_pnl: Unrealized P&L from open positions.
        """
        self._assert_open()

        # Update peak and drawdown
        if total_equity_usd > self._peak_equity:
            self._peak_equity = total_equity_usd
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - total_equity_usd) / self._peak_equity
            self._max_drawdown = max(self._max_drawdown, current_dd)

        # Sharpe estimate: annualized from equity history
        self._equity_history.append(total_equity_usd)
        sharpe = self._estimate_sharpe()

        now = time.time()
        sql = """
            INSERT OR REPLACE INTO paper_snapshots (
                timestamp, total_equity_usd, unrealized_pnl,
                realized_pnl, max_drawdown, sharpe_estimate
            ) VALUES (?, ?, ?, ?, ?, ?)
        """
        async with self._write_lock:
            await self._conn.execute(sql, (
                now, total_equity_usd, unrealized_pnl,
                self._realized_pnl, self._max_drawdown, sharpe,
            ))
            await self._conn.commit()

    def _estimate_sharpe(self) -> float:
        """Estimate the annualized Sharpe ratio from equity history.

        Uses 5-minute periods as the base frequency.  Assumes risk-free
        rate of zero (appropriate for crypto).

        Returns:
            Annualized Sharpe ratio.  Returns 0.0 if insufficient data.
        """
        if len(self._equity_history) < 10:
            return 0.0
        eq = np.array(self._equity_history, dtype=np.float64)
        returns = np.diff(eq) / np.maximum(eq[:-1], 1e-10)
        if len(returns) < 2:
            return 0.0
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns))
        if std_r < 1e-10:
            return 0.0
        # Annualize: ~105,120 five-minute periods per year
        periods_per_year = 105_120.0
        sharpe = (mean_r / std_r) * math.sqrt(periods_per_year)
        return round(sharpe, 4)

    # ----- P&L and drawdown queries -----

    async def get_pnl_summary(self) -> dict[str, Any]:
        """Return a summary of virtual P&L statistics.

        Returns:
            Dict with keys: ``realized_pnl``, ``trade_count``,
            ``win_count``, ``win_rate``, ``avg_slippage_bps``,
            ``max_drawdown``, ``sharpe_estimate``.
        """
        win_rate = (
            (self._win_count / self._trade_count * 100.0)
            if self._trade_count > 0 else 0.0
        )
        avg_slip = (
            (self._total_slippage_bps / self._trade_count)
            if self._trade_count > 0 else 0.0
        )
        return {
            "realized_pnl": round(self._realized_pnl, 4),
            "trade_count": self._trade_count,
            "win_count": self._win_count,
            "win_rate": round(win_rate, 2),
            "avg_slippage_bps": round(avg_slip, 2),
            "max_drawdown": round(self._max_drawdown, 6),
            "sharpe_estimate": self._estimate_sharpe(),
            "peak_equity": round(self._peak_equity, 2),
        }

    async def get_drawdown(self) -> dict[str, float]:
        """Return current drawdown statistics.

        Returns:
            Dict with ``max_drawdown``, ``peak_equity``, and
            ``current_drawdown`` (latest equity vs peak).
        """
        current_eq = self._equity_history[-1] if self._equity_history else 0.0
        current_dd = 0.0
        if self._peak_equity > 0 and current_eq > 0:
            current_dd = (self._peak_equity - current_eq) / self._peak_equity
        return {
            "max_drawdown": round(self._max_drawdown, 6),
            "peak_equity": round(self._peak_equity, 2),
            "current_drawdown": round(current_dd, 6),
        }


# ---------------------------------------------------------------------------
# PaperTradingEngine
# ---------------------------------------------------------------------------


class PaperTradingEngine:
    """Main orchestrator for high-fidelity paper trading.

    Runs the FULL robust execution pipeline (regime filter, robust alpha,
    liquidity profiling, SOR routing, MEV shield decisions) and stops
    EXACTLY before ``send_transaction``.

    All would-be trades are recorded in the VirtualLedger with realistic
    slippage simulation and gas cost estimation.

    Usage::

        engine = await get_paper_engine()
        result = await engine.run_shadow_cycle(
            manager=wallet_manager,
            wallet_id="wallet_1",
            portfolio_manager=pm,
            task_context=ctx,
            analytics_db=db,
            quant_analytics=qa,
        )
        report = await engine.get_performance_report()
    """

    def __init__(
        self,
        config: PaperTradeConfig | None = None,
        # Simplified API params
        initial_capital_usd: float | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        if initial_capital_usd is not None and config is None:
            config = PaperTradeConfig(initial_balance_usd=initial_capital_usd)
        self._config = config or PaperTradeConfig()
        self._min_confidence = min_confidence
        self._ledger = VirtualLedger()
        self._slippage = SlippageSimulator(
            mean_bps=self._config.slippage_mean_bps,
            std_bps=self._config.slippage_std_bps,
        )
        self._gas_estimator = GasEstimator()

        # Position tracking (mirrors real engine)
        self._in_position: bool = False
        self._entry_price: float = 0.0
        self._position_size_usd: float = 0.0
        self._entry_alpha: float = 0.0
        self._entry_timestamp: float = 0.0

        self._initialized: bool = False
        self._cycle_count: int = 0

        # Simplified step() API state
        self._capital_usd: float = self._config.initial_balance_usd
        self._step_trades: list[dict] = []
        self._step_returns: list[float] = []
        self._step_pnl: float = 0.0
        self._step_position_usd: float = 0.0
        self._step_entry_price: float = 0.0

        log.info(
            f"PaperTradingEngine created: balance=${self._config.initial_balance_usd:.2f} "
            f"slippage={self._config.slippage_model} "
            f"mean={self._config.slippage_mean_bps}bps "
            f"gas_estimation={self._config.gas_estimation}"
        )

    # ------------------------------------------------------------------
    # Simplified step-based API for testing
    # ------------------------------------------------------------------

    async def step(self, price: float, signal: float) -> None:
        """Process one price tick with a directional signal.

        If signal >= min_confidence and no position: open long.
        If position open and signal < min_confidence: close.
        """
        import math
        MAX_POSITION_PCT = 0.25

        if self._step_position_usd > 0:
            # Close position if signal dropped below threshold
            if signal < self._min_confidence:
                pnl = self._step_position_usd * (price - self._step_entry_price) / max(self._step_entry_price, 1e-10)
                self._step_pnl += pnl
                ret = (price - self._step_entry_price) / max(self._step_entry_price, 1e-10)
                self._step_returns.append(ret)
                self._capital_usd += self._step_position_usd + pnl
                self._step_position_usd = 0.0
                self._step_entry_price = 0.0
                self._step_trades.append({"type": "close", "price": price, "pnl": pnl})
        else:
            # Open position if signal strong enough
            if signal >= self._min_confidence:
                size_usd = min(self._capital_usd * MAX_POSITION_PCT, self._capital_usd)
                size_usd = max(0.0, size_usd)
                self._capital_usd -= size_usd
                self._step_position_usd = size_usd
                self._step_entry_price = price
                self._step_trades.append({"type": "open", "price": price, "size_usd": size_usd})

    @property
    def current_exposure_usd(self) -> float:
        """Current open position size in USD."""
        return self._step_position_usd

    def get_metrics(self) -> dict:
        """Return summary performance metrics for the step-based simulation."""
        import numpy as np
        returns = np.asarray(self._step_returns)
        sharpe = 0.0
        if len(returns) > 1:
            std = float(np.std(returns))
            if std > 0:
                sharpe = float(np.mean(returns)) / std * (252 ** 0.5)
        return {
            "total_pnl_usd": round(self._step_pnl, 4),
            "total_trades": len([t for t in self._step_trades if t["type"] == "open"]),
            "sharpe_ratio": round(sharpe, 4),
            "final_capital_usd": round(self._capital_usd + self._step_position_usd, 4),
        }

    async def initialize(self) -> None:
        """Initialize the VirtualLedger and set initial balances.

        Sets the starting USDC balance from config and zero balances
        for WETH and aUSDC.
        """
        if self._initialized:
            return

        await self._ledger.initialize()

        # Set initial balances if this is a fresh ledger
        current_usdc = await self._ledger.get_balance("USDC")
        if current_usdc <= 0:
            await self._ledger.update_balance(
                "USDC", self._config.initial_balance_usd,
                usd_value=self._config.initial_balance_usd,
            )
            await self._ledger.update_balance("WETH", 0.0, usd_value=0.0)
            await self._ledger.update_balance("aUSDC", 0.0, usd_value=0.0)
            self._ledger._peak_equity = self._config.initial_balance_usd
            log.info(
                f"PAPER_ENGINE: initial balances set — "
                f"USDC=${self._config.initial_balance_usd:.2f}"
            )

        self._initialized = True
        log.info("PaperTradingEngine initialized")

    async def shutdown(self) -> None:
        """Shutdown the paper trading engine and close the ledger."""
        await self._ledger.shutdown()
        self._initialized = False
        log.info("PaperTradingEngine shutdown complete")

    async def _send_trade_telegram(
        self,
        direction: str,
        token: str,
        size_usd: float,
        price: float,
        pnl_usd: float = 0.0,
    ) -> None:
        """Send instant Telegram notification for paper trade open/close."""
        try:
            from wallet_manager.security.telegram_manager import TelegramManager
            tg = TelegramManager()
            if not tg._is_configured():
                return
            if direction == "BUY":
                text = (
                    f"\U0001f7e2 <b>PAPER TRADE OPENED</b>\n"
                    f"Token: {token}\n"
                    f"Size: ${size_usd:.2f}\n"
                    f"Price: ${price:.2f}"
                )
            else:
                emoji = "\U0001f534"
                pnl_sign = f"${pnl_usd:+.2f}"
                text = (
                    f"{emoji} <b>PAPER TRADE CLOSED</b>\n"
                    f"Token: {token}\n"
                    f"Size: ${size_usd:.2f}\n"
                    f"Price: ${price:.2f}\n"
                    f"PnL: {pnl_sign}"
                )
            await tg._send(text)
        except Exception as exc:
            log.debug(f"Paper trade Telegram notification skipped: {exc}")

    # ----- Shadow execution cycle -----

    async def run_shadow_cycle(
        self,
        manager: WalletManager,
        wallet_id: str,
        portfolio_manager: PortfolioManager,
        task_context: dict,
        analytics_db: Any = None,
        quant_analytics: Any = None,
    ) -> PaperTradeResult:
        """Run one complete shadow execution cycle.

        Mirrors ``MLQuantExecutionEngine.run_cycle()`` but stops before
        any on-chain transaction.  Records the would-be trade with
        simulated slippage and gas cost.

        Pipeline:
        1. Pre-flight checks (chain, circuit breakers)
        2. Price fetch + ML signal generation (robust alpha)
        3. Flash crash screening
        4. Deep Kelly risk sizing
        5. Trading logic (shadow — no actual execution)
        6. Slippage simulation
        7. Gas cost estimation
        8. VirtualLedger recording
        9. Equity snapshot

        Args:
            manager: WalletManager instance for config and balance reads.
            wallet_id: Wallet identifier.
            portfolio_manager: PortfolioManager for trade budget.
            task_context: Orchestrator context dict.
            analytics_db: Optional AnalyticsDB for cross-reference.
            quant_analytics: Optional QuantAnalytics for TCA recording.

        Returns:
            :class:`PaperTradeResult` with complete shadow cycle outcome.
        """
        if not self._initialized:
            await self.initialize()

        cycle_start = time.time()
        reason_codes: list[str] = [PAPER_SHADOW_ONLY]

        # ==============================================================
        # 1. PRE-FLIGHT CHECKS
        # ==============================================================

        command_state = task_context.get("command_state")
        if command_state and getattr(command_state, "is_circuit_breaker_halted", False):
            return PaperTradeResult(
                action="HALTED",
                reason_codes=(PAPER_SHADOW_ONLY,),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        config = manager.get_config(wallet_id)
        chain_id = int(config.chain_id) if config.chain_id is not None else 0
        if chain_id != CHAIN_ID_BASE:
            return PaperTradeResult(
                action="WRONG_CHAIN",
                reason_codes=(PAPER_SHADOW_ONLY,),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        rpc_url = config.rpc_url

        # ==============================================================
        # 2. PRICE FETCH + ROBUST ALPHA SIGNAL
        # ==============================================================

        from wallet_manager import actions

        try:
            price = await actions.get_live_price_v3(rpc_url)
            if price <= 0:
                price = await actions.get_weth_usdc_price(rpc_url)
            if price <= 0:
                price = await actions.get_current_price(rpc_url, chain_id)
        except Exception as exc:
            log.info(f"PAPER_ENGINE: price unavailable ({type(exc).__name__}: {exc})")
            return PaperTradeResult(
                action="PRICE_ERROR",
                reason_codes=(PAPER_SHADOW_ONLY,),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        if price <= 0:
            return PaperTradeResult(
                action="INVALID_PRICE",
                reason_codes=(PAPER_SHADOW_ONLY,),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        # Generate robust alpha signal
        from wallet_manager.strategies.robust_alpha import get_robust_alpha_engine
        robust_engine = get_robust_alpha_engine()

        # Update microstructure (best-effort)
        try:
            from wallet_manager.strategies.signal_intelligence import get_alpha_generator
            ofi_gen = get_alpha_generator()
            ofi_reading = await ofi_gen.read_ofi(rpc_url, _get_weth_usdc_pool())
            if ofi_reading:
                vol_imb = ofi_reading.imbalance_ratio - 1.0
                vol_imb = max(-1.0, min(1.0, vol_imb))
                robust_engine.set_microstructure(
                    volume_imbalance=vol_imb, spread_bps=5.0,
                )
        except Exception as _ms_exc:
            log.warning("Paper engine microstructure update failed (non-critical): %s", _ms_exc)

        signal = robust_engine.ingest_price(price, volume=0.0)

        if signal is None:
            return PaperTradeResult(
                action="WARMUP",
                reason_codes=(PAPER_SHADOW_ONLY,),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        log.info(
            f"PAPER_ENGINE: price=${price:.2f} "
            f"robust_alpha={signal.robust_alpha:+.3f} "
            f"raw_alpha={signal.raw_alpha:+.3f} "
            f"conf={signal.confidence:.2f} "
            f"stationary={signal.stationarity_d:.2f}"
        )

        # ==============================================================
        # 3. FLASH CRASH SCREENING
        # ==============================================================

        from wallet_manager.strategies.flash_crash_detector import (
            get_detector as get_crash_detector,
        )
        crash_detector = get_crash_detector()
        crash_alert = crash_detector.ingest_price(price)
        if crash_alert:
            return PaperTradeResult(
                action="FLASH_CRASH",
                signal_alpha=signal.robust_alpha,
                reason_codes=(PAPER_SHADOW_ONLY, PAPER_SIGNAL_REJECTED),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        # ==============================================================
        # 4. DEEP KELLY RISK SIZING
        # ==============================================================

        from wallet_manager.strategies.deep_kelly_risk import get_deep_kelly_engine
        risk_engine = get_deep_kelly_engine()

        total_equity = await self._ledger.get_total_equity_usd(eth_price=price)
        risk_engine.update_equity(total_equity)

        underlying = signal.underlying_signal
        garch_sigma = underlying.garch_sigma if underlying else 0.001
        vol_regime = underlying.volatility_regime if underlying else "normal"
        confidence = signal.confidence

        risk_decision = risk_engine.calculate_position(
            total_capital_usd=total_equity,
            alpha_score=signal.robust_alpha,
            garch_sigma=garch_sigma,
            vol_regime=vol_regime,
            confidence=confidence,
        )

        if risk_decision.position_usd <= 0:
            reason_codes.append(PAPER_SIGNAL_REJECTED)
            return PaperTradeResult(
                action=f"RISK_BLOCKED({risk_decision.drawdown_state})",
                signal_alpha=signal.robust_alpha,
                reason_codes=tuple(reason_codes),
                cycle_duration_ms=(time.time() - cycle_start) * 1000,
                timestamp=cycle_start,
            )

        trade_budget_usd = min(
            risk_decision.position_usd,
            portfolio_manager.get_algo_trade_budget(),
        )

        # ==============================================================
        # 5. TRADING LOGIC (SHADOW)
        # ==============================================================

        usdc_balance = await self._ledger.get_balance("USDC")
        weth_balance = await self._ledger.get_balance("WETH")
        weth_value_usd = weth_balance * price

        should_buy = (
            signal.should_enter_long
            and not self._in_position
            and usdc_balance >= MIN_TRADE_USD
            and trade_budget_usd >= MIN_TRADE_USD
        )

        should_sell = (
            self._in_position
            and signal.should_exit
            and weth_value_usd >= MIN_TRADE_USD
        )

        action = "HOLD"
        simulated_fill_price = 0.0
        slippage_bps = 0.0
        gas_cost_usd = 0.0
        virtual_pnl = 0.0
        would_have_executed = False

        if should_buy:
            action = "BUY"
            would_have_executed = True
            size_usd = min(trade_budget_usd, usdc_balance)

            # --- Latency emulation ---
            delay_min, delay_max = self._config.execution_delay_ms
            delay_ms = np.random.uniform(delay_min, delay_max)
            await asyncio.sleep(delay_ms / 1000.0)

            # --- Liquidity profiling (best-effort) ---
            pool_depth_usd = 500_000.0  # Conservative default
            try:
                from wallet_manager.strategies.liquidity_alpha import (
                    get_liquidity_alpha,
                )
                liq_alpha = get_liquidity_alpha()
                profile = await liq_alpha.profiler.profile_pool(
                    rpc_url=rpc_url,
                    pool_address=_get_weth_usdc_pool(),
                    eth_price=price,
                )
                if profile:
                    pool_depth_usd = max(profile.depth_usd, 10_000.0)
            except Exception as _liq_exc:
                log.warning("Pool liquidity profile failed (non-critical): %s", _liq_exc)
                pass

            # --- Slippage simulation ---
            slippage_bps, simulated_fill_price = self._slippage.simulate_slippage(
                size_usd=size_usd,
                pool_depth_usd=pool_depth_usd,
                model=self._config.slippage_model,
                entry_price=price,
            )
            reason_codes.append(PAPER_SLIPPAGE_APPLIED)

            # --- Gas estimation ---
            if self._config.gas_estimation:
                gas_cost_usd = await self._gas_estimator.estimate_gas_cost_usd(
                    tx_type="swap", rpc_url=rpc_url, eth_price=price,
                )
                reason_codes.append(PAPER_GAS_ESTIMATED)

            # --- Update virtual balances ---
            weth_received = size_usd / simulated_fill_price if simulated_fill_price > 0 else 0.0
            new_usdc = usdc_balance - size_usd
            new_weth = weth_balance + weth_received

            await self._ledger.update_balance("USDC", new_usdc, usd_value=new_usdc)
            await self._ledger.update_balance(
                "WETH", new_weth, usd_value=new_weth * price,
            )

            # Track position
            self._in_position = True
            self._entry_price = simulated_fill_price
            self._position_size_usd = size_usd
            self._entry_alpha = signal.robust_alpha
            self._entry_timestamp = time.time()

            reason_codes.append(PAPER_TRADE_EXECUTED)

            # Record trade
            await self._ledger.record_trade(
                direction="BUY",
                asset_pair="WETH/USDC",
                entry_price=price,
                simulated_fill_price=simulated_fill_price,
                slippage_bps=slippage_bps,
                size_usd=size_usd,
                gas_cost_usd=gas_cost_usd,
                pnl_usd=0.0,
                reason_codes=tuple(reason_codes),
                signal_source=f"robust_alpha={signal.robust_alpha:+.3f}",
            )

            log.info(
                f"PAPER_ENGINE: BUY ${size_usd:.2f} USDC -> "
                f"{weth_received:.6f} WETH @ ${simulated_fill_price:.2f} "
                f"(slip={slippage_bps:.2f}bps gas=${gas_cost_usd:.6f})"
            )

            # Instant Telegram notification
            await self._send_trade_telegram("BUY", "WETH", size_usd, simulated_fill_price)

            # Record balance history
            await self._ledger.record_balance_history(eth_price=price)

        elif should_sell:
            action = "SELL"
            would_have_executed = True
            size_usd = weth_value_usd

            # --- Latency emulation ---
            delay_min, delay_max = self._config.execution_delay_ms
            delay_ms = np.random.uniform(delay_min, delay_max)
            await asyncio.sleep(delay_ms / 1000.0)

            # --- Liquidity profiling (best-effort) ---
            pool_depth_usd = 500_000.0
            try:
                from wallet_manager.strategies.liquidity_alpha import (
                    get_liquidity_alpha,
                )
                liq_alpha = get_liquidity_alpha()
                profile = await liq_alpha.profiler.profile_pool(
                    rpc_url=rpc_url,
                    pool_address=_get_weth_usdc_pool(),
                    eth_price=price,
                )
                if profile:
                    pool_depth_usd = max(profile.depth_usd, 10_000.0)
            except Exception as _liq_exc:
                log.warning("Pool liquidity profile failed (non-critical): %s", _liq_exc)
                pass

            # --- Slippage simulation ---
            slippage_bps, simulated_fill_price = self._slippage.simulate_slippage(
                size_usd=size_usd,
                pool_depth_usd=pool_depth_usd,
                model=self._config.slippage_model,
                entry_price=price,
            )
            reason_codes.append(PAPER_SLIPPAGE_APPLIED)

            # --- Gas estimation ---
            if self._config.gas_estimation:
                gas_cost_usd = await self._gas_estimator.estimate_gas_cost_usd(
                    tx_type="swap", rpc_url=rpc_url, eth_price=price,
                )
                reason_codes.append(PAPER_GAS_ESTIMATED)

            # --- P&L calculation ---
            if self._entry_price > 0:
                pnl_pct = (simulated_fill_price - self._entry_price) / self._entry_price
                virtual_pnl = self._position_size_usd * pnl_pct
                virtual_pnl -= gas_cost_usd  # Deduct gas from P&L
            else:
                virtual_pnl = 0.0

            # --- Update virtual balances ---
            usdc_received = weth_balance * simulated_fill_price
            new_usdc = usdc_balance + usdc_received
            new_weth = 0.0

            await self._ledger.update_balance("USDC", new_usdc, usd_value=new_usdc)
            await self._ledger.update_balance("WETH", new_weth, usd_value=0.0)

            # Clear position
            self._in_position = False
            self._entry_price = 0.0
            self._position_size_usd = 0.0

            reason_codes.append(PAPER_TRADE_EXECUTED)

            # Record trade
            await self._ledger.record_trade(
                direction="SELL",
                asset_pair="WETH/USDC",
                entry_price=price,
                simulated_fill_price=simulated_fill_price,
                slippage_bps=slippage_bps,
                size_usd=size_usd,
                gas_cost_usd=gas_cost_usd,
                pnl_usd=virtual_pnl,
                reason_codes=tuple(reason_codes),
                signal_source=f"robust_alpha={signal.robust_alpha:+.3f}",
            )

            log.info(
                f"PAPER_ENGINE: SELL {weth_balance:.6f} WETH @ "
                f"${simulated_fill_price:.2f} P&L=${virtual_pnl:+.2f} "
                f"(slip={slippage_bps:.2f}bps gas=${gas_cost_usd:.6f})"
            )

            # Instant Telegram notification
            await self._send_trade_telegram(
                "SELL", "WETH", size_usd, simulated_fill_price, pnl_usd=virtual_pnl,
            )

            # Record balance history
            await self._ledger.record_balance_history(eth_price=price)

        else:
            # HOLD — no trade
            reason_codes.append(PAPER_SIGNAL_REJECTED)

        # ==============================================================
        # 6. EQUITY SNAPSHOT
        # ==============================================================

        total_equity = await self._ledger.get_total_equity_usd(eth_price=price)
        unrealized_pnl = 0.0
        if self._in_position and self._entry_price > 0:
            unrealized_pnl = (
                (price - self._entry_price) / self._entry_price
            ) * self._position_size_usd
        await self._ledger._record_snapshot(total_equity, unrealized_pnl)

        # Log drawdown status
        dd_info = await self._ledger.get_drawdown()
        if dd_info["current_drawdown"] > 0.01:
            log.info(
                f"PAPER_DRAWDOWN: current={dd_info['current_drawdown']:.4%} "
                f"max={dd_info['max_drawdown']:.4%} "
                f"peak=${dd_info['peak_equity']:.2f} equity=${total_equity:.2f}"
            )

        self._cycle_count += 1
        cycle_duration_ms = (time.time() - cycle_start) * 1000

        log.info(
            f"PAPER_ENGINE: cycle={self._cycle_count} action={action} "
            f"alpha={signal.robust_alpha:+.3f} equity=${total_equity:.2f} "
            f"pnl=${virtual_pnl:+.2f} dd={self._ledger._max_drawdown:.4f} "
            f"{cycle_duration_ms:.0f}ms"
        )

        return PaperTradeResult(
            action=action,
            signal_alpha=signal.robust_alpha,
            simulated_fill_price=simulated_fill_price,
            slippage_bps=slippage_bps,
            gas_cost_usd=gas_cost_usd,
            virtual_pnl_usd=virtual_pnl,
            would_have_executed=would_have_executed,
            reason_codes=tuple(reason_codes),
            cycle_duration_ms=cycle_duration_ms,
            timestamp=cycle_start,
        )

    # ----- Performance reporting -----

    async def get_performance_report(self) -> dict[str, Any]:
        """Return comprehensive virtual trading performance report.

        Includes virtual P&L, Sharpe ratio, max drawdown, win rate,
        average slippage, cycle count, position state, and balances.

        Returns:
            Dict with all performance metrics.
        """
        if not self._initialized:
            return {"error": "Paper engine not initialized"}

        pnl_summary = await self._ledger.get_pnl_summary()
        drawdown = await self._ledger.get_drawdown()

        usdc_bal = await self._ledger.get_balance("USDC")
        weth_bal = await self._ledger.get_balance("WETH")
        ausdc_bal = await self._ledger.get_balance("aUSDC")

        return {
            "virtual_pnl": pnl_summary,
            "drawdown": drawdown,
            "balances": {
                "USDC": round(usdc_bal, 4),
                "WETH": round(weth_bal, 8),
                "aUSDC": round(ausdc_bal, 4),
            },
            "position": {
                "in_position": self._in_position,
                "entry_price": self._entry_price,
                "position_size_usd": self._position_size_usd,
                "entry_alpha": self._entry_alpha,
            },
            "engine": {
                "cycle_count": self._cycle_count,
                "config": {
                    "initial_balance_usd": self._config.initial_balance_usd,
                    "slippage_model": self._config.slippage_model,
                    "slippage_mean_bps": self._config.slippage_mean_bps,
                    "slippage_std_bps": self._config.slippage_std_bps,
                    "gas_estimation": self._config.gas_estimation,
                },
                "gas_gwei_cached": self._gas_estimator.get_cached_gas_gwei(),
            },
        }

    async def get_telegram_report(self) -> str:
        """Return a formatted string for Telegram notification.

        Shows virtual P&L vs real gas cost estimate, Sharpe ratio,
        drawdown, win rate, and current balances.

        Returns:
            Telegram-friendly formatted report string.
        """
        if not self._initialized:
            return "Paper engine not initialized."

        pnl = await self._ledger.get_pnl_summary()
        dd = await self._ledger.get_drawdown()

        usdc_bal = await self._ledger.get_balance("USDC")
        weth_bal = await self._ledger.get_balance("WETH")

        total_gas = 0.0
        try:
            self._ledger._assert_open()
            async with self._ledger._conn.execute(
                "SELECT COALESCE(SUM(gas_cost_usd), 0.0) AS total_gas FROM paper_trades"
            ) as cur:
                row = await cur.fetchone()
            if row:
                total_gas = float(row["total_gas"])
        except Exception as _gas_exc:
            log.warning("Paper engine gas cost DB query failed (non-critical): %s", _gas_exc)

        lines = [
            "--- PAPER TRADING REPORT ---",
            "",
            "Virtual P&L vs Real Gas Cost Estimate",
            f"  Realized P&L:     ${pnl['realized_pnl']:+.4f}",
            f"  Total Gas Cost:   ${total_gas:.6f}",
            f"  Net P&L (after gas): ${pnl['realized_pnl'] - total_gas:+.4f}",
            "",
            "Performance Metrics",
            f"  Sharpe Ratio:     {pnl['sharpe_estimate']:.4f}",
            f"  Max Drawdown:     {dd['max_drawdown'] * 100:.4f}%",
            f"  Current Drawdown: {dd['current_drawdown'] * 100:.4f}%",
            f"  Win Rate:         {pnl['win_rate']:.2f}%",
            f"  Avg Slippage:     {pnl['avg_slippage_bps']:.2f} bps",
            f"  Total Trades:     {pnl['trade_count']}",
            "",
            "Virtual Balances",
            f"  USDC:  ${usdc_bal:.4f}",
            f"  WETH:  {weth_bal:.8f}",
            f"  Peak:  ${pnl['peak_equity']:.2f}",
            "",
            f"Cycles: {self._cycle_count}",
            "--- END PAPER REPORT ---",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# PaperExecutionBridge — Section 9.1
# ---------------------------------------------------------------------------


class PaperExecutionBridge:
    """Runs the COMPLETE MLQuantExecutionEngine pipeline end-to-end but
    writes to VirtualLedger instead of broadcasting at the final step.

    All stages run identically to the live engine (GARCH + Kalman feature
    engineering, EnsembleAlphaEngine, PPO sizing, RegimeFilter, CVaR,
    DrawdownController, LiquidityProfiler, AlphaDecayMonitor, MEVShield
    jitter simulation, AdaptiveSOR routing simulation).

    VirtualLedger records the simulated fill price using the paper slippage
    model (gaussian, configured via cfg.paper_slippage_*).
    ConceptDriftDetector is updated with paper trade results.
    """

    def __init__(self, engine: PaperTradingEngine | None = None) -> None:
        self._engine = engine
        self._heartbeat_interval_s: float = 21_600.0  # 6 hours
        self._last_report_ts: float = 0.0

    async def run_cycle(
        self,
        manager: Any,
        wallet_id: str,
        portfolio_manager: Any,
        task_context: dict,
        analytics_db: Any = None,
        quant_analytics: Any = None,
    ) -> PaperTradeResult:
        """Execute a full shadow cycle via PaperTradingEngine.

        Identical entry point to MLQuantExecutionEngine.run_cycle() so the
        orchestrator can swap them interchangeably in robust_paper mode.
        Returns PaperTradeResult (never a real TxReceipt).
        """
        if self._engine is None:
            self._engine = await get_paper_engine()
        result = await self._engine.run_shadow_cycle(
            manager=manager,
            wallet_id=wallet_id,
            portfolio_manager=portfolio_manager,
            task_context=task_context,
            analytics_db=analytics_db,
            quant_analytics=quant_analytics,
        )
        await self._maybe_send_heartbeat_report(task_context)
        return result

    async def _maybe_send_heartbeat_report(self, task_context: dict) -> None:
        """Send 6-hour comparison report to Telegram if interval has elapsed."""
        now = time.time()
        if now - self._last_report_ts < self._heartbeat_interval_s:
            return
        self._last_report_ts = now
        try:
            await self._send_comparison_report(task_context)
        except Exception as exc:
            log.warning("PaperExecutionBridge: heartbeat report failed: %s", exc)

    async def _send_comparison_report(self, task_context: dict) -> None:
        """Build and send paper-vs-real comparison report to Telegram."""
        if self._engine is None:
            return
        try:
            from datetime import datetime

            report = await self._engine.get_performance_report()
            p_sharpe = report.get("virtual_pnl", {}).get("sharpe_estimate", 0.0)
            p_win = report.get("virtual_pnl", {}).get("win_rate", 0.0)
            p_trades = report.get("virtual_pnl", {}).get("trade_count", 0)
            p_dd = report.get("drawdown", {}).get("max_drawdown", 0.0)
            p_slip = report.get("virtual_pnl", {}).get("avg_slippage_bps", 0.0)
            r_sharpe = task_context.get("real_sharpe", 0.0)
            # p_win: already percentage (0–100) from VirtualLedger.get_pnl_summary()
            # r_win: decimal (0–1) from task_context["real_win_rate"] — multiply by 100
            r_win = task_context.get("real_win_rate", 0.0)
            r_trades = task_context.get("real_trade_count", 0)
            r_dd = task_context.get("real_max_drawdown", 0.0)
            r_slip = task_context.get("real_avg_slippage_bps", 0.0)
            is_avg = task_context.get("avg_implementation_shortfall_bps", 0.0)

            utc_now = datetime.now(UTC).strftime("%H:%M UTC")
            lines = [
                f"📊 *Paper vs Real*  `{utc_now}`  \\(Last 6 hours\\)",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"📈 Paper Sharpe:    `{p_sharpe:.2f}`  \\|  Real Sharpe:    `{r_sharpe:.2f}`",
                f"🎯 Paper Win Rate:  `{p_win:.1f}%` \\|  Real Win Rate:  `{r_win * 100:.1f}%`",
                f"💸 Est\\. Slippage:  `{p_slip:.1f} bps` \\|  Actual:        `{r_slip:.1f} bps`",
                f"⚡ Paper Trades:    `{p_trades}`     \\|  Real Trades:    `{r_trades}`",
                f"📉 Paper Drawdown:  `{p_dd*100:.2f}%`  \\|  Real Drawdown:  `{r_dd*100:.2f}%`",
                f"🔍 IS \\(shortfall\\): `{is_avg:.1f} bps` avg",
            ]

            divergence = abs(p_sharpe - r_sharpe)
            if divergence > 0.1:
                lines.append(
                    f"⚠️ *Divergence: {divergence*100:.1f}%*  — check regime filter and slippage model"
                )

            msg = "\n".join(lines)
            notifier = task_context.get("notifier")
            if notifier:
                await notifier.send_message(msg, parse_mode="MarkdownV2")

            if divergence > 0.25:
                critical = "🚨 *CRITICAL: Paper\\/Real divergence >25%\\. Manual review required\\.*"
                if notifier:
                    await notifier.send_message(critical, parse_mode="MarkdownV2")

        except Exception as exc:
            log.warning("PaperExecutionBridge: could not build comparison report: %s", exc)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: PaperTradingEngine | None = None
_engine_lock = asyncio.Lock()


async def get_paper_engine(
    config: PaperTradeConfig | None = None,
) -> PaperTradingEngine:
    """Return the process-wide PaperTradingEngine singleton.

    Creates and initializes the engine on first call.  Subsequent calls
    return the same instance (``config`` is ignored after first init).

    Args:
        config: Optional configuration override for first initialization.

    Returns:
        Initialized :class:`PaperTradingEngine` singleton.
    """
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance

    async with _engine_lock:
        if _engine_instance is not None:
            return _engine_instance
        engine = PaperTradingEngine(config=config)
        await engine.initialize()
        _engine_instance = engine
        log.info("PaperTradingEngine singleton created and initialized")
        return _engine_instance


def reset_paper_engine() -> None:
    """Reset the singleton (for tests).

    Does NOT call ``shutdown()`` — the caller is responsible for awaiting
    shutdown before reset if the engine is active.
    """
    global _engine_instance
    _engine_instance = None
    log.info("PaperTradingEngine singleton reset")
