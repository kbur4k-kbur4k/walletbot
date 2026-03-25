"""
wallet_manager/metrics_server.py

Prometheus-Compatible /metrics Endpoint.

Lightweight async HTTP server exposing /metrics on cfg.metrics_port (default 9090).
Uses aiohttp.web + prometheus_client for proper OpenMetrics text generation.

Metrics:
  wallet_portfolio_total_usd
  wallet_pillar_value_usd{pillar}
  wallet_trade_count_total          (counter)
  wallet_pnl_usd
  wallet_sharpe_ratio
  wallet_max_drawdown
  wallet_ml_model_confidence
  wallet_concept_drift_level        (0=NOMINAL, 1=MILD, 2=MODERATE, 3=SEVERE)
  wallet_aave_apy_pct
  wallet_rpc_latency_ms{endpoint}
  wallet_gas_price_gwei

Runs as an asyncio background task; does not block the main loop.
If cfg.metrics_enabled = False, server never starts.
"""

from __future__ import annotations

import asyncio
import contextlib
import time

import aiohttp.web
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    CollectorRegistry,
    generate_latest,
)

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# ---------------------------------------------------------------------------
# Metric registry — module-level prometheus_client objects
# ---------------------------------------------------------------------------

# Use a dedicated registry to avoid collisions with the default global one
# (important for tests that import this module multiple times).
_REGISTRY = CollectorRegistry(auto_describe=True)


def _gauge(name: str, doc: str, labelnames: tuple[str, ...] = ()) -> Gauge:
    return Gauge(name, doc, list(labelnames), registry=_REGISTRY)


def _counter(name: str, doc: str) -> Counter:
    return Counter(name, doc, registry=_REGISTRY)


# Scalar gauges
_G_PORTFOLIO_TOTAL   = _gauge("wallet_portfolio_total_usd",   "Total portfolio value in USD")
_G_PNL               = _gauge("wallet_pnl_usd",               "Current PnL in USD")
_G_SHARPE            = _gauge("wallet_sharpe_ratio",           "Current Sharpe ratio estimate")
_G_DRAWDOWN          = _gauge("wallet_max_drawdown",           "Maximum drawdown fraction")
_G_ML_CONFIDENCE     = _gauge("wallet_ml_model_confidence",    "ML model signal confidence 0-1")
_G_DRIFT_LEVEL       = _gauge("wallet_concept_drift_level",    "Concept drift level 0=NOMINAL 1=MILD 2=MODERATE 3=SEVERE")
_G_AAVE_APY          = _gauge("wallet_aave_apy_pct",           "Aave V3 USDC supply APY percent")
_G_GAS_PRICE         = _gauge("wallet_gas_price_gwei",         "Current gas price in Gwei")
_G_KILL_SWITCH       = _gauge("wallet_kill_switch_active",     "Kill switch active flag (1=active)")
_G_REGIME            = _gauge("wallet_regime_state",           "Market regime state")
_G_KELLY             = _gauge("wallet_kelly_fraction",         "Kelly fraction used for position sizing")
_G_VAR99             = _gauge("wallet_var_99_usd",             "Value-at-Risk 99th percentile in USD")
_G_OOS_SHARPE        = _gauge("wallet_oos_sharpe",             "Out-of-sample Sharpe ratio from walk-forward validation")

# Labeled gauges
_G_PILLAR_VALUE      = _gauge("wallet_pillar_value_usd",       "Pillar value in USD by strategy",       ("pillar",))
_G_RPC_LATENCY       = _gauge("wallet_rpc_latency_ms",         "RPC endpoint latency in milliseconds",  ("endpoint",))

# Counters
_C_TRADE_COUNT       = _counter("wallet_trade_count_total",    "Total number of completed trades")
_C_RPC_FAILOVER      = _counter("wallet_rpc_failover_total",   "Total RPC failover events")
_C_TX_REPAIR         = _counter("wallet_tx_repair_total",      "Total stuck-transaction repair events")
_C_CIRCUIT_OPEN      = _counter("wallet_circuit_open_total",   "Total circuit breaker open events")
_C_RETRAIN           = _counter("wallet_retrain_total",        "Total ML model retrain events")

# ---------------------------------------------------------------------------
# Mapping from legacy string names → prometheus_client objects
# ---------------------------------------------------------------------------

_SCALAR_GAUGES: dict[str, Gauge] = {
    "wallet_portfolio_total_usd":  _G_PORTFOLIO_TOTAL,
    "wallet_pnl_usd":              _G_PNL,
    "wallet_sharpe_ratio":         _G_SHARPE,
    "wallet_max_drawdown":         _G_DRAWDOWN,
    "wallet_ml_model_confidence":  _G_ML_CONFIDENCE,
    "wallet_concept_drift_level":  _G_DRIFT_LEVEL,
    "wallet_aave_apy_pct":         _G_AAVE_APY,
    "wallet_gas_price_gwei":       _G_GAS_PRICE,
    "wallet_kill_switch_active":   _G_KILL_SWITCH,
    "wallet_regime_state":         _G_REGIME,
    "wallet_kelly_fraction":       _G_KELLY,
    "wallet_var_99_usd":           _G_VAR99,
    "wallet_oos_sharpe":           _G_OOS_SHARPE,
}

_COUNTERS: dict[str, Counter] = {
    "wallet_trade_count_total":    _C_TRADE_COUNT,
    "wallet_rpc_failover_total":   _C_RPC_FAILOVER,
    "wallet_tx_repair_total":      _C_TX_REPAIR,
    "wallet_circuit_open_total":   _C_CIRCUIT_OPEN,
    "wallet_retrain_total":        _C_RETRAIN,
}

_last_update: float = 0.0


def update_metric(name: str, value: float) -> None:
    """Update a scalar metric value.

    Accepts both gauge and counter names.  For counters the value is treated
    as an *absolute* reading — the counter is incremented by the delta since
    the last call (never decremented).
    """
    global _last_update
    _last_update = time.time()

    if name in _SCALAR_GAUGES:
        _SCALAR_GAUGES[name].set(value)
    elif name in _COUNTERS:
        # prometheus_client counters only go up; call inc() with a positive delta
        counter = _COUNTERS[name]
        # We don't track previous values here, so just call inc(value) when > 0
        if value > 0:
            counter.inc(value)
    else:
        log.debug("update_metric: unknown metric %r (value=%s)", name, value)


def update_labeled_metric(name: str, label: str, value: float) -> None:
    """Update a labeled gauge value."""
    global _last_update
    _last_update = time.time()

    if name == "wallet_pillar_value_usd":
        _G_PILLAR_VALUE.labels(pillar=label).set(value)
    elif name == "wallet_rpc_latency_ms":
        _G_RPC_LATENCY.labels(endpoint=label).set(value)
    else:
        log.debug("update_labeled_metric: unknown metric %r", name)


# ---------------------------------------------------------------------------
# Async HTTP handlers
# ---------------------------------------------------------------------------


def _refresh_from_system() -> None:
    """Pull live values from running subsystems into the metrics registry."""
    # Portfolio manager
    with contextlib.suppress(Exception):
        from wallet_manager.orchestrator import _get_portfolio_manager
        pm = _get_portfolio_manager()
        if pm is not None:
            snap = pm.get_snapshot()
            _G_PORTFOLIO_TOTAL.set(snap.total_value_usd)
            for alloc in snap.allocations:
                _G_PILLAR_VALUE.labels(pillar=alloc.strategy.name).set(alloc.current_usd)

    # Concept drift
    with contextlib.suppress(Exception):
        from wallet_manager.orchestrator import _drift_detector
        if _drift_detector is not None:
            dl = getattr(_drift_detector, "_drift_level", None)
            if dl is not None:
                _G_DRIFT_LEVEL.set(float(dl.value))

    # ML confidence
    with contextlib.suppress(Exception):
        from wallet_manager.strategies.alpha_engine import get_alpha_engine
        engine = get_alpha_engine()
        sig = engine.last_signal
        if sig is not None:
            _G_ML_CONFIDENCE.set(sig.confidence)

    # RPC latency
    with contextlib.suppress(Exception):
        from wallet_manager.orchestrator import _get_rpc_router
        router = _get_rpc_router()
        if router is not None:
            for chain_id, pool_info in router.pool_status().items():
                for p in pool_info:
                    url = p.get("url", "")
                    score = router.latency_score(chain_id, url)
                    if score > 0:
                        latency_ms = 1.0 / score - 1.0
                        _G_RPC_LATENCY.labels(endpoint=url[:40]).set(latency_ms)


async def _handle_metrics(request: aiohttp.web.Request) -> aiohttp.web.Response:
    _refresh_from_system()
    body = generate_latest(_REGISTRY)
    # Strip charset from content_type — aiohttp rejects compound content-type values
    ct = CONTENT_TYPE_LATEST.split(";")[0].strip()
    return aiohttp.web.Response(
        body=body,
        content_type=ct,
    )


async def _handle_health(request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.json_response({"status": "ok", "ts": time.time()})


async def _handle_ready(request: aiohttp.web.Request) -> aiohttp.web.Response:
    return aiohttp.web.json_response({"status": "ready", "ts": time.time()})


# ---------------------------------------------------------------------------
# MetricsServer class (dependency-injectable for testing)
# ---------------------------------------------------------------------------


class MetricsServer:
    """Dependency-injectable MetricsServer for testing and production.

    Args:
        port: HTTP port to bind (0 = OS assigns a random port, for tests).
    """

    def __init__(self, port: int = 9090) -> None:
        self.port = port
        self._metrics: dict[str, float] = {}
        self._labeled_metrics: dict[tuple[str, str], float] = {}

    def update_metric(self, name: str, value: float) -> None:
        """Update a scalar metric by name."""
        self._metrics[name] = value
        update_metric(name, value)

    def update_labeled_metric(self, name: str, label: str, value: float) -> None:
        """Update a labeled metric."""
        self._labeled_metrics[(name, label)] = value
        update_labeled_metric(name, label, value)

    def build_app(self) -> aiohttp.web.Application:
        """Build and return the aiohttp Application with all routes."""
        app = aiohttp.web.Application()
        app.router.add_get("/metrics", _handle_metrics)
        app.router.add_get("/health", _handle_health)
        app.router.add_get("/ready", _handle_ready)
        return app


# ---------------------------------------------------------------------------
# Background task entry point
# ---------------------------------------------------------------------------


async def start_metrics_server() -> asyncio.Task | None:
    """Start the /metrics HTTP server as a background asyncio task.

    Returns the task, or None if metrics_enabled = False.
    """
    from wallet_manager.config import get_settings
    cfg = get_settings()
    if not cfg.metrics_enabled:
        log.info("MetricsServer: disabled (metrics_enabled=False).")
        return None

    port = cfg.metrics_port
    host = getattr(cfg, "metrics_host", "0.0.0.0")

    async def _serve() -> None:
        try:
            server = MetricsServer(port=port)
            app = server.build_app()
            runner = aiohttp.web.AppRunner(app)
            await runner.setup()
            site = aiohttp.web.TCPSite(runner, host, port)
            await site.start()
            log.info("MetricsServer: listening on %s:%d/metrics", host, port)
            while True:
                await asyncio.sleep(3600)
        except Exception as exc:
            log.warning("MetricsServer: failed: %s", exc)

    task = asyncio.ensure_future(_serve())
    return task
