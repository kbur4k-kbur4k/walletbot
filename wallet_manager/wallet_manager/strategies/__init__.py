"""
wallet_manager/strategies/__init__.py

Portfolio strategy modules — QUANT-X Institutional Grade DeFi Engine.

Pillars:
  - YIELD   (40%): Passive Aave V3 USDC supply — zero lazy capital.
  - ALGO    (35%): QUANT-X Alpha Engine (Z-Score 2.0, OFI, VWAP, Kelly Criterion).
  - SHORT   (10%): Hedged delta-neutral synthetic short via Aave V3 borrow.
  - AIRDROP (15%): Ping-pong ETH<->USDC farming for protocol activity.

QUANT-X modules:
  - signal_intelligence: Z-Score 2.0 + OFI + VWAP composite alpha generator.
  - smart_order_router: Multi-DEX aggregation (Uniswap V3, SushiSwap, Aerodrome).
  - mev_shield: Private RPC routing + Gaussian Jitter + Dynamic RBF.
  - kelly_criterion: Optimal position sizing based on historical win/loss.
  - quantx_engine: Unified execution engine integrating all QUANT-X modules.

Support modules:
  - idle_sweep: Automatically sweeps idle USDC into Aave V3.
  - flash_crash_detector: 8% / 5min flash crash detection + KILL_SWITCH.

Robustness Shield (Phase 4 — Anti-Overfit):
  - robust_alpha: Fractional Differentiation + Permutation Feature Importance.
  - wf_validator: Purged Walk-Forward Cross-Validation engine.
  - regime_filter: Hidden Markov Model regime detection + Economic Calendar.
  - liquidity_alpha: V3 tick-level liquidity profiling + split-order execution.

Final Optimization (Phase 5 — Maximum Potency):
  - paper_engine: High-Fidelity Paper Trading with slippage sim + virtual ledger.
  - stochastic_stress_test: Monte Carlo chaos layer + Recursive Feature Elimination.
  - latency_shield: Network latency emulation + stale quote protection.
  - oci_deployer: OCI watchdog + health-check API + systemd template.
"""
from __future__ import annotations

import contextlib
import importlib
import logging

log = logging.getLogger(__name__)

# Defensive imports — missing optional ML dependencies must not crash startup
_STRATEGY_MODULES = [
    "alpha_engine",
    "deep_kelly_risk",
    "execution_logic",
    "regime_filter",
    "signal_intelligence",
    "smart_order_router",
    "market_microstructure",
    "alpha_decay_monitor",
    "wf_validator",
    "robust_alpha",
    "paper_engine",
    "liquidity_alpha",
    "mev_shield",
    "flash_crash_detector",
    "latency_shield",
    "stochastic_stress_test",
]

for _module_name in _STRATEGY_MODULES:
    try:
        importlib.import_module(f".{_module_name}", package=__name__)
    except ImportError as _exc:
        log.warning("Strategy import failed: %s — %s", _module_name, _exc)

# Always-required strategy modules
from . import yield_strategy, algo_strategy, airdrop_strategy  # noqa: E402, F401
from . import short_strategy, idle_sweep  # noqa: E402, F401
from . import kelly_criterion, quantx_engine  # noqa: E402, F401
from . import quant_analytics  # noqa: E402, F401
from . import oci_deployer  # noqa: E402, F401

# Conditional re-exports (graceful if ML libs absent)
with contextlib.suppress(ImportError):
    from wallet_manager.strategies.yield_strategy import (  # noqa: F401
        supply_usdc,  # noqa: F401
        withdraw_usdc,  # noqa: F401
        get_apy,  # noqa: F401
        get_yield_pillar_value_usd,  # noqa: F401
        get_user_position,  # noqa: F401
        UserPosition,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.algo_strategy import (  # noqa: F401
        run_algo_strategy,  # noqa: F401
        compute_bollinger_bands,  # noqa: F401
        bollinger_band_proximity,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.short_strategy import run_short_strategy  # noqa: F401

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.airdrop_strategy import run_airdrop_strategy  # noqa: F401

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.idle_sweep import run_idle_sweep  # noqa: F401

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.flash_crash_detector import (  # noqa: F401
        FlashCrashDetector,  # noqa: F401
        get_detector,  # noqa: F401
        CrashAlert,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.signal_intelligence import (  # noqa: F401
        AlphaGenerator,  # noqa: F401
        SignalOutput,  # noqa: F401
        get_alpha_generator,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.smart_order_router import (  # noqa: F401
        SmartOrderRouter,  # noqa: F401
        get_smart_order_router,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.mev_shield import (  # noqa: F401
        MEVShield,  # noqa: F401
        get_mev_shield,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.kelly_criterion import (  # noqa: F401
        KellyCriterion,  # noqa: F401
        get_kelly_criterion,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.quantx_engine import (  # noqa: F401
        QuantXEngine,  # noqa: F401
        get_quantx_engine,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.alpha_engine import (  # noqa: F401
        StochasticMLAlphaEngine,  # noqa: F401
        get_alpha_engine,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.deep_kelly_risk import (  # noqa: F401
        DeepKellyRiskEngine,  # noqa: F401
        get_deep_kelly_engine,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.execution_logic import (  # noqa: F401
        MLQuantExecutionEngine,  # noqa: F401
        get_ml_execution_engine,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.quant_analytics import (  # noqa: F401
        QuantAnalytics,  # noqa: F401
        get_quant_analytics,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.liquidity_alpha import (  # noqa: F401
        LiquidityAlpha,  # noqa: F401
        get_liquidity_alpha,  # noqa: F401
        reset_liquidity_alpha,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.robust_alpha import (  # noqa: F401
        RobustAlphaEngine,  # noqa: F401
        get_robust_alpha_engine,  # noqa: F401
        reset_robust_alpha_engine,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.wf_validator import (  # noqa: F401
        WalkForwardValidator,  # noqa: F401
        get_wf_validator,  # noqa: F401
        reset_wf_validator,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.regime_filter import (  # noqa: F401
        RegimeFilter,  # noqa: F401
        get_regime_filter,  # noqa: F401
        reset_regime_filter,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.paper_engine import (  # noqa: F401
        PaperTradingEngine,  # noqa: F401
        PaperExecutionBridge,  # noqa: F401
        get_paper_engine,  # noqa: F401
        reset_paper_engine,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.stochastic_stress_test import (  # noqa: F401
        StochasticStressTester,  # noqa: F401
        get_stress_tester,  # noqa: F401
        reset_stress_tester,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.latency_shield import (  # noqa: F401
        LatencyShield,  # noqa: F401
        get_latency_shield,  # noqa: F401
        reset_latency_shield,  # noqa: F401
    )

with contextlib.suppress(ImportError):
    from wallet_manager.strategies.oci_deployer import (  # noqa: F401
        OCIDeployer,  # noqa: F401
        get_oci_deployer,  # noqa: F401
        reset_oci_deployer,  # noqa: F401
    )
