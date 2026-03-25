# Changelog

## [Unreleased] — 2026-03-25

### Added
- Unit tests for core/manager.py (WalletManager)
- Unit tests for network/tx_broadcaster.py, balance_provider.py, session_factory.py
- Unit tests for security/logging_filter.py, vault_manager.py
- Unit tests for strategies/execution_logic.py, mev_shield.py, latency_shield.py
- Unit tests for strategies/liquidity_alpha.py, market_microstructure.py, short_strategy.py
- Unit tests for strategies/quant_analytics.py, quantx_engine.py
- Unit tests for utils/health_check.py, rate_limiter.py, cooldown.py
- Expanded coverage for airdrop_strategy, yield_strategy, robust_alpha, idle_sweep, rpc_router, tx_repair
- SessionFactory.get_session() and close_all() methods for per-wallet session management
- CooldownManager.trigger_cooldown(), is_in_cooldown(), clear(), clear_all() methods
- Docker and docker-compose support (`Dockerfile`, `docker-compose.yml`, `.dockerignore`)
- Monitoring stack (`monitoring/docker-compose.monitoring.yml`, `monitoring/prometheus.yml`)
- Grafana dashboard JSON (`monitoring/grafana_dashboard.json`) for all Prometheus metrics
- New Telegram commands: `/setmode`, `/wallets`, `/allocation`, `/reset_circuit`, `/report`, `/gas`
- Per-user rate limiting for Telegram commands (10 commands/minute, 1 heavy command/hour)
- Portfolio rebalancing threshold (`REBALANCE_DRIFT_THRESHOLD` env var)
- `needs_rebalancing()` method on `PortfolioManager` — rebalance only when drift exceeds threshold
- Configurable allocation weights via environment variables (`YIELD_WEIGHT`, `ALGO_WEIGHT`, `SHORT_WEIGHT`, `AIRDROP_WEIGHT`, `MONTHLY_INJECTION_USD`)
- Weight sum validation in `Settings.__post_init__` — raises `ConfigurationError` if weights ≠ 1.0
- `metrics_host` and `log_format` fields in `Settings`
- Wallet config JSON schema validation with full error collection before raising
- `SIGTERM` graceful shutdown support (`_setup_signal_handlers`)
- Structured JSON logging option (`LOG_FORMAT=json` env var) via `build_logger()`
- `wallets.json.example` template (no real keys)
- Orchestrator integration tests (`tests/integration/test_orchestrator.py`)
- Wallet config validation tests (`tests/unit/test_wallet_config_validation.py`)
- Extended Telegram command tests (`tests/unit/test_telegram_commands.py`)
- Portfolio manager rebalancing threshold tests

### Changed
- pyproject.toml: updated ruff lint rules, mypy strict=false with disallow_untyped_defs, coverage fail_under=75
- pytest-xdist added for parallel test execution

### Fixed
- SessionFactory missing get_session()/close_all() methods
- CooldownManager incomplete implementation (missing cooldown tracking methods)
- `deep_kelly_history.json` removed from version control (runtime-generated file)
- `allcode.txt` debugging artifact deleted
- `wallets.json` added to `.gitignore`
- Portfolio weights now loaded from config/env vars instead of hardcoded constants
- Telegram `/help` updated to list all new commands

### Removed
- `wallet_manager/allcode.txt` (34 KB debugging artifact)
- `wallet_manager/deep_kelly_history.json` (runtime state file)
