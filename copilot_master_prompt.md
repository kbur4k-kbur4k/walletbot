# COPILOT AGENTS — walletbot Master Improvement Prompt

Paste everything below this line into GitHub Copilot Agents.

---

You are a senior Python/DeFi/ML engineer working autonomously on the `walletbot` repository. This is a fully automated DeFi portfolio management system running on Base chain (ID 8453). Your job is to work through every task below without stopping, without asking questions, and without waiting for human input. After every task, run the relevant tests and fix any failures before moving on. Commit after every completed task with a clear message. If a task hits an unexpected blocker, log it in `AGENT_LOG.md` and continue to the next task.

**Hard constraints that must never be violated:**
- Chain ID is always 8453 (Base). Never add support for other chains.
- Never modify ABC signatures in `wallet_manager/core/interfaces.py`.
- All I/O must be async/await. No blocking calls in the event loop.
- Private keys must never appear in logs. `SensitiveDataFilter` is mandatory on every new logger.
- Every new module must have a corresponding test file.
- Coverage must stay at or above 85% at all times.
- After every task: `ruff check . && ruff format . --check && mypy wallet_manager/ --ignore-missing-imports && pytest tests/ -v --cov=wallet_manager --cov-fail-under=85`

---

## PHASE 1 — STABILITY & CORRECTNESS

### Task 1.1 — Eliminate all silent exception swallowing

Search every Python file in `wallet_manager/` for bare `except: pass` and `except Exception: pass` blocks that discard errors without logging. Replace each one with a properly logged warning or error. Rules:
- Use `log.warning("...: %s", exc)` for non-critical paths (optional feature computation, enrichment steps).
- Use `log.error("...: %s", exc)` for anything involving balances, transactions, or state changes.
- If swallowing is genuinely correct, add an inline comment explaining exactly why.
- Files known to have this problem: `execution_logic.py`, `smart_order_router.py`, `paper_engine.py`, `quantx_engine.py`, `algo_strategy.py`, `orchestrator.py`, `idle_sweep.py`.
- After fixing, run `pytest tests/unit/ -v` and fix any regressions.
- Commit: `fix: replace silent exception swallowing with structured logging`

### Task 1.2 — Complete Compound V3 integration

In `wallet_manager/actions.py` around line 1063 there is a TODO:
`# TODO: Add Compound V3 cUSDCv3 Base contract when addresses provided`

Complete it:
1. Compound III USDC on Base contract address: `0xb125E6687d4313864e53df431d5425969c15Eb2`
2. Add ABI fragments for `getUtilization()` (returns uint), `getSupplyRate(uint utilization)` (returns uint per second in 1e18 scale).
3. Implement `async def _get_compound_v3_apy(rpc_url: str) -> float | None` — returns APY as percentage (e.g. 4.2 for 4.2%) or None on failure.
4. Update `get_alternative_apy()` to try Compound V3 first, fall back to the conservative constant only on failure.
5. Write `tests/unit/test_compound_integration.py` with: mock RPC returning valid utilization, mock RPC raising an exception (fallback test), APY value in 0–50 range assertion.
6. Commit: `feat: complete Compound V3 APY integration on Base`

### Task 1.3 — Fix type annotations and remove mypy exclusions

`wallet_manager/strategies/oci_deployer.py` is excluded from mypy in `pyproject.toml`. Fix it:
1. Add complete type annotations to every function and method.
2. Remove the mypy exclusion entry from `pyproject.toml`.
3. Run `mypy wallet_manager/strategies/oci_deployer.py --ignore-missing-imports` — zero errors allowed.
4. Write `tests/unit/test_oci_deployer.py`: test OCIDeployer instantiation, health check endpoint returns expected schema, watchdog RSS threshold logic.
5. Commit: `fix: add type annotations to oci_deployer, remove mypy exclusion`

---

## PHASE 2 — ML EXCELLENCE

### Task 2.1 — Dynamic ensemble weight adaptation

Currently in `wallet_manager/strategies/alpha_engine.py` the ensemble weights are hardcoded constants:
```
ENSEMBLE_WEIGHT_XGB = 0.40
ENSEMBLE_WEIGHT_LGBM = 0.25
ENSEMBLE_WEIGHT_LSTM = 0.25
ENSEMBLE_WEIGHT_TRANSFORMER = 0.10
```

Replace this with a dynamic weighting system:
1. Create `wallet_manager/strategies/ensemble_weighter.py` with class `DynamicEnsembleWeighter`.
2. It reads `ml_models/ensemble_weights.json` at startup (falls back to hardcoded defaults if file absent).
3. It tracks a rolling 30-day OOS Sharpe per model using trade outcomes from the analytics DB.
4. It recomputes weights proportional to OOS Sharpe, with floor of 0.05 per model. Models with Sharpe below 0.3 are temporarily zeroed and the weight is redistributed.
5. It persists updated weights back to `ml_models/ensemble_weights.json` after each recomputation.
6. `StochasticMLAlphaEngine.generate_signal()` must call `DynamicEnsembleWeighter.get_weights()` instead of using the constants.
7. Add Prometheus gauge `ensemble_weight_{model}` for each of the four models.
8. Write `tests/unit/test_ensemble_weighter.py`: weight normalization, low-Sharpe model exclusion, fallback when JSON absent, persistence round-trip.
9. Commit: `feat: dynamic ensemble weight adaptation based on rolling OOS Sharpe`

### Task 2.2 — Feature engineering expansion

In `scripts/train_models.py` and `wallet_manager/strategies/alpha_engine.py`, the feature set has 30 dimensions. Add 6 more:

| Feature name | Description |
|---|---|
| `rsi_14` | 14-period RSI normalized to [0, 1] |
| `bb_position` | Price position within Bollinger Bands: (price − lower) / (upper − lower), clipped [0, 1] |
| `volume_trend` | EMA(volume, 20) / EMA(volume, 5) ratio, clipped [0.1, 10] |
| `price_acceleration` | Kalman 3-state acceleration component |
| `spread_ratio` | current spread_bps / rolling_mean(spread_bps, 50) |
| `microstructure_score` | Kyle lambda × Amihud ratio, log-scaled |

Rules:
- Update `FEATURE_NAMES` list and `N_FEATURES = 36` in both files.
- Add vectorized implementations in `scripts/train_models.py::build_feature_matrix()`.
- Update `FeatureEngineer.ingest()` in `alpha_engine.py` to compute and append these features.
- Update `FeatureNormalizer` to handle 36 features.
- All new features must be NaN-safe (use `np.nan_to_num` with appropriate fill values).
- Write `tests/unit/test_feature_expansion.py`: shape assertion (n_samples, 36), no NaN in output, each new feature within expected range.
- Commit: `feat: expand ML feature set from 30 to 36 dimensions`

### Task 2.3 — Model versioning and archival

1. After every successful `train_models.py` run, copy all model files from `ml_models/` into `ml_models/archive/YYYYMMDD_HHMMSS/`.
2. Keep only the 3 most recent archive snapshots — delete older ones automatically.
3. Add `version`, `trained_at`, `archive_path` fields to `training_report.json`.
4. In `wallet_manager/security/telegram_commands.py`, update the `/retrain` status message to show the current model version and training timestamp.
5. Write `tests/unit/test_model_versioning.py`: archive creation, old-archive pruning (keep 3), report JSON fields.
6. Commit: `feat: model versioning with automatic archive rotation (keep 3)`

### Task 2.4 — Stacking meta-learner improvement

In `wallet_manager/strategies/alpha_engine.py` (or wherever the stacking meta-learner lives), the current stacking is a simple weighted average. Replace it with a proper logistic regression meta-learner:
1. The meta-learner takes the four model probability outputs as features.
2. It is trained online using a sliding window of the last 500 predictions vs actual outcomes (stored in analytics DB).
3. It updates its weights every 100 new observations using SGD.
4. Falls back to weighted average when fewer than 100 observations exist.
5. Expose `meta_learner_accuracy` as a Prometheus gauge.
6. Write `tests/unit/test_meta_learner.py`: online update, fallback behavior, prediction in [0,1].
7. Commit: `feat: replace static weighted average with online logistic regression meta-learner`

---

## PHASE 3 — NEW CAPABILITIES

### Task 3.1 — Backtesting engine

Create `scripts/backtest.py`:

```
Usage: python scripts/backtest.py [OPTIONS]
  --mode          legacy|quantx|ml_quant  (default: ml_quant)
  --start-date    YYYY-MM-DD              (default: 90 days ago)
  --end-date      YYYY-MM-DD              (default: today)
  --initial-usd   float                   (default: 1000.0)
  --output        path                    (default: results/backtest_YYYYMMDD_HHMMSS.json)
  --plot          flag                    (if set, save equity curve to results/*.png using matplotlib)
```

Requirements:
1. Load historical OHLCV data from the SQLite analytics DB or `scripts/data/` CSVs.
2. Use `PaperTradingEngine` and `VirtualLedger` wherever possible — do not reimplement execution logic.
3. Compute and report: total return %, annualised Sharpe, max drawdown %, win rate %, profit factor, avg trade duration, breakdown by pillar (YIELD/ALGO/SHORT/AIRDROP).
4. Print a clean formatted summary table to stdout.
5. Save full results to JSON including per-trade log.
6. If `--plot` is set, generate equity curve PNG using matplotlib (graceful import — if matplotlib absent, skip with a warning).
7. Write `tests/unit/test_backtest.py`: mock historical data, verify Sharpe computation, verify drawdown computation, verify JSON output schema.
8. Commit: `feat: add backtesting engine with Sharpe/drawdown/winrate metrics`

### Task 3.2 — Prometheus alert rules

Create `monitoring/alert_rules.yml` with the following alerts:

| Alert name | Condition | Severity | For |
|---|---|---|---|
| `PortfolioValueDrop5pct` | portfolio_value_usd drops >5% in 1h | critical | 5m |
| `PortfolioValueDrop10pct` | portfolio_value_usd drops >10% in 1h | critical | 2m |
| `WalletCircuitOpen` | circuit_breaker_state == 2 (OPEN) for any wallet | critical | 1m |
| `MLConfidenceLow` | ml_confidence_score < 0.50 sustained | warning | 15m |
| `MLEnsembleWeightDrift` | any ensemble_weight_{model} > 0.70 (model dominance) | warning | 10m |
| `RPCLatencyHigh` | rpc_latency_p99_ms > 2000 | warning | 5m |
| `RPCAllEndpointsDegraded` | all RPC endpoints health_score < 0.5 | critical | 2m |
| `GasPriceSpike` | gas_price_gwei > 200 | warning | 3m |
| `TxBroadcastFailureRate` | rate(tx_broadcast_failures[5m]) / rate(tx_broadcast_total[5m]) > 0.2 | critical | 3m |
| `ConceptDriftDetected` | feature_drift_score > 0.15 | warning | 10m |
| `AaveAPYDropped` | aave_supply_apy_pct < 1.0 | info | 30m |
| `FlashCrashKillSwitch` | kill_switch_active == 1 | critical | 0m |
| `DrawdownControllerHalt` | drawdown_controller_state == "HALTED" | critical | 1m |

Update `monitoring/prometheus.yml` to include `alert_rules.yml` via `rule_files`.
Validate: `docker run --rm -v $(pwd)/monitoring:/monitoring prom/prometheus promtool check rules /monitoring/alert_rules.yml`
Commit: `feat: add 13 Prometheus alert rules for portfolio, ML, and network health`

### Task 3.3 — Telegram webhook for Prometheus alerts

Create `wallet_manager/alerting/prometheus_webhook.py`:
1. A lightweight `aiohttp.web` server listening on `0.0.0.0:9093`.
2. Accepts POST `/alerts` from Alertmanager (standard Prometheus webhook format).
3. Formats each alert into a readable Telegram message and sends it via the existing `TelegramNotifier`.
4. Groups alerts: send a single message for multiple simultaneous alerts, not one per alert.
5. Rate-limit: max 1 alert message per 30 seconds per alert name.
6. Add `ALERTMANAGER_WEBHOOK_ENABLED=false` and `ALERTMANAGER_WEBHOOK_PORT=9093` to `.env.example` and `config.py`.
7. Write `tests/unit/test_prometheus_webhook.py`: POST with sample payload, rate-limit test, message formatting test.
8. Commit: `feat: Prometheus Alertmanager webhook → Telegram notifications`

---

## PHASE 4 — REFACTORING & MAINTAINABILITY

### Task 4.1 — Split orchestrator.py

`orchestrator.py` is 1713 lines. Split it:

1. `wallet_manager/orchestrator_gas.py` — extract everything related to `GasMonitor` polling, gas-based wallet suspension/resumption, and gas event notifications. Expose `GasGuard` class.
2. `wallet_manager/orchestrator_lifecycle.py` — extract boot sequence, signal handlers (`SIGTERM`/`SIGINT`), graceful shutdown, and startup/shutdown notifications. Expose `LifecycleManager` class.
3. `wallet_manager/orchestrator.py` — now imports from the two new modules. Must remain the entry point (`python -m wallet_manager.orchestrator` still works).
4. All existing tests must pass unchanged: `pytest tests/integration/test_orchestrator.py tests/unit/ -v`
5. Commit: `refactor: split orchestrator.py into gas, lifecycle, and core modules`

### Task 4.2 — Split task_factories.py

`task_factories.py` is 1636 lines. Split by pillar:

1. `wallet_manager/task_factories_yield.py` — all YIELD pillar task builders.
2. `wallet_manager/task_factories_algo.py` — all ALGO pillar task builders.
3. `wallet_manager/task_factories_short.py` — all SHORT pillar task builders.
4. `wallet_manager/task_factories_airdrop.py` — all AIRDROP pillar task builders.
5. `wallet_manager/task_factories.py` — becomes a re-export shim only:
   ```python
   from wallet_manager.task_factories_yield import *   # noqa: F401, F403
   from wallet_manager.task_factories_algo import *    # noqa: F401, F403
   from wallet_manager.task_factories_short import *   # noqa: F401, F403
   from wallet_manager.task_factories_airdrop import * # noqa: F401, F403
   ```
6. All existing tests pass unchanged.
7. Commit: `refactor: split task_factories.py into per-pillar modules`

---

## PHASE 5 — TEST COVERAGE COMPLETION

### Task 5.1 — Tests for untested critical modules

The following modules have zero test coverage. Write tests for each:

**`tests/unit/test_flash_crash_detector.py`**
- Price drop below threshold does not trigger (9.9%)
- Price drop above threshold triggers kill switch (8.1% in window)
- Kill switch cooldown prevents re-triggering within 1 hour
- Recovery after cooldown

**`tests/unit/test_stochastic_stress_test.py`**
- Monte Carlo passes when >55% of simulations are profitable
- Monte Carlo fails when <55% of simulations are profitable
- RFE prunes high-variance features
- RFE respects minimum feature floor (5 features)

**`tests/unit/test_telegram_notifier.py`** (the full notifier, not just commands)
- `send_notification` formats EventKind messages correctly
- Token bucket rate limiting: rapid fire messages get throttled
- HTTP failure is caught and logged, not propagated

**`tests/unit/test_algo_strategy.py`**
- Daily drawdown circuit breaker triggers at configured threshold
- Strategy returns correct action for bullish/bearish signals
- Emergency halt flag is respected

**`tests/unit/test_task_factories.py`**
- Each pillar's task factory returns a non-empty list of coroutines
- Task factories respect gas price limits from config

After all five files are written, run:
```bash
pytest tests/ -v --cov=wallet_manager --cov-report=term-missing
```
Coverage must be ≥ 87%.
Commit: `test: add coverage for flash_crash, stress_test, telegram, algo_strategy, task_factories`

### Task 5.2 — Integration test for full execution cycle

Write `tests/integration/test_full_cycle.py`:
1. Boot a mocked orchestrator with all RPC calls mocked.
2. Run one full execution cycle in `legacy` mode — no real network calls.
3. Assert: no exceptions raised, at least one task was dispatched per pillar, Prometheus metrics were updated, no private keys in any log output.
4. Run the same cycle in `quantx` mode.
5. Verify graceful shutdown: `SIGTERM` handler completes without error.
6. Commit: `test: add full execution cycle integration test (legacy + quantx modes)`

---

## PHASE 6 — OPERATIONAL EXCELLENCE

### Task 6.1 — Structured JSON logging improvements

`LOG_FORMAT=json` mode exists but the JSON schema is inconsistent across modules. Standardize it:
1. Every log record in JSON mode must include: `timestamp` (ISO 8601), `level`, `logger`, `message`, `wallet_id` (if applicable), `strategy` (if applicable), `tx_hash` (if applicable).
2. Create `wallet_manager/utils/log_schema.py` with a `WalletManagerLogRecord` TypedDict defining the schema.
3. Update `build_logger()` in `logging_filter.py` to attach a `JsonFormatter` when `LOG_FORMAT=json`.
4. Write `tests/unit/test_log_schema.py`: emit a log record in JSON mode, parse the output, assert all required fields present.
5. Commit: `feat: standardize JSON log schema with required fields`

### Task 6.2 — Health check endpoint improvements

`wallet_manager/utils/health_check.py` exists but is basic. Enhance it:
1. Add `/ready` endpoint (Kubernetes readiness probe) — returns 200 only when: RPC is reachable, vault is initialized, at least one wallet is configured.
2. Add `/live` endpoint (Kubernetes liveness probe) — returns 200 as long as the event loop is running.
3. Add `/metrics/summary` endpoint — returns JSON with: portfolio_value_usd, active_wallets, last_cycle_timestamp, ml_confidence_last, current_execution_mode.
4. Document all endpoints in `wallet_manager/README.md` under a new "Health Endpoints" section.
5. Write `tests/unit/test_health_endpoints.py`: mock all dependencies, test each endpoint status code and response schema.
6. Commit: `feat: add /ready, /live, /metrics/summary health endpoints`

### Task 6.3 — Docker and deployment improvements

1. Update `Dockerfile`:
   - Use multi-stage build: `builder` stage installs deps, `runtime` stage copies only what's needed.
   - Final image must be non-root: `USER wallet:wallet`.
   - Add `HEALTHCHECK CMD curl -f http://localhost:9090/health || exit 1`.
2. Update `docker-compose.yml`:
   - Add `restart: unless-stopped` to the `wallet_manager` service.
   - Add resource limits: `cpus: '1.0'`, `memory: 2g`.
   - Add `depends_on` with health check condition for monitoring services.
3. Create `deployment/k8s/` directory with minimal Kubernetes manifests:
   - `deployment.yaml` — Deployment with readiness/liveness probes, resource requests/limits.
   - `configmap.yaml` — Non-secret config.
   - `secret.yaml` — Template (values redacted) for VAULT_KEY, TELEGRAM_BOT_TOKEN, RPC keys.
   - `service.yaml` — ClusterIP for metrics port.
4. Commit: `feat: multi-stage Dockerfile, docker-compose improvements, k8s manifests`

---

## PHASE 7 — FINAL VALIDATION AND DOCUMENTATION

### Task 7.1 — Full test suite and quality gate

Run the complete quality pipeline and fix every failure:
```bash
ruff check . --output-format=github
ruff format . --check
mypy wallet_manager/ --ignore-missing-imports
pytest tests/ -v -n auto \
  --cov=wallet_manager \
  --cov-report=term-missing \
  --cov-report=xml \
  --cov-fail-under=87
```

If coverage is below 87%, identify the uncovered lines and add tests until it passes. If mypy has errors, fix them. If ruff has errors, fix them. Do not proceed until all four commands pass cleanly.
Commit: `ci: all quality gates passing — ruff, mypy, pytest ≥87% coverage`

### Task 7.2 — Smoke test in legacy mode

Run a 10-second smoke test with mocked credentials:
```bash
VAULT_KEY="kVdPFDoKAHLY4q8X_UrFNVYp5OjLw7fIqO8YtK-3rWo=" \
EXECUTION_MODE=legacy \
TELEGRAM_BOT_TOKEN=test_token \
TELEGRAM_CHAT_ID=123456789 \
timeout 10 python -m wallet_manager.orchestrator 2>&1 | head -50
```
The output must not contain: `ImportError`, `ConfigurationError`, `AttributeError`, `ModuleNotFoundError`. It may contain `ConnectionError` or `RPC` errors (expected without real credentials).
Commit: `test: smoke test passes in legacy mode`

### Task 7.3 — CHANGELOG and agent log

1. Update `CHANGELOG.md` with all changes made during this agent run, organized by phase.
2. Write `AGENT_LOG.md` with:
   - Tasks completed successfully
   - Any tasks that were skipped or partially completed, with reasons
   - Total files modified
   - Coverage before and after
   - Any decisions made that deviate from the task spec (with justification)
3. Final commit: `docs: update CHANGELOG and write AGENT_LOG for autonomous run`

---

## EXECUTION NOTES

- Work through tasks in order: Phase 1 → 2 → 3 → 4 → 5 → 6 → 7.
- Never skip the post-task test run. If tests fail, fix before moving on.
- Commit after every individual task — not at the end of each phase.
- If a task requires a dependency not in `requirements.txt`, add it and note in `AGENT_LOG.md`.
- Do not ask for clarification. Make a reasonable decision and note it in `AGENT_LOG.md`.
- The goal is a production-ready, fully tested, well-documented DeFi portfolio bot.
