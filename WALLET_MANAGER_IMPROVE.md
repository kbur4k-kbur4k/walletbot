# wallet_manager — Code Quality Improvement Prompt

> **Instructions for Claude Code:** Apply every section in order.
> Run the VERIFY block after each section. If any assertion fails, re-apply
> that section. After all sections pass, run FINAL_VERIFY.
> Loop until it prints "ALL CHECKS PASSED". Do not stop early.

---

## Issue map

| # | Category | Problem |
|---|----------|---------|
| 1 | File size | `telegram_notifier.py` 2145 lines — CommandCenter (843 lines) extracted to own module |
| 2 | File size | `orchestrator.py` 3320 lines — task factories extracted to own module |
| 3 | Test quality | 88 tests with 0–1 assertions ("does not raise" = not actually testing anything) |

---

## SECTION 1 — Split telegram_notifier.py

### Why this split is safe

`telegram_notifier.py` contains two completely separate concerns:

1. **HTTP transport + formatting** (lines 1–596): `_TelegramCredentials`,
   `_TokenBucket`, `_MessageFormatter`, `_TelegramClient`
2. **Command dispatch** (lines 597–2062): `CommandState`, `ConfirmationGate`,
   `StatusProvider`, `LiveStatusProvider`, `CommandCenter`
3. **Public API** (lines 1867–2145): `TelegramNotifier`, `ProactiveAlerter`

`CommandCenter` alone is 843 lines and handles every `/command`. It can move
to `wallet_manager/security/telegram_commands.py` while `telegram_notifier.py`
keeps the public-facing classes. Nothing outside the package imports
`CommandCenter` directly.

### Step 1 — Create wallet_manager/security/telegram_commands.py

Read `wallet_manager/telegram_notifier.py` carefully, then create a new file
`wallet_manager/security/telegram_commands.py` containing **exactly** these
classes (copied verbatim from telegram_notifier.py, preserving all docstrings
and logic — do not change any logic):

- `CommandState`
- `ConfirmationGate`
- `StatusProvider`
- `LiveStatusProvider`
- `CommandCenter`

The new file needs these imports at the top (collect all imports that these
classes use from telegram_notifier.py's import block plus any new ones needed):

```python
"""
wallet_manager/security/telegram_commands.py

Telegram command dispatch — CommandState, CommandCenter, and supporting types.

Extracted from telegram_notifier.py to keep that module under 600 lines.
All public names are re-exported from telegram_notifier for backward compat.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from wallet_manager.config import get_settings
from wallet_manager.security.logging_filter import build_logger

if TYPE_CHECKING:
    from wallet_manager.telegram_notifier import _TelegramClient, _MessageFormatter

log = build_logger(__name__)
```

### Step 2 — Update telegram_notifier.py

After creating `telegram_commands.py`:

1. **Remove** the class bodies of `CommandState`, `ConfirmationGate`,
   `StatusProvider`, `LiveStatusProvider`, `CommandCenter` from
   `telegram_notifier.py`.

2. **Add** this import block immediately after the existing imports in
   `telegram_notifier.py`:

```python
from wallet_manager.security.telegram_commands import (
    CommandState,
    ConfirmationGate,
    StatusProvider,
    LiveStatusProvider,
    CommandCenter,
)
```

3. These names are now re-exported — any code doing
   `from wallet_manager.telegram_notifier import CommandState` continues to
   work without changes.

### Step 3 — Verify line counts improved

**VERIFY:**
```bash
python3 -c "
import ast

# New file must exist and be valid Python
src_cmd = open('wallet_manager/security/telegram_commands.py').read()
ast.parse(src_cmd)
for cls in ['CommandState', 'ConfirmationGate', 'StatusProvider',
            'LiveStatusProvider', 'CommandCenter']:
    assert f'class {cls}' in src_cmd, f'FAIL: {cls} not found in telegram_commands.py'

# telegram_notifier.py must still be valid and re-export these names
src_tn = open('wallet_manager/telegram_notifier.py').read()
ast.parse(src_tn)
assert 'from wallet_manager.security.telegram_commands import' in src_tn, \
    'FAIL: re-export import missing from telegram_notifier.py'

# telegram_notifier.py must be shorter
lines = len(src_tn.splitlines())
assert lines < 1500, f'FAIL: telegram_notifier.py still {lines} lines (target <1500)'

# All Python files still compile
import glob
for pf in glob.glob('wallet_manager/**/*.py', recursive=True):
    if '__pycache__' in pf: continue
    try:
        ast.parse(open(pf).read())
    except SyntaxError as e:
        raise AssertionError(f'FAIL: syntax error in {pf}: {e}')

print(f'PASS: Section 1 — telegram_notifier.py now {lines} lines, CommandCenter extracted')
"
```

---

## SECTION 2 — Split orchestrator.py: extract task factories

### Why this split is safe

`orchestrator.py` contains two very different concerns:

1. **Infrastructure**: `GasMonitor`, `WalletOrchestrator`, `MasterOrchestrator`,
   startup validation, wallet loading — this must stay in `orchestrator.py`
   because many strategy modules do
   `from wallet_manager.orchestrator import GasMonitor`.

2. **Task factories**: `_default_task_factory`, `_robust_paper_task_factory`,
   `_robust_task_factory`, `_ml_quant_task_factory`, `_quantx_task_factory`,
   `_three_pillar_task_factory`, `_airdrop_task_factory`,
   `_execute_usdc_to_eth`, `_execute_eth_to_usdc` — these are pure async
   functions with no class membership. They can move to a new module.

### Step 1 — Create wallet_manager/task_factories.py

Read `wallet_manager/orchestrator.py` carefully, then create
`wallet_manager/task_factories.py` containing **exactly** these functions
(copied verbatim, no logic changes):

- `_default_task_factory`
- `_robust_paper_task_factory`
- `_robust_task_factory`
- `_ml_quant_task_factory`
- `_quantx_task_factory`
- `_three_pillar_task_factory`
- `_airdrop_task_factory`
- `_execute_usdc_to_eth`
- `_execute_eth_to_usdc`

The new file header:

```python
"""
wallet_manager/task_factories.py

Per-execution-mode task factory coroutines.

Extracted from orchestrator.py to keep that module under 1500 lines.
All names are re-exported from orchestrator for backward compatibility.
"""
from __future__ import annotations
```

Collect **all** imports these functions use from `orchestrator.py`'s import
block and add them to `task_factories.py`. For any reference to
`WalletOrchestrator`, `GasMonitor`, or other orchestrator-defined names, use
a `TYPE_CHECKING` guard or a late import inside the function body (pattern
already used throughout orchestrator.py: `from wallet_manager.orchestrator
import X` inside the function).

### Step 2 — Update orchestrator.py

1. **Remove** the function bodies of the nine factories listed above from
   `orchestrator.py`.

2. **Add** this import immediately after orchestrator.py's existing imports:

```python
from wallet_manager.task_factories import (
    _default_task_factory,
    _robust_paper_task_factory,
    _robust_task_factory,
    _ml_quant_task_factory,
    _quantx_task_factory,
    _three_pillar_task_factory,
    _airdrop_task_factory,
    _execute_usdc_to_eth,
    _execute_eth_to_usdc,
)
```

3. The `MasterOrchestrator._select_task_factory` method references these
   names — it will find them through the import above without any changes.

**VERIFY:**
```bash
python3 -c "
import ast, glob

# New file must exist and be valid
src_tf = open('wallet_manager/task_factories.py').read()
ast.parse(src_tf)
for fn in ['_default_task_factory', '_robust_paper_task_factory',
           '_robust_task_factory', '_ml_quant_task_factory',
           '_quantx_task_factory', '_three_pillar_task_factory',
           '_airdrop_task_factory', '_execute_usdc_to_eth', '_execute_eth_to_usdc']:
    assert f'async def {fn}' in src_tf or f'def {fn}' in src_tf, \
        f'FAIL: {fn} not found in task_factories.py'

# orchestrator.py must be shorter and still valid
src_o = open('wallet_manager/orchestrator.py').read()
ast.parse(src_o)
lines = len(src_o.splitlines())
assert lines < 1800, f'FAIL: orchestrator.py still {lines} lines (target <1800)'
assert 'from wallet_manager.task_factories import' in src_o, \
    'FAIL: re-export import missing from orchestrator.py'

# All files still compile
for pf in glob.glob('wallet_manager/**/*.py', recursive=True):
    if '__pycache__' in pf: continue
    try:
        ast.parse(open(pf).read())
    except SyntaxError as e:
        raise AssertionError(f'FAIL: syntax error in {pf}: {e}')

print(f'PASS: Section 2 — orchestrator.py now {lines} lines, task factories extracted')
"
```

---

## SECTION 3 — Fix 88 weak tests

### The pattern to fix

There are two classes of weak tests:

**Class A — "does not raise" tests (0 assertions):**
These call a function and trust that no exception is the proof of correctness.
That is not a test — it just proves the code doesn't crash on that exact input.
They need at least one assertion about the *return value* or *side effect*.

**Class B — mock-heavy tests with 1 vague assertion:**
These create MagicMock objects, call a function, and assert something trivially
true like `assert result is None`. They need assertions about *what the mock
was called with*.

### Files to fix and what assertions to add

Fix **every** test listed below. The rule: after the action, assert at least
one meaningful property of the outcome.

---

#### tests/unit/test_cooldown.py

**`test_wait_does_not_raise`** — add assertion that sleep was called once with
a non-negative float:
```python
# After await mgr.wait():
mock_sleep.assert_called_once()
delay = mock_sleep.call_args[0][0]
assert delay >= 0.0
```

**`test_invalid_lambda_raises_value_error`** — already raises, but add
assertion about the message:
```python
with pytest.raises(ValueError, match="positive"):
    ...
```

**`test_invalid_sigma_raises_value_error`** — same:
```python
with pytest.raises(ValueError, match="non-negative|sigma"):
    ...
```

**`test_clear_nonexistent_wallet_does_not_raise`** — add assertion that the
manager state is unchanged:
```python
mgr.clear("nonexistent_wallet")
assert not mgr.is_in_cooldown("nonexistent_wallet")
```

---

#### tests/unit/test_balance_provider.py

**`test_fetch_balance_checksums_address`** — assert that
`to_checksum_address` was called with the raw input:
```python
mock_cls.to_checksum_address.assert_called_once_with("0xabcd")
# or assert the checksum was passed to get_balance
```

**`test_fetch_balance_creates_provider_with_rpc_url`** — assert that
`AsyncHTTPProvider` was instantiated with the correct URL:
```python
mock_cls.AsyncHTTPProvider.assert_called_once()
call_args = mock_cls.AsyncHTTPProvider.call_args
assert "https://rpc.example.com" in str(call_args)
```

**`test_fetch_native_balance_rpc_error_propagates`** — assert the exception
type propagates:
```python
with pytest.raises(Exception, match="RPC connection refused"):
    await fetch_native_balance(...)
```

**`test_fetch_native_balance_returns_wei`** — assert the return value equals
the mock's return value:
```python
result = await fetch_native_balance(...)
assert result == 12345  # whatever the mock returns
```

---

#### tests/integration/test_execution_mode.py

**`test_garch_stationarity_validated`** — assert exception message mentions
the constraint:
```python
with pytest.raises(ConfigurationError, match="stationar|alpha.*beta|garch"):
    ...
```

---

#### tests/integration/test_orchestrator.py

**`test_orchestrator_single_cycle_completes`** — after the cycle, assert at
least one observable outcome (notifier was called, or no exception raised is
insufficient alone). Add:
```python
# Assert the notifier received at least one call
assert mock_notifier.notify.called or mock_notifier.send.called
```

---

#### tests/integration/test_paper_engine.py

**`test_paper_engine_never_calls_broadcast`** — this is the critical one.
After running the paper engine, assert the broadcast mock was *never* called:
```python
mock_broadcaster.broadcast.assert_not_called()
# or
assert mock_wallet_manager.sign_and_broadcast_tx.call_count == 0
```

---

#### tests/unit/test_analytics_db.py

**`test_close_session`** — assert the session is actually closed:
```python
await db.close()
assert not db.is_open()  # or assert db._conn is None
```

**`test_update_tx_status`** — assert the status was actually updated:
```python
await db.update_tx_status(tx_hash, TxStatus.MINED)
record = await db.get_tx(tx_hash)
assert record.status == TxStatus.MINED
```

---

#### tests/unit/test_config.py

**`test_garch_stationarity_boundary_raises`** and
**`test_garch_stationarity_violation_raises`** — add match pattern:
```python
with pytest.raises(ConfigurationError, match="stationar|garch|alpha"):
    ...
```

---

#### tests/unit/test_execution_logic.py

For each weak test (`test_analytics_not_called_when_no_signal`,
`test_execution_records_trade_to_analytics`,
`test_circuit_breaker_halt_skips_execution`,
`test_halt_on_drawdown_no_broadcast`,
`test_sizing_multiplier_zero_no_broadcast`):

Pattern — assert the mock was or was not called:
```python
# "not called" pattern:
mock_analytics.record_trade.assert_not_called()

# "called once with" pattern:
mock_analytics.record_trade.assert_called_once()
args = mock_analytics.record_trade.call_args[0]
assert args[0].strategy_type == "ALGO"
```

---

#### tests/unit/test_flash_crash.py

**`test_ingesting_price_does_not_raise`** — add assertion on state:
```python
detector.ingest(price=3000.0, timestamp=time.time())
# Assert the price was recorded
assert detector.last_price == pytest.approx(3000.0)
# or assert the buffer has one entry
assert len(detector._prices) >= 1
```

---

#### tests/unit/test_feature_drift_monitor.py

**`test_update_does_not_raise`** — assert the monitor updated its state:
```python
monitor.update(features)
assert monitor.n_updates >= 1
# or assert some metric changed
```

---

#### tests/unit/test_alpha_decay_monitor.py

**`test_crowding_detector_update_does_not_raise`** — assert state changed:
```python
detector.update(signal_score=0.7, timestamp=time.time())
assert detector.n_samples >= 1
```

---

### General rule for any remaining weak test

For any test not listed above that has 0 assertions:

1. If it tests that an exception is raised → add `match=` to narrow it.
2. If it tests a void function → assert the mock was called correctly:
   `mock_x.assert_called_once_with(expected_arg)`.
3. If it tests a function with a return value → assert `result == expected`.
4. Never leave a test that only proves "it didn't crash".

**VERIFY:**
```bash
python3 -c "
import ast, glob

zero_assert = []
for path in glob.glob('tests/**/*.py', recursive=True):
    if '__pycache__' in path: continue
    src = open(path).read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            continue
        if not node.name.startswith('test_'): continue
        body = ast.unparse(node)
        # Count real assertions (not just 'with pytest.raises' without a match)
        asserts = body.count('assert ')
        raises = body.count('pytest.raises')
        mock_assert = body.count('.assert_called') + body.count('.assert_not_called')
        total = asserts + mock_assert
        if total == 0 and raises == 0:
            zero_assert.append(f'{path}::{node.name}')

if zero_assert:
    print(f'FAIL: {len(zero_assert)} tests still have zero assertions:')
    for t in zero_assert[:15]:
        print(f'  {t}')
    raise AssertionError('Fix the tests above')

print(f'PASS: Section 3 — all tests have at least one assertion or pytest.raises')
"
```

---

## FINAL_VERIFY

```bash
python3 << 'PYEOF'
import ast, os, re, sys, glob

results = []
def check(name, cond, detail=""):
    results.append(("PASS" if cond else "FAIL", name, detail))

# ── Section 1: telegram split ──────────────────────────────────────────────
cmd_path = "wallet_manager/security/telegram_commands.py"
check("S1 telegram_commands.py exists", os.path.isfile(cmd_path))
if os.path.isfile(cmd_path):
    src = open(cmd_path).read()
    try: ast.parse(src); check("S1 telegram_commands.py compiles", True)
    except SyntaxError as e: check("S1 telegram_commands.py compiles", False, str(e))
    for cls in ["CommandState","ConfirmationGate","StatusProvider",
                "LiveStatusProvider","CommandCenter"]:
        check(f"S1 {cls} in telegram_commands.py", f"class {cls}" in src)

tn_src = open("wallet_manager/telegram_notifier.py").read()
tn_lines = len(tn_src.splitlines())
check("S1 telegram_notifier.py < 1500 lines", tn_lines < 1500, f"{tn_lines} lines")
check("S1 telegram_notifier.py re-exports commands",
      "from wallet_manager.security.telegram_commands import" in tn_src)

# ── Section 2: orchestrator split ─────────────────────────────────────────
tf_path = "wallet_manager/task_factories.py"
check("S2 task_factories.py exists", os.path.isfile(tf_path))
if os.path.isfile(tf_path):
    src = open(tf_path).read()
    try: ast.parse(src); check("S2 task_factories.py compiles", True)
    except SyntaxError as e: check("S2 task_factories.py compiles", False, str(e))
    for fn in ["_default_task_factory","_robust_task_factory",
               "_ml_quant_task_factory","_airdrop_task_factory"]:
        check(f"S2 {fn} in task_factories.py",
              f"def {fn}" in src)

orch_src = open("wallet_manager/orchestrator.py").read()
orch_lines = len(orch_src.splitlines())
check("S2 orchestrator.py < 1800 lines", orch_lines < 1800, f"{orch_lines} lines")
check("S2 orchestrator.py re-exports factories",
      "from wallet_manager.task_factories import" in orch_src)

# ── Section 3: test quality ────────────────────────────────────────────────
zero = []
for path in glob.glob("tests/**/*.py", recursive=True):
    if "__pycache__" in path: continue
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if not isinstance(node,(ast.AsyncFunctionDef,ast.FunctionDef)): continue
        if not node.name.startswith("test_"): continue
        body = ast.unparse(node)
        total = body.count("assert ") + body.count(".assert_called") + \
                body.count(".assert_not_called") + body.count("pytest.raises")
        if total == 0:
            zero.append(f"{path}::{node.name}")
check("S3 no zero-assertion tests", len(zero)==0,
      f"{len(zero)} tests still empty: " + "; ".join(zero[:5]))

# ── All Python files compile ───────────────────────────────────────────────
bad = []
for pf in (glob.glob("wallet_manager/**/*.py", recursive=True) +
           glob.glob("tests/**/*.py", recursive=True)):
    if "__pycache__" in pf or "allcode" in pf: continue
    try: ast.parse(open(pf).read())
    except SyntaxError as e: bad.append(f"{pf}: {e}")
check("All Python files compile", len(bad)==0, "; ".join(bad))

# ── Regression: previous fixes ────────────────────────────────────────────
src = open("wallet_manager/resilience/retry.py").read()
check("retry.py: MaxRetriesExceededError", "MaxRetriesExceededError" in src)
vsrc = open("wallet_manager/security/vault.py").read()
check("vault.py: fernet on early return",
      "self._fernet = new_fernet" in vsrc.split("not path.exists")[1][:300])
tsrc = open("tests/unit/test_vault_key_rotation.py").read()
check("vault rotation: async+await",
      "async def test_" in tsrc and "await vault.rotate_key" in tsrc)

# ── Summary ────────────────────────────────────────────────────────────────
fails  = [r for r in results if r[0]=="FAIL"]
passes = [r for r in results if r[0]=="PASS"]

print(f"\n{'='*64}")
print(f"  FINAL VERIFICATION — {len(passes)} passed, {len(fails)} failed")
print(f"{'='*64}")
for status, name, detail in results:
    print(f"  {'✓' if status=='PASS' else '✗'}  {name}")
    if detail: print(f"       → {detail}")

if fails:
    print(f"\n  ❌  {len(fails)} FAILED — re-apply those sections and loop")
    sys.exit(1)
else:
    print(f"\n  ✅  ALL {len(passes)} CHECKS PASSED")
    sys.exit(0)
PYEOF
```

---

## Loop control

```
Exit 0  →  Done.
Exit 1  →  Read the FAIL lines, re-apply those sections, run FINAL_VERIFY again.
```

---

## Summary of changes

| Section | Files changed | What |
|---------|---------------|------|
| 1 | `telegram_notifier.py` → **1 file created**: `security/telegram_commands.py` | CommandCenter (843 lines) extracted. telegram_notifier.py drops from 2145 to ~700 lines |
| 2 | `orchestrator.py` → **1 file created**: `task_factories.py` | 9 task factories extracted. orchestrator.py drops from 3320 to ~1400 lines |
| 3 | 15+ test files | 88 zero/near-zero-assertion tests get real assertions — actually test the behavior |

**No logic changes. No API changes. All existing imports continue to work.**
