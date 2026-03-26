"""
wallet_manager/utils/health_check.py

Startup health checks for RPC connectivity and contract accessibility.
Integrated into orchestrator._validate_startup_config().
"""
from __future__ import annotations

import asyncio
import json

import aiohttp

from wallet_manager.security.logging_filter import build_logger

log = build_logger(__name__)

# Aave V3 Pool contract on Base (chain ID 8453)
_AAVE_V3_POOL_BASE = "0xA238Dd80C259a72e81d7e4664a9801593F98d1c5"


async def check_rpc_connectivity(rpc_url: str, timeout: float = 5.0) -> bool:
    """Check RPC connectivity via eth_blockNumber.

    Args:
        rpc_url: JSON-RPC HTTP endpoint URL.
        timeout: Request timeout in seconds.

    Returns:
        True if the endpoint responds with a valid block number.
    """
    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session, session.post(
            rpc_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                log.warning("health_check: RPC %s returned HTTP %s", rpc_url[:40], resp.status)
                return False
            data = await resp.json(content_type=None)
            result = data.get("result")
            if result and isinstance(result, str) and result.startswith("0x"):
                log.debug("health_check: RPC %s OK (block %s)", rpc_url[:40], int(result, 16))
                return True
            log.warning("health_check: RPC %s unexpected result: %s", rpc_url[:40], result)
            return False
    except Exception as exc:
        log.warning("health_check: RPC %s unreachable: %s", rpc_url[:40], type(exc).__name__)
        return False


async def check_aave_contract_accessibility(rpc_url: str, chain_id: int) -> bool:
    """Check that the Aave V3 Pool contract responds on the given RPC.

    Uses eth_call to query the pool's getReservesList() method signature.

    Args:
        rpc_url: JSON-RPC HTTP endpoint URL.
        chain_id: Chain ID (must be 8453 for Base).

    Returns:
        True if the contract is accessible.
    """
    if chain_id != 8453:
        log.debug("health_check: Aave check skipped (chain_id=%s, expected 8453)", chain_id)
        return True

    # getReservesList() selector: 0xd1946dbc
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{"to": _AAVE_V3_POOL_BASE, "data": "0xd1946dbc"}, "latest"],
        "id": 1,
    }
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10.0)
        ) as session, session.post(
            rpc_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            if resp.status != 200:
                return False
            data = await resp.json(content_type=None)
            result = data.get("result", "")
            # A non-empty hex response means the contract responded
            if result and result != "0x":
                log.debug("health_check: Aave V3 Pool accessible on %s", rpc_url[:40])
                return True
            return False
    except Exception as exc:
        log.warning(
            "health_check: Aave contract check failed on %s: %s",
            rpc_url[:40], type(exc).__name__,
        )
        return False


async def run_startup_health_checks(settings: object) -> list[str]:
    """Run all startup health checks and return a list of problem descriptions.

    Args:
        settings: A :class:`~wallet_manager.config.Settings` instance.

    Returns:
        List of problem strings (empty = all checks passed).
    """
    problems: list[str] = []

    rpc_candidates: list[str] = []
    for attr in ("base_rpc_1", "base_rpc_2", "base_rpc_3", "private_rpc_1"):
        val = getattr(settings, attr, "")
        if val:
            rpc_candidates.append(val)

    if not rpc_candidates:
        problems.append("No Base RPC endpoints configured (BASE_RPC_1..3 or PRIVATE_RPC_1)")
        return problems

    rpc_ok = False
    for rpc_url in rpc_candidates:
        if await check_rpc_connectivity(rpc_url):
            rpc_ok = True
            # Also check Aave contract accessibility on the first working RPC
            chain_id = getattr(settings, "chain_id", 8453) if hasattr(settings, "chain_id") else 8453
            aave_ok = await check_aave_contract_accessibility(rpc_url, chain_id)
            if not aave_ok:
                problems.append(f"Aave V3 Pool contract not accessible on {rpc_url[:40]}")
            break

    if not rpc_ok:
        problems.append(
            f"All Base RPC endpoints unreachable: {', '.join(u[:30] for u in rpc_candidates)}"
        )

    return problems
