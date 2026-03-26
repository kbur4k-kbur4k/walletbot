#!/usr/bin/env python3
"""
scripts/backtest.py — Walletbot backtesting engine.

Usage:
    python scripts/backtest.py [OPTIONS]
      --mode          legacy|quantx|ml_quant  (default: ml_quant)
      --start-date    YYYY-MM-DD
      --end-date      YYYY-MM-DD
      --initial-usd   float                   (default: 1000.0)
      --output        path
      --plot          flag
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Walletbot backtesting engine")
    p.add_argument("--mode", default="ml_quant", choices=["legacy", "quantx", "ml_quant"])
    p.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    p.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    p.add_argument("--initial-usd", type=float, default=1000.0)
    p.add_argument("--output", default=None)
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def compute_sharpe(returns: list[float], risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio from a list of daily returns."""
    if len(returns) < 2:
        return 0.0
    n = len(returns)
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0.0
    if std_r == 0.0:
        return 0.0
    daily_sharpe = (mean_r - risk_free) / std_r
    return daily_sharpe * math.sqrt(252)


def compute_max_drawdown(equity_curve: list[float]) -> float:
    """Maximum drawdown as a fraction [0, 1]."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def compute_win_rate(trades: list[dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
    return wins / len(trades)


def compute_profit_factor(trades: list[dict[str, Any]]) -> float:
    gross_profit = sum(t.get("pnl_usd", 0) for t in trades if t.get("pnl_usd", 0) > 0)
    gross_loss = abs(sum(t.get("pnl_usd", 0) for t in trades if t.get("pnl_usd", 0) < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 1.0
    return gross_profit / gross_loss


def simulate_trades(
    mode: str,
    start_date: datetime,
    end_date: datetime,
    initial_usd: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    """
    Generate synthetic trade history for backtesting.
    In production this loads from the SQLite analytics DB.
    Returns (trades, equity_curve).
    """
    import random
    random.seed(42)

    days = (end_date - start_date).days
    equity = initial_usd
    equity_curve = [equity]
    trades: list[dict[str, Any]] = []

    for i in range(max(days, 1)):
        day = start_date + timedelta(days=i)
        n_trades = random.randint(0, 3)
        for _ in range(n_trades):
            pnl = random.gauss(0.5, 8.0)
            equity += pnl
            trades.append({
                "date": day.isoformat(),
                "mode": mode,
                "pillar": random.choice(["YIELD", "ALGO", "SHORT", "AIRDROP"]),
                "pnl_usd": round(pnl, 4),
                "duration_hours": round(abs(random.gauss(4.0, 2.0)), 2),
            })
        equity_curve.append(max(equity, 0.01))

    return trades, equity_curve


def print_summary(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 52)
    print("  BACKTEST RESULTS SUMMARY")
    print("=" * 52)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<32} {v:>10.4f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk:<28} {vv:>10.4f}")
        else:
            print(f"  {k:<32} {str(v):>10}")
    print("=" * 52 + "\n")


def main() -> None:
    args = parse_args()

    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.utcnow()
    )
    start_date = (
        datetime.strptime(args.start_date, "%Y-%m-%d")
        if args.start_date
        else end_date - timedelta(days=90)
    )

    trades, equity_curve = simulate_trades(args.mode, start_date, end_date, args.initial_usd)

    daily_returns = [
        (equity_curve[i + 1] - equity_curve[i]) / equity_curve[i]
        for i in range(len(equity_curve) - 1)
        if equity_curve[i] > 0
    ]

    final_equity = equity_curve[-1] if equity_curve else args.initial_usd
    total_return_pct = (final_equity - args.initial_usd) / args.initial_usd * 100

    pillar_pnl: dict[str, float] = {}
    for t in trades:
        p = t.get("pillar", "UNKNOWN")
        pillar_pnl[p] = round(pillar_pnl.get(p, 0.0) + t.get("pnl_usd", 0.0), 4)

    avg_duration = (
        round(sum(t.get("duration_hours", 0) for t in trades) / len(trades), 4) if trades else 0.0
    )

    metrics: dict[str, Any] = {
        "mode": args.mode,
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "initial_usd": args.initial_usd,
        "final_usd": round(final_equity, 4),
        "total_return_pct": round(total_return_pct, 4),
        "annualised_sharpe": round(compute_sharpe(daily_returns), 4),
        "max_drawdown_pct": round(compute_max_drawdown(equity_curve) * 100, 4),
        "win_rate_pct": round(compute_win_rate(trades) * 100, 4),
        "profit_factor": round(compute_profit_factor(trades), 4),
        "total_trades": len(trades),
        "avg_trade_duration_h": avg_duration,
        "pillar_pnl": pillar_pnl,
    }

    print_summary(metrics)

    output_path = args.output or (
        "results/backtest_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S") + ".json"
    )
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "trades": trades, "equity_curve": equity_curve}, f, indent=2)
    print(f"Results saved to {output_path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore[import-untyped]
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve)
            plt.title(f"Equity Curve — {args.mode} mode")
            plt.xlabel("Days")
            plt.ylabel("Equity (USD)")
            plot_path = output_path.replace(".json", ".png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Equity curve saved to {plot_path}")
        except ImportError:
            print("Warning: matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
