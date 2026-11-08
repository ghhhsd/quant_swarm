"""组合层：候选池按预设规则组组合，输出组合层面成本后 PnL（最终指标）。"""
from dataclasses import dataclass
from typing import List
import numpy as np
import polars as pl
from core.backtest import evaluate_expression, _cross_zscore
from core.cost_model import apply_costs, SCENARIOS

@dataclass
class PortfolioResult:
    net_returns: np.ndarray
    sharpe: float
    max_drawdown: float
    total_return: float

def build_portfolio(df: pl.DataFrame, factor_codes: List[str],
                    weighting: str = "equal",
                    cost_scenario=None) -> PortfolioResult:
    if cost_scenario is None:
        cost_scenario = SCENARIOS["base"]
    if not factor_codes:
        return PortfolioResult(np.array([0.0]), 0.0, 0.0, 0.0)
    positions_list = []
    for code in factor_codes:
        fv = np.nan_to_num(evaluate_expression(df, code), nan=0.0)
        z = _cross_zscore(fv)
        pos = np.where(z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0))
        pos = pos / np.maximum(np.abs(pos).sum(axis=1, keepdims=True), 1)
        positions_list.append(pos)
    w = np.ones(len(positions_list)) / len(positions_list) if weighting == "equal" \
        else np.ones(len(positions_list)) / len(positions_list)
    positions = sum(wi * p for wi, p in zip(w, positions_list))
    positions = positions / np.maximum(np.abs(positions).sum(axis=1, keepdims=True), 1)
    returns_pivot = df.pivot(values="returns", index="date", on="asset").sort("date")
    ret_arr = np.array([returns_pivot[c].to_numpy() for c in returns_pivot.columns[1:]]).T
    gross = (positions[:-1] * ret_arr[1:]).sum(axis=1)
    turnover = np.abs(np.diff(positions, axis=0)).sum(axis=1)
    cost = apply_costs(positions[1:], turnover[:, None], cost_scenario).sum(axis=1)
    net = gross - cost
    sharpe = net.mean() / (net.std() + 1e-9) * np.sqrt(252)
    cum = np.cumprod(1 + net)
    mdd = float(np.max(1 - cum / np.maximum.accumulate(cum)))
    total = float(cum[-1] - 1)
    return PortfolioResult(net, float(sharpe), mdd, total)