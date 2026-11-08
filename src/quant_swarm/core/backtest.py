"""向量化回测。"""
from dataclasses import dataclass
import numpy as np
import polars as pl
from core.cost_model import apply_costs, SCENARIOS

@dataclass
class BacktestResult:
    factor_values: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    turnover: np.ndarray
    sharpe: float
    net_sharpe: float

def evaluate_expression(df: pl.DataFrame, code: str) -> np.ndarray:
    from core.data import DATA_CONTRACT
    panel = {}
    for field in DATA_CONTRACT:
        pivot = df.pivot(values=field, index="date", on="asset").sort("date")
        panel[field] = np.array([pivot[c].to_numpy() for c in pivot.columns[1:]]).T
    env = {
        "ts_mean": lambda x, w: _ts_rolling(x, w, np.mean),
        "ts_std":  lambda x, w: _ts_rolling(x, w, np.std),
        "ts_sum":  lambda x, w: _ts_rolling(x, w, np.sum),
        "ts_max":  lambda x, w: _ts_rolling(x, w, np.max),
        "ts_min":  lambda x, w: _ts_rolling(x, w, np.min),
        "ts_rank": lambda x, w: _ts_rolling(x, w, lambda a: _rank_last(a)),
        "abs": np.abs, "log": lambda x: np.log(np.abs(x) + 1e-9),
        "sqrt": lambda x: np.sqrt(np.abs(x)), "sign": np.sign,
        "clip": lambda x, a, b: np.clip(x, a, b),
        "rank": _cross_rank, "zscore": _cross_zscore,
    }
    env.update(panel)
    env["__builtins__"] = {}
    return eval(code, env, {})

def _ts_rolling(x, w, fn):
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(w, len(x)):
        out[t] = fn(x[t - w:t], axis=0)
    return out

def _rank_last(a):
    return (a[-1] > a[:-1]).sum() / len(a)

def _cross_rank(x):
    return np.apply_along_axis(
        lambda row: (row.argsort().argsort() + 1) / len(row), 1, x)

def _cross_zscore(x):
    m, s = np.nanmean(x, axis=1, keepdims=True), np.nanstd(x, axis=1, keepdims=True)
    return (x - m) / (s + 1e-9)

def backtest_factor(df: pl.DataFrame, code: str, cost_scenario=None) -> BacktestResult:
    if cost_scenario is None:
        cost_scenario = SCENARIOS["base"]
    fv = np.nan_to_num(evaluate_expression(df, code), nan=0.0)
    z = _cross_zscore(fv)
    positions = np.where(z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0))
    positions = positions / np.maximum(np.abs(positions).sum(axis=1, keepdims=True), 1)
    returns = df.pivot(values="returns", index="date", on="asset").sort("date")
    ret_arr = np.array([returns[c].to_numpy() for c in returns.columns[1:]]).T
    gross = (positions[:-1] * ret_arr[1:]).sum(axis=1)
    turnover = np.abs(np.diff(positions, axis=0)).sum(axis=1)
    cost = apply_costs(positions[1:], turnover[:, None], cost_scenario).sum(axis=1)
    net = gross - cost
    sharpe = gross.mean() / (gross.std() + 1e-9) * np.sqrt(252)
    net_sharpe = net.mean() / (net.std() + 1e-9) * np.sqrt(252)
    return BacktestResult(fv, gross, net, turnover, sharpe, net_sharpe)