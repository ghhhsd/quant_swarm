"""
回测结果数据结构。

关键设计：
1. 单因子回测结果 + 批量回测结果。
2. 含成本分解（佣金、价差、冲击）。
3. 含成本后 Sharpe（最终指标）。
4. 可序列化（供事件溯源）。
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class FactorBacktestResult:
    """单因子回测结果。"""
    fid: str
    code: str
    factor_values: np.ndarray      # (T, A)
    positions: np.ndarray          # (T, A)
    gross_returns: np.ndarray      # (T,)
    net_returns: np.ndarray        # (T,)
    turnover: np.ndarray           # (T,)
    gross_sharpe: float
    net_sharpe: float
    total_cost: float
    n_obs: int

    def to_dict(self):
        return {
            "fid": self.fid,
            "code": self.code,
            "gross_sharpe": self.gross_sharpe,
            "net_sharpe": self.net_sharpe,
            "total_cost": self.total_cost,
            "n_obs": self.n_obs,
        }


@dataclass
class BatchBacktestResult:
    """批量回测结果。"""
    results: List[FactorBacktestResult]
    n_factors: int

    def to_dict(self):
        return {
            "n_factors": self.n_factors,
            "results": [r.to_dict() for r in self.results],
        }

    def get(self, fid: str) -> Optional[FactorBacktestResult]:
        for r in self.results:
            if r.fid == fid:
                return r
        return None