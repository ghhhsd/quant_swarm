"""
组合结果数据结构。

关键设计：
1. 单批次组合结果 + 跨批次汇总。
2. 含成本分解（佣金、价差、冲击）。
3. 含换手分解（组合换手、单因子换手、抵消效应）。
4. 可序列化（供事件溯源）。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


@dataclass
class PortfolioResult:
    """单批次组合结果。"""
    batch_id: str
    n_factors: int
    weighting: str
    positions: np.ndarray           # (T, A) 组合持仓
    gross_returns: np.ndarray       # (T,)
    net_returns: np.ndarray         # (T,)
    combined_turnover: np.ndarray   # (T,) 组合换手
    sum_single_turnover: np.ndarray # (T,) 单因子换手之和
    offset_ratio: float             # 抵消比例
    total_cost: float
    gross_sharpe: float
    net_sharpe: float
    max_drawdown: float
    total_return: float
    n_obs: int

    def to_dict(self):
        return {
            "batch_id": self.batch_id,
            "n_factors": self.n_factors,
            "weighting": self.weighting,
            "total_cost": self.total_cost,
            "gross_sharpe": self.gross_sharpe,
            "net_sharpe": self.net_sharpe,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "n_obs": self.n_obs,
            "offset_ratio": self.offset_ratio,
        }


@dataclass
class BatchPortfolioSummary:
    """跨批次组合汇总。"""
    results: List[PortfolioResult]
    n_batches: int

    def to_dict(self):
        return {
            "n_batches": self.n_batches,
            "results": [r.to_dict() for r in self.results],
        }

    def mean_net_sharpe(self) -> float:
        """跨批次平均净 Sharpe。"""
        sharpes = [r.net_sharpe for r in self.results]
        return float(np.mean(sharpes)) if sharpes else 0.0

    def std_net_sharpe(self) -> float:
        """跨批次净 Sharpe 标准差。"""
        sharpes = [r.net_sharpe for r in self.results]
        return float(np.std(sharpes)) if sharpes else 0.0