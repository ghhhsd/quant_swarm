"""
组合构建规则（v3.1 修正③：组合层是最终指标，规则必须显式声明）。

关键设计：
1. 开/关比较必须用同一 PortfolioConfig，否则比较的不是"层的贡献"。
2. 组合规则：
   - equal：等权
   - ic_weighted：按滚动 IC 加权
   - risk_parity：风险平价（简化：按波动倒数加权）
3. 成本模型复用 C5.5 的三档场景。
4. 组合 PnL 是成本后净收益。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import numpy as np
import polars as pl

from core.backtest import evaluate_expression, _cross_zscore
from candidate_tier.c5_5_economic.cost_scenarios import (
    CostScenario, get_scenario, CostScenarioName,
)


class WeightingScheme(str, Enum):
    EQUAL = "equal"
    IC_WEIGHTED = "ic_weighted"
    RISK_PARITY = "risk_parity"


@dataclass(frozen=True)
class PortfolioConfig:
    """
    组合构建配置。frozen=True，开/关必须用同一配置。
    """
    weighting: str = WeightingScheme.EQUAL.value
    cost_scenario: str = CostScenarioName.BASE.value
    # IC 加权的滚动窗口
    ic_window: int = 60

    def to_dict(self):
        return {
            "weighting": self.weighting,
            "cost_scenario": self.cost_scenario,
            "ic_window": self.ic_window,
        }


@dataclass
class PortfolioPnLResult:
    """组合 PnL 结果（单 OOS 批次）。"""
    sharpe: float
    max_drawdown: float
    total_return: float
    net_returns: np.ndarray
    n_factors: int
    weighting: str

    def to_dict(self):
        return {
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "total_return": self.total_return,
            "n_factors": self.n_factors,
            "weighting": self.weighting,
        }


class PortfolioBuilder:
    """
    组合构建器。

    输入：候选池因子代码列表 + 数据
    输出：组合 PnL（成本后）
    """

    def __init__(self, df: pl.DataFrame, config: PortfolioConfig):
        self.df = df
        self.config = config

    def build(self, codes: List[str],
              oos_start: int, oos_end: int) -> PortfolioPnLResult:
        """
        在指定 OOS 窗口内构建组合并计算 PnL。

        codes：候选池因子代码
        oos_start / oos_end：OOS 窗口的起止索引
        """
        if not codes:
            return PortfolioPnLResult(
                sharpe=0.0, max_drawdown=0.0, total_return=0.0,
                net_returns=np.array([0.0]), n_factors=0,
                weighting=self.config.weighting,
            )

        # 1) 每个因子的持仓
        positions_list = []
        factor_returns_list = []
        for code in codes:
            fv = np.nan_to_num(evaluate_expression(self.df, code), nan=0.0)
            z = _cross_zscore(fv)
            pos = np.where(z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0))
            pos = pos / np.maximum(np.abs(pos).sum(axis=1, keepdims=True), 1)
            positions_list.append(pos)
            # 因子收益（用于 IC 加权）
            returns = self.df.pivot(values="returns", index="date",
                                    on="asset").sort("date")
            ret_arr = np.array([returns[c].to_numpy()
                                for c in returns.columns[1:]]).T
            factor_returns_list.append(
                (pos[:-1] * ret_arr[1:]).sum(axis=1))

        # 2) 按配置加权
        if self.config.weighting == WeightingScheme.EQUAL.value:
            w = np.ones(len(positions_list)) / len(positions_list)
        elif self.config.weighting == WeightingScheme.IC_WEIGHTED.value:
            w = self._ic_weights(factor_returns_list, oos_start, oos_end)
        elif self.config.weighting == WeightingScheme.RISK_PARITY.value:
            w = self._risk_parity_weights(factor_returns_list, oos_start, oos_end)
        else:
            raise ValueError(f"未知加权方式: {self.config.weighting}")

        # 3) 组合持仓
        positions = sum(wi * p for wi, p in zip(w, positions_list))
        positions = positions / np.maximum(
            np.abs(positions).sum(axis=1, keepdims=True), 1)

        # 4) 组合净收益（成本后）
        returns = self.df.pivot(values="returns", index="date",
                                on="asset").sort("date")
        ret_arr = np.array([returns[c].to_numpy()
                            for c in returns.columns[1:]]).T
        # 只在 OOS 窗口内计算
        pos_oos = positions[oos_start:oos_end]
        ret_oos = ret_arr[oos_start:oos_end]
        if len(pos_oos) < 2:
            return PortfolioPnLResult(
                sharpe=0.0, max_drawdown=0.0, total_return=0.0,
                net_returns=np.array([0.0]), n_factors=len(codes),
                weighting=self.config.weighting,
            )

        gross = (pos_oos[:-1] * ret_oos[1:]).sum(axis=1)
        turnover = np.abs(np.diff(pos_oos, axis=0)).sum(axis=1)

        # 成本（用 C5.5 的场景参数）
        sc = get_scenario(self.config.cost_scenario)
        tc = (sc.commission_bps + sc.spread_bps) / 1e4
        impact = sc.impact_coef * sc.impact_stress * np.sqrt(
            np.abs(turnover)) / 1e4
        cost = np.abs(turnover) * (tc + impact)
        net = gross - cost

        sharpe = float(np.mean(net) / (np.std(net) + 1e-9) * np.sqrt(252))
        cum = np.cumprod(1 + net)
        mdd = float(np.max(1 - cum / np.maximum.accumulate(cum)))
        total = float(cum[-1] - 1)

        return PortfolioPnLResult(
            sharpe=sharpe, max_drawdown=mdd, total_return=total,
            net_returns=net, n_factors=len(codes),
            weighting=self.config.weighting,
        )

    def _ic_weights(self, factor_returns: List[np.ndarray],
                    oos_start: int, oos_end: int) -> np.ndarray:
        """
        IC 加权：用 OOS 窗口前的滚动 IC 绝对值作为权重。
        （OOS 窗口内不可用未来信息）
        """
        ic_window = self.config.ic_window
        # 用 OOS 窗口前的 IC（训练段）
        train_end = oos_start
        train_start = max(0, train_end - ic_window)
        if train_end <= train_start + 1:
            return np.ones(len(factor_returns)) / len(factor_returns)

        ics = []
        for fr in factor_returns:
            seg = fr[train_start:train_end]
            if len(seg) < 2 or np.std(seg) < 1e-9:
                ics.append(0.0)
            else:
                # 简化：用收益自相关作为 IC 代理
                ics.append(abs(float(np.mean(seg))))
        ics = np.array(ics)
        if ics.sum() < 1e-9:
            return np.ones(len(ics)) / len(ics)
        return ics / ics.sum()

    def _risk_parity_weights(self, factor_returns: List[np.ndarray],
                             oos_start: int, oos_end: int) -> np.ndarray:
        """风险平价：按波动倒数加权。"""
        vols = np.array([np.std(fr) + 1e-9 for fr in factor_returns])
        inv_vols = 1.0 / vols
        return inv_vols / inv_vols.sum()