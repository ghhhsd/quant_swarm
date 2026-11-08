"""
组合构建（v3.1 修正③：单批次组合 PnL）。

关键设计：
1. 输入：N 个因子的持仓 + 收益面板。
2. 输出：组合持仓 + 组合 PnL + 成本 + 换手分解。
3. 加权规则显式声明（来自 PortfolioConfig）。
4. 成本模型来自 core.cost。
5. OOS 切片：在全样本上构建组合，切片计算 PnL。
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from core.portfolio.config import PortfolioConfig, WeightingScheme
from core.portfolio.weighting import compute_weights
from core.portfolio.turnover import compute_turnover, TurnoverBreakdown
from core.portfolio.result import PortfolioResult
from core.cost import CostModel, get_scenario, CostScenarioName
from core.backtest import (
    PanelData, FactorEvaluator, PositionBuilder,
)


class PortfolioBuilder:
    """
    组合构建器。

    在全样本上构建组合，OOS 切片计算 PnL。
    """

    def __init__(self, panel: PanelData, config: PortfolioConfig):
        self.panel = panel
        self.config = config
        self.evaluator = FactorEvaluator(panel)
        self.position_builder = PositionBuilder(
            config.long_threshold, config.short_threshold,
        )
        scenario = get_scenario(config.cost_scenario)
        self.cost_model = CostModel(scenario)

    def build(self, codes: List[str],
              batch_id: str = "full",
              oos_start: int = 0,
              oos_end: Optional[int] = None) -> PortfolioResult:
        """
        构建组合。

        codes：因子表达式列表
        oos_start / oos_end：OOS 批次切片
        """
        if not codes:
            return self._empty_result(batch_id)
        if oos_end is None:
            oos_end = len(self.panel.dates)

        # 1) 因子值 + 持仓
        factor_tensor = self.evaluator.evaluate_batch(codes)  # (N, T, A)
        positions_tensor = self.position_builder.build_batch(factor_tensor)

        # 2) 单因子收益（全样本）
        returns = self.panel.get("returns")
        factor_returns_list = []
        for i in range(len(codes)):
            pos = positions_tensor[i]
            if len(pos) < 2:
                factor_returns_list.append(np.array([0.0]))
            else:
                fr = (pos[:-1] * returns[1:]).sum(axis=1)
                factor_returns_list.append(fr)

        # 3) 加权
        weights = compute_weights(
            self.config.weighting,
            factor_returns_list,
            oos_start,
            self.config.ic_window,
        )

        # 4) 组合持仓（全样本）
        combined_positions = sum(
            w * positions_tensor[i] for i, w in enumerate(weights)
        )
        # 行归一化
        row_sums = np.abs(combined_positions).sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
        combined_positions = combined_positions / row_sums

        # 5) OOS 切片
        pos_oos = combined_positions[oos_start:oos_end]
        ret_oos = returns[oos_start:oos_end]

        if len(pos_oos) < 2:
            return self._empty_result(batch_id)

        # 6) 组合毛收益
        gross = (pos_oos[:-1] * ret_oos[1:]).sum(axis=1)

        # 7) 组合换手 + 单因子换手
        single_positions_oos = [
            positions_tensor[i][oos_start:oos_end]
            for i in range(len(codes))
        ]
        turnover_breakdown = compute_turnover(pos_oos, single_positions_oos)

        # 8) 成本
        costs = self.cost_model.compute_costs(
            pos_oos[1:], turnover_breakdown.combined_turnover,
        )
        net = gross - costs[:len(gross)]

        # 9) 指标
        gross_sharpe = _sharpe(gross)
        net_sharpe = _sharpe(net)
        cum = np.cumprod(1 + net)
        mdd = float(np.max(1 - cum / np.maximum.accumulate(cum)))
        total_ret = float(cum[-1] - 1) if len(cum) > 0 else 0.0

        return PortfolioResult(
            batch_id=batch_id,
            n_factors=len(codes),
            weighting=self.config.weighting,
            positions=pos_oos,
            gross_returns=gross,
            net_returns=net,
            combined_turnover=turnover_breakdown.combined_turnover,
            sum_single_turnover=turnover_breakdown.sum_single_turnover,
            offset_ratio=turnover_breakdown.offset_ratio,
            total_cost=float(np.sum(costs)),
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            max_drawdown=mdd,
            total_return=total_ret,
            n_obs=len(gross),
        )

    def _empty_result(self, batch_id: str) -> PortfolioResult:
        return PortfolioResult(
            batch_id=batch_id, n_factors=0,
            weighting=self.config.weighting,
            positions=np.array([]),
            gross_returns=np.array([0.0]),
            net_returns=np.array([0.0]),
            combined_turnover=np.array([0.0]),
            sum_single_turnover=np.array([0.0]),
            offset_ratio=0.0, total_cost=0.0,
            gross_sharpe=0.0, net_sharpe=0.0,
            max_drawdown=0.0, total_return=0.0, n_obs=0,
        )


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2 or np.std(returns) < 1e-9:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))