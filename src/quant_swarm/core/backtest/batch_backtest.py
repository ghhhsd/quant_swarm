"""
批量回测（v3.1 修正③：多因子共享面板）。

关键设计：
1. 一次回测多个因子，共享面板数据。
2. 向量化：因子值 → 持仓 → 换手 → 成本 → 净收益，全程 numpy。
3. 与成本模型对接：CostModel.compute_costs。
4. OOS 批次：支持按索引切片。

为什么批量：
- 组合层 PnL 需要 N 个因子同时计算。
- 消融框架需要跨 B 个 OOS 批次重复计算。
- 逐因子独立回测的 O(N × T × A) 会被放大到 O(N × B × T × A)。
- 批量回测把因子层循环降为向量化。
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from core.backtest.panel import PanelData
from core.backtest.factor_eval import FactorEvaluator
from core.backtest.positions import PositionBuilder
from core.backtest.result import FactorBacktestResult, BatchBacktestResult
from core.cost import (
    CostModel, get_scenario, CostScenarioName,
)


class BatchBacktester:
    """
    批量回测器。

    一次回测多个因子，共享面板数据。
    """

    def __init__(self,
                 panel: PanelData,
                 cost_scenario: str = CostScenarioName.BASE.value,
                 long_threshold: float = 1.0,
                 short_threshold: float = -1.0):
        self.panel = panel
        self.evaluator = FactorEvaluator(panel)
        self.position_builder = PositionBuilder(long_threshold, short_threshold)
        scenario = get_scenario(cost_scenario)
        self.cost_model = CostModel(scenario)

    def backtest(self, codes: List[str],
                 fids: Optional[List[str]] = None,
                 oos_start: int = 0,
                 oos_end: Optional[int] = None) -> BatchBacktestResult:
        """
        批量回测。

        codes：因子表达式列表
        fids：因子 ID 列表（默认用索引）
        oos_start / oos_end：OOS 批次切片
        """
        if fids is None:
            fids = [f"f_{i:03d}" for i in range(len(codes))]
        if oos_end is None:
            oos_end = len(self.panel.dates)

        # 1) 批量求值：(N, T, A)
        factor_tensor = self.evaluator.evaluate_batch(codes)

        # 2) 批量持仓：(N, T, A)
        positions_tensor = self.position_builder.build_batch(factor_tensor)

        # 3) 收益面板：(T, A)
        returns = self.panel.get("returns")

        # 4) 对每个因子计算 PnL
        results = []
        for i, (fid, code) in enumerate(zip(fids, codes)):
            fv = factor_tensor[i]
            pos = positions_tensor[i]

            # OOS 切片
            fv_oos = fv[oos_start:oos_end]
            pos_oos = pos[oos_start:oos_end]
            ret_oos = returns[oos_start:oos_end]

            # 毛收益（持仓 t 期 × 收益 t+1 期）
            if len(pos_oos) < 2:
                gross = np.array([0.0])
                turnover = np.array([0.0])
            else:
                gross = (pos_oos[:-1] * ret_oos[1:]).sum(axis=1)
                turnover = np.abs(np.diff(pos_oos, axis=0)).sum(axis=1)

            # 成本
            costs = self.cost_model.compute_costs(pos_oos[1:], turnover)
            net = gross - costs[:len(gross)]

            # Sharpe
            gross_sharpe = _sharpe(gross)
            net_sharpe = _sharpe(net)
            total_cost = float(np.sum(costs))

            results.append(FactorBacktestResult(
                fid=fid,
                code=code,
                factor_values=fv_oos,
                positions=pos_oos,
                gross_returns=gross,
                net_returns=net,
                turnover=turnover,
                gross_sharpe=gross_sharpe,
                net_sharpe=net_sharpe,
                total_cost=total_cost,
                n_obs=len(gross),
            ))

        return BatchBacktestResult(results=results, n_factors=len(codes))


def _sharpe(returns: np.ndarray) -> float:
    """计算 Sharpe。"""
    if len(returns) < 2 or np.std(returns) < 1e-9:
        return 0.0
    return float(np.mean(returns) / np.std(returns) * np.sqrt(252))