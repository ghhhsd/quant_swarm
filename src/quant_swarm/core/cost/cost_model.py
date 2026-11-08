"""
成本计算（佣金 + 价差 + 冲击）。

关键设计：
1. 固定成本：佣金 + 价差（线性于换手）。
2. 冲击成本：平方根模型（k * sqrt(turnover) * impact_stress）。
3. 三档情景共享同一计算逻辑，只是参数不同。
4. 输入：持仓矩阵 + 换手率矩阵。
5. 输出：每日成本（收益单位）。

平方根冲击模型的依据：
- Almgren-Chriss (2000)：冲击 ∝ sqrt(交易量)
- Frazzini-Israel-Moskowitz (2018)：实证验证
- 线性模型会低估大资金冲击
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from core.cost.scenario import (
    CostScenario, get_scenario, CostScenarioName, DEFAULT_SCENARIOS,
)


@dataclass
class CostBreakdown:
    """成本分解（单期）。"""
    commission: float       # 佣金
    spread: float           # 价差
    impact: float           # 冲击
    total: float            # 合计

    def to_dict(self):
        return {
            "commission": self.commission,
            "spread": self.spread,
            "impact": self.impact,
            "total": self.total,
        }


class CostModel:
    """
    成本模型。

    三档情景共享同一计算逻辑。
    """

    def __init__(self, scenario: CostScenario):
        """
        scenario：成本情景（冻结）。
        """
        self.scenario = scenario

    # ---- 单期成本 -------------------------------------------------------
    def compute_period_cost(self, turnover: np.ndarray) -> CostBreakdown:
        """
        单期成本。

        turnover：换手率矩阵（形状 (N,) 或 (T, N)）
        """
        # 固定成本：佣金 + 价差
        fixed_bps = self.scenario.total_fixed_bps() / 1e4
        commission = self.scenario.commission_bps / 1e4 * np.abs(turnover)
        spread = self.scenario.spread_bps / 1e4 * np.abs(turnover)

        # 冲击成本：平方根模型
        # impact = k * stress * sqrt(|turnover|) / 1e4
        impact_rate = (
            self.scenario.impact_coef
            * self.scenario.impact_stress
            * np.sqrt(np.abs(turnover)) / 1e4
        )
        impact = impact_rate * np.abs(turnover)

        # 汇总
        if isinstance(turnover, np.ndarray) and turnover.ndim > 1:
            # 多期：每期汇总
            return CostBreakdown(
                commission=float(np.sum(commission)),
                spread=float(np.sum(spread)),
                impact=float(np.sum(impact)),
                total=float(np.sum(commission + spread + impact)),
            )
        else:
            return CostBreakdown(
                commission=float(np.sum(commission)),
                spread=float(np.sum(spread)),
                impact=float(np.sum(impact)),
                total=float(np.sum(commission + spread + impact)),
            )

    # ---- 多期成本 -------------------------------------------------------
    def compute_costs(self, positions: np.ndarray,
                      turnover: np.ndarray) -> np.ndarray:
        """
        多期成本（每日一个成本值）。

        positions：(T, N) 持仓权重
        turnover：(T,) 每期组合换手（或 (T, N) 每资产换手）

        返回：(T,) 每期成本（收益单位）
        """
        if turnover.ndim == 1:
            turnover = turnover[:, None]

        # 佣金 + 价差（线性）
        fixed_bps = self.scenario.total_fixed_bps() / 1e4
        fixed_cost = fixed_bps * np.abs(turnover).sum(axis=1)

        # 冲击（平方根）
        impact_rate = (
            self.scenario.impact_coef
            * self.scenario.impact_stress
            * np.sqrt(np.abs(turnover)) / 1e4
        )
        impact_cost = (impact_rate * np.abs(turnover)).sum(axis=1)

        return fixed_cost + impact_cost

    # ---- 成本后收益 -----------------------------------------------------
    def net_returns(self, gross_returns: np.ndarray,
                    positions: np.ndarray,
                    turnover: np.ndarray) -> np.ndarray:
        """
        成本后收益。

        gross_returns：(T,) 毛收益
        positions：(T, N) 持仓
        turnover：(T,) 或 (T, N) 换手

        返回：(T,) 净收益
        """
        costs = self.compute_costs(positions, turnover)
        # 对齐长度（毛收益可能与成本长度不同）
        min_len = min(len(gross_returns), len(costs))
        return gross_returns[:min_len] - costs[:min_len]

    # ---- 成本后 Sharpe --------------------------------------------------
    def net_sharpe(self, gross_returns: np.ndarray,
                   positions: np.ndarray,
                   turnover: np.ndarray) -> float:
        """成本后 Sharpe。"""
        net = self.net_returns(gross_returns, positions, turnover)
        if len(net) < 2 or np.std(net) < 1e-9:
            return 0.0
        return float(np.mean(net) / np.std(net) * np.sqrt(252))


# =============================================================================
# 便捷函数
# =============================================================================

def apply_costs(positions: np.ndarray,
                turnover: np.ndarray,
                scenario_name: str = CostScenarioName.BASE.value) -> np.ndarray:
    """
    便捷函数：应用成本。

    兼容旧接口（之前 core/cost_model.py 的 apply_costs）。
    """
    scenario = get_scenario(scenario_name)
    model = CostModel(scenario)
    return model.compute_costs(positions, turnover)