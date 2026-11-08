"""
情景敏感性分析（v3.1 修正⑥：残余乐观显性化）。

关键设计：
1. 三档情景分别计算成本后 Sharpe。
2. 输出：
   - base_sharpe：正常市场
   - stress_50_sharpe：中等拥挤
   - stress_100_sharpe：极端拥挤
   - sensitivity：base 到 stress_100 的衰减比例
   - 入库依据：stress_100 下是否达标
3. 一阶修正的诚实标注：三档情景是"残余偏差的显性化"，
   不是"消除残余偏差"。

为什么分三档：
- 单一 base 会掩盖极端市场下的成本爆炸。
- stress_100 转负 → base 的乐观不可接受。
- 敏感性高 → 因子对成本敏感，容量有限。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from core.cost.scenario import (
    CostScenarioName, get_scenario, all_scenarios,
)
from core.cost.cost_model import CostModel


@dataclass
class ScenarioSharpe:
    """单场景 Sharpe。"""
    scenario_name: str
    gross_sharpe: float
    net_sharpe: float
    total_cost: float
    turnover_mean: float

    def to_dict(self):
        return {
            "scenario_name": self.scenario_name,
            "gross_sharpe": self.gross_sharpe,
            "net_sharpe": self.net_sharpe,
            "total_cost": self.total_cost,
            "turnover_mean": self.turnover_mean,
        }


@dataclass
class SensitivityReport:
    """
    情景敏感性报告。

    - scenario_sharpes：三档情景的 Sharpe
    - sensitivity：base 到 stress_100 的衰减比例
    - base_net_sharpe：base 成本后 Sharpe
    - stress_100_net_sharpe：stress_100 成本后 Sharpe
    - residual_optimism_note：一阶修正的诚实标注
    """
    scenario_sharpes: Dict[str, ScenarioSharpe]
    sensitivity: float
    base_net_sharpe: float
    stress_50_net_sharpe: float
    stress_100_net_sharpe: float
    residual_optimism_note: str = (
        "成本模型是对回测乐观的一阶修正，不消除残余偏差。"
        "若 stress_100 下 Sharpe 转负，说明 base 的乐观不可接受。"
    )

    def to_dict(self):
        return {
            "scenario_sharpes": {
                k: v.to_dict() for k, v in self.scenario_sharpes.items()
            },
            "sensitivity": self.sensitivity,
            "base_net_sharpe": self.base_net_sharpe,
            "stress_50_net_sharpe": self.stress_50_net_sharpe,
            "stress_100_net_sharpe": self.stress_100_net_sharpe,
            "residual_optimism_note": self.residual_optimism_note,
        }


class SensitivityAnalyzer:
    """
    情景敏感性分析器。
    """

    def __init__(self,
                 scenarios: Optional[Dict] = None):
        """
        scenarios：三档情景（默认 DEFAULT_SCENARIOS）。
        """
        self.scenarios = scenarios or all_scenarios()

    def analyze(self,
                gross_returns: np.ndarray,
                positions: np.ndarray,
                turnover: np.ndarray) -> SensitivityReport:
        """
        分析三档情景。

        gross_returns：(T,) 毛收益
        positions：(T, N) 持仓
        turnover：(T,) 或 (T, N) 换手
        """
        # 毛 Sharpe（无成本）
        gross_sharpe = 0.0
        if len(gross_returns) >= 2 and np.std(gross_returns) > 1e-9:
            gross_sharpe = float(
                np.mean(gross_returns) / np.std(gross_returns) * np.sqrt(252)
            )

        # 三档情景
        scenario_sharpes: Dict[str, ScenarioSharpe] = {}
        for name, scenario in self.scenarios.items():
            model = CostModel(scenario)
            net_ret = model.net_returns(gross_returns, positions, turnover)
            net_sharpe = (
                float(np.mean(net_ret) / np.std(net_ret) * np.sqrt(252))
                if len(net_ret) >= 2 and np.std(net_ret) > 1e-9
                else 0.0
            )
            total_cost = float(np.sum(model.compute_costs(positions, turnover)))
            turnover_mean = float(np.mean(np.abs(turnover)))

            scenario_sharpes[name] = ScenarioSharpe(
                scenario_name=name,
                gross_sharpe=gross_sharpe,
                net_sharpe=net_sharpe,
                total_cost=total_cost,
                turnover_mean=turnover_mean,
            )

        # 敏感性：base → stress_100 的衰减比例
        base_sharpe = scenario_sharpes[
            CostScenarioName.BASE.value].net_sharpe
        stress_100_sharpe = scenario_sharpes[
            CostScenarioName.STRESS_100.value].net_sharpe
        stress_50_sharpe = scenario_sharpes[
            CostScenarioName.STRESS_50.value].net_sharpe

        sensitivity = 0.0
        if abs(base_sharpe) > 1e-9:
            sensitivity = (base_sharpe - stress_100_sharpe) / abs(base_sharpe)

        return SensitivityReport(
            scenario_sharpes=scenario_sharpes,
            sensitivity=float(sensitivity),
            base_net_sharpe=base_sharpe,
            stress_50_net_sharpe=stress_50_sharpe,
            stress_100_net_sharpe=stress_100_sharpe,
        )