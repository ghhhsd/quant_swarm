"""
一阶修正的诚实标注（v3.1 修正⑥）。

关键设计：
1. 成本模型是"一阶修正"，不是"消除"。
2. 显式声明残余偏差：
   - 冲击函数用历史数据标定，拥挤时成交结构已变（循环依赖）
   - 容量曲线用参数化模型，真实容量需 tick 级数据
   - 情景敏感性只覆盖三档，不是"所有可能情景"
3. 不宣称"消除"——绝对承诺不可证。
4. 提供"残余偏差的情景分析"（如冲击参数上浮 200% 时 Sharpe 如何）。

为什么需要诚实标注：
- 第三方批评："用历史成交估冲击，但拥挤时成交结构已变"。
- 循环依赖：用历史数据标定冲击 → 策略拥挤 → 成交结构变化 → 历史标定失效。
- 必须显式声明，不能假装"有了成本模型就等于实盘"。
"""
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from core.cost.scenario import CostScenario, get_scenario, CostScenarioName
from core.cost.cost_model import CostModel


@dataclass
class ResidualOptimismReport:
    """
    残余乐观报告。

    - note：诚实标注（文字）
    - circular_dependency：循环依赖说明
    - scenario_analysis：残余偏差的情景分析
    """
    note: str
    circular_dependency: str
    scenario_analysis: Dict[str, float]

    def to_dict(self):
        return {
            "note": self.note,
            "circular_dependency": self.circular_dependency,
            "scenario_analysis": self.scenario_analysis,
        }


class HonestyAnalyzer:
    """
    诚实标注分析器。

    提供：
1. 标准诚实标注（文字）。
2. 残余偏差的情景分析（冲击参数上浮 200% / 300% 时 Sharpe 如何）。
    """

    def __init__(self, base_scenario: CostScenario = None):
        self.base_scenario = base_scenario or get_scenario(
            CostScenarioName.BASE.value)

    def get_note(self) -> str:
        """返回标准诚实标注。"""
        return (
            "成本模型是对回测乐观的一阶修正，不消除残余偏差。"
            "残余偏差来源："
            "(1) 冲击函数用历史数据标定，拥挤时成交结构已变（循环依赖）；"
            "(2) 容量曲线用参数化模型，真实容量需 tick 级数据；"
            "(3) 情景敏感性只覆盖三档，不是所有可能情景。"
            "若 stress_100 下 Sharpe 转负，说明 base 的乐观不可接受。"
        )

    def get_circular_dependency(self) -> str:
        """返回循环依赖说明。"""
        return (
            "循环依赖：用历史数据标定冲击函数 → 策略拥挤 → "
            "成交结构变化 → 历史标定失效。这是成本模型的根本局限，"
            "无法通过增加参数解决，只能通过实时监控 + 动态调整缓解。"
        )

    def scenario_analysis(self,
                          gross_returns: np.ndarray,
                          positions: np.ndarray,
                          turnover: np.ndarray) -> ResidualOptimismReport:
        """
        残余偏差的情景分析。

        在 base 情景基础上，冲击参数上浮 200% / 300%，
        观察 Sharpe 如何变化。
        """
        # 基准情景
        base_model = CostModel(self.base_scenario)
        base_net = base_model.net_returns(gross_returns, positions, turnover)
        base_sharpe = (
            float(np.mean(base_net) / np.std(base_net) * np.sqrt(252))
            if len(base_net) >= 2 and np.std(base_net) > 1e-9
            else 0.0
        )

        # 极端情景：冲击上浮 200% / 300%
        scenarios = {
            "base": self.base_scenario,
            "stress_200": CostScenario(
                name="stress_200",
                commission_bps=self.base_scenario.commission_bps,
                spread_bps=self.base_scenario.spread_bps,
                impact_coef=self.base_scenario.impact_coef,
                impact_stress=3.0,  # +200%
                description="冲击上浮 200%（极端极端）",
            ),
            "stress_300": CostScenario(
                name="stress_300",
                commission_bps=self.base_scenario.commission_bps,
                spread_bps=self.base_scenario.spread_bps,
                impact_coef=self.base_scenario.impact_coef,
                impact_stress=4.0,  # +300%
                description="冲击上浮 300%（极端极端极端）",
            ),
        }

        analysis = {}
        for name, scenario in scenarios.items():
            model = CostModel(scenario)
            net = model.net_returns(gross_returns, positions, turnover)
            sharpe = (
                float(np.mean(net) / np.std(net) * np.sqrt(252))
                if len(net) >= 2 and np.std(net) > 1e-9
                else 0.0
            )
            analysis[name] = sharpe

        return ResidualOptimismReport(
            note=self.get_note(),
            circular_dependency=self.get_circular_dependency(),
            scenario_analysis=analysis,
        )