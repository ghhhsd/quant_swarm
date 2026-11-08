"""
三档成本场景（v3.1 修正⑥：情景敏感性显性化）。

关键设计：
1. 三档：base / stress_50 / stress_100。
2. 每档含佣金 + 价差 + 冲击系数 + 冲击上浮。
3. 入库依据是 stress_100 下的成本后 Sharpe，不是 base——
   "最坏情况下还能活"比"正常情况好"更重要。
4. 一阶修正的诚实标注：这是对回测乐观的一阶修正，不消除残余偏差。
   残余偏差通过三档情景显性化（若 stress_100 下 Sharpe 转负，
   说明 base 的乐观不可接受）。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict


class CostScenarioName(str, Enum):
    BASE = "base"
    STRESS_50 = "stress_50"
    STRESS_100 = "stress_100"


@dataclass(frozen=True)
class CostScenario:
    """
    成本场景参数。

    - commission_bps：佣金（基点）
    - spread_bps：价差（基点）
    - impact_coef：冲击系数（相对成交量的平方根模型）
    - impact_stress：冲击上浮倍数（1.0 = base，1.5 = +50%，2.0 = +100%）
    """
    name: str
    commission_bps: float
    spread_bps: float
    impact_coef: float
    impact_stress: float

    def to_dict(self):
        return {
            "name": self.name,
            "commission_bps": self.commission_bps,
            "spread_bps": self.spread_bps,
            "impact_coef": self.impact_coef,
            "impact_stress": self.impact_stress,
        }


# 三档场景（冻结）
DEFAULT_SCENARIOS: Dict[str, CostScenario] = {
    CostScenarioName.BASE.value: CostScenario(
        name=CostScenarioName.BASE.value,
        commission_bps=5.0,
        spread_bps=2.0,
        impact_coef=0.1,
        impact_stress=1.0,
    ),
    CostScenarioName.STRESS_50.value: CostScenario(
        name=CostScenarioName.STRESS_50.value,
        commission_bps=5.0,
        spread_bps=2.0,
        impact_coef=0.1,
        impact_stress=1.5,
    ),
    CostScenarioName.STRESS_100.value: CostScenario(
        name=CostScenarioName.STRESS_100.value,
        commission_bps=5.0,
        spread_bps=2.0,
        impact_coef=0.1,
        impact_stress=2.0,
    ),
}


def get_scenario(name: str) -> CostScenario:
    """按名称获取场景。"""
    if name not in DEFAULT_SCENARIOS:
        raise ValueError(f"未知成本场景: {name}")
    return DEFAULT_SCENARIOS[name]


def all_scenarios() -> Dict[str, CostScenario]:
    return dict(DEFAULT_SCENARIOS)