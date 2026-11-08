"""
三档成本情景（v3.1 修正⑥：情景敏感性显式化 + 冻结）。

关键设计：
1. 三档：base / stress_50 / stress_100。
2. 每档含：佣金、价差、冲击系数、冲击上浮。
3. 冻结：构造后不可修改（frozen=True）。
4. 运行期只读，禁止自适应。
5. 不提供"下调参数"的接口——成本高是事实，不是 bug。

为什么三档：
- 单一 base 会掩盖极端市场下的成本爆炸。
- 三档显式化残余乐观：若 stress_100 下 Sharpe 转负，
  说明 base 的乐观不可接受。
- 入库依据是 stress_100，不是 base——"最坏情况下还能活"比"正常情况好"更重要。
"""
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict


class CostScenarioName(str, Enum):
    """成本情景名称。"""
    BASE = "base"
    STRESS_50 = "stress_50"
    STRESS_100 = "stress_100"


@dataclass(frozen=True)
class CostScenario:
    """
    成本情景参数（冻结）。

    - name：情景名称
    - commission_bps：佣金（基点）
    - spread_bps：价差（基点）
    - impact_coef：冲击系数（平方根模型的 k）
    - impact_stress：冲击上浮倍数（1.0 = base，1.5 = +50%，2.0 = +100%）
    - description：情景描述
    """
    name: str
    commission_bps: float
    spread_bps: float
    impact_coef: float
    impact_stress: float
    description: str = ""

    def __post_init__(self):
        """构造时校验。"""
        if self.commission_bps < 0:
            raise ValueError(f"佣金不能为负: {self.commission_bps}")
        if self.spread_bps < 0:
            raise ValueError(f"价差不能为负: {self.spread_bps}")
        if self.impact_coef < 0:
            raise ValueError(f"冲击系数不能为负: {self.impact_coef}")
        if self.impact_stress < 1.0:
            raise ValueError(
                f"冲击上浮倍数必须 >= 1.0（不能下调），"
                f"实际 {self.impact_stress}"
            )

    def to_dict(self) -> Dict:
        return asdict(self)

    def total_fixed_bps(self) -> float:
        """固定成本（佣金 + 价差），单位基点。"""
        return self.commission_bps + self.spread_bps


# =============================================================================
# 三档情景（冻结）
# =============================================================================

DEFAULT_SCENARIOS: Dict[str, CostScenario] = {
    CostScenarioName.BASE.value: CostScenario(
        name=CostScenarioName.BASE.value,
        commission_bps=5.0,
        spread_bps=2.0,
        impact_coef=0.1,
        impact_stress=1.0,
        description="正常市场条件",
    ),
    CostScenarioName.STRESS_50.value: CostScenario(
        name=CostScenarioName.STRESS_50.value,
        commission_bps=5.0,
        spread_bps=2.0,
        impact_coef=0.1,
        impact_stress=1.5,
        description="冲击上浮 50%（中等拥挤/流动性下降）",
    ),
    CostScenarioName.STRESS_100.value: CostScenario(
        name=CostScenarioName.STRESS_100.value,
        commission_bps=5.0,
        spread_bps=2.0,
        impact_coef=0.1,
        impact_stress=2.0,
        description="冲击上浮 100%（极端拥挤/流动性枯竭）",
    ),
}


def get_scenario(name: str) -> CostScenario:
    """按名称获取情景。"""
    if name not in DEFAULT_SCENARIOS:
        raise ValueError(f"未知成本情景: {name}")
    return DEFAULT_SCENARIOS[name]


def all_scenarios() -> Dict[str, CostScenario]:
    """返回所有情景（深拷贝）。"""
    return {k: v for k, v in DEFAULT_SCENARIOS.items()}