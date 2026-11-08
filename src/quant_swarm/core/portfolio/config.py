"""
组合配置（v3.1 修正③：加权规则显式声明）。

关键设计：
1. 加权规则显式声明（equal / ic_weighted / risk_parity）。
2. 成本场景显式声明（base / stress_50 / stress_100）。
3. frozen=True，开/关必须用同一配置。
4. IC 加权用 OOS 窗口前的 IC（无未来信息）。
"""
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict


class WeightingScheme(str, Enum):
    """加权规则。"""
    EQUAL = "equal"
    IC_WEIGHTED = "ic_weighted"
    RISK_PARITY = "risk_parity"


@dataclass(frozen=True)
class PortfolioConfig:
    """
    组合配置（冻结）。

    - weighting：加权规则
    - cost_scenario：成本场景名
    - ic_window：IC 加权的滚动窗口（天数）
    - long_threshold / short_threshold：多空阈值
    """
    weighting: str = WeightingScheme.EQUAL.value
    cost_scenario: str = "base"
    ic_window: int = 60
    long_threshold: float = 1.0
    short_threshold: float = -1.0

    def __post_init__(self):
        if self.weighting not in [w.value for w in WeightingScheme]:
            raise ValueError(f"未知加权规则: {self.weighting}")
        if self.ic_window < 1:
            raise ValueError(f"ic_window 必须 >= 1")
        if self.long_threshold <= self.short_threshold:
            raise ValueError(
                f"long_threshold ({self.long_threshold}) 必须 > "
                f"short_threshold ({self.short_threshold})"
            )

    def to_dict(self) -> Dict:
        return asdict(self)