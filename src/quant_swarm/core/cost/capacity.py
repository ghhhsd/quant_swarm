"""
容量曲线（v3.1 修正④：容量档位而非布尔）。

关键设计：
1. 容量曲线：Sharpe 随资金规模衰减。
2. 参数化模型：
   sharpe(aum) = base_sharpe / (1 + (aum / half_life)^exp)
3. 档位映射：
   - TIER_SMALL：< 1000 万
   - TIER_MEDIUM：1000 万 – 1 亿
   - TIER_LARGE：> 1 亿
   - TIER_REJECT：容量不足以覆盖成本
4. 档位不是"质量分"，而是"资金规模适配度"。

为什么参数化：
- 真实容量曲线需要 tick 级数据 + 撮合模拟。
- 参数化模型提供"可调"的接口，真实部署时替换参数。
- 参数化模型也能做"敏感性分析"（参数变化时档位变化）。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict
import numpy as np


class CapacityTier(str, Enum):
    """容量档位。"""
    TIER_SMALL = "tier_small"      # < 1000 万
    TIER_MEDIUM = "tier_medium"    # 1000 万 – 1 亿
    TIER_LARGE = "tier_large"      # > 1 亿
    TIER_REJECT = "tier_reject"    # 容量不足以覆盖成本


# 档位边界（资金规模，单位：元）
TIER_BOUNDARIES = {
    "small_upper": 1e7,    # 1000 万
    "medium_upper": 1e8,   # 1 亿
}

# 档位 → 衰减监控频率（天）
TIER_MONITOR_FREQUENCY_DAYS = {
    CapacityTier.TIER_SMALL: 7,
    CapacityTier.TIER_MEDIUM: 30,
    CapacityTier.TIER_LARGE: 90,
    CapacityTier.TIER_REJECT: 0,
}


@dataclass(frozen=True)
class CapacityCurve:
    """
    容量曲线（参数化）。

    sharpe(aum) = base_sharpe / (1 + (aum / capacity_half_life) ** decay_exponent)
    """
    base_sharpe: float
    capacity_half_life: float
    decay_exponent: float = 0.5

    def sharpe_at(self, aum: float) -> float:
        if aum <= 0:
            return self.base_sharpe
        return self.base_sharpe / (
            1 + (aum / self.capacity_half_life) ** self.decay_exponent
        )

    def to_dict(self):
        return {
            "base_sharpe": self.base_sharpe,
            "capacity_half_life": self.capacity_half_life,
            "decay_exponent": self.decay_exponent,
        }


@dataclass
class CapacityReport:
    """容量报告。"""
    tier: CapacityTier
    base_sharpe: float
    sharpe_at_boundaries: Dict[str, float]
    max_capacity_aum: float
    monitor_frequency_days: int
    reason: str

    def to_dict(self):
        return {
            "tier": self.tier.value,
            "base_sharpe": self.base_sharpe,
            "sharpe_at_boundaries": self.sharpe_at_boundaries,
            "max_capacity_aum": self.max_capacity_aum,
            "monitor_frequency_days": self.monitor_frequency_days,
            "reason": self.reason,
        }


class CapacityEstimator:
    """
    容量估计器。

    真实部署应用实盘成交数据标定容量曲线参数。
    本实现用参数化模型 + 简单标定。
    """

    def __init__(self, min_sharpe_threshold: float = 0.3):
        self.min_sharpe_threshold = min_sharpe_threshold

    def estimate(self,
                 base_sharpe: float,
                 capacity_half_life: float = 5e7,
                 decay_exponent: float = 0.5) -> CapacityReport:
        curve = CapacityCurve(
            base_sharpe=base_sharpe,
            capacity_half_life=capacity_half_life,
            decay_exponent=decay_exponent,
        )

        sharpe_small = curve.sharpe_at(TIER_BOUNDARIES["small_upper"])
        sharpe_medium = curve.sharpe_at(TIER_BOUNDARIES["medium_upper"])
        sharpe_large = curve.sharpe_at(TIER_BOUNDARIES["medium_upper"] * 2)

        max_aum = self._solve_max_capacity(curve)

        if sharpe_small < self.min_sharpe_threshold:
            tier = CapacityTier.TIER_REJECT
            reason = f"小资金（<1000万）Sharpe {sharpe_small:.3f} 已低于阈值"
        elif sharpe_medium < self.min_sharpe_threshold:
            tier = CapacityTier.TIER_SMALL
            reason = f"中资金（>1000万）Sharpe {sharpe_medium:.3f} 低于阈值"
        elif sharpe_large < self.min_sharpe_threshold:
            tier = CapacityTier.TIER_MEDIUM
            reason = f"大资金（>2亿）Sharpe {sharpe_large:.3f} 低于阈值"
        else:
            tier = CapacityTier.TIER_LARGE
            reason = "大资金仍可承受"

        return CapacityReport(
            tier=tier,
            base_sharpe=base_sharpe,
            sharpe_at_boundaries={
                "small": sharpe_small,
                "medium": sharpe_medium,
                "large": sharpe_large,
            },
            max_capacity_aum=max_aum,
            monitor_frequency_days=TIER_MONITOR_FREQUENCY_DAYS[tier],
            reason=reason,
        )

    def _solve_max_capacity(self, curve: CapacityCurve) -> float:
        if curve.base_sharpe <= self.min_sharpe_threshold:
            return 0.0
        ratio = curve.base_sharpe / self.min_sharpe_threshold - 1
        if ratio <= 0:
            return 0.0
        return curve.capacity_half_life * (ratio ** (1 / curve.decay_exponent))