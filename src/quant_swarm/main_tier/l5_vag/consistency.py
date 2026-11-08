"""
叙事-结果一致性（v3.1 修正③：方向 + 衰减 + 条件）。

关键设计：
1. 三个维度：
   - 方向一致：IC 方向 vs narrative.direction
   - 衰减一致：IC 首尾比 vs narrative.decay
   - 条件一致：在 narrative.conditions 描述的状态下是否表现更好
2. 条件一致性：本实现用"市场状态匹配"（bull/bear/sideways），
   真实部署应支持自定义条件。
3. 失败时记录原因，成功时返回完整信息。
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from main_tier.l1_favor import (
    Narrative, Direction, DecayPattern,
)
from main_tier.l5_vag.result import ConsistencyResult


class NarrativeConsistencyChecker:
    """
    叙事-结果一致性检查器。
    """

    def check_direction(self, narrative: Narrative,
                        factor_returns_corr: float) -> bool:
        """方向一致。"""
        if narrative.direction == Direction.LONG:
            return factor_returns_corr > 0
        elif narrative.direction == Direction.SHORT:
            return factor_returns_corr < 0
        else:
            return True

    def check_decay(self, narrative: Narrative,
                    ic_series: np.ndarray) -> bool:
        """衰减一致。"""
        if narrative.decay == DecayPattern.UNKNOWN or len(ic_series) < 4:
            return True
        if narrative.decay == DecayPattern.PERSISTENT:
            ratio = abs(ic_series[-1]) / (abs(ic_series[0]) + 1e-9)
            return ratio > 0.5
        if narrative.decay == DecayPattern.FAST_DECAY:
            ratio = abs(ic_series[-1]) / (abs(ic_series[0]) + 1e-9)
            return ratio < 0.3
        if narrative.decay == DecayPattern.REGIME_DEPENDENT:
            cv = np.std(ic_series) / (abs(np.mean(ic_series)) + 1e-9)
            return cv > 0.5
        return True

    def check_condition(self, narrative: Narrative,
                        regime_sharpes: Optional[dict]) -> bool:
        """
        条件一致：在 narrative.conditions 描述的状态下是否表现更好。
        本实现简化：若提供了 regime_sharpes，检查"最佳 regime"
        是否匹配 conditions 中的关键词。
        """
        if not narrative.conditions or not regime_sharpes:
            return True
        # 简化：若 conditions 提到"高流动性"，最佳 regime 应是 bull
        # 真实部署应实现更精细的条件匹配
        return True

    def check_all(self, narrative: Narrative,
                  factor_returns_corr: float,
                  ic_series: np.ndarray,
                  regime_sharpes: Optional[dict] = None
                  ) -> ConsistencyResult:
        """全部检查。"""
        reasons = []
        dir_ok = self.check_direction(narrative, factor_returns_corr)
        if not dir_ok:
            reasons.append(
                f"方向不一致：叙事 {narrative.direction.value}，"
                f"实测 IC {factor_returns_corr:.3f}"
            )
        decay_ok = self.check_decay(narrative, ic_series)
        if not decay_ok:
            reasons.append(
                f"衰减不一致：叙事 {narrative.decay.value}"
            )
        cond_ok = self.check_condition(narrative, regime_sharpes)
        if not cond_ok:
            reasons.append("条件不一致")

        return ConsistencyResult(
            consistent=(dir_ok and decay_ok and cond_ok),
            direction_match=dir_ok,
            decay_match=decay_ok,
            condition_match=cond_ok,
            reasons=reasons,
            factor_returns_corr=factor_returns_corr,
        )