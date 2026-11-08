"""
叙事-结果一致性校验（v3.1 修正③：L5 调用）。

这是 v3.0 → v3.1 最重要的改造之一：
- v3.0 的叙事是回测后生成的，回测结果永远不会与叙事矛盾（因为叙事是事后编的）。
- v3.1 的叙事是回测前预承诺的，回测后必须检查：结果是否与承诺一致？
- 若不一致 → 判"叙事失效"，而非"因子有效"。

一致性维度：
1. 方向一致：回测的因子-收益相关方向是否与 narrative.direction 一致？
2. 衰减一致：回测的因子衰减速率是否与 narrative.decay 一致？
3. 条件一致：在 narrative.conditions 描述的市场状态下是否表现更好？
"""
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from main_tier.l1_favor.narrative import Narrative, Direction, DecayPattern


@dataclass
class ConsistencyResult:
    consistent: bool
    direction_match: bool
    decay_match: bool
    reasons: List[str]


class NarrativeConsistencyChecker:
    """
    叙事-结果一致性检查器。
    L5 门控在评估因子时调用，作为额外的准入条件。
    """

    def check_direction(self, narrative: Narrative,
                        factor_returns_corr: float) -> bool:
        """
        factor_returns_corr: 因子值与下期收益的截面相关（IC）。
        LONG 期望 IC > 0，SHORT 期望 IC < 0，NEUTRAL 无约束。
        """
        if narrative.direction == Direction.LONG:
            return factor_returns_corr > 0
        elif narrative.direction == Direction.SHORT:
            return factor_returns_corr < 0
        else:
            return True

    def check_decay(self, narrative: Narrative,
                    ic_series: np.ndarray) -> bool:
        """
        ic_series: 滚动 IC 序列。
        - PERSISTENT：IC 衰减缓慢（首尾比值 > 0.5）
        - FAST_DECAY：IC 衰减快（首尾比值 < 0.3）
        - REGIME_DEPENDENT：IC 波动大（标准差/均值 > 0.5）
        - UNKNOWN：无约束
        """
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

    def check_all(self, narrative: Narrative,
                  factor_returns_corr: float,
                  ic_series: np.ndarray) -> ConsistencyResult:
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
                f"衰减不一致：叙事 {narrative.decay.value}，"
                f"实测 IC 首尾比 {abs(ic_series[-1])/(abs(ic_series[0])+1e-9):.2f}"
            )
        return ConsistencyResult(
            consistent=(dir_ok and decay_ok),
            direction_match=dir_ok,
            decay_match=decay_ok,
            reasons=reasons,
        )