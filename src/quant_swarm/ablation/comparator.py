# 开/关比较器（组合 PnL 为最终指标）
"""
开/关比较器（v3.1 修正③：组合 PnL 为最终指标）。

关键设计：
1. 比较对象不是"层 vs 层"，而是"接组合层后的最终 PnL"。
2. 分市场状态分别比较。
3. 输出证据强度：strong / weak。
4. 弱证据不作为晋升充分条件。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from ablation.efficacy_budget import EfficacyBudget, EffectSize
from ablation.power_analysis import PowerAnalyzer, PowerResult
from ablation.regime_split import (
    split_batch_by_regime, MarketRegime,
)


@dataclass
class AblationResult:
    """消融结果。"""
    layer: str
    overall_power: PowerResult
    by_regime: Dict[str, PowerResult]   # regime → PowerResult
    evidence_strength: str               # "strong" / "weak"
    note: str

    def to_dict(self):
        return {
            "layer": self.layer,
            "overall_power": self.overall_power.to_dict(),
            "by_regime": {k: v.to_dict() for k, v in self.by_regime.items()},
            "evidence_strength": self.evidence_strength,
            "note": self.note,
        }


class AblationComparator:
    """
    消融比较器。

    流程：
    1. 用 PowerAnalyzer 计算整体功效。
    2. 按 regime 分别计算功效。
    3. 证据强度：整体功效 >= 目标 AND 任一 regime 功效 >= 目标。
    """

    def __init__(self):
        self.analyzer = PowerAnalyzer()

    def compare(self,
                budget: EfficacyBudget,
                on_sharpes: List[float],
                off_sharpes: List[float],
                on_regimes: Optional[List[str]] = None,
                off_regimes: Optional[List[str]] = None) -> AblationResult:
        """
        比较启用 vs 关闭该层的组合 PnL。

        on_sharpes / off_sharpes：跨批次的组合 Sharpe 序列。
        on_regimes / off_regimes：对应批次的市场状态标签（可选）。
        """
        # 1) 整体功效
        overall = self.analyzer.analyze(budget, on_sharpes, off_sharpes)

        # 2) 分 regime 功效
        by_regime: Dict[str, PowerResult] = {}
        if on_regimes and off_regimes:
            on_by_reg, off_by_reg = split_batch_by_regime(
                on_sharpes, off_sharpes, on_regimes, off_regimes,
            )
            for regime in budget.market_regimes:
                on_list = on_by_reg.get(regime, [])
                off_list = off_by_reg.get(regime, [])
                by_regime[regime] = self.analyzer.analyze(
                    budget, on_list, off_list,
                )

        # 3) 证据强度判定
        # 强证据：整体 >= 目标 AND 任一 regime >= 目标
        overall_strong = (overall.evidence_strength == "strong")
        any_regime_strong = any(
            r.evidence_strength == "strong" for r in by_regime.values()
        ) if by_regime else False

        if overall_strong and any_regime_strong:
            strength = "strong"
            note = "整体 + 至少一个 regime 均达目标功效"
        elif overall_strong and not by_regime:
            # 无 regime 数据时，只看整体
            strength = "strong"
            note = "整体达目标功效（无 regime 数据）"
        else:
            strength = "weak"
            reasons = [f"整体: {overall.note}"]
            if by_regime:
                weak_regimes = [
                    r for r, p in by_regime.items()
                    if p.evidence_strength == "weak"
                ]
                if weak_regimes:
                    reasons.append(f"弱 regime: {weak_regimes}")
            note = "; ".join(reasons)

        return AblationResult(
            layer=budget.layer,
            overall_power=overall,
            by_regime=by_regime,
            evidence_strength=strength,
            note=note,
        )