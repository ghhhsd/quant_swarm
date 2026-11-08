# 实际功效计算（事前 vs 事后对照）
"""
实际功效计算（v3.1 修正②：事前 vs 事后对照）。

关键设计：
1. 事前声明 EfficacyBudget（预期效应量、目标功效）。
2. 事后用 PowerAnalyzer 计算实际功效。
3. 对照：
   - 实际功效 >= 目标功效 → 强证据
   - 实际功效 < 目标功效 → 弱证据（不作为晋升充分条件）
4. 弱证据不是"失败"，而是"证据不足"——候选层可继续积累数据。
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy import stats as scipy_stats

from ablation.efficacy_budget import (
    EfficacyBudget, EffectSize,
)


@dataclass
class PowerResult:
    """实际功效计算结果。"""
    layer: str
    observed_effect: float       # 观测到的效应量
    observed_power: float        # 实际统计功效
    target_power: float          # 目标功效（来自 budget）
    n_obs_used: int              # 实际使用的观测数
    available_batches: int       # 实际可用批次数
    required_batches: int        # 按 budget 估算的所需批次数
    evidence_strength: str       # "strong" / "weak"
    note: str

    def to_dict(self):
        return {
            "layer": self.layer,
            "observed_effect": self.observed_effect,
            "observed_power": self.observed_power,
            "target_power": self.target_power,
            "n_obs_used": self.n_obs_used,
            "available_batches": self.available_batches,
            "required_batches": self.required_batches,
            "evidence_strength": self.evidence_strength,
            "note": self.note,
        }


class PowerAnalyzer:
    """
    实际功效分析器。

    用两独立样本 t 检验的功效公式：
    power = P(reject H0 | H1 true)
          = 1 - Phi(z_alpha/2 - observed_effect * sqrt(n) / 2)
                - Phi(-z_alpha/2 - observed_effect * sqrt(n) / 2)
    其中 n 是总样本量（on + off 各 n/2）。
    """

    def analyze(self,
                budget: EfficacyBudget,
                on_sharpes: List[float],
                off_sharpes: List[float]) -> PowerResult:
        """
        计算实际功效。

        on_sharpes：启用该层时的组合 Sharpe 序列（跨批次）
        off_sharpes：关闭该层时的组合 Sharpe 序列
        """
        n_on = len(on_sharpes)
        n_off = len(off_sharpes)
        n_total = n_on + n_off
        available_batches = min(n_on, n_off)
        required_batches = budget.required_batches()

        # 观测效应量：两组均值差
        if n_on == 0 or n_off == 0:
            return PowerResult(
                layer=budget.layer,
                observed_effect=0.0,
                observed_power=0.0,
                target_power=budget.power,
                n_obs_used=n_total,
                available_batches=available_batches,
                required_batches=required_batches,
                evidence_strength="weak",
                note="样本不足（on 或 off 为空）",
            )

        mean_on = float(np.mean(on_sharpes))
        mean_off = float(np.mean(off_sharpes))
        observed_effect = mean_on - mean_off

        # 实际功效：用观测效应量 + 实际样本量
        observed_power = self._compute_power(
            effect=abs(observed_effect),
            n_per_group=n_total / 2,
            alpha=budget.alpha,
        )

        # 证据强度判定
        if (observed_power >= budget.power
                and abs(observed_effect) >= budget.mde
                and available_batches >= required_batches):
            strength = "strong"
            note = (f"观测效应 {observed_effect:.3f} >= mde {budget.mde}, "
                    f"实际功效 {observed_power:.2f} >= 目标 {budget.power}")
        else:
            strength = "weak"
            reasons = []
            if observed_power < budget.power:
                reasons.append(
                    f"实际功效 {observed_power:.2f} < 目标 {budget.power}"
                )
            if abs(observed_effect) < budget.mde:
                reasons.append(
                    f"观测效应 {observed_effect:.3f} < mde {budget.mde}"
                )
            if available_batches < required_batches:
                reasons.append(
                    f"可用批次 {available_batches} < 所需 {required_batches}"
                )
            note = "; ".join(reasons)

        return PowerResult(
            layer=budget.layer,
            observed_effect=observed_effect,
            observed_power=observed_power,
            target_power=budget.power,
            n_obs_used=n_total,
            available_batches=available_batches,
            required_batches=required_batches,
            evidence_strength=strength,
            note=note,
        )

    @staticmethod
    def _compute_power(effect: float, n_per_group: float,
                       alpha: float = 0.05) -> float:
        """两独立样本 t 检验的功效。"""
        if n_per_group < 2:
            return 0.0
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
        # 非中心参数
        ncp = effect * np.sqrt(n_per_group / 2)
        power = (1 - scipy_stats.norm.cdf(z_alpha - ncp)
                 + scipy_stats.norm.cdf(-z_alpha - ncp))
        return float(power)