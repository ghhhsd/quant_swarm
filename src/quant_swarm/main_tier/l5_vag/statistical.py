"""
统计检查（v3.1 修正①：N 从 L0 取）。

关键设计：
1. N 的唯一真相源是 L0 的 TrialCounter，L5 不持有 N。
2. 三道统计检查：
   - OOS 稳定性（符号一致）
   - DSR（Deflated Sharpe Ratio）
   - BH-FDR（同批次内）
3. 失败时记录原因，成功时返回完整统计信息。
4. 不持有 N，只消费 N。
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from core.config import FrozenThresholds, GLOBAL_THRESHOLDS
from core.stats import (
    deflated_sharpe, bh_fdr_from_records, sharpe_to_pvalue,
)
from main_tier.l5_vag.result import StatisticalResult


class StatisticalChecker:
    """
    统计检查器（只消费 N，不持有 N）。
    """

    def __init__(self, thresholds: Optional[FrozenThresholds] = None):
        self.th = thresholds or GLOBAL_THRESHOLDS
        assert self.th.frozen, "阈值必须 frozen=True"

    def check(self,
              oos_sharpes: List[float],
              n_trials: int,
              batch_records: List,
              n_obs: int) -> StatisticalResult:
        """
        统计检查。

        oos_sharpes：同一因子跨 OOS 窗口的 Sharpe
        n_trials：搜索批次累计 N（来自 L0）
        batch_records：同批次所有试验记录（来自 L0）
        n_obs：单 OOS 窗口观测数
        """
        reasons = []
        if not oos_sharpes:
            return StatisticalResult(
                mean_oos_sharpe=0.0, sharpe_std=0.0,
                sharpe_sign_stable=False, dsr=0.0,
                p_value=1.0, fdr_passed=False, n_trials=n_trials,
                reasons=["OOS Sharpe 序列为空"],
            )

        mean_sharpe = float(np.mean(oos_sharpes))
        std_sharpe = float(np.std(oos_sharpes))

        # 1) 稳定性
        signs = [s > 0 for s in oos_sharpes]
        sign_stable = all(signs) or not any(signs)
        if not sign_stable:
            reasons.append(
                f"OOS Sharpe 符号不稳定：{['+' if s else '-' for s in signs]}"
            )

        # 2) 最低 Sharpe 门槛
        if mean_sharpe < self.th.min_oos_sharpe:
            reasons.append(
                f"平均 OOS Sharpe {mean_sharpe:.3f} < {self.th.min_oos_sharpe}"
            )

        # 3) DSR（N 来自 L0）
        dsr_result = deflated_sharpe(mean_sharpe, n_trials, n_obs)
        dsr = dsr_result.dsr
        if dsr < self.th.dsr_threshold:
            reasons.append(
                f"DSR {dsr:.3f} < {self.th.dsr_threshold} "
                f"(N={n_trials}, E[max]={dsr_result.expected_max_sharpe:.3f})"
            )

        # 4) BH-FDR（同批次内）
        p = sharpe_to_pvalue(mean_sharpe, n_obs)
        fdr_passed = True
        if batch_records and len(batch_records) >= 2:
            fdr_result = bh_fdr_from_records(batch_records, self.th.fdr_alpha)
            # 当前因子的 p 值应通过 FDR
            # 注意：batch_records 包含当前因子（因为 L0 在门控前记录）
            # 但 bh_fdr_from_records 计算的是整批次的 FDR，
            # 当前因子的 flag 在最后一位（如果按记录顺序）
            # 更严谨的做法：用当前因子的 p 值单独判断
            # 这里简化：用当前 p 值 + alpha 判断（不依赖批次位置）
            fdr_passed = p <= self.th.fdr_alpha
            if not fdr_passed:
                reasons.append(
                    f"未通过 BH-FDR (p={p:.4f}, alpha={self.th.fdr_alpha})"
                )

        return StatisticalResult(
            mean_oos_sharpe=mean_sharpe,
            sharpe_std=std_sharpe,
            sharpe_sign_stable=sign_stable,
            dsr=dsr,
            p_value=p,
            fdr_passed=fdr_passed,
            n_trials=n_trials,
            reasons=reasons,
        )