"""
组合边际增益（v2.0 冻结）。

关键设计：
1. 基准组合明确化：等权现有精英池组合。
2. 边际增益 = 加入新因子后的组合 Sharpe - 基准组合 Sharpe。
3. 阈值来自 L0 的 CommittedThreshold（冻结）。
"""
from dataclasses import dataclass
from typing import List, Callable

from main_tier.l5_vag.result import MarginalResult


class MarginalGainChecker:
    """
    组合边际增益检查器。
    """

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold

    def check(self,
              existing_codes: List[str],
              new_code: str,
              portfolio_sharpe_fn: Callable[[List[str]], float]
              ) -> MarginalResult:
        """计算加入新因子后的组合 Sharpe 增量。"""
        base = portfolio_sharpe_fn(existing_codes) if existing_codes else 0.0
        new = portfolio_sharpe_fn(existing_codes + [new_code])
        gain = new - base
        passed = gain >= self.threshold
        return MarginalResult(
            base_sharpe=float(base),
            new_sharpe=float(new),
            marginal_gain=float(gain),
            threshold=self.threshold,
            passed=passed,
            reason=("ok" if passed
                    else f"边际增益 {gain:.4f} < {self.threshold:.4f}"),
        )