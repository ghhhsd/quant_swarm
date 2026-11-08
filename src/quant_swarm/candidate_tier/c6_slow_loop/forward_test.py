# 前向验证（周/月滚动）
"""
前向验证（v3.1 修正⑤：慢环前向验证，周/月滚动）。

关键设计：
1. 前向验证是"纸面交易"——因子在候选池中，用真实/模拟数据按周/月滚动出结果。
2. 前向 Sharpe 的置信区间用 bootstrap 或正态近似。
3. 输出结构含 CI，供审批协议使用。
4. 前向验证是"异步"的——它不阻塞快环，快环继续产出新候选。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy import stats as scipy_stats


@dataclass
class ForwardTestResult:
    """前向验证结果。"""
    fid: str
    forward_sharpe: float
    forward_ci_low: float
    forward_ci_high: float
    n_obs: int
    confidence: float
    test_period: str  # "weekly" / "monthly"

    def to_dict(self):
        return {
            "fid": self.fid,
            "forward_sharpe": self.forward_sharpe,
            "forward_ci_low": self.forward_ci_low,
            "forward_ci_high": self.forward_ci_high,
            "n_obs": self.n_obs,
            "confidence": self.confidence,
            "test_period": self.test_period,
        }


class ForwardTester:
    """
    前向验证器。
    """

    def __init__(self,
                 confidence: float = 0.95,
                 min_obs: int = 30):
        self.confidence = confidence
        self.min_obs = min_obs

    def test(self, fid: str, forward_returns: np.ndarray,
             test_period: str = "monthly") -> ForwardTestResult:
        """
        前向验证。
        forward_returns：前向模拟盘的日收益序列。
        """
        n = len(forward_returns)
        if n < self.min_obs:
            return ForwardTestResult(
                fid=fid, forward_sharpe=0.0,
                forward_ci_low=0.0, forward_ci_high=0.0,
                n_obs=n, confidence=self.confidence,
                test_period=test_period,
            )

        sharpe = float(np.mean(forward_returns)
                       / (np.std(forward_returns) + 1e-9) * np.sqrt(252))
        se = np.sqrt(252) / np.sqrt(n)
        z = scipy_stats.norm.ppf(1 - (1 - self.confidence) / 2)
        ci_low = sharpe - z * se
        ci_high = sharpe + z * se

        return ForwardTestResult(
            fid=fid, forward_sharpe=sharpe,
            forward_ci_low=ci_low, forward_ci_high=ci_high,
            n_obs=n, confidence=self.confidence,
            test_period=test_period,
        )

    def batch_test(self, returns_per_fid: Dict[str, np.ndarray],
                   test_period: str = "monthly"
                   ) -> List[ForwardTestResult]:
        """批量前向验证。"""
        return [
            self.test(fid, rets, test_period)
            for fid, rets in returns_per_fid.items()
        ]