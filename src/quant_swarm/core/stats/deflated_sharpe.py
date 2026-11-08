"""
Deflated Sharpe Ratio（v3.1 修正①：考虑偏度/峰度）。

关键设计：
1. DSR 是"校正多重检验后，Sharpe 是真实的概率"。
2. 期望最大 Sharpe：
   E[max] = (1 - γ) * Φ^(-1)(1 - 1/N) + γ * Φ^(-1)(1 - 1/(N*e))
   其中 γ 是 Euler-Mascheroni 常数，N 是搜索批次累计试验数。
3. 标准误：
   SE = sqrt((1 - skew*sharpe + (kurt-1)/4 * sharpe^2) / (n_obs - 1))
4. DSR = Φ((sharpe - E[max]) / SE)
5. 偏度/峰度：
   - 若传入 returns，从收益序列计算（推荐）。
   - 若不传，用正态假设（skew=0, kurt=3），并在结果中标注。

为什么考虑偏度/峰度：
- 肥尾收益的峰度 > 3，导致 SE 被低估，DSR 被高估。
- 负偏度收益的 skew < 0，导致 SE 被高估，DSR 被低估。
- 不校正会导致"肥尾因子被误判为真实"。
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats as scipy_stats

from core.stats.result import DSRResult


# Euler-Mascheroni 常数
EULER_GAMMA = 0.5772156649015329


def deflated_sharpe(sharpe: float,
                    n_trials: int,
                    n_obs: int,
                    returns: Optional[np.ndarray] = None,
                    skew: Optional[float] = None,
                    kurt: Optional[float] = None) -> DSRResult:
    """
    Deflated Sharpe Ratio。

    sharpe：观测到的 Sharpe
    n_trials：搜索批次累计试验数 N（来自 SearchBatchCounter）
    n_obs：观测数
    returns：收益序列（可选，用于计算偏度/峰度）
    skew / kurt：偏度/峰度（可选，显式传入）
    """
    # 1) 边界检查
    if n_trials < 1:
        return DSRResult(
            sharpe=sharpe, dsr=0.0, n_trials=n_trials, n_obs=n_obs,
            expected_max_sharpe=0.0, skew=0.0, kurt=3.0,
            skew_kurt_source="normal_assumption",
            note=f"n_trials={n_trials} < 1，无法校正",
        )
    if n_obs < 10:
        return DSRResult(
            sharpe=sharpe, dsr=0.0, n_trials=n_trials, n_obs=n_obs,
            expected_max_sharpe=0.0, skew=0.0, kurt=3.0,
            skew_kurt_source="normal_assumption",
            note=f"n_obs={n_obs} < 10，样本不足",
        )

    # 2) 偏度/峰度
    source = "normal_assumption"
    if returns is not None and len(returns) >= 10:
        skew_val = float(scipy_stats.skew(returns))
        kurt_val = float(scipy_stats.kurtosis(returns, fisher=False))
        source = "empirical"
    elif skew is not None and kurt is not None:
        skew_val = float(skew)
        kurt_val = float(kurt)
        source = "explicit"
    else:
        skew_val = 0.0
        kurt_val = 3.0
        source = "normal_assumption"

    # 3) 期望最大 Sharpe
    # E[max] = (1-γ) * Φ^(-1)(1 - 1/N) + γ * Φ^(-1)(1 - 1/(N*e))
    if n_trials == 1:
        e_max = 0.0  # 单次试验，无多重检验校正
    else:
        q1 = scipy_stats.norm.ppf(1 - 1 / n_trials)
        q2 = scipy_stats.norm.ppf(1 - 1 / (n_trials * np.e))
        e_max = (1 - EULER_GAMMA) * q1 + EULER_GAMMA * q2

    # 4) 标准误
    # SE = sqrt((1 - skew*sharpe + (kurt-1)/4 * sharpe^2) / (n_obs - 1))
    var_term = (1 - skew_val * sharpe
                + (kurt_val - 1) / 4 * sharpe ** 2)
    se = np.sqrt(max(var_term, 1e-9) / (n_obs - 1))

    # 5) DSR
    dsr = float(scipy_stats.norm.cdf((sharpe - e_max) / se))

    note = ""
    if source == "normal_assumption":
        note = "未校正偏度/峰度（用正态假设），肥尾收益可能高估 DSR"

    return DSRResult(
        sharpe=sharpe,
        dsr=dsr,
        n_trials=n_trials,
        n_obs=n_obs,
        expected_max_sharpe=float(e_max),
        skew=skew_val,
        kurt=kurt_val,
        skew_kurt_source=source,
        note=note,
    )


def expected_max_sharpe(n_trials: int) -> float:
    """
    零假设下最大 Sharpe 期望（独立函数，便于测试）。
    """
    if n_trials < 1:
        return 0.0
    if n_trials == 1:
        return 0.0
    q1 = scipy_stats.norm.ppf(1 - 1 / n_trials)
    q2 = scipy_stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float((1 - EULER_GAMMA) * q1 + EULER_GAMMA * q2)