"""
Sharpe → p 值转换（v3.1 修正①：BH-FDR 的 p 值来源）。

关键设计：
1. 单边检验：H0: Sharpe <= 0, H1: Sharpe > 0。
2. p 值 = P(T > t | H0)，其中 t = sharpe * sqrt(n_obs)。
3. 用正态近似（n_obs 足够大时）。
4. 校验 p 值在 [0, 1]。
5. 批量转换：一次处理多个 Sharpe。

为什么用正态近似：
- Sharpe 的 t 统计量在 n_obs 足够大时趋近正态。
- 小样本时应用 t 分布（本实现用正态，真实部署可切换）。
- 单边检验（>0）符合"因子是否有效"的业务语义。
"""
from typing import List
import numpy as np
from scipy import stats as scipy_stats


def sharpe_to_pvalue(sharpe: float, n_obs: int,
                     one_sided: bool = True) -> float:
    """
    Sharpe → p 值。

    sharpe：净 Sharpe
    n_obs：观测数
    one_sided：是否单边检验（默认 True）
    """
    if n_obs < 2:
        return 1.0
    t_stat = sharpe * np.sqrt(n_obs)
    if one_sided:
        p = 1 - scipy_stats.norm.cdf(t_stat)
    else:
        p = 2 * (1 - scipy_stats.norm.cdf(abs(t_stat)))
    # 校验 p 值在 [0, 1]
    return float(max(0.0, min(1.0, p)))


def sharpes_to_pvalues(sharpes: List[float], n_obs: int,
                       one_sided: bool = True) -> List[float]:
    """
    批量 Sharpe → p 值。
    """
    return [sharpe_to_pvalue(s, n_obs, one_sided) for s in sharpes]


def validate_pvalues(p_values: List[float]) -> None:
    """
    校验 p 值在 [0, 1]。

    抛 ValueError 如果超出范围。
    """
    for i, p in enumerate(p_values):
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"第 {i} 个 p 值 {p} 不在 [0, 1]")