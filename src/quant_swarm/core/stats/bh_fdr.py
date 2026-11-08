"""
Benjamini-Hochberg FDR（v3.1 修正①：批次内 FDR）。

关键设计：
1. FDR 是"批次内"的多重检验校正，不是"全局"。
2. 输入：同批次的 p 值列表（来自 TrialRecord）。
3. 输出：每个 p 值是否通过 FDR 校正。
4. 校验：p 值在 [0, 1]，批次内至少有 2 个试验。
5. 失败时抛 ValueError，不静默返回。

为什么批次内：
- 跨批次混在一起做 FDR 会破坏批次独立性。
- 每个批次的试验是"同一搜索过程"的产物，应批次内校正。
- 批次间比较用 DSR（N 按批次），不用 FDR。

BH-FDR 算法：
1. 排序 p 值：p_(1) <= p_(2) <= ... <= p_(m)
2. 找最大 k 使得 p_(k) <= k/m * alpha
3. 前 k 个 p 值通过
"""
from typing import List, Optional
from dataclasses import dataclass

from core.stats.result import FDRResult
from core.stats.pvalue import validate_pvalues


def bh_fdr(p_values: List[float],
           alpha: float = 0.10,
           batch_id: str = "unknown") -> FDRResult:
    """
    BH-FDR 校正（批次内）。

    p_values：同批次的 p 值列表
    alpha：FDR 显著性水平（默认 0.10）
    batch_id：批次 ID（用于审计）
    """
    # 1) 校验
    if not (0 < alpha < 1):
        raise ValueError(f"alpha 必须在 (0,1)，实际 {alpha}")
    validate_pvalues(p_values)
    m = len(p_values)
    if m < 2:
        # 单次试验无需 FDR
        return FDRResult(
            p_values=p_values,
            flags=[p <= alpha for p in p_values],
            alpha=alpha,
            n_tests=m,
            n_significant=sum(1 for p in p_values if p <= alpha),
            batch_id=batch_id,
        )

    # 2) 排序
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # 3) 找最大 k
    thresh_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= rank / m * alpha:
            thresh_rank = rank

    # 4) 标记
    flags = [False] * m
    for rank, (idx, _) in enumerate(indexed, start=1):
        if rank <= thresh_rank:
            flags[idx] = True

    return FDRResult(
        p_values=p_values,
        flags=flags,
        alpha=alpha,
        n_tests=m,
        n_significant=sum(flags),
        batch_id=batch_id,
    )


def bh_fdr_from_records(records: List,
                        alpha: float = 0.10) -> FDRResult:
    """
    从 TrialRecord 列表做 BH-FDR。

    records：同批次的 TrialRecord 列表
    """
    if not records:
        raise ValueError("records 不能为空")
    batch_id = records[0].batch_id
    # 校验所有记录同批次
    for r in records:
        if r.batch_id != batch_id:
            raise ValueError(
                f"records 中有不同批次: {batch_id} vs {r.batch_id}"
            )
    # 转换 p 值
    from core.stats.pvalue import sharpe_to_pvalue
    p_values = [
        sharpe_to_pvalue(r.net_sharpe, r.n_obs) for r in records
    ]
    return bh_fdr(p_values, alpha, batch_id)