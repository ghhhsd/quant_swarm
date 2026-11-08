"""
组合换手计算（v3.1 修正③：组合换手 ≠ 单因子换手之和）。

关键设计：
1. 组合换手 = |Δ组合持仓| 之和。
2. 单因子换手之和 = Σ |Δ单因子持仓| 之和。
3. 抵消效应 = 单因子换手之和 - 组合换手。
4. 抵消效应越大，组合层面的成本越低。
5. 报告两者，供审计。

为什么组合换手 ≠ 单因子换手之和：
- 两个因子一个加仓一个减仓，单因子换手都增加，
  但组合换手可能不变（抵消）。
- 用单因子换手之和算成本会高估成本。
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class TurnoverBreakdown:
    """换手分解。"""
    combined_turnover: np.ndarray       # (T,) 组合换手
    sum_single_turnover: np.ndarray     # (T,) 单因子换手之和
    offset: np.ndarray                  # (T,) 抵消效应
    offset_ratio: float                 # 抵消比例（均值）

    def to_dict(self):
        return {
            "combined_turnover_mean": float(np.mean(self.combined_turnover)),
            "sum_single_turnover_mean": float(np.mean(self.sum_single_turnover)),
            "offset_mean": float(np.mean(self.offset)),
            "offset_ratio": self.offset_ratio,
        }


def compute_turnover(combined_positions: np.ndarray,
                     single_positions: list) -> TurnoverBreakdown:
    """
    计算组合换手 + 单因子换手之和 + 抵消效应。

    combined_positions：(T, A) 组合持仓
    single_positions：[N 个 (T, A) 单因子持仓]
    """
    # 组合换手
    if len(combined_positions) < 2:
        combined_turnover = np.array([0.0])
    else:
        combined_turnover = np.abs(
            np.diff(combined_positions, axis=0)
        ).sum(axis=1)

    # 单因子换手之和
    if not single_positions:
        sum_single = np.zeros_like(combined_turnover)
    else:
        sum_single_list = []
        for pos in single_positions:
            if len(pos) < 2:
                sum_single_list.append(np.array([0.0]))
            else:
                sum_single_list.append(np.abs(np.diff(pos, axis=0)).sum(axis=1))
        # 对齐长度
        min_len = min(len(s) for s in sum_single_list)
        sum_single = np.sum([s[:min_len] for s in sum_single_list], axis=0)

    # 抵消效应
    min_len = min(len(combined_turnover), len(sum_single))
    combined_turnover = combined_turnover[:min_len]
    sum_single = sum_single[:min_len]
    offset = sum_single - combined_turnover
    offset_ratio = (
        float(np.mean(offset) / (np.mean(sum_single) + 1e-9))
        if np.mean(sum_single) > 0 else 0.0
    )

    return TurnoverBreakdown(
        combined_turnover=combined_turnover,
        sum_single_turnover=sum_single,
        offset=offset,
        offset_ratio=offset_ratio,
    )