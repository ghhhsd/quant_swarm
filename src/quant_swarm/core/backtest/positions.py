"""
持仓构造（截面 zscore → 多空）。

关键设计：
1. 因子值 → 截面 zscore → 多空持仓。
2. z > 1.0 → 多头 1.0；z < -1.0 → 空头 -1.0；否则 0。
3. 行归一化到权重和为 1（多空各 0.5）。
4. 向量化：批量处理 N 个因子。

为什么用 zscore：
- 因子值量纲不同，直接排序会受量纲影响。
- zscore 标准化后，截面可比。
- 阈值 1.0 是多空分界，对应约 16% 分位。
"""
import numpy as np


class PositionBuilder:
    """
    持仓构造器。
    """

    def __init__(self,
                 long_threshold: float = 1.0,
                 short_threshold: float = -1.0):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold

    def build(self, factor_values: np.ndarray) -> np.ndarray:
        """
        单因子：因子值 → 持仓。

        factor_values：(T, A)
        返回：(T, A) 持仓
        """
        z = _cross_zscore(factor_values)
        positions = np.where(
            z > self.long_threshold, 1.0,
            np.where(z < self.short_threshold, -1.0, 0.0),
        )
        # 行归一化
        row_sums = np.abs(positions).sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
        return positions / row_sums

    def build_batch(self, factor_tensor: np.ndarray) -> np.ndarray:
        """
        批量：因子值张量 → 持仓张量。

        factor_tensor：(N, T, A)
        返回：(N, T, A) 持仓
        """
        z = _cross_zscore_batch(factor_tensor)
        positions = np.where(
            z > self.long_threshold, 1.0,
            np.where(z < self.short_threshold, -1.0, 0.0),
        )
        # 每行归一化
        row_sums = np.abs(positions).sum(axis=2, keepdims=True)
        row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
        return positions / row_sums


def _cross_zscore(x: np.ndarray) -> np.ndarray:
    """截面 zscore。"""
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    return (x - m) / (s + 1e-9)


def _cross_zscore_batch(x: np.ndarray) -> np.ndarray:
    """批量截面 zscore。x: (N, T, A)"""
    m = np.nanmean(x, axis=2, keepdims=True)
    s = np.nanstd(x, axis=2, keepdims=True)
    return (x - m) / (s + 1e-9)