"""
持仓矩阵提取 + 采样降维（v3.1 工程约束）。

关键设计：
1. 只采样最近 N 天（默认 63），不取全历史。
2. 持仓矩阵形状 (sample_days, n_assets)。
3. 提供增量缓存：新因子只计算自己的持仓，不重算已有因子。
4. 持仓矩阵归一化到"权重和为 1"（多空各 0.5），便于余弦相似度计算。
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import polars as pl

from core.backtest import evaluate_expression, _cross_zscore


@dataclass
class PositionMatrix:
    """因子持仓矩阵。"""
    fid: str
    positions: np.ndarray       # (sample_days, n_assets)
    sample_days: int
    n_assets: int

    def normalized(self) -> np.ndarray:
        """
        归一化持仓：多空各 0.5，权重和为 1。
        用于余弦相似度计算。
        """
        pos = self.positions.copy()
        # 分行归一化（每行权重和为 1）
        row_sums = np.abs(pos).sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-9, 1.0, row_sums)
        return pos / row_sums

    def to_dict(self):
        return {
            "fid": self.fid,
            "sample_days": self.sample_days,
            "n_assets": self.n_assets,
            "positions_shape": list(self.positions.shape),
        }


class PositionExtractor:
    """
    持仓矩阵提取器。

    采样降维：
    - 只取最近 sample_days 天（默认 63）。
    - 缓存已提取的持仓，新因子只计算自己。

    持仓构造：
    - 因子值 → 截面 zscore
    - z > 1.0 → 多头 1.0；z < -1.0 → 空头 -1.0；否则 0
    - 行归一化到权重和为 1
    """

    def __init__(self, df: pl.DataFrame, sample_days: int = 63):
        self.df = df
        self.sample_days = sample_days
        self._cache: Dict[str, PositionMatrix] = {}

    def extract(self, fid: str, code: str) -> PositionMatrix:
        """提取持仓矩阵。缓存命中则直接返回。"""
        if fid in self._cache:
            return self._cache[fid]

        # 因子值
        fv = np.nan_to_num(evaluate_expression(self.df, code), nan=0.0)
        z = _cross_zscore(fv)
        positions = np.where(z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0))

        # 采样降维：只取最近 sample_days 天
        if positions.shape[0] > self.sample_days:
            positions = positions[-self.sample_days:]

        pm = PositionMatrix(
            fid=fid,
            positions=positions,
            sample_days=positions.shape[0],
            n_assets=positions.shape[1],
        )
        self._cache[fid] = pm
        return pm

    def extract_batch(self, fids_codes: Dict[str, str]) -> Dict[str, PositionMatrix]:
        """批量提取。"""
        return {fid: self.extract(fid, code) for fid, code in fids_codes.items()}

    def invalidate(self, fid: str) -> None:
        """清除缓存（因子被淘汰时）。"""
        self._cache.pop(fid, None)

    def size(self) -> int:
        return len(self._cache)