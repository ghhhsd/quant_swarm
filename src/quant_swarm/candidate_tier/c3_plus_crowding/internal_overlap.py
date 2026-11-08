"""
内部持仓重叠（v3.1 口径修正：只算"能看见的"）。

关键设计：
1. 只对精英池内因子两两计算，不涉及全市场。
2. 相似度用余弦相似度（归一化持仓的行向量）。
3. 返回内部重叠均值 + 最大重叠对（用于诊断）。
4. 增量计算：新因子入池时，只与现有因子两两计算，不重算全部。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

from candidate_tier.c3_plus_crowding.positions import (
    PositionMatrix, PositionExtractor,
)


@dataclass
class OverlapPair:
    """一对因子的重叠。"""
    fid_a: str
    fid_b: str
    cosine_similarity: float   # [0, 1]，越高越重叠

    def to_dict(self):
        return {
            "fid_a": self.fid_a, "fid_b": self.fid_b,
            "cosine_similarity": self.cosine_similarity,
        }


@dataclass
class InternalOverlapReport:
    """内部持仓重叠报告。"""
    mean_overlap: float
    max_overlap: float
    max_pair: Optional[OverlapPair]
    n_factors: int
    n_pairs: int
    pairwise: List[OverlapPair] = field(default_factory=list)

    def to_dict(self):
        return {
            "mean_overlap": self.mean_overlap,
            "max_overlap": self.max_overlap,
            "max_pair": self.max_pair.to_dict() if self.max_pair else None,
            "n_factors": self.n_factors,
            "n_pairs": self.n_pairs,
            "pairwise": [p.to_dict() for p in self.pairwise],
        }


class InternalOverlapChecker:
    """
    内部持仓重叠检查器。

    v3.1 精确化：口径为"内部持仓重叠"，不是"拥挤度检测"。
    """

    def __init__(self, max_internal_overlap: float = 0.70):
        self.max_internal_overlap = max_internal_overlap

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        两个持仓矩阵的余弦相似度。
        展平后计算，值域 [0, 1]（取绝对值）。
        """
        a_flat = a.flatten()
        b_flat = b.flatten()
        norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
        if norm < 1e-9:
            return 0.0
        return abs(float(np.dot(a_flat, b_flat) / norm))

    def check(self, positions: Dict[str, PositionMatrix]) -> InternalOverlapReport:
        """
        计算内部持仓重叠。
        """
        fids = list(positions.keys())
        n = len(fids)
        if n < 2:
            return InternalOverlapReport(
                mean_overlap=0.0, max_overlap=0.0,
                max_pair=None, n_factors=n, n_pairs=0,
            )

        pairwise: List[OverlapPair] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.cosine_similarity(
                    positions[fids[i]].positions,
                    positions[fids[j]].positions,
                )
                pairwise.append(OverlapPair(fids[i], fids[j], sim))

        sims = [p.cosine_similarity for p in pairwise]
        max_idx = int(np.argmax(sims))
        max_pair = pairwise[max_idx]

        return InternalOverlapReport(
            mean_overlap=float(np.mean(sims)),
            max_overlap=float(np.max(sims)),
            max_pair=max_pair,
            n_factors=n,
            n_pairs=len(pairwise),
            pairwise=pairwise,
        )

    def incremental_check(self,
                          new_fid: str,
                          new_position: PositionMatrix,
                          existing_positions: Dict[str, PositionMatrix]
                          ) -> List[OverlapPair]:
        """
        增量检查：新因子与现有因子的重叠。
        不重算现有因子之间的重叠。
        """
        pairs = []
        for fid, pos in existing_positions.items():
            sim = self.cosine_similarity(new_position.positions, pos.positions)
            pairs.append(OverlapPair(new_fid, fid, sim))
        return pairs

    def would_flag(self, report: InternalOverlapReport) -> bool:
        """是否应 flagged（内部重叠超标）。"""
        return report.mean_overlap > self.max_internal_overlap

    def would_flag_incremental(self,
                               pairs: List[OverlapPair]) -> bool:
        """增量检查：新因子是否与现有因子重叠超标。"""
        if not pairs:
            return False
        max_sim = max(p.cosine_similarity for p in pairs)
        return max_sim > self.max_internal_overlap