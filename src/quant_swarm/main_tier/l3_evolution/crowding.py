"""
持仓拥挤度（v3.1 新增，分层口径）。

关键设计（v3.1 精确化）：
1. 内部持仓重叠：本系统因子之间的持仓余弦相似度。
   这是系统"能看见"的踩踏风险。
2. 外部拥挤代理：全市场其他资金的同向持仓，系统"看不见"，
   只能用代理指标（融券余额、同业披露、因子回撤聚集）间接观测。
   单独列出，不混在"拥挤度"里，避免口径夸大。

工程约束（v3.1 批评指出）：
持仓重叠度计算需完整持仓序列，存储/计算成本高。
本实现用"最近 63 天采样"降维，而非全历史两两计算。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import polars as pl

from main_tier.l3_evolution.factor import Factor
from core.backtest import evaluate_expression, _cross_zscore


@dataclass
class CrowdingReport:
    """分层拥挤度报告。"""
    internal_overlap: float                 # 内部持仓重叠（系统能看见）
    external_proxy: Dict[str, float]        # 外部拥挤代理（单独列）
    flagged: bool
    reason: str
    sample_days: int


class CrowdingChecker:
    """
    持仓拥挤度检查（分层）。

    内部重叠：两两因子持仓余弦相似度均值，超过 max_internal_overlap 则 flagged。
    外部代理：从数据中提取可观测的代理指标，单独输出，不参与 flagged 判定
             （因为外部拥挤是本系统不可控的盲区，只能观测不能约束）。
    """

    def __init__(self, max_internal_overlap: float = 0.70,
                 sample_days: int = 63):
        self.max_internal_overlap = max_internal_overlap
        self.sample_days = sample_days

    def _get_positions(self, df: pl.DataFrame, code: str) -> np.ndarray:
        """
        计算因子持仓矩阵，只取最近 sample_days 天。
        返回 (sample_days, N)。
        """
        fv = np.nan_to_num(evaluate_expression(df, code), nan=0.0)
        z = _cross_zscore(fv)
        positions = np.where(z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0))
        # 只取最近 sample_days 天
        return positions[-self.sample_days:]

    def evaluate(self, df: pl.DataFrame,
                 factors: List[Factor]) -> CrowdingReport:
        """
        评估一组因子的内部重叠 + 外部代理。
        """
        if len(factors) < 2:
            return CrowdingReport(
                0.0, self._external_proxy(df), False,
                "候选不足 2 个，无法计算内部重叠", self.sample_days,
            )

        # 内部持仓重叠：两两余弦相似度均值
        positions = [self._get_positions(df, f.code) for f in factors]
        sims = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                a = positions[i].flatten()
                b = positions[j].flatten()
                norm = np.linalg.norm(a) * np.linalg.norm(b)
                if norm < 1e-9:
                    continue
                sims.append(abs(float(np.dot(a, b) / norm)))
        overlap = float(np.mean(sims)) if sims else 0.0

        # 外部代理（单独列）
        external = self._external_proxy(df)

        flagged = overlap > self.max_internal_overlap
        reason = (f"内部持仓重叠 {overlap:.2f} > {self.max_internal_overlap}"
                  if flagged else "ok")
        return CrowdingReport(overlap, external, flagged, reason, self.sample_days)

    @staticmethod
    def _external_proxy(df: pl.DataFrame) -> Dict[str, float]:
        """
        外部拥挤代理（v3.1：单独列，不混在内部重叠里）。
        本实现用可观测指标：
        - 换手率 90 分位（市场活跃度）
        - 收益截面离散度（市场分化程度）
        - 成交量集中度（资金抱团程度）
        """
        proxies = {}
        try:
            turnover = df.pivot(values="turnover", index="date",
                                on="asset").sort("date")
            turn_arr = np.array([turnover[c].to_numpy()
                                 for c in turnover.columns[1:]]).T
            proxies["turnover_q90"] = float(np.nanpercentile(turn_arr, 90))

            returns = df.pivot(values="returns", index="date",
                               on="asset").sort("date")
            ret_arr = np.array([returns[c].to_numpy()
                                for c in returns.columns[1:]]).T
            cross_dispersion = np.nanstd(ret_arr, axis=1)
            proxies["return_dispersion"] = float(np.nanmean(cross_dispersion))

            volume = df.pivot(values="volume", index="date",
                              on="asset").sort("date")
            vol_arr = np.array([volume[c].to_numpy()
                                for c in volume.columns[1:]]).T
            # 集中度：每行 top 20% 成交量占比
            concentration = []
            for row in vol_arr:
                total = np.nansum(row)
                if total > 0:
                    top_n = max(1, int(len(row) * 0.2))
                    top_sum = np.nansum(np.sort(row)[-top_n:])
                    concentration.append(top_sum / total)
            proxies["volume_concentration"] = float(np.nanmean(concentration))
        except Exception:
            pass
        return proxies