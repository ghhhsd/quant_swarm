"""
外部拥挤代理（v3.1 精确化：单独列，不约束）。

关键设计：
1. 外部拥挤是本系统"看不见"的——其他资金的同向持仓。
2. 只能用代理指标间接观测：融券余额、同业披露、因子回撤聚集、
   换手率分位、收益离散度、成交量集中度等。
3. 代理指标**写入事件流供审计**，但**不参与 flagged 判定**。
   - 理由：外部拥挤不可控，强行约束=用不可观测的东西做决策。
4. 代理指标可被后续层（如 L6 慢环）用于"上线前参考"。
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np
import polars as pl


@dataclass
class ExternalProxyReport:
    """外部拥挤代理报告。"""
    metrics: Dict[str, float]
    n_metrics: int
    note: str = "外部拥挤代理仅用于观测，不参与约束判定"

    def to_dict(self):
        return {
            "metrics": dict(self.metrics),
            "n_metrics": self.n_metrics,
            "note": self.note,
        }


class ExternalProxyExtractor:
    """
    外部拥挤代理提取器。

    真实部署应接入：
    - 融券余额（市场数据）
    - 同业持仓披露（季度 13F / 龙虎榜等）
    - 因子收益回撤聚集（多因子同时回撤 = 拥挤信号）

    本实现用可观测的代理：
    - 换手率 90 分位
    - 收益截面离散度
    - 成交量集中度（top 20% 占比）
    - 波动率分位
    """

    def extract(self, df: pl.DataFrame) -> ExternalProxyReport:
        metrics = {}
        try:
            turnover = df.pivot(values="turnover", index="date",
                                on="asset").sort("date")
            turn_arr = np.array([turnover[c].to_numpy()
                                 for c in turnover.columns[1:]]).T
            metrics["turnover_q90"] = float(np.nanpercentile(turn_arr, 90))
            metrics["turnover_q50"] = float(np.nanpercentile(turn_arr, 50))

            returns = df.pivot(values="returns", index="date",
                               on="asset").sort("date")
            ret_arr = np.array([returns[c].to_numpy()
                                for c in returns.columns[1:]]).T
            cross_dispersion = np.nanstd(ret_arr, axis=1)
            metrics["return_dispersion"] = float(np.nanmean(cross_dispersion))
            metrics["return_dispersion_q90"] = float(np.nanpercentile(
                cross_dispersion, 90))

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
            metrics["volume_concentration"] = float(np.nanmean(concentration))

            # 波动率分位（市场波动状态）
            market_vol = np.nanstd(ret_arr, axis=1)
            metrics["market_vol_q90"] = float(np.nanpercentile(market_vol, 90))

        except Exception:
            pass

        return ExternalProxyReport(
            metrics=metrics, n_metrics=len(metrics),
        )