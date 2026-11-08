"""
跨 OOS 批次组合 PnL（v3.1 修正②：消融需要跨批次）。

关键设计：
1. 用 walk-forward 切分生成 OOS 批次。
2. 每个批次独立构建组合 PnL。
3. 汇总跨批次结果（均值、标准差）。
4. 这是消融的"最终指标"来源。
"""
from typing import List, Optional
import numpy as np

from core.portfolio.portfolio_builder import PortfolioBuilder
from core.portfolio.config import PortfolioConfig
from core.portfolio.result import (
    PortfolioResult, BatchPortfolioSummary,
)
from core.backtest import PanelData, generate_oos_batches


class BatchPortfolioBuilder:
    """
    跨 OOS 批次组合构建器。

    对每个 OOS 批次独立构建组合 PnL。
    """

    def __init__(self, panel: PanelData, config: PortfolioConfig):
        self.panel = panel
        self.config = config
        self.builder = PortfolioBuilder(panel, config)

    def build(self, codes: List[str],
              train_window: int = 500,
              oos_window: int = 252,
              step: int = 63,
              regimes: Optional[np.ndarray] = None
              ) -> BatchPortfolioSummary:
        """
        跨 OOS 批次构建组合。
        """
        n_obs = len(self.panel.dates)
        batches = generate_oos_batches(
            n_obs, train_window, oos_window, step, regimes,
        )
        results = []
        for batch in batches:
            result = self.builder.build(
                codes,
                batch_id=batch.batch_id,
                oos_start=batch.oos_start,
                oos_end=batch.oos_end,
            )
            results.append(result)
        return BatchPortfolioSummary(
            results=results, n_batches=len(results),
        )