"""
OOS 批次切片（v3.1 修正②：消融比较需要跨批次）。

关键设计：
1. walk-forward 切分：多个 OOS 批次。
2. 每个批次是面板的索引切片。
3. 批次之间可以有重叠（step < oos_window）。
4. 消融框架对每个批次独立计算组合 PnL。
"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class OOSBatch:
    """单个 OOS 批次。"""
    batch_id: str
    oos_start: int
    oos_end: int
    regime: Optional[str] = None

    def to_dict(self):
        return {
            "batch_id": self.batch_id,
            "oos_start": self.oos_start,
            "oos_end": self.oos_end,
            "regime": self.regime,
        }


def generate_oos_batches(n_obs: int,
                         train_window: int = 500,
                         oos_window: int = 252,
                         step: int = 63,
                         regimes: Optional[np.ndarray] = None
                         ) -> List[OOSBatch]:
    """
    生成 walk-forward OOS 批次。
    """
    batches = []
    start = 0
    idx = 0
    while start + train_window + oos_window <= n_obs:
        oos_start = start + train_window
        oos_end = start + train_window + oos_window
        regime = None
        if regimes is not None and oos_end <= len(regimes):
            oos_regimes = regimes[oos_start:oos_end]
            if len(oos_regimes) > 0:
                values, counts = np.unique(oos_regimes, return_counts=True)
                regime = str(values[np.argmax(counts)])
        batches.append(OOSBatch(
            batch_id=f"oos_{idx:03d}",
            oos_start=oos_start,
            oos_end=oos_end,
            regime=regime,
        ))
        start += step
        idx += 1
    return batches