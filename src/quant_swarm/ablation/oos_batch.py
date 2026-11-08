"""
OOS 批次生成（walk-forward 切分）。

关键设计：
1. 每个 OOS 批次是一个独立的 walk-forward 窗口。
2. 消融比较的是"跨多个 OOS 批次的组合 Sharpe 分布"。
3. OOS 窗口之间可以有重叠，但每个窗口独立计算组合 PnL。
4. 批次 ID 与 L0 的搜索批次 ID 不同——搜索批次是"一轮演化产生多少试验"，
   OOS 批次是"同一组合在不同时间段的评估"。
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
    regime: Optional[str] = None  # 该批次所属市场状态

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

    n_obs：总观测数
    train_window：训练窗口长度
    oos_window：OOS 窗口长度
    step：窗口前移步长
    regimes：每个时点的市场状态（可选，用于标注每个 OOS 批次的 regime）
    """
    batches = []
    start = 0
    idx = 0
    while start + train_window + oos_window <= n_obs:
        oos_start = start + train_window
        oos_end = start + train_window + oos_window
        regime = None
        if regimes is not None and oos_end <= len(regimes):
            # 取 OOS 窗口内主导 regime
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