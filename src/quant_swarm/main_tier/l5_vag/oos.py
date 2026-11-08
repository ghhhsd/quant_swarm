"""
滚动 OOS + walk-forward 切分（v2.0 冻结）。

关键设计：
1. OOS 不是固定一段，而是滚动窗口——每次用不同 OOS 段评估，
   降低对单一 OOS 段的隐式过拟合。
2. walk-forward：训练窗口前移，OOS 窗口跟随。
3. 返回多个 OOS 批次的评估结果，L5 的 DSR/FDR 用"跨批次"的
   分布而非"单批次"的 Sharpe。

注意：这里的"批次"是 OOS 滚动批次，与 L0 的"搜索批次"不同。
搜索批次是"一轮演化产生多少试验"，OOS 批次是"同一因子在不同时间段的评估"。
DSR 的 N 用搜索批次（L0），BH-FDR 的 p 值用 OOS 批次（L5 内部）。
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class OOSWindow:
    """单个 OOS 窗口。"""
    train_start: int
    train_end: int
    oos_start: int
    oos_end: int

    def to_dict(self):
        return {
            "train_start": self.train_start, "train_end": self.train_end,
            "oos_start": self.oos_start, "oos_end": self.oos_end,
        }


def walk_forward_split(n_obs: int,
                       train_window: int = 500,
                       oos_window: int = 252,
                       step: int = 63) -> List[OOSWindow]:
    """
    Walk-forward 切分。

    返回多个 OOS 窗口，每个窗口有独立的训练段和 OOS 段。
    窗口之间可能有重叠（step < oos_window 时），这是设计意图——
    重叠降低对单一 OOS 段的依赖。
    """
    windows = []
    start = 0
    while start + train_window + oos_window <= n_obs:
        windows.append(OOSWindow(
            train_start=start,
            train_end=start + train_window,
            oos_start=start + train_window,
            oos_end=start + train_window + oos_window,
        ))
        start += step
    if not windows:
        # 数据太短：至少给一个窗口
        windows.append(OOSWindow(
            train_start=0,
            train_end=max(1, n_obs - oos_window),
            oos_start=max(1, n_obs - oos_window),
            oos_end=n_obs,
        ))
    return windows


def rolling_oos_sharpes(returns: np.ndarray,
                        windows: List[OOSWindow]) -> List[float]:
    """
    对给定收益序列，在每个 OOS 窗口内计算 Sharpe。
    返回跨窗口的 Sharpe 列表。
    """
    sharpes = []
    for w in windows:
        seg = returns[w.oos_start:w.oos_end]
        if len(seg) < 2:
            sharpes.append(0.0)
            continue
        mu = float(np.mean(seg))
        sigma = float(np.std(seg))
        if sigma < 1e-9:
            sharpes.append(0.0)
        else:
            sharpes.append(mu / sigma * np.sqrt(252))
    return sharpes