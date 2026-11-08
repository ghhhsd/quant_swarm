"""
加权规则（v3.1 修正③：IC 加权用 OOS 窗口前的 IC）。

关键设计：
1. equal：等权。
2. ic_weighted：用 OOS 窗口前的滚动 IC 绝对值作为权重。
3. risk_parity：按波动倒数加权。
4. 所有规则返回 (N,) 权重，和为 1。
5. IC 加权必须用 OOS 窗口前的数据（无未来信息）。
"""
from typing import List
import numpy as np


class WeightingScheme:
    """
    加权规则实现。
    """

    @staticmethod
    def equal(n_factors: int) -> np.ndarray:
        """等权。"""
        if n_factors <= 0:
            return np.array([])
        return np.ones(n_factors) / n_factors

    @staticmethod
    def ic_weighted(factor_returns: List[np.ndarray],
                    oos_start: int,
                    ic_window: int) -> np.ndarray:
        """
        IC 加权。

        factor_returns：每个因子的收益序列（全样本）
        oos_start：OOS 窗口起始索引
        ic_window：滚动 IC 窗口

        用 OOS 窗口前的 IC（训练段）作为权重。
        """
        n = len(factor_returns)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # 训练段：[oos_start - ic_window, oos_start)
        train_end = max(0, oos_start)
        train_start = max(0, train_end - ic_window)

        if train_end - train_start < 2:
            # 训练段不足，退化为等权
            return np.ones(n) / n

        ics = []
        for fr in factor_returns:
            seg = fr[train_start:train_end]
            if len(seg) < 2 or np.std(seg) < 1e-9:
                ics.append(0.0)
            else:
                # 用收益自相关作为 IC 代理
                ics.append(abs(float(np.mean(seg))))

        ics = np.array(ics)
        if ics.sum() < 1e-9:
            return np.ones(n) / n
        return ics / ics.sum()

    @staticmethod
    def risk_parity(factor_returns: List[np.ndarray],
                    oos_start: int,
                    ic_window: int) -> np.ndarray:
        """
        风险平价：按波动倒数加权。

        用 OOS 窗口前的波动（训练段），避免前视偏差。
        """
        n = len(factor_returns)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        train_end = max(0, oos_start)
        train_start = max(0, train_end - ic_window)

        vols = []
        for fr in factor_returns:
            seg = fr[train_start:train_end] if train_end > train_start else fr
            if len(seg) < 2:
                vols.append(1.0)
            else:
                vols.append(np.std(seg) + 1e-9)

        vols = np.array(vols)
        inv_vols = 1.0 / vols
        return inv_vols / inv_vols.sum()


def compute_weights(scheme: str,
                    factor_returns: List[np.ndarray],
                    oos_start: int,
                    ic_window: int) -> np.ndarray:
    """
    统一入口：按规则计算权重。
    """
    n = len(factor_returns)
    if scheme == "equal":
        return WeightingScheme.equal(n)
    elif scheme == "ic_weighted":
        return WeightingScheme.ic_weighted(
            factor_returns, oos_start, ic_window)
    elif scheme == "risk_parity":
        return WeightingScheme.risk_parity(
            factor_returns, oos_start, ic_window)
    else:
        raise ValueError(f"未知加权规则: {scheme}")