"""
因子值计算（向量化）。

关键设计：
1. 在面板上求值表达式，返回 (T, A) 因子值矩阵。
2. 函数实现与 DSL 签名一致（ts_mean / ts_std / ...）。
3. 向量化：优先用 numpy 向量化，避免 Python 循环。
4. 时间序列函数（ts_*）必须用滚动窗口，无法完全向量化，用 numpy 优化。

为什么向量化：
- 之前的 evaluate_expression 用 Python 循环计算滚动窗口。
- 批量回测时，N 个因子 × T 天 × 窗口 = O(N × T × W) 次 Python 操作。
- 向量化后，numpy 内部 C 实现，快 10-100 倍。
"""
from typing import Dict, Any
import numpy as np

from core.backtest.panel import PanelData


class FactorEvaluator:
    """
    因子值计算器。

    在面板上求值表达式。
    """

    def __init__(self, panel: PanelData):
        self.panel = panel
        self._funcs = self._build_funcs()

    def _build_funcs(self) -> Dict[str, Any]:
        """构建函数字典（与 DSL 签名一致）。"""
        return {
            "ts_mean": lambda x, w: _ts_rolling(x, int(w), np.mean),
            "ts_std": lambda x, w: _ts_rolling(x, int(w), np.std),
            "ts_sum": lambda x, w: _ts_rolling(x, int(w), np.sum),
            "ts_max": lambda x, w: _ts_rolling(x, int(w), np.max),
            "ts_min": lambda x, w: _ts_rolling(x, int(w), np.min),
            "ts_rank": lambda x, w: _ts_rolling(x, int(w), _rank_last),
            "abs": np.abs,
            "log": lambda x: np.log(np.abs(x) + 1e-9),
            "sqrt": lambda x: np.sqrt(np.abs(x)),
            "sign": np.sign,
            "clip": lambda x, lo, hi: np.clip(x, lo, hi),
            "rank": _cross_rank,
            "zscore": _cross_zscore,
        }

    def evaluate(self, code: str) -> np.ndarray:
        """
        求值表达式，返回 (T, A) 因子值矩阵。
        """
        env = {"__builtins__": {}}
        env.update(self._funcs)
        env.update(self.panel.fields)
        try:
            result = eval(code, env, {})
        except Exception as e:
            raise ValueError(f"表达式求值失败: {e}")
        return np.asarray(result, dtype=float)

    def evaluate_batch(self, codes: list) -> np.ndarray:
        """
        批量求值，返回 (N, T, A) 因子值张量。
        """
        results = [self.evaluate(code) for code in codes]
        return np.stack(results, axis=0)


# =============================================================================
# 辅助函数（向量化实现）
# =============================================================================

def _ts_rolling(x: np.ndarray, w: int, fn) -> np.ndarray:
    """
    时间序列滚动窗口（向量化）。

    用 numpy sliding_window_view 避免 Python 循环。
    """
    if w <= 0 or w >= len(x):
        return np.full_like(x, np.nan, dtype=float)
    # 构造滑动窗口：(T - w + 1, w, A)
    windows = np.lib.stride_tricks.sliding_window_view(x, w, axis=0)
    # 对每个窗口应用 fn
    out = np.full_like(x, np.nan, dtype=float)
    out[w - 1:] = fn(windows, axis=1)
    return out


def _rank_last(a: np.ndarray) -> np.ndarray:
    """滚动窗口内最后一个元素的排名。"""
    # a: (n_windows, w, A)
    last = a[:, -1, :]  # (n_windows, A)
    # 排名 = 小于 last 的元素数 / w
    return (a < last[:, None, :]).sum(axis=1) / a.shape[1]


def _cross_rank(x: np.ndarray) -> np.ndarray:
    """截面排名。"""
    return np.apply_along_axis(
        lambda row: (row.argsort().argsort() + 1) / len(row), 1, x
    )


def _cross_zscore(x: np.ndarray) -> np.ndarray:
    """截面 zscore。"""
    m = np.nanmean(x, axis=1, keepdims=True)
    s = np.nanstd(x, axis=1, keepdims=True)
    return (x - m) / (s + 1e-9)