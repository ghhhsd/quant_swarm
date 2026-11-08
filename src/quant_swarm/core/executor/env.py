"""
受限执行环境（第 1 层：空 builtins + 白名单）。

关键设计：
1. 空 builtins：禁用 __import__、eval、exec、open 等。
2. 白名单函数：ts_mean / ts_std / ... （与 DSL 函数签名一致）。
3. 白名单字段：来自数据契约。
4. 这只是"环境最小化"，不是"安全边界"——真正兜底靠子进程/Docker。
5. 提供"环境指纹"用于审计（哪些函数被允许）。

为什么是"环境最小化"而非"安全边界"：
- 空 builtins 可被绕过（如通过 __class__ 链获取 object）。
- 受限 DSL 下没有通用代码逃逸面，因为表达式不是通用程序。
- 但如果未来放开到通用代码生成，必须叠加 seccomp/gVisor。
"""
from typing import Dict, Set, Any, Optional
import numpy as np

from core.data import DEFAULT_DATA_CONTRACT
from core.dsl.functions import FUNCTION_SIGNATURES


class RestrictedEnv:
    """
    受限执行环境。

    - 空 builtins。
    - 白名单函数（与 DSL 签名一致）。
    - 白名单字段（来自数据契约）。
    """

    def __init__(self, panel: Dict[str, np.ndarray],
                 contract=None):
        """
        panel：{字段名: 面板数据 (T, N)}，来自回测引擎。
        contract：数据契约（默认 DEFAULT_DATA_CONTRACT）。
        """
        self.panel = panel
        self.contract = contract or DEFAULT_DATA_CONTRACT
        self.allowed_funcs = self._build_allowed_funcs()
        self.allowed_fields = set(panel.keys())

    def _build_allowed_funcs(self) -> Dict[str, Any]:
        """构建白名单函数（与 DSL 签名一致）。"""
        return {
            "ts_mean": lambda x, w: _ts_rolling(x, w, np.mean),
            "ts_std": lambda x, w: _ts_rolling(x, w, np.std),
            "ts_sum": lambda x, w: _ts_rolling(x, w, np.sum),
            "ts_max": lambda x, w: _ts_rolling(x, w, np.max),
            "ts_min": lambda x, w: _ts_rolling(x, w, np.min),
            "ts_rank": lambda x, w: _ts_rolling(x, w, _rank_last),
            "abs": np.abs,
            "log": lambda x: np.log(np.abs(x) + 1e-9),
            "sqrt": lambda x: np.sqrt(np.abs(x)),
            "sign": np.sign,
            "clip": lambda x, lo, hi: np.clip(x, lo, hi),
            "rank": _cross_rank,
            "zscore": _cross_zscore,
        }

    def build_globals(self) -> Dict[str, Any]:
        """
        构建 eval 的 globals。

        - 空 builtins。
        - 白名单函数 + 字段。
        """
        g = {"__builtins__": {}}
        g.update(self.allowed_funcs)
        g.update(self.panel)
        return g

    def fingerprint(self) -> Dict[str, Any]:
        """
        环境指纹：记录哪些函数/字段被允许。
        用于审计"这次执行用了什么环境"。
        """
        return {
            "allowed_funcs": sorted(self.allowed_funcs.keys()),
            "allowed_fields": sorted(self.allowed_fields),
            "has_builtins": False,
            "contract_hash": self.contract.content_hash(),
        }


# =============================================================================
# 辅助函数（与回测引擎一致）
# =============================================================================

def _ts_rolling(x, w, fn):
    out = np.full_like(x, np.nan, dtype=float)
    for t in range(int(w), len(x)):
        out[t] = fn(x[t - int(w):t], axis=0)
    return out


def _rank_last(a):
    return (a[-1] > a[:-1]).sum() / len(a)


def _cross_rank(x):
    return np.apply_along_axis(
        lambda row: (row.argsort().argsort() + 1) / len(row), 1, x)


def _cross_zscore(x):
    m, s = np.nanmean(x, axis=1, keepdims=True), np.nanstd(x, axis=1, keepdims=True)
    return (x - m) / (s + 1e-9)