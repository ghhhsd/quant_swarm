"""
统计结果数据结构。

关键设计：
1. 每个统计工具有独立的结果结构。
2. 结果可序列化（供事件溯源）。
3. 失败时带 reason，成功时带完整推导信息。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class DSRResult:
    """Deflated Sharpe Ratio 结果。"""
    sharpe: float
    dsr: float                    # [0, 1]，校正多重检验后的真实概率
    n_trials: int                 # 搜索批次累计 N
    n_obs: int
    expected_max_sharpe: float    # 零假设下最大 Sharpe 期望
    skew: float
    kurt: float
    skew_kurt_source: str         # "empirical" / "normal_assumption"
    note: str = ""

    def to_dict(self):
        return {
            "sharpe": self.sharpe,
            "dsr": self.dsr,
            "n_trials": self.n_trials,
            "n_obs": self.n_obs,
            "expected_max_sharpe": self.expected_max_sharpe,
            "skew": self.skew,
            "kurt": self.kurt,
            "skew_kurt_source": self.skew_kurt_source,
            "note": self.note,
        }


@dataclass
class FDRResult:
    """BH-FDR 结果。"""
    p_values: List[float]
    flags: List[bool]             # 每个 p 值是否通过 FDR
    alpha: float
    n_tests: int
    n_significant: int
    batch_id: str

    def to_dict(self):
        return {
            "p_values": self.p_values,
            "flags": self.flags,
            "alpha": self.alpha,
            "n_tests": self.n_tests,
            "n_significant": self.n_significant,
            "batch_id": self.batch_id,
        }


@dataclass
class TrialRecord:
    """单次试验记录。"""
    batch_id: str
    hid: str
    trial_index: int              # 该批次内第几次试验（从 1 开始）
    net_sharpe: float
    n_obs: int
    timestamp: float

    def to_dict(self):
        return {
            "batch_id": self.batch_id,
            "hid": self.hid,
            "trial_index": self.trial_index,
            "net_sharpe": self.net_sharpe,
            "n_obs": self.n_obs,
            "timestamp": self.timestamp,
        }