"""
五类失败计数与差异化剪枝。

差异化策略（v2.0 冻结，v3.1 保留）：
- gate_rejected：直接剪枝，不重试（避免反复撞同一门控）
- 其他四类：累计到阈值才剪枝
- 不同失败类型的阈值可不同（语法错误重试便宜，逻辑错误重试贵）
"""
from dataclasses import dataclass, field
from typing import Dict
from main_tier.l1_favor import FailureType


@dataclass
class FailurePolicy:
    """每类失败的重试上限。"""
    max_retries: Dict[FailureType, int] = field(default_factory=lambda: {
        FailureType.SYNTAX: 3,        # 语法错误重试便宜
        FailureType.LOGIC: 2,         # 逻辑错误重试贵
        FailureType.DATA: 2,          # 数据契约修正有限
        FailureType.BACKTEST: 2,      # 回测异常降复杂度重试
        FailureType.GATE_REJECTED: 0, # 门控拒绝不重试
    })


class FailureTracker:
    """
    按"因子 hid + 失败类型"记账。不是全局计数——同一个因子反复失败才剪枝，
    不同因子各自独立。
    """

    def __init__(self, policy: FailurePolicy = None):
        self.policy = policy or FailurePolicy()
        # hid -> {failure_type: count}
        self._counts: Dict[str, Dict[FailureType, int]] = {}
        # 已剪枝的 hid 集合（避免重复剪枝）
        self._pruned: set = set()

    def record(self, hid: str, failure_type: FailureType) -> bool:
        """
        记录一次失败。返回 True 表示应剪枝该因子。
        """
        if hid in self._pruned:
            return True  # 已剪枝，幂等
        counts = self._counts.setdefault(hid, {})
        counts[failure_type] = counts.get(failure_type, 0) + 1
        limit = self.policy.max_retries.get(failure_type, 2)
        # gate_rejected 的 limit=0，第一次就剪
        if counts[failure_type] > limit:
            self._pruned.add(hid)
            return True
        return False

    def is_pruned(self, hid: str) -> bool:
        return hid in self._pruned

    def get_counts(self, hid: str) -> Dict[FailureType, int]:
        return dict(self._counts.get(hid, {}))

    def reset(self) -> None:
        """轮次切换时清理（每轮独立）。"""
        self._counts.clear()
        self._pruned.clear()