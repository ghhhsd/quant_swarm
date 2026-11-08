"""
main_tier/l0_governance/threshold_lifecycle.py — L0 阈值生命周期管理。

职责：
1. 委托 core/threshold_commit.py 做实际 commit（不重复实现）。
2. 在 L0 启动轮次时触发 commit，并记录轮次语义（这是 L0 专属，
   不属于 core/threshold_commit.py 的职责）。
3. 后续轮次幂等复用已 commit 的阈值。

为什么保留这一层薄包装：
- "轮次生命周期"是真实的 L0 语义，core 版本不知道"轮次"概念。
"""
from typing import Optional
from core.threshold_commit import (
    ThresholdCommitter, ThresholdDerivation, CommittedThreshold,
)
from core.config import FrozenThresholds
from core.event_log import EventLog, GLOBAL_EVENT_LOG


class L0ThresholdLifecycle:
    """L0 阈值生命周期管理（唯一保留的薄包装）。"""

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        # 委托 core 版本做实际 commit
        self._committer = ThresholdCommitter(self.event_log)
        self._round_idx = 0

    def commit_for_round(self,
                         round_idx: int,
                         thresholds: FrozenThresholds,
                         derivation: ThresholdDerivation,
                         triggered_by: str) -> CommittedThreshold:
        """
        在指定轮次 commit 阈值。
        记录 L0 治理事件：哪个轮次、谁触发、推导依据。
        """
        self._round_idx = round_idx
        committed = self._committer.commit(
            thresholds, derivation,
            governance_context={
                "round_idx": round_idx,
                "triggered_by": triggered_by,
            },
        )
        return committed

    def get_committed(self) -> CommittedThreshold:
        return self._committer.get()

    def is_committed(self) -> bool:
        return self._committer.is_committed()

    def verify(self) -> bool:
        return self._committer.verify()