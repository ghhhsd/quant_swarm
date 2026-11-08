"""
core/threshold_commit.py — 冻结阈值 commit（唯一实现，v3.1）。

职责：
1. commit 不是"提交数字"，而是"提交数字 + 推导依据"。
2. commit hash = sha256(thresholds + derivation)，防止"事后改推导依据"。
3. 二次 commit 被拒（防事后调参）。
4. verify_integrity() 可校验内存层面未被篡改。

"""
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import hashlib
import json
import time

from core.config import FrozenThresholds
from core.event_log import EventLog, GLOBAL_EVENT_LOG


class ThresholdViolation(Exception):
    """运行期试图修改已 commit 的阈值。"""
    pass


@dataclass(frozen=True)
class ThresholdDerivation:
    """
    阈值推导依据（必须显式声明）。

    禁止"阈值是拍脑袋的但没记录为什么"。
    """
    budget_n: int
    acceptable_fdr: float
    acceptable_dsr: float
    calibration_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CommittedThreshold:
    """
    commit 后的阈值快照。frozen=True，运行期不可改。
    """
    thresholds: FrozenThresholds
    derivation: ThresholdDerivation
    commit_hash: str
    committed_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thresholds": self.thresholds.to_dict(),
            "derivation": self.derivation.to_dict(),
            "commit_hash": self.commit_hash,
            "committed_at": self.committed_at,
        }

    def verify_integrity(self) -> bool:
        """校验 commit_hash 与内容一致（防内存篡改）。"""
        expected = self._compute_hash(self.thresholds, self.derivation)
        return expected == self.commit_hash

    @staticmethod
    def _compute_hash(thresholds: FrozenThresholds,
                      derivation: ThresholdDerivation) -> str:
        payload = {
            "thresholds": thresholds.to_dict(),
            "derivation": derivation.to_dict(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()


class ThresholdCommitter:
    """
    阈值 commit 器。

    使用流程：
    1. 运行前调用 commit(thresholds, derivation)。
    2. 得到 CommittedThreshold。
    3. 运行期只读取，任何修改尝试抛 ThresholdViolation。
    4. 二次 commit 抛 ThresholdViolation。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self._committed: Optional[CommittedThreshold] = None

    def commit(self,
               thresholds: FrozenThresholds,
               derivation: ThresholdDerivation,
               governance_context: Optional[Dict[str, Any]] = None
               ) -> CommittedThreshold:
        """
        事前 commit 阈值 + 推导依据。

        governance_context：可选的治理上下文（供 L0 记录轮次语义）。
        """
        if not thresholds.frozen:
            raise ThresholdViolation("阈值对象必须 frozen=True")
        if self._committed is not None:
            raise ThresholdViolation(
                "阈值已 commit，禁止二次 commit（v3.1 修正①：防事后调整）"
            )

        commit_hash = CommittedThreshold._compute_hash(thresholds, derivation)
        committed = CommittedThreshold(
            thresholds=thresholds,
            derivation=derivation,
            commit_hash=commit_hash,
            committed_at=time.time(),
        )
        self._committed = committed
        payload = committed.to_dict()
        if governance_context:
            payload["governance_context"] = governance_context
        self.event_log.append("threshold_committed", payload)
        return committed

    def get(self) -> CommittedThreshold:
        if self._committed is None:
            raise ThresholdViolation("阈值尚未 commit，运行前必须先 commit()")
        return self._committed

    def is_committed(self) -> bool:
        return self._committed is not None

    def verify(self) -> bool:
        """校验已 commit 的阈值未被篡改（内存层面）。"""
        if self._committed is None:
            return False
        return self._committed.verify_integrity()