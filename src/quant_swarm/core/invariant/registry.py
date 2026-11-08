"""
InvariantRegistry（v2.0 冻结：只读访问 + 校验）。

关键设计：
1. 深拷贝返回，外部修改不影响注册表。
2. 版本号 + 内容哈希双重锚定。
3. assert_unchanged 先类型检查，再值检查。
4. diff 递归到列表。
5. 访问可审计（可选 governance_context）。
6. 纯函数：不依赖外部状态（除 event_log）。
"""
import copy
from typing import Any, Dict, Optional

from core.config import INVARIANT_CORE
from core.invariant.hash import compute_hash
from core.invariant.diff import diff, DiffResult
from core.invariant.violation import InvariantViolation
from core.invariant.version import InvariantVersion, create_version
from core.event_log import EventLog, GLOBAL_EVENT_LOG


class InvariantRegistry:
    """
    不变核心注册表。

    - 只读访问。
    - 篡改检测。
    - 版本锚定。
    """

    def __init__(self,
                 core: Optional[dict] = None,
                 event_log: Optional[EventLog] = None):
        """
        core：不变核心（默认 INVARIANT_CORE）
        event_log：事件日志
        """
        self._core = copy.deepcopy(core or INVARIANT_CORE)
        self._version = create_version(self._core)
        self._hash = self._version.content_hash
        self.event_log = event_log or GLOBAL_EVENT_LOG

    # ---- 只读访问 -------------------------------------------------------
    def get(self, log_access: bool = True,
            governance_context: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
        """
        返回深拷贝。

        governance_context：可选的治理上下文（如 {"requester": "L0"}）。
        """
        if log_access:
            payload = {
                "version": self._version.version,
                "hash": self._hash,
            }
            if governance_context:
                payload["governance_context"] = governance_context
            self.event_log.append("invariant_accessed", payload)
        return copy.deepcopy(self._core)

    def get_version(self) -> str:
        return self._version.version

    def get_hash(self) -> str:
        return self._hash

    def get_version_info(self) -> InvariantVersion:
        return self._version

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        返回某一部分的深拷贝（如 "risk_breaker"）。
        """
        if section not in self._core:
            raise KeyError(f"不变核心没有 section: {section}")
        return copy.deepcopy(self._core[section])

    # ---- 篡改检测 -------------------------------------------------------
    def assert_unchanged(self, candidate: Dict[str, Any],
                         governance_context: Optional[Dict[str, Any]] = None
                         ) -> None:
        """
        校验候选产物没有篡改不变核心。

        - 先类型检查，再值检查（由 diff 实现）。
        - 失败时抛 InvariantViolation，带 diff 信息。
        """
        candidate_hash = compute_hash(candidate)
        if candidate_hash == self._hash:
            return

        diff_result = diff(self._core, candidate)
        violation = InvariantViolation(
            message=f"不变核心被篡改（期望哈希 {self._hash[:12]}，"
                    f"实际 {candidate_hash[:12]}）",
            expected_hash=self._hash,
            actual_hash=candidate_hash,
            diff_result=diff_result,
        )
        # 失败事件
        payload = violation.to_dict()
        if governance_context:
            payload["governance_context"] = governance_context
        self.event_log.append("invariant_violation", payload)
        raise violation

    def is_unchanged(self, candidate: Dict[str, Any]) -> bool:
        """不抛异常的校验（返回 bool）。"""
        return compute_hash(candidate) == self._hash

    # ---- 版本校验 -------------------------------------------------------
    def verify_version(self) -> bool:
        """
        校验版本号 + 哈希一致。
        """
        return self._version.verify(self._core)