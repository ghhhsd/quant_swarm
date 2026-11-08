"""
core/invariant.py — 不变核心只读 + 篡改检测（唯一实现，v3.1）。

职责：
1. 深拷贝返回，外部修改不影响注册表。
2. 版本号 + 内容哈希 + 逐键 diff 三重锚定。
3. 篡改时抛 InvariantViolation，带逐键 diff 诊断。
4. 支持可选 governance_context 参数（供 L0 记录治理语义，
   不引入薄包装层）。

"""
import copy
from typing import Dict, Any, Optional
from core.config import (
    INVARIANT_CORE, INVARIANT_CORE_HASH, compute_invariant_hash,
)
from core.event_log import EventLog, GLOBAL_EVENT_LOG


class InvariantViolation(Exception):
    """试图修改不变核心。"""
    pass


class InvariantRegistry:
    """
    不变核心注册表。

    - get() 返回深拷贝。
    - assert_unchanged() 校验候选产物未篡改。
    - diff() 输出逐键差异。
    - 支持 governance_context 参数（可选，用于 L0 记录治理语义）。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self._core = copy.deepcopy(INVARIANT_CORE)
        self._hash = INVARIANT_CORE_HASH

    def get(self, log_access: bool = True,
            governance_context: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
        """
        返回深拷贝。

        governance_context：可选的治理上下文（如 {"requester": "L0",
        "round_idx": 3}）。若提供，写入访问事件供审计。
        """
        if log_access:
            payload = {
                "version": self._core["version"],
                "hash": self._hash,
            }
            if governance_context:
                payload["governance_context"] = governance_context
            self.event_log.append("invariant_accessed", payload)
        return copy.deepcopy(self._core)

    def get_version(self) -> str:
        return self._core["version"]

    def get_hash(self) -> str:
        return self._hash

    def assert_unchanged(self, candidate: Dict[str, Any],
                         governance_context: Optional[Dict[str, Any]] = None
                         ) -> None:
        """
        校验候选产物没有篡改不变核心。

        L3 演化产物、L4 编码产物、L6 技能库写入前应调用此方法。
        governance_context：可选的治理上下文，失败时写入事件流。
        """
        candidate_hash = compute_invariant_hash(candidate)
        if candidate_hash == self._hash:
            return
        diff = self.diff(self._core, candidate)
        # 失败事件（含治理上下文）
        self.event_log.append("invariant_violation", {
            "expected_hash": self._hash,
            "actual_hash": candidate_hash,
            "diff": diff,
            "governance_context": governance_context or {},
        })
        raise InvariantViolation(
            f"不变核心被篡改（hash 期望 {self._hash[:12]}，"
            f"实际 {candidate_hash[:12]}）: {diff}"
        )

    @staticmethod
    def diff(expected: Dict[str, Any],
             actual: Dict[str, Any]) -> Dict[str, Any]:
        """
        逐键 diff。返回：
        {
            "missing": [...],
            "extra": [...],
            "changed": {key: {"expected": ..., "actual": ...}}
        }
        """
        result = {"missing": [], "extra": [], "changed": {}}
        result.update(InvariantRegistry._diff_recursive(expected, actual, ""))
        return result

    @staticmethod
    def _diff_recursive(expected: Any, actual: Any,
                        prefix: str) -> Dict[str, Any]:
        result = {"missing": [], "extra": [], "changed": {}}
        if type(expected) != type(actual):
            result["changed"][prefix] = {
                "expected": expected, "actual": actual,
                "reason": "type mismatch",
            }
            return result
        if isinstance(expected, dict):
            expected_keys = set(expected.keys())
            actual_keys = set(actual.keys())
            for k in expected_keys - actual_keys:
                result["missing"].append(f"{prefix}.{k}" if prefix else k)
            for k in actual_keys - expected_keys:
                result["extra"].append(f"{prefix}.{k}" if prefix else k)
            for k in expected_keys & actual_keys:
                sub = InvariantRegistry._diff_recursive(
                    expected[k], actual[k],
                    f"{prefix}.{k}" if prefix else k,
                )
                result["missing"].extend(sub["missing"])
                result["extra"].extend(sub["extra"])
                result["changed"].update(sub["changed"])
        elif expected != actual:
            result["changed"][prefix] = {
                "expected": expected, "actual": actual,
            }
        return result