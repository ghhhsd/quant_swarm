"""
InvariantViolation 异常（v2.0 冻结）。

关键设计：
1. 携带 diff 信息（供审计）。
2. 携带期望哈希和实际哈希（供快速定位）。
3. 异常消息可读。
"""
from typing import Optional
from core.invariant.diff import DiffResult


class InvariantViolation(Exception):
    """
    不变核心被篡改。
    """

    def __init__(self,
                 message: str,
                 expected_hash: Optional[str] = None,
                 actual_hash: Optional[str] = None,
                 diff_result: Optional[DiffResult] = None):
        super().__init__(message)
        self.message = message
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        self.diff_result = diff_result

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "expected_hash": self.expected_hash,
            "actual_hash": self.actual_hash,
            "diff": (self.diff_result.to_dict()
                     if self.diff_result else None),
        }