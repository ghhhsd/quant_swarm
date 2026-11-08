"""
递归 diff（v2.0 冻结：含列表）。

关键设计：
1. 递归比较 dict / list / 标量。
2. dict：报告 missing / extra / changed。
3. list：报告 missing / extra / changed（按元素比较）。
4. 标量：报告类型不匹配或值不匹配。
5. 输出结构化 diff（供审计）。
"""
from typing import Any, Dict


class DiffResult:
    """
    Diff 结果。

    - missing：期望有但实际没有
    - extra：实际有但期望没有
    - changed：都有但不同
    - type_mismatch：类型不匹配
    """

    def __init__(self):
        self.missing: list = []
        self.extra: list = []
        self.changed: Dict[str, dict] = {}
        self.type_mismatch: Dict[str, dict] = {}

    def is_empty(self) -> bool:
        return (not self.missing and not self.extra
                and not self.changed and not self.type_mismatch)

    def to_dict(self) -> dict:
        return {
            "missing": self.missing,
            "extra": self.extra,
            "changed": self.changed,
            "type_mismatch": self.type_mismatch,
        }

    def __str__(self):
        parts = []
        if self.missing:
            parts.append(f"missing={self.missing}")
        if self.extra:
            parts.append(f"extra={self.extra}")
        if self.changed:
            parts.append(f"changed={list(self.changed.keys())}")
        if self.type_mismatch:
            parts.append(f"type_mismatch={list(self.type_mismatch.keys())}")
        return "; ".join(parts) if parts else "no diff"


def diff(expected: Any, actual: Any,
         prefix: str = "") -> DiffResult:
    """
    递归 diff。

    expected：期望值（不变核心）
    actual：实际值（候选产物）
    prefix：路径前缀（用于嵌套）
    """
    result = DiffResult()
    _diff_recursive(expected, actual, prefix, result)
    return result


def _diff_recursive(expected: Any, actual: Any,
                    prefix: str, result: DiffResult) -> None:
    """递归 diff 实现。"""
    path = prefix or "<root>"

    # 1) 类型检查
    if type(expected) != type(actual):
        # 特殊处理：int 和 float 视为兼容（如 0.20 vs 0）
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            pass  # 继续比较值
        else:
            result.type_mismatch[path] = {
                "expected_type": type(expected).__name__,
                "actual_type": type(actual).__name__,
                "expected_value": expected,
                "actual_value": actual,
            }
            return

    # 2) dict
    if isinstance(expected, dict):
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        for k in expected_keys - actual_keys:
            result.missing.append(f"{prefix}.{k}" if prefix else k)
        for k in actual_keys - expected_keys:
            result.extra.append(f"{prefix}.{k}" if prefix else k)
        for k in expected_keys & actual_keys:
            _diff_recursive(
                expected[k], actual[k],
                f"{prefix}.{k}" if prefix else k,
                result,
            )
        return

    # 3) list / tuple
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            result.changed[path] = {
                "reason": "length_mismatch",
                "expected_len": len(expected),
                "actual_len": len(actual),
                "missing": [e for e in expected if e not in actual],
                "extra": [a for a in actual if a not in expected],
            }
            return
        # 逐元素比较
        for i, (e, a) in enumerate(zip(expected, actual)):
            _diff_recursive(e, a, f"{prefix}[{i}]", result)
        return

    # 4) 标量
    if expected != actual:
        result.changed[path] = {
            "expected": expected,
            "actual": actual,
        }