"""
递归规范化哈希（v3.1 修正⑥：覆盖嵌套结构）。

关键设计：
1. 递归规范化任意嵌套结构（dict / list / 标量）。
2. dict 的键排序，确保顺序不影响哈希。
3. list 保持顺序（顺序有意义）。
4. 标量按类型 + 值编码。
5. 输出规范化 JSON 字符串，再 sha256。

为什么不用 json.dumps(sort_keys=True)：
- sort_keys=True 只排序顶层 dict 的键。
- 嵌套 dict 的键仍按插入顺序。
- 两个"语义相同但嵌套顺序不同"的不变核心会产生不同哈希。
- 递归规范化确保"语义相同 → 哈希相同"。
"""
import hashlib
import json
from typing import Any


def canonicalize(obj: Any) -> Any:
    """
    递归规范化。

    - dict → 键排序后的 dict
    - list → 保持顺序，逐元素规范化
    - 标量 → 原样返回
    """
    if isinstance(obj, dict):
        return {
            k: canonicalize(v)
            for k, v in sorted(obj.items())
        }
    elif isinstance(obj, list):
        return [canonicalize(v) for v in obj]
    elif isinstance(obj, tuple):
        return [canonicalize(v) for v in obj]
    else:
        return obj


def canonical_json(obj: Any) -> str:
    """
    规范化 JSON 字符串。

    - 递归规范化。
    - 排序键。
    - 无空格。
    """
    canonical = canonicalize(obj)
    return json.dumps(
        canonical, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"),
    )


def compute_hash(obj: Any) -> str:
    """
    计算内容的 sha256 哈希。

    递归规范化后 sha256。
    """
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def compute_hash_with_prefix(obj: Any, prefix: str = "") -> str:
    """
    带前缀的哈希（用于版本锚定）。

    格式："{prefix}:{hash}"
    """
    h = compute_hash(obj)
    return f"{prefix}:{h}" if prefix else h