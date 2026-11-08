"""
版本锚定（v3.1 修正⑥：版本号 + 哈希）。

关键设计：
1. 版本号（人工可见，如 "ic-1.0.0"）。
2. 内容哈希（机器可校验）。
3. 双重锚定：版本号变化必须伴随哈希变化，反之亦然。
4. 提供 verify_version_hash() 校验两者一致。
"""
from dataclasses import dataclass
from typing import Any, Optional
from core.invariant.hash import compute_hash


@dataclass(frozen=True)
class InvariantVersion:
    """
    不变核心版本锚定。

    - version：版本号（如 "ic-1.0.0"）
    - content_hash：内容哈希
    """
    version: str
    content_hash: str

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "content_hash": self.content_hash,
        }

    def verify(self, core: dict) -> bool:
        """
        校验：
        1. core 的 version 字段与 self.version 一致。
        2. core 的哈希与 self.content_hash 一致。
        """
        if core.get("version") != self.version:
            return False
        actual_hash = compute_hash(core)
        return actual_hash == self.content_hash


def create_version(core: dict) -> InvariantVersion:
    """从 core 创建版本锚定。"""
    return InvariantVersion(
        version=core.get("version", "unknown"),
        content_hash=compute_hash(core),
    )