"""
Event 数据结构（v3.1 修正⑥：不可变）。

关键设计：
1. frozen=True，构造后不可修改。
2. content_hash 在构造后由 EventLog 赋值（用 object.__setattr__）。
3. 序列化时包含所有字段（含 prev_hash 和 content_hash）。
4. 反序列化时从 JSON 重建。
"""
from dataclasses import dataclass, asdict, field
from typing import Dict, Any
import json


@dataclass(frozen=True)
class Event:
    """
    事件（不可变）。

    - event_type：事件类型（如 "threshold_committed"）
    - payload：事件载荷（任意 JSON 可序列化 dict）
    - timestamp：时间戳（Unix 秒）
    - prev_hash：前一个事件的 content_hash（哈希链）
    - content_hash：本事件的 content_hash（由 EventLog 赋值）
    """
    event_type: str
    payload: Dict[str, Any]
    timestamp: float
    prev_hash: str
    content_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        return cls(
            event_type=d["event_type"],
            payload=d["payload"],
            timestamp=d["timestamp"],
            prev_hash=d["prev_hash"],
            content_hash=d.get("content_hash", ""),
        )

    @classmethod
    def from_json(cls, s: str) -> "Event":
        return cls.from_dict(json.loads(s))