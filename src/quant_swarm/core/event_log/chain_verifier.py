"""
链完整性校验（v3.1 修正⑥：从第一条重放验证）。

关键设计：
1. 从第一条开始重放，验证每条 prev_hash 是否等于前一条 content_hash。
2. 验证每条 content_hash 是否等于"重新计算的哈希"。
3. 返回错误列表（空表示完整）。
4. 可检测：
   - 单条篡改（content_hash 不匹配）
   - 链断裂（prev_hash 不匹配）
   - 顺序错乱（timestamp 倒序）
"""
import hashlib
import json
from typing import List

from core.event_log.event import Event


class ChainVerifier:
    """
    链完整性校验器。
    """

    @staticmethod
    def compute_hash(event: Event) -> str:
        """重新计算事件的 content_hash。"""
        # 注意：content_hash 本身不参与哈希计算
        payload = {
            "event_type": event.event_type,
            "payload": event.payload,
            "timestamp": event.timestamp,
            "prev_hash": event.prev_hash,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    @classmethod
    def verify(cls, events: List[Event]) -> List[str]:
        """
        校验事件链。

        返回错误列表（空表示完整）。
        """
        errors = []
        prev_content_hash = "GENESIS"
        prev_timestamp = 0.0

        for i, event in enumerate(events):
            # 1) prev_hash 校验
            if event.prev_hash != prev_content_hash:
                errors.append(
                    f"事件 {i} ({event.event_type}): "
                    f"prev_hash 不匹配（期望 {prev_content_hash[:12]}，"
                    f"实际 {event.prev_hash[:12]}）"
                )

            # 2) content_hash 校验
            computed = cls.compute_hash(event)
            if event.content_hash != computed:
                errors.append(
                    f"事件 {i} ({event.event_type}): "
                    f"content_hash 不匹配（期望 {computed[:12]}，"
                    f"实际 {event.content_hash[:12]}）"
                )

            # 3) 时间顺序校验
            if event.timestamp < prev_timestamp:
                errors.append(
                    f"事件 {i} ({event.event_type}): "
                    f"timestamp 倒序（{event.timestamp} < {prev_timestamp}）"
                )

            prev_content_hash = event.content_hash
            prev_timestamp = event.timestamp

        return errors

    @classmethod
    def verify_from_file(cls, path: str) -> List[str]:
        """从文件读取并校验链。"""
        events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(Event.from_json(line))
        return cls.verify(events)