"""
EventLog（v3.1 修正⑥：append-only + 哈希链 + 并发安全）。

关键设计：
1. append-only：不提供删除/修改接口。
2. 哈希链：每个事件携带 prev_hash，形成链。
3. 并发安全：内存锁 + 文件锁。
4. 链完整性校验：verify_chain()。
5. 按类型/时间查询：query()。
6. 可持久化：写入 JSONL 文件。
"""
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import List, Optional

from core.event_log.event import Event
from core.event_log.chain_verifier import ChainVerifier
from core.event_log.query import EventQuery
from core.event_log.concurrency import EventLogLock


class EventLog:
    """
    不可变事件日志。

    - append-only。
    - 哈希链。
    - 并发安全。
    """

    def __init__(self, path: str = "events.jsonl"):
        self.path = Path(path)
        self.events: List[Event] = []
        self._last_hash = "GENESIS"
        self._last_timestamp = 0.0
        self._memory_lock = threading.Lock()

        # 加载已有事件
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        """从文件加载事件。"""
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    event = Event.from_json(line)
                    self.events.append(event)
                    self._last_hash = event.content_hash
                    self._last_timestamp = event.timestamp

    def append(self, event_type: str,
               payload: dict) -> Event:
        """
        追加事件。

        - 生成 content_hash。
        - 写入文件（加锁）。
        - 返回 Event（frozen）。
        """
        with EventLogLock(str(self.path), self._memory_lock) as f:
            # 1) 构造事件（不含 content_hash）
            timestamp = time.time()
            prev_hash = self._last_hash
            event = Event(
                event_type=event_type,
                payload=payload,
                timestamp=timestamp,
                prev_hash=prev_hash,
                content_hash="",
            )

            # 2) 计算 content_hash
            content_hash = ChainVerifier.compute_hash(event)
            # frozen dataclass 用 object.__setattr__ 绕过
            object.__setattr__(event, "content_hash", content_hash)

            # 3) 更新状态
            self.events.append(event)
            self._last_hash = content_hash
            self._last_timestamp = timestamp

            # 4) 写入文件
            f.write(event.to_json() + "\n")
            f.flush()
            # 确保写入磁盘（防崩溃丢数据）
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

        return event

    def query(self, event_type: Optional[str] = None,
              start_time: Optional[float] = None,
              end_time: Optional[float] = None) -> List[Event]:
        """查询事件（委托给 EventQuery）。"""
        with self._memory_lock:
            return EventQuery.query(
                self.events, event_type, start_time, end_time,
            )

    def query_latest(self, event_type: str) -> Optional[Event]:
        """查询最新的指定类型事件。"""
        with self._memory_lock:
            return EventQuery.query_latest(self.events, event_type)

    def verify_chain(self) -> List[str]:
        """校验链完整性。"""
        with self._memory_lock:
            return ChainVerifier.verify(self.events)

    def verify_chain_from_file(self) -> List[str]:
        """从文件校验链完整性（不依赖内存）。"""
        return ChainVerifier.verify_from_file(str(self.path))

    def size(self) -> int:
        """事件数量。"""
        with self._memory_lock:
            return len(self.events)

    def get_last_hash(self) -> str:
        """最后一条事件的 content_hash。"""
        with self._memory_lock:
            return self._last_hash

    def get_events(self) -> List[Event]:
        """所有事件（副本）。"""
        with self._memory_lock:
            return list(self.events)


# 导入 os（在 append 中用）
import os


# =============================================================================
# 全局单例
# =============================================================================

GLOBAL_EVENT_LOG = EventLog()