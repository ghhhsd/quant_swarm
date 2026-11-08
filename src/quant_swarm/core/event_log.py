"""不可变事件日志（append-only，带哈希链）。"""
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class Event:
    event_type: str
    payload: Dict[str, Any]
    timestamp: float
    prev_hash: str
    content_hash: str = ""

class EventLog:
    def __init__(self, path: str = "events.jsonl"):
        self.path = Path(path)
        self.events: List[Event] = []
        self._last_hash = "GENESIS"
        if self.path.exists():
            self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                ev = Event(**d)
                self.events.append(ev)
                self._last_hash = ev.content_hash

    def append(self, event_type: str, payload: Dict[str, Any]) -> Event:
        ev = Event(
            event_type=event_type,
            payload=payload,
            timestamp=time.time(),
            prev_hash=self._last_hash,
            content_hash="",
        )
        h = hashlib.sha256(
            json.dumps(asdict(ev), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        ev.content_hash = h
        self._last_hash = h
        self.events.append(ev)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        return ev

    def query(self, event_type: Optional[str] = None) -> List[Event]:
        if event_type is None:
            return self.events
        return [e for e in self.events if e.event_type == event_type]

GLOBAL_EVENT_LOG = EventLog()