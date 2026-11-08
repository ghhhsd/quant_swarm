"""
查询接口（v3.1：按类型/时间查询）。

关键设计：
1. 按类型查询：query_by_type(event_type)。
2. 按时间范围查询：query_by_time_range(start, end)。
3. 组合查询：query(event_type=None, start=None, end=None)。
4. 返回事件列表（frozen，不可变）。
"""
from typing import List, Optional

from core.event_log.event import Event


class EventQuery:
    """
    事件查询器。
    """

    @staticmethod
    def query(events: List[Event],
              event_type: Optional[str] = None,
              start_time: Optional[float] = None,
              end_time: Optional[float] = None) -> List[Event]:
        """
        组合查询。

        event_type：按类型过滤（None 表示不过滤）
        start_time：起始时间（包含）
        end_time：结束时间（包含）
        """
        result = []
        for e in events:
            if event_type is not None and e.event_type != event_type:
                continue
            if start_time is not None and e.timestamp < start_time:
                continue
            if end_time is not None and e.timestamp > end_time:
                continue
            result.append(e)
        return result

    @staticmethod
    def query_by_type(events: List[Event],
                      event_type: str) -> List[Event]:
        """按类型查询。"""
        return [e for e in events if e.event_type == event_type]

    @staticmethod
    def query_by_time_range(events: List[Event],
                            start_time: float,
                            end_time: float) -> List[Event]:
        """按时间范围查询。"""
        return [
            e for e in events
            if start_time <= e.timestamp <= end_time
        ]

    @staticmethod
    def query_latest(events: List[Event],
                     event_type: str) -> Optional[Event]:
        """查询最新的指定类型事件。"""
        filtered = [e for e in events if e.event_type == event_type]
        return filtered[-1] if filtered else None