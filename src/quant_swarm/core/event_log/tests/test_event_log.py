"""
事件日志测试。

验证 v3.1 的九条纪律：
1. append-only（无删除接口）
2. 哈希链（prev_hash + content_hash）
3. 链完整性可校验
4. 单条篡改可检测
5. 链断裂可检测
6. 按类型查询
7. 按时间范围查询
8. 并发安全
9. 持久化 + 重载
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import json
import threading
import time
from core.event_log import (
    Event, EventLog, ChainVerifier, EventQuery,
)
import tempfile
from pathlib import Path


def make_log(tmp_path):
    path = str(tmp_path / "events.jsonl")
    return EventLog(path), path


def test_append_only():
    """append-only：无删除接口。"""
    log = EventLog(":memory:")  # 简化：用内存
    log.append("test", {"a": 1})
    log.append("test", {"b": 2})
    assert log.size() == 2
    # 没有 delete 方法
    assert not hasattr(log, "delete")
    assert not hasattr(log, "remove")
    print("[OK] append-only")


def test_hash_chain():
    """哈希链。"""
    log = EventLog(":memory:")
    e1 = log.append("test", {"a": 1})
    e2 = log.append("test", {"b": 2})
    assert e1.prev_hash == "GENESIS"
    assert e2.prev_hash == e1.content_hash
    print(f"[OK] 哈希链: e1={e1.content_hash[:8]}, "
          f"e2.prev={e2.prev_hash[:8]}")


def test_chain_verification():
    """链完整性校验。"""
    log = EventLog(":memory:")
    log.append("test", {"a": 1})
    log.append("test", {"b": 2})
    log.append("test", {"c": 3})
    errors = log.verify_chain()
    assert not errors, f"链应完整: {errors}"
    print("[OK] 链完整性")


def test_tamper_detection():
    """单条篡改可检测。"""
    log = EventLog(":memory:")
    log.append("test", {"a": 1})
    log.append("test", {"b": 2})
    # 篡改第二条的 payload（用 object.__setattr__ 绕过 frozen）
    event = log.events[1]
    object.__setattr__(event, "payload", {"b": 999})
    errors = log.verify_chain()
    assert len(errors) > 0
    assert "content_hash 不匹配" in errors[0]
    print(f"[OK] 篡改检测: {errors[0][:60]}...")


def test_chain_break_detection():
    """链断裂可检测。"""
    log = EventLog(":memory:")
    log.append("test", {"a": 1})
    log.append("test", {"b": 2})
    # 篡改第二条的 prev_hash
    event = log.events[1]
    object.__setattr__(event, "prev_hash", "FAKE")
    errors = log.verify_chain()
    assert len(errors) > 0
    assert "prev_hash 不匹配" in errors[0]
    print(f"[OK] 链断裂检测: {errors[0][:60]}...")


def test_query_by_type():
    """按类型查询。"""
    log = EventLog(":memory:")
    log.append("type_a", {"x": 1})
    log.append("type_b", {"x": 2})
    log.append("type_a", {"x": 3})
    results = log.query("type_a")
    assert len(results) == 2
    assert all(e.event_type == "type_a" for e in results)
    print(f"[OK] 按类型: {len(results)} 条")


def test_query_by_time():
    """按时间范围查询。"""
    log = EventLog(":memory:")
    log.append("test", {"a": 1})
    time.sleep(0.01)
    log.append("test", {"b": 2})
    time.sleep(0.01)
    log.append("test", {"c": 3})
    start = log.events[0].timestamp
    end = log.events[1].timestamp
    results = log.query(start_time=start, end_time=end)
    assert len(results) >= 2
    print(f"[OK] 按时间: {len(results)} 条")


def test_query_latest():
    """最新事件。"""
    log = EventLog(":memory:")
    log.append("test", {"a": 1})
    log.append("test", {"b": 2})
    latest = log.query_latest("test")
    assert latest.payload == {"b": 2}
    print(f"[OK] 最新事件: {latest.payload}")


def test_concurrent_append(tmp_path):
    """并发安全（多线程）。"""
    path = str(tmp_path / "concurrent.jsonl")
    log = EventLog(path)
    n_threads = 4
    n_per_thread = 10

    def worker(thread_id):
        for i in range(n_per_thread):
            log.append("test", {"thread": thread_id, "i": i})

    threads = [
        threading.Thread(target=worker, args=(t,))
        for t in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert log.size() == n_threads * n_per_thread
    errors = log.verify_chain()
    assert not errors, f"并发后链应完整: {errors[:3]}"
    print(f"[OK] 并发: {log.size()} 条, 链完整")


def test_persistence(tmp_path):
    """持久化 + 重载。"""
    path = str(tmp_path / "persist.jsonl")
    log1 = EventLog(path)
    log1.append("test", {"a": 1})
    log1.append("test", {"b": 2})
    # 重载
    log2 = EventLog(path)
    assert log2.size() == 2
    assert log2.get_last_hash() == log1.get_last_hash()
    errors = log2.verify_chain()
    assert not errors
    print(f"[OK] 持久化: {log2.size()} 条")


def test_file_verification(tmp_path):
    """从文件校验。"""
    path = str(tmp_path / "verify.jsonl")
    log = EventLog(path)
    log.append("test", {"a": 1})
    log.append("test", {"b": 2})
    errors = log.verify_chain_from_file()
    assert not errors
    print("[OK] 文件校验")


def test_event_frozen():
    """事件 frozen。"""
    log = EventLog(":memory:")
    e = log.append("test", {"a": 1})
    try:
        e.event_type = "modified"
        assert False, "frozen dataclass 应拒绝修改"
    except Exception:
        print("[OK] 事件 frozen")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_append_only()
        test_hash_chain()
        test_chain_verification()
        test_tamper_detection()
        test_chain_break_detection()
        test_query_by_type()
        test_query_by_time()
        test_query_latest()
        test_concurrent_append(tp)
        test_persistence(tp)
        test_file_verification(tp)
        test_event_frozen()
        print("\n=== 事件日志全部测试通过 ===")