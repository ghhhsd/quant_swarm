"""
并发安全（v3.1：多进程/多线程保护）。

关键设计：
1. 内存锁：threading.Lock，保护 self.events 和 self._last_hash。
2. 文件锁：fcntl.flock（Unix）/ msvcrt.locking（Windows），保护文件写入。
3. 跨平台：Windows 用 msvcrt，Unix 用 fcntl。
4. 锁的粒度：append 操作全程持锁（内存锁 + 文件锁）。
"""
import os
import threading
from contextlib import contextmanager
from typing import Optional

# 跨平台文件锁
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False


class FileLock:
    """
    跨平台文件锁。

    - Unix：fcntl.flock
    - Windows：msvcrt.locking
    - 其他：无锁（仅内存锁）
    """

    def __init__(self, file_handle):
        self.file_handle = file_handle

    def __enter__(self):
        if HAS_FCNTL:
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_EX)
        elif HAS_MSVCRT:
            msvcrt.locking(
                self.file_handle.fileno(), msvcrt.LK_LOCK, 1,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if HAS_FCNTL:
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
        elif HAS_MSVCRT:
            try:
                msvcrt.locking(
                    self.file_handle.fileno(), msvcrt.LK_UNLCK, 1,
                )
            except OSError:
                pass
        return False


class EventLogLock:
    """
    事件日志锁（内存锁 + 文件锁）。

    使用：
    with EventLogLock(path, memory_lock):
        # 写文件
    """

    def __init__(self, path: str, memory_lock: threading.Lock):
        self.path = path
        self.memory_lock = memory_lock

    @contextmanager
    def acquire(self):
        """获取内存锁 + 文件锁。"""
        with self.memory_lock:
            # 打开文件（append 模式）
            with open(self.path, "a", encoding="utf-8") as f:
                with FileLock(f):
                    yield f