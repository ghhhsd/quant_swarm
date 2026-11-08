"""
执行结果数据结构。

关键设计：
1. 成功/失败都带原因，失败归为 BACKTEST（运行时异常）。
2. 带执行耗时，用于监控。
3. 带模式标识（subprocess / docker），便于审计。
"""
from dataclasses import dataclass, field
from typing import Optional, Any
from main_tier.l1_favor.failure import FailureType


@dataclass
class ExecutorResult:
    """执行结果。"""
    ok: bool
    mode: str  # "subprocess" / "docker"
    reason: Optional[str] = None
    failure_type: Optional[FailureType] = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    return_value: Any = None
    timed_out: bool = False
    oom: bool = False

    def to_dict(self):
        return {
            "ok": self.ok,
            "mode": self.mode,
            "reason": self.reason,
            "failure_type": (self.failure_type.value
                             if self.failure_type else None),
            "duration_s": self.duration_s,
            "timed_out": self.timed_out,
            "oom": self.oom,
            "stdout_len": len(self.stdout),
            "stderr_len": len(self.stderr),
        }