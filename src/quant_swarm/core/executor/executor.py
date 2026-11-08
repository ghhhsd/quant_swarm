"""
执行器门面（统一接口）。

关键设计：
1. 统一接口：execute(code, env_globals) -> ExecutorResult。
2. 子进程模式 / Docker 模式可切换。
3. 执行前不重复做 DSL 检查（DSL 由 L4 沙箱负责）。
4. 失败归为 BACKTEST。
"""
from typing import Optional, Dict, Any
from enum import Enum

from core.executor.result import ExecutorResult
from core.executor.subprocess_executor import SubprocessExecutor
from core.executor.docker_executor import DockerExecutor


class ExecutorMode(str, Enum):
    SUBPROCESS = "subprocess"
    DOCKER = "docker"


class Executor:
    """
    执行器门面。

    构造时选择模式，运行期统一接口。
    """

    def __init__(self,
                 mode: ExecutorMode = ExecutorMode.SUBPROCESS,
                 timeout_s: float = 30.0,
                 max_memory_mb: int = 4096,
                 **kwargs):
        self.mode = mode
        if mode == ExecutorMode.SUBPROCESS:
            self._impl = SubprocessExecutor(
                timeout_s=timeout_s,
                max_memory_mb=max_memory_mb,
            )
        elif mode == ExecutorMode.DOCKER:
            self._impl = DockerExecutor(
                timeout_s=timeout_s,
                max_memory_mb=max_memory_mb,
                **kwargs,
            )
        else:
            raise ValueError(f"未知执行模式: {mode}")

    def execute(self, code: str,
                env_globals: Optional[Dict[str, Any]] = None
                ) -> ExecutorResult:
        """执行表达式。"""
        return self._impl.execute(code, env_globals)