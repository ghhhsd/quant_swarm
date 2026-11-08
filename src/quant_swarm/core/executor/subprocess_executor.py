"""
子进程模式（第 2 层：超时 + 资源限制）。

关键设计：
1. 子进程 + signal.alarm：超时兜底。
2. 资源限制（Unix）：RLIMIT_CPU / RLIMIT_AS。
3. 空 builtins + 白名单函数：环境最小化。
4. 失败归为 BACKTEST（运行时异常）。
5. 不宣称"绝对安全"——子进程仍可访问文件系统（除非在容器内）。

局限：
- 子进程模式不是"安全边界"——代码仍可访问文件系统/网络。
- 生产环境必须用 Docker 模式。
- 子进程模式只用于开发/测试。
"""
import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Optional, Dict, Any

from core.executor.result import ExecutorResult
from main_tier.l1_favor.failure import FailureType


class SubprocessExecutor:
    """
    子进程执行器（开发/测试用）。

    生产环境请用 DockerExecutor。
    """

    mode = "subprocess"

    def __init__(self,
                 timeout_s: float = 30.0,
                 max_memory_mb: int = 4096,
                 max_cpu_seconds: int = 60):
        self.timeout_s = timeout_s
        self.max_memory_mb = max_memory_mb
        self.max_cpu_seconds = max_cpu_seconds

    def execute(self, code: str,
                env_globals: Optional[Dict[str, Any]] = None) -> ExecutorResult:
        """
        执行表达式。

        code：因子表达式（已通过 DSL 三层检查）
        env_globals：受限环境（来自 RestrictedEnv.build_globals()）
        """
        if env_globals is None:
            return ExecutorResult(
                ok=False, mode=self.mode,
                reason="env_globals 不能为空（必须先通过 RestrictedEnv）",
                failure_type=FailureType.BACKTEST,
            )

        # 构造 wrapper：子进程 + 超时 + 资源限制
        wrapper = self._build_wrapper(code, env_globals)
        with tempfile.NamedTemporaryFile("w", suffix=".py",
                                         delete=False) as f:
            f.write(wrapper)
            path = f.name

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                timeout=self.timeout_s + 2,
            )
            duration = time.time() - start
            if proc.returncode != 0:
                stderr = proc.stderr.decode()[:500]
                return ExecutorResult(
                    ok=False, mode=self.mode,
                    reason=f"执行失败: {stderr}",
                    failure_type=FailureType.BACKTEST,
                    stderr=stderr,
                    duration_s=duration,
                )
            return ExecutorResult(
                ok=True, mode=self.mode,
                stdout=proc.stdout.decode()[:500],
                duration_s=duration,
            )
        except subprocess.TimeoutExpired:
            return ExecutorResult(
                ok=False, mode=self.mode,
                reason=f"执行超时（{self.timeout_s}s）",
                failure_type=FailureType.BACKTEST,
                duration_s=time.time() - start,
                timed_out=True,
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _build_wrapper(self, code: str,
                       env_globals: Dict[str, Any]) -> str:
        """
        构造 wrapper：子进程 + 超时 + 资源限制。
        """
        # 把 env_globals 的 key 序列化为字符串，子进程重建
        # 注意：env_globals 里的函数不能跨进程传递，子进程需要重新构建
        # 所以这里只传 code，子进程从受限环境模块 import
        wrapper = f"""
import signal
import sys

# 资源限制（Unix）
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU,
                       ({self.max_cpu_seconds}, {self.max_cpu_seconds}))
    mem_bytes = {self.max_memory_mb} * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
except (ImportError, ValueError):
    pass

# 超时
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(
    TimeoutError("execution timeout")))
signal.alarm({int(self.timeout_s)})

# 受限环境（子进程重建）
from core.executor.env import _ts_rolling, _rank_last, _cross_rank, _cross_zscore
import numpy as np

env = {{
    "__builtins__": {{}},
    "ts_mean": lambda x, w: _ts_rolling(x, w, np.mean),
    "ts_std": lambda x, w: _ts_rolling(x, w, np.std),
    "ts_sum": lambda x, w: _ts_rolling(x, w, np.sum),
    "ts_max": lambda x, w: _ts_rolling(x, w, np.max),
    "ts_min": lambda x, w: _ts_rolling(x, w, np.min),
    "ts_rank": lambda x, w: _ts_rolling(x, w, _rank_last),
    "abs": np.abs,
    "log": lambda x: np.log(np.abs(x) + 1e-9),
    "sqrt": lambda x: np.sqrt(np.abs(x)),
    "sign": np.sign,
    "clip": lambda x, lo, hi: np.clip(x, lo, hi),
    "rank": _cross_rank,
    "zscore": _cross_zscore,
}}

try:
    result = eval({code!r}, env, {{}})
    print("EXEC_OK")
except Exception as e:
    print(f"EXEC_ERR:{{type(e).__name__}}:{{e}}")
    sys.exit(1)
"""
        return wrapper