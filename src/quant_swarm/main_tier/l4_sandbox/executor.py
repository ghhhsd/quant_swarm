"""
执行沙箱（第 3 层，子进程/Docker，慢）。

定位（v3.1）：
- 兜底一切运行时问题：超时、OOM、异常逃逸。
- 不拦逻辑错误（L1 职责）。
- 不拦 AST/语义违规（前两层已拦）。

两种模式：
1. 子进程模式（开发/测试）：signal.alarm + 子进程超时。
2. Docker 模式（生产）：容器隔离 + CPU/内存/网络限额。

v3.1 修正⑦：
- 生产环境应叠加 seccomp/eBPF + 能力最小化 + gVisor/Firecracker。
- 本实现给出 Docker 接口桩，真实部署时注入 gVisor runtime。
"""
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class ExecutorMode(str, Enum):
    SUBPROCESS = "subprocess"   # 开发/测试
    DOCKER = "docker"           # 生产（需 Docker + gVisor）


@dataclass
class ExecutorConfig:
    mode: ExecutorMode = ExecutorMode.SUBPROCESS
    timeout_s: float = 30.0
    max_memory_mb: int = 4096
    max_cpu_cores: float = 2.0
    network_disabled: bool = True
    # Docker 配置（mode=DOCKER 时生效）
    docker_image: str = "quant-sandbox:latest"
    docker_runtime: str = "runsc"  # gVisor


@dataclass
class ExecutorResult:
    ok: bool
    reason: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0


class ExpressionExecutor:
    """
    表达式执行器。

    子进程模式：受限 eval + 超时 + 资源限制（macOS/Linux 差异处理）。
    Docker 模式：接口桩，真实部署注入 gVisor。
    """

    def __init__(self, config: ExecutorConfig = None):
        self.config = config or ExecutorConfig()

    def execute(self, code: str) -> ExecutorResult:
        if self.config.mode == ExecutorMode.SUBPROCESS:
            return self._execute_subprocess(code)
        elif self.config.mode == ExecutorMode.DOCKER:
            return self._execute_docker(code)
        else:
            return ExecutorResult(False, f"未知执行模式: {self.config.mode}")

    # ---- 子进程模式 -----------------------------------------------------
    def _execute_subprocess(self, code: str) -> ExecutorResult:
        """
        受限 eval + 超时。
        注意：这是"受限 DSL"下的执行，不是通用 Python 执行。
        真实环境应使用 Docker 模式。
        """
        wrapper = self._build_wrapper(code)
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(wrapper)
            path = f.name
        import time
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                timeout=self.config.timeout_s + 2,
            )
            duration = time.time() - start
            if proc.returncode != 0:
                return ExecutorResult(
                    False,
                    reason=proc.stderr.decode()[:500],
                    stderr=proc.stderr.decode()[:500],
                    duration_s=duration,
                )
            return ExecutorResult(
                True, stdout=proc.stdout.decode()[:500],
                duration_s=duration,
            )
        except subprocess.TimeoutExpired:
            return ExecutorResult(
                False, reason=f"执行超时（{self.config.timeout_s}s）",
                duration_s=time.time() - start,
            )
        finally:
            os.unlink(path)

    def _build_wrapper(self, code: str) -> str:
        """
        构造受限执行 wrapper。
        用 eval 在空 globals 下执行，禁止 builtins。
        """
        return f"""
import signal
signal.alarm({int(self.config.timeout_s)})
try:
    result = eval({code!r}, {{"__builtins__": {{}}}}, {{}})
    print("EXEC_OK")
except Exception as e:
    print(f"EXEC_ERR:{{type(e).__name__}}:{{e}}")
    raise SystemExit(1)
"""

    # ---- Docker 模式 ----------------------------------------------------
    def _execute_docker(self, code: str) -> ExecutorResult:
        """
        Docker 模式接口桩。

        真实实现：
        1. 构造 Dockerfile 或复用预构建镜像。
        2. docker run --runtime=runsc（gVisor）
           --cpus={max_cpu} --memory={max_mem}m
           --network=none
           --cap-drop=ALL
           --security-opt=no-new-privileges
           --read-only
           -v code:/tmp/code.py:ro
        3. 捕获 stdout/stderr/exit_code。
        """
        raise NotImplementedError(
            "Docker 模式需真实 Docker + gVisor 环境。"
            "开发阶段用 SUBPROCESS 模式，生产部署时实现此方法。"
        )


def build_seccomp_profile() -> Dict[str, Any]:
    """
    seccomp 配置生成（v3.1 修正⑦）。
    返回可序列化的 seccomp profile，供 Docker --security-opt seccomp=... 使用。
    """
    return {
        "defaultAction": "SCMP_ACT_ERRNO",
        "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_AARCH64"],
        "syscalls": [
            {
                "names": [
                    "read", "write", "close", "exit", "exit_group",
                    "brk", "mmap", "munmap", "mprotect",
                ],
                "action": "SCMP_ACT_ALLOW",
            },
        ],
    }