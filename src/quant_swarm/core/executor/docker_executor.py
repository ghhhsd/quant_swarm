"""
Docker 模式（第 3 层：容器隔离 + seccomp + gVisor）。

关键设计：
1. Docker 容器隔离：CPU / 内存 / 网络限额。
2. seccomp 过滤：限制系统调用。
3. gVisor / Firecracker：强隔离（即使容器逃逸也碰不到宿主）。
4. 网络禁用：--network=none。
5. 能力最小化：--cap-drop=ALL。

本实现是接口桩，真实部署需实现：
- Dockerfile 构建（含 Python + numpy）
- docker run 参数（--runtime=runsc, --cpus, --memory, --network=none）
- 结果捕获（stdout / stderr / exit_code）
"""
import json
import subprocess
import time
from typing import Optional, Dict, Any

from core.executor.result import ExecutorResult
from main_tier.l1_favor.failure import FailureType


class DockerExecutor:
    """
    Docker 执行器（生产用）。

    接口桩——真实部署需实现 _build_docker_command 和 _parse_result。
    """

    mode = "docker"

    def __init__(self,
                 image: str = "quant-sandbox:latest",
                 runtime: str = "runsc",  # gVisor
                 timeout_s: float = 30.0,
                 max_memory_mb: int = 4096,
                 max_cpu_cores: float = 2.0,
                 network_disabled: bool = True):
        self.image = image
        self.runtime = runtime
        self.timeout_s = timeout_s
        self.max_memory_mb = max_memory_mb
        self.max_cpu_cores = max_cpu_cores
        self.network_disabled = network_disabled

    def execute(self, code: str,
                env_globals: Optional[Dict[str, Any]] = None) -> ExecutorResult:
        """
        执行表达式。

        接口桩：真实部署时实现 docker run 调用。
        """
        # TODO: 真实实现
        # 1. 构造 Dockerfile（含 Python + numpy + 受限环境）
        # 2. docker run --runtime=runsc --cpus={max_cpu_cores}
        #    --memory={max_memory_mb}m --network=none
        #    --cap-drop=ALL --security-opt=no-new-privileges
        #    --read-only -v code:/tmp/code.py:ro
        # 3. 捕获 stdout/stderr/exit_code
        raise NotImplementedError(
            "DockerExecutor.execute() 需实现："
            "docker run --runtime={} --cpus={} --memory={}m "
            "--network={} --cap-drop=ALL ...".format(
                self.runtime, self.max_cpu_cores, self.max_memory_mb,
                "none" if self.network_disabled else "bridge",
            )
        )

    def _build_seccomp_profile(self) -> Dict[str, Any]:
        """
        构造 seccomp profile。

        只允许必要的系统调用（read / write / exit / mmap 等）。
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