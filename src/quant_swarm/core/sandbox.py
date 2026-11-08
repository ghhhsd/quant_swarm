"""受限执行：AST + 语义预检 + 子进程超时。"""
import subprocess, sys, tempfile, os
from dataclasses import dataclass
from typing import Optional
from core.dsl import check_ast, semantic_check

@dataclass
class SandboxResult:
    ok: bool
    reason: Optional[str] = None

def execute_expression(code: str, timeout_s: float = 30.0) -> SandboxResult:
    r = check_ast(code)
    if not r.ok:
        return SandboxResult(False, r.reason)
    r = semantic_check(code)
    if not r.ok:
        return SandboxResult(False, r.reason)
    wrapper = f"""
import signal
signal.alarm({int(timeout_s)})
try:
    result = eval({code!r}, {{"__builtins__": {{}}}}, {{}})
    print("OK")
except Exception as e:
    print(f"ERR:{{e}}")
    raise SystemExit(1)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(wrapper); path = f.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              timeout=timeout_s + 2)
        if proc.returncode != 0:
            return SandboxResult(False, proc.stderr.decode()[:300])
        return SandboxResult(True)
    except subprocess.TimeoutExpired:
        return SandboxResult(False, "执行超时")
    finally:
        os.unlink(path)