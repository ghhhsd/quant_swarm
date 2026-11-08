"""
L4 门面：三层组合 + 失败分类。

对外接口：
- check(code) -> CheckResult：只做 AST + 语义检查（快，用于 L2 编码后立即检查）
- execute(code) -> ExecutorResult：完整执行（慢，用于 L5 回测前）

失败分类：
- AST 层失败 → FailureType.SYNTAX
- 语义层失败 → FailureType.BACKTEST
- 执行层失败 → FailureType.BACKTEST
"""
from dataclasses import dataclass, field
from typing import Optional, Set

from main_tier.l4_sandbox.ast_layer import check_ast, ASTCheckResult
from main_tier.l4_sandbox.semantic_layer import SemanticChecker, SemanticCheckResult
from main_tier.l4_sandbox.executor import (
    ExpressionExecutor, ExecutorConfig, ExecutorResult, ExecutorMode,
)
from main_tier.l1_favor.failure import FailureType


@dataclass
class CheckResult:
    ok: bool
    stage: str  # "ast" / "semantic" / "ok"
    reason: Optional[str] = None
    failure_type: Optional[FailureType] = None
    ast_used_fields: Set[str] = field(default_factory=set)
    ast_used_funcs: Set[str] = field(default_factory=set)
    max_depth: int = 0


class SandboxLayer:
    """
    L4 门面。

    两层快检查（check）+ 一层慢执行（execute）。
    """

    def __init__(self,
                 semantic_checker: Optional[SemanticChecker] = None,
                 executor_config: Optional[ExecutorConfig] = None):
        self.semantic_checker = semantic_checker or SemanticChecker()
        self.executor = ExpressionExecutor(
            executor_config or ExecutorConfig(mode=ExecutorMode.SUBPROCESS)
        )

    # ---- 快检查（AST + 语义）-------------------------------------------
    def check(self, code: str) -> CheckResult:
        """
        两层快检查。不执行代码。
        """
        # 第 1 层：AST 合规
        ast_result = check_ast(code)
        if not ast_result.ok:
            return CheckResult(
                ok=False, stage="ast", reason=ast_result.reason,
                failure_type=FailureType.SYNTAX,
                max_depth=ast_result.max_depth,
            )

        # 第 2 层：语义预检
        sem_result = self.semantic_checker.check(code, ast_result)
        if not sem_result.ok:
            return CheckResult(
                ok=False, stage="semantic", reason=sem_result.reason,
                failure_type=FailureType.BACKTEST,
                ast_used_fields=ast_result.used_fields,
                ast_used_funcs=ast_result.used_funcs,
                max_depth=ast_result.max_depth,
            )

        return CheckResult(
            ok=True, stage="ok",
            ast_used_fields=ast_result.used_fields,
            ast_used_funcs=ast_result.used_funcs,
            max_depth=ast_result.max_depth,
        )

    # ---- 慢执行（完整）--------------------------------------------------
    def execute(self, code: str) -> ExecutorResult:
        """
        完整执行。先做快检查，通过后才执行。
        """
        check = self.check(code)
        if not check.ok:
            return ExecutorResult(
                ok=False, reason=f"[{check.stage}] {check.reason}",
            )
        return self.executor.execute(code)

    # ---- 指标 -----------------------------------------------------------
    def security_metrics(self):
        from main_tier.l4_sandbox.metrics import compute_metrics
        return compute_metrics(self)