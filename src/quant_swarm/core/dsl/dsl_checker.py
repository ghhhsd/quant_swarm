"""
DSL 检查门面（三层组合）。

关键设计：
1. 三层顺序执行：AST → 语义 → 契约。
2. 任一层失败即返回，但带 stage 标记哪层失败。
3. 失败时带 failure_type，供 L1 自愈分流：
   - AST 失败 → SYNTAX
   - 语义失败 → BACKTEST
   - 契约失败 → DATA
4. 纯函数：不依赖外部状态。
"""
from typing import Optional

from core.data import DEFAULT_DATA_CONTRACT
from main_tier.l1_favor.failure import FailureType

from core.dsl.ast_layer import check_ast
from core.dsl.semantic_layer import SemanticChecker, SemanticConfig
from core.dsl.contract_layer import ContractChecker
from core.dsl.result import DSLCheckResult


class DSLChecker:
    """
    DSL 检查器（三层组合）。

    构造时注入数据契约，运行期纯函数。
    """

    def __init__(self, contract=None,
                 semantic_config: Optional[SemanticConfig] = None):
        self.contract = contract or DEFAULT_DATA_CONTRACT
        self.field_types = {
            f.name: f.dtype for f in self.contract.get_data_fields()
        }
        self.allowed_fields = set(self.field_types.keys())
        self.semantic = SemanticChecker(semantic_config)
        self.contract_checker = ContractChecker(self.field_types)

    def check(self, code: str) -> DSLCheckResult:
        """
        三层检查。

        返回 DSLCheckResult，失败时 stage 标记哪层失败。
        """
        # 1) AST 层
        ast_result = check_ast(code, self.allowed_fields)
        if not ast_result.ok:
            return DSLCheckResult(
                ok=False, stage="ast",
                failure_type=FailureType.SYNTAX,
                reason=ast_result.reason,
                ast_result=ast_result,
            )

        # 2) 语义层
        sem_result = self.semantic.check(code, ast_result)
        if not sem_result.ok:
            return DSLCheckResult(
                ok=False, stage="semantic",
                failure_type=FailureType.BACKTEST,
                reason=sem_result.reason,
                ast_result=ast_result,
                semantic_result=sem_result,
            )

        # 3) 契约层
        contract_result = self.contract_checker.check(
            code, ast_result.used_fields,
        )
        if not contract_result.ok:
            return DSLCheckResult(
                ok=False, stage="contract",
                failure_type=FailureType.DATA,
                reason=contract_result.reason,
                ast_result=ast_result,
                semantic_result=sem_result,
                contract_result=contract_result,
            )

        return DSLCheckResult(
            ok=True, stage="ok",
            ast_result=ast_result,
            semantic_result=sem_result,
            contract_result=contract_result,
            inferred_type=contract_result.inferred_type,
        )