"""
契约对齐（第 3 层：数据契约层）。

关键设计：
1. 检查表达式引用的字段是否在数据契约中。
2. 检查字段类型是否支持所用操作（通过类型推导）。
3. 失败归为 DATA（数据错误）。
4. 不拦"语法错误"（那是 AST 层的职责）。
5. 不拦"计算爆炸"（那是语义层的职责）。
"""
from typing import Dict, Set

from core.dsl.result import ContractCheckResult
from core.dsl.type_inference import TypeInference


class ContractChecker:
    """
    契约对齐检查器。
    """

    def __init__(self, field_types: Dict[str, str]):
        """
        field_types：{字段名: 类型}，来自数据契约。
        """
        self.field_types = field_types
        self.inference = TypeInference(field_types)

    def check(self, code: str,
              ast_used_fields: Set[str] = None) -> ContractCheckResult:
        """
        契约对齐检查。

        code：表达式
        ast_used_fields：AST 层已识别的字段（可选，用于交叉验证）
        """
        # 1) 类型推导（内部会校验字段存在性和类型兼容性）
        inference_result = self.inference.infer(code)
        if not inference_result.ok:
            return ContractCheckResult(
                False, reason=inference_result.reason,
                field_types=dict(self.field_types),
            )

        # 2) 交叉校验：AST 层识别的字段应都在契约中
        if ast_used_fields:
            missing = ast_used_fields - set(self.field_types.keys())
            if missing:
                return ContractCheckResult(
                    False,
                    reason=f"AST 识别的字段不在契约中: {missing}",
                    field_types=dict(self.field_types),
                )

        return ContractCheckResult(
            True, inferred_type=inference_result.inferred_type,
            field_types=dict(self.field_types),
        )