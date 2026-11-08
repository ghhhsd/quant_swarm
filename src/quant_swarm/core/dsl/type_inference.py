"""
类型推导（第 3 层：契约层的核心）。

关键设计：
1. 静态类型推导：给定表达式和数据契约，推导表达式的返回类型。
2. 规则：
   - 字段 → 契约声明的类型
   - 字面量 → 推断类型
   - 函数 → 按函数签名推导
   - 二元操作 → 按操作符类型规则推导
   - 比较操作 → 返回 bool
3. 推导失败时返回 None + 错误原因（供契约层报错）。
4. 纯函数：不依赖外部状态。
"""
import ast
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from core.dsl.functions import (
    FUNCTION_SIGNATURES, BINOP_TYPE_RULES, COMPARE_TYPE_RULES,
)


@dataclass
class TypeInferenceResult:
    """类型推导结果。"""
    ok: bool
    inferred_type: Optional[str] = None
    reason: Optional[str] = None
    field_types: Dict[str, str] = None

    def __post_init__(self):
        if self.field_types is None:
            self.field_types = {}


class TypeInference:
    """
    类型推导器。
    """

    def __init__(self, field_types: Dict[str, str]):
        """
        field_types：{字段名: 类型}，来自数据契约。
        """
        self.field_types = field_types

    def infer(self, code: str) -> TypeInferenceResult:
        """推导表达式类型。"""
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError as e:
            return TypeInferenceResult(False, reason=f"语法错误: {e}")

        try:
            inferred = self._infer_node(tree.body)
        except TypeError as e:
            return TypeInferenceResult(
                False, reason=str(e), field_types=dict(self.field_types),
            )
        return TypeInferenceResult(
            True, inferred_type=inferred,
            field_types=dict(self.field_types),
        )

    def _infer_node(self, node) -> str:
        """递归推导节点类型。"""
        # 字面量
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "bool"
            elif isinstance(node.value, int):
                return "int64"
            elif isinstance(node.value, float):
                return "float64"
            elif isinstance(node.value, str):
                return "string"
            else:
                raise TypeError(f"不支持的字面量类型: {type(node.value).__name__}")

        # 字段名
        if isinstance(node, ast.Name):
            if node.id in self.field_types:
                return self.field_types[node.id]
            if node.id in {"True", "False"}:
                return "bool"
            if node.id == "None":
                return "none"
            raise TypeError(f"未知标识符: {node.id}")

        # 一元操作
        if isinstance(node, ast.UnaryOp):
            operand_type = self._infer_node(node.operand)
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                if operand_type not in ("float64", "int64"):
                    raise TypeError(
                        f"一元操作符不支持类型 {operand_type}"
                    )
                return operand_type
            raise TypeError(f"不支持的一元操作符: {type(node.op).__name__}")

        # 二元操作
        if isinstance(node, ast.BinOp):
            left_type = self._infer_node(node.left)
            right_type = self._infer_node(node.right)
            op_name = type(node.op).__name__
            rules = BINOP_TYPE_RULES.get(op_name, {})
            result = rules.get((left_type, right_type))
            if result is None:
                raise TypeError(
                    f"二元操作 {op_name} 不支持类型组合 "
                    f"({left_type}, {right_type})"
                )
            return result

        # 比较操作
        if isinstance(node, ast.Compare):
            left_type = self._infer_node(node.left)
            for comparator in node.comparators:
                right_type = self._infer_node(comparator)
                op_name = type(node.ops[0]).__name__
                rules = COMPARE_TYPE_RULES.get(op_name, {})
                result = rules.get((left_type, right_type))
                if result is None:
                    raise TypeError(
                        f"比较操作 {op_name} 不支持类型组合 "
                        f"({left_type}, {right_type})"
                    )
            return "bool"

        # 布尔操作
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                value_type = self._infer_node(value)
                if value_type != "bool":
                    raise TypeError(
                        f"布尔操作不支持类型 {value_type}"
                    )
            return "bool"

        # 条件表达式
        if isinstance(node, ast.IfExp):
            test_type = self._infer_node(node.test)
            if test_type != "bool":
                raise TypeError(f"条件表达式 test 应为 bool，实际 {test_type}")
            body_type = self._infer_node(node.body)
            orelse_type = self._infer_node(node.orelse)
            if body_type != orelse_type:
                raise TypeError(
                    f"条件表达式分支类型不一致: {body_type} vs {orelse_type}"
                )
            return body_type

        # 函数调用
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise TypeError("函数调用必须是简单名称")
            func_name = node.func.id
            if func_name not in FUNCTION_SIGNATURES:
                raise TypeError(f"未授权函数: {func_name}")
            sig = FUNCTION_SIGNATURES[func_name]
            # 参数个数校验
            n_args = len(node.args)
            if not sig.variadic and n_args != len(sig.param_types):
                raise TypeError(
                    f"函数 {func_name} 需要 {len(sig.param_types)} 个参数，"
                    f"实际 {n_args} 个"
                )
            # 参数类型校验
            for i, arg in enumerate(node.args):
                arg_type = self._infer_node(arg)
                if i < len(sig.param_types):
                    expected = sig.param_types[i]
                    if expected != "any" and not self._type_compatible(
                            arg_type, expected):
                        raise TypeError(
                            f"函数 {func_name} 参数 {i} 期望 {expected}，"
                            f"实际 {arg_type}"
                        )
            return sig.return_type

        raise TypeError(f"不支持的节点类型: {type(node).__name__}")

    @staticmethod
    def _type_compatible(actual: str, expected: str) -> bool:
        """类型兼容性：int64 可兼容 float64。"""
        if actual == expected:
            return True
        if expected == "float64" and actual == "int64":
            return True
        return False