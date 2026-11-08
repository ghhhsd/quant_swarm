"""
AST 白名单检查（第 1 层：语法层）。

关键设计：
1. 只允许声明的节点类型、函数名、字段名。
2. 拦截的是"不合规"，不是"逃逸"——受限 DSL 下没有通用代码逃逸面。
3. 失败归为 SYNTAX（语法错误）。
4. 纯语法检查，不依赖数据契约。
"""
import ast
from typing import Set

from core.dsl.functions import ALLOWED_NODES, ALLOWED_FUNCS
from core.dsl.result import ASTCheckResult


def check_ast(code: str, allowed_fields: Set[str]) -> ASTCheckResult:
    """
    AST 白名单检查。

    code：因子表达式
    allowed_fields：允许的字段名（来自数据契约）
    """
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        return ASTCheckResult(False, f"语法错误: {e}")

    used_fields: Set[str] = set()
    used_funcs: Set[str] = set()

    for node in ast.walk(tree):
        node_type = type(node).__name__
        if node_type not in ALLOWED_NODES:
            return ASTCheckResult(
                False,
                f"AST 白名单拦截：不允许的节点 {node_type}",
            )

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return ASTCheckResult(
                    False, "AST 白名单拦截：函数调用必须是简单名称"
                )
            func_name = node.func.id
            if func_name not in ALLOWED_FUNCS:
                return ASTCheckResult(
                    False, f"AST 白名单拦截：未授权函数 {func_name}"
                )
            used_funcs.add(func_name)

        if isinstance(node, ast.Name):
            name = node.id
            if name in ALLOWED_FUNCS:
                continue
            if name in allowed_fields:
                used_fields.add(name)
            elif name not in {"True", "False", "None"}:
                return ASTCheckResult(False, f"未声明字段: {name}")

    max_depth = _max_depth(tree)
    return ASTCheckResult(
        True, max_depth=max_depth,
        used_fields=used_fields, used_funcs=used_funcs,
    )


def _max_depth(tree) -> int:
    def walk(node, d):
        children = list(ast.iter_child_nodes(node))
        if not children:
            return d
        return max(walk(c, d + 1) for c in children)
    return walk(tree, 0)