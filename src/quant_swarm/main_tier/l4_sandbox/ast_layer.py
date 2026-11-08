"""
AST 合规检查（第 1 层，纯 Python，快）。

定位（v3.1 修正⑦）：
- 拦"不合规表达式"：未声明字段、非法函数、非法节点类型。
- 不拦"计算爆炸"：那是 semantic_layer 的职责。
- 不拦"逃逸"：受限 DSL 下没有通用代码逃逸面，因为表达式不是通用程序。

威胁模型辨析：
在受限 DSL 下，AST 检查的威胁模型是"LLM 生成不合规表达式"
（如编造字段名、调用未授权函数），而不是"LLM 生成恶意代码逃逸"。
后者只在放开到通用代码生成时才成立，那时必须叠加 seccomp/gVisor。
"""
import ast
from dataclasses import dataclass
from typing import Optional, Set, Dict

from core.data import DATA_CONTRACT


# 允许的 AST 节点类型（白名单，扩展需审批）
ALLOWED_NODES: Set[type] = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
    ast.Compare, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq,
    ast.BoolOp, ast.And, ast.Or, ast.IfExp,
}

# 允许的函数调用（白名单）
ALLOWED_FUNCS: Set[str] = {
    "ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min", "ts_rank",
    "abs", "log", "sqrt", "sign", "clip", "rank", "zscore",
}

# 允许的字段（从数据契约推导）
ALLOWED_FIELDS: Set[str] = set(DATA_CONTRACT.keys())


@dataclass
class ASTCheckResult:
    ok: bool
    reason: Optional[str] = None
    max_depth: int = 0
    used_fields: Set[str] = None
    used_funcs: Set[str] = None

    def __post_init__(self):
        if self.used_fields is None:
            self.used_fields = set()
        if self.used_funcs is None:
            self.used_funcs = set()


def check_ast(code: str) -> ASTCheckResult:
    """
    检查表达式是否符合受限 DSL 白名单。
    """
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        return ASTCheckResult(False, f"语法错误: {e}")

    used_fields = set()
    used_funcs = set()

    for node in ast.walk(tree):
        node_type = type(node)
        if node_type not in ALLOWED_NODES:
            return ASTCheckResult(
                False,
                f"AST 白名单拦截：不允许的节点 {node_type.__name__}",
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
            # 函数名（调用时已检查）和字段名区分
            if name in ALLOWED_FUNCS:
                continue
            if name in ALLOWED_FIELDS:
                used_fields.add(name)
            elif name not in {"True", "False", "None"}:
                return ASTCheckResult(
                    False, f"未声明字段: {name}"
                )

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