"""
语义预检（第 2 层，纯 Python，快）。

定位（v3.1）：
- 拦"合规但计算危险"：嵌套深度、中间结果规模、全表扫描/笛卡尔积。
- 不拦"运行时异常"：那是 executor 的职责。
- 不拦"逻辑错误"：那是 L1 的职责。

三类预检：
1. 嵌套深度上限：防 AST 过深导致递归爆栈。
2. 中间结果规模估算：防 O(n^k) 计算爆炸。
3. 全表扫描/笛卡尔积模式识别：防因子与因子之间意外做笛卡尔积。
"""
import ast
from dataclasses import dataclass
from typing import Optional, List

from main_tier.l4_sandbox.ast_layer import ASTCheckResult


@dataclass
class SemanticCheckResult:
    ok: bool
    reason: Optional[str] = None
    max_depth: int = 0
    estimated_intermediate: float = 0.0
    violation_type: Optional[str] = None  # "depth" / "intermediate" / "scan"


class SemanticChecker:
    """
    语义预检器。
    """

    def __init__(self,
                 max_nesting_depth: int = 12,
                 max_intermediate: float = 5e8,
                 max_field_count: int = 10):
        self.max_nesting_depth = max_nesting_depth
        self.max_intermediate = max_intermediate
        self.max_field_count = max_field_count

    def check(self, code: str, ast_result: Optional[ASTCheckResult] = None) -> SemanticCheckResult:
        """
        语义检查。若传入 ast_result，复用其 max_depth 和 used_fields。
        """
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError as e:
            return SemanticCheckResult(False, f"语法错误: {e}")

        # 1) 嵌套深度
        depth = ast_result.max_depth if ast_result else _max_depth(tree)
        if depth > self.max_nesting_depth:
            return SemanticCheckResult(
                False,
                f"嵌套深度 {depth} > {self.max_nesting_depth}",
                max_depth=depth, violation_type="depth",
            )

        # 2) 全表扫描/笛卡尔积模式
        scan_result = self._check_scan_patterns(code, tree)
        if not scan_result.ok:
            return scan_result

        # 3) 中间结果规模估算
        est = self._estimate_intermediate(code, tree)
        if est > self.max_intermediate:
            return SemanticCheckResult(
                False,
                f"预估中间结果规模 {est:.2e} > {self.max_intermediate:.2e}",
                max_depth=depth, estimated_intermediate=est,
                violation_type="intermediate",
            )

        return SemanticCheckResult(
            True, max_depth=depth, estimated_intermediate=est,
        )

    def _check_scan_patterns(self, code: str, tree) -> SemanticCheckResult:
        """
        识别全表扫描/笛卡尔积模式。
        本实现用关键词 + AST 结构双重检查：
        - 关键词：cross_join / cartesian / product
        - AST 结构：连续两次 ast.Call 直接相乘（可能是因子间笛卡尔积）
        """
        keywords = ["cross_join", "cartesian", "product", "outer_join"]
        for kw in keywords:
            if kw in code.lower():
                return SemanticCheckResult(
                    False, f"检测到疑似全表扫描/笛卡尔积关键词: {kw}",
                    violation_type="scan",
                )
        # AST 结构：Call * Call 模式（两个因子相乘，可能导致 O(N²)）
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if isinstance(node.left, ast.Call) and isinstance(node.right, ast.Call):
                    # 若两个 call 都是 ts_ 函数，可能是因子交叉而非笛卡尔积
                    # 若其中之一是 rank/zscore（截面函数），判为可疑
                    suspicious = False
                    for call in (node.left, node.right):
                        if isinstance(call.func, ast.Name) and call.func.id in {"rank", "zscore"}:
                            suspicious = True
                    if suspicious:
                        return SemanticCheckResult(
                            False, "检测到疑似因子间笛卡尔积（截面函数相乘）",
                            violation_type="scan",
                        )
        return SemanticCheckResult(True)

    def _estimate_intermediate(self, code: str, tree) -> float:
        """
        粗估中间结果规模。
        公式：10^depth * (1 + 0.5 * n_calls)
        - depth 越大，中间结果越多层
        - n_calls 越多，中间变量越多
        """
        depth = _max_depth(tree)
        n_calls = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.Call))
        return (10 ** min(depth, 10)) * (1 + 0.5 * n_calls)


def _max_depth(tree) -> int:
    def walk(node, d):
        children = list(ast.iter_child_nodes(node))
        if not children:
            return d
        return max(walk(c, d + 1) for c in children)
    return walk(tree, 0)