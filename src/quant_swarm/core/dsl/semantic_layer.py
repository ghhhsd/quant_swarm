"""
语义预检（第 2 层：计算爆炸层）。

关键设计：
1. 拦截"合规但计算危险"的表达式。
2. 三类检查：
   - 嵌套深度上限
   - 中间结果规模估算
   - 全表扫描/笛卡尔积模式
3. 失败归为 BACKTEST（回测异常）。
4. 不拦"运行时异常"（那是 L4 沙箱的职责）。
5. 不拦"契约不匹配"（那是第 3 层契约层的职责）。
"""
import ast
from dataclasses import dataclass
from typing import Optional

from core.dsl.result import SemanticCheckResult


@dataclass
class SemanticConfig:
    """语义检查配置。"""
    max_nesting_depth: int = 12
    max_intermediate: float = 5e8
    scan_keywords: tuple = ("cross_join", "cartesian", "product", "outer_join")


class SemanticChecker:
    """语义预检器。"""

    def __init__(self, config: Optional[SemanticConfig] = None):
        self.config = config or SemanticConfig()

    def check(self, code: str, ast_result=None) -> SemanticCheckResult:
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError as e:
            return SemanticCheckResult(False, f"语法错误: {e}")

        depth = ast_result.max_depth if ast_result else _max_depth(tree)

        # 1) 嵌套深度
        if depth > self.config.max_nesting_depth:
            return SemanticCheckResult(
                False,
                f"嵌套深度 {depth} > {self.config.max_nesting_depth}",
                violation_type="depth", max_depth=depth,
            )

        # 2) 全表扫描/笛卡尔积
        scan_result = self._check_scan_patterns(code, tree)
        if not scan_result.ok:
            return scan_result

        # 3) 中间结果规模
        est = self._estimate_intermediate(code, tree)
        if est > self.config.max_intermediate:
            return SemanticCheckResult(
                False,
                f"预估中间结果规模 {est:.2e} > {self.config.max_intermediate:.2e}",
                violation_type="intermediate", max_depth=depth,
                estimated_intermediate=est,
            )

        return SemanticCheckResult(
            True, max_depth=depth, estimated_intermediate=est,
        )

    def _check_scan_patterns(self, code: str, tree) -> SemanticCheckResult:
        for kw in self.config.scan_keywords:
            if kw in code.lower():
                return SemanticCheckResult(
                    False, f"检测到疑似全表扫描/笛卡尔积关键词: {kw}",
                    violation_type="scan",
                )
        # AST 结构：Call * Call（两个截面函数相乘，疑似笛卡尔积）
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if (isinstance(node.left, ast.Call)
                        and isinstance(node.right, ast.Call)):
                    suspicious = False
                    for call in (node.left, node.right):
                        if isinstance(call.func, ast.Name) and \
                           call.func.id in {"rank", "zscore"}:
                            suspicious = True
                    if suspicious:
                        return SemanticCheckResult(
                            False, "检测到疑似因子间笛卡尔积（截面函数相乘）",
                            violation_type="scan",
                        )
        return SemanticCheckResult(True)

    def _estimate_intermediate(self, code: str, tree) -> float:
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