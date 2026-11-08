"""
AST 结构距离（v3.1 修正②：语义 + AST 双重近邻）。

关键设计：
1. AST 距离用树编辑距离（Zhang-Shasha）或 token 近似。
2. 本实现用"归一化 token 序列 + Levenshtein 距离"近似树编辑距离。
3. AST 距离与语义距离互补：
   - AST 近 = 表达式结构相似（可能是窗口参数变体）
   - 语义近 = 机制描述相似（可能是同族不同实现）
"""
import ast
import re
from typing import List, Optional


class ASTDistance:
    """
    AST 结构距离计算器。
    """

    def __init__(self, use_tree_edit: bool = False):
        self.use_tree_edit = use_tree_edit
        if use_tree_edit:
            try:
                import zhang_shasha  # 可选依赖
            except ImportError:
                raise ImportError(
                    "use_tree_edit=True 需要 zhang-shasha 库。"
                    "pip install zhang-shasha，或设 use_tree_edit=False"
                )

    def distance(self, code_a: str, code_b: str) -> float:
        """返回 [0, 1] 的 AST 距离。"""
        if self.use_tree_edit:
            return self._tree_edit_distance(code_a, code_b)
        return self._normalized_token_distance(code_a, code_b)

    @staticmethod
    def _tree_edit_distance(a: str, b: str) -> float:
        """
        Zhang-Shasha 树编辑距离归一化。
        """
        from zhang_shasha import simple_distance
        try:
            ta = ast.parse(a, mode="eval")
            tb = ast.parse(b, mode="eval")
        except SyntaxError:
            return 1.0
        # 把 AST 转为标签树
        def to_label_tree(node):
            label = type(node).__name__
            children = [to_label_tree(c) for c in ast.iter_child_nodes(node)]
            return (label, children)
        # simple_distance 需要适配
        # 这里简化：用节点标签序列的 Levenshtein 距离
        ta_nodes = [type(n).__name__ for n in ast.walk(ta)]
        tb_nodes = [type(n).__name__ for n in ast.walk(tb)]
        return _levenshtein_normalized(ta_nodes, tb_nodes)

    @staticmethod
    def _normalized_token_distance(a: str, b: str) -> float:
        """
        归一化 token 序列的 Levenshtein 距离。
        """
        ta = _tokenize(a)
        tb = _tokenize(b)
        return _levenshtein_normalized(ta, tb)

    def nearest(self, code: str,
                library_codes: List[str]) -> tuple:
        """返回 (最小距离, 最近邻代码, 最近邻索引)。"""
        if not library_codes:
            return 1.0, None, -1
        min_d, nearest, idx = 1.0, None, -1
        for i, lib in enumerate(library_codes):
            d = self.distance(code, lib)
            if d < min_d:
                min_d, nearest, idx = d, lib, i
        return min_d, nearest, idx


def _tokenize(code: str) -> List[str]:
    """把表达式拆为 token 序列。"""
    return re.findall(r"[a-zA-Z_]+|\d+|[-+*/(),]", code)


def _levenshtein_normalized(a: List[str], b: List[str]) -> float:
    """归一化 Levenshtein 距离 [0, 1]。"""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    m, n = len(a), len(b)
    # 空间优化：只保留两行
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    raw = prev[n]
    return raw / max(m, n)