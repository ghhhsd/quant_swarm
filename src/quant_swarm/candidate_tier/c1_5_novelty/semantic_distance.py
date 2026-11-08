"""
语义距离（v3.1 修正②：语义 + AST 双重近邻）。

关键设计：
1. 语义距离优先用 embedding（真实部署），本实现用 token 近似。
2. 语义距离阈值由已知因子库内部分布标定（见 novelty_gate.py）。
3. 语义距离与 AST 距离"互补"：
   - 语义近但 AST 远：同机制不同实现 → 变体
   - 语义远但 AST 近：同表达式不同描述 → 重复
   - 两个都近：重复
   - 两个都远：新颖
"""
import re
from typing import List, Optional, Set
import hashlib


class SemanticDistance:
    """
    语义距离计算器。

    真实部署应替换为：
    - sentence-transformers 的 embedding + cosine 距离
    - 或 OpenAI/本地 embedding API

    本实现用 token Jaccard 近似（受限 DSL 下足够）。
    """

    def __init__(self, embedding_fn=None):
        self.embedding_fn = embedding_fn  # 可选，None 时用 token 近似

    def distance(self, text_a: str, text_b: str) -> float:
        """返回 [0, 1] 的距离，0 = 相同，1 = 完全不同。"""
        if self.embedding_fn is not None:
            vec_a = self.embedding_fn(text_a)
            vec_b = self.embedding_fn(text_b)
            # cosine 距离
            import numpy as np
            sim = float(np.dot(vec_a, vec_b) /
                        (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9))
            return 1.0 - sim
        # 回退：token Jaccard
        return self._token_jaccard_distance(text_a, text_b)

    @staticmethod
    def _token_jaccard_distance(a: str, b: str) -> float:
        ta = set(re.findall(r"[a-zA-Z_]+|\d+", a))
        tb = set(re.findall(r"[a-zA-Z_]+|\d+", b))
        if not ta or not tb:
            return 1.0
        return 1.0 - len(ta & tb) / len(ta | tb)

    def nearest(self, text: str,
                library_texts: List[str]) -> tuple:
        """返回 (最小距离, 最近邻文本, 最近邻索引)。"""
        if not library_texts:
            return 1.0, None, -1
        min_d, nearest, idx = 1.0, None, -1
        for i, lib in enumerate(library_texts):
            d = self.distance(text, lib)
            if d < min_d:
                min_d, nearest, idx = d, lib, i
        return min_d, nearest, idx