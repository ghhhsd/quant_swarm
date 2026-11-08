"""
Factor 数据结构与内容哈希。

关键设计：
1. Factor 是不可变的（frozen=True），演化产生新 Factor 而非修改旧 Factor。
2. 内容哈希 = hash(AST + family + parent_fids)，用于演化树引用和版本管理。
3. sharpe / marginal_gain 是"评估结果"，可变——但通过 with_metrics() 产生
   新实例，而非原地修改，保证"评估前"和"评估后"的 Factor 可区分。
"""
from dataclasses import dataclass, field, replace
from typing import List, Optional, Dict, Any
import hashlib
import json


@dataclass(frozen=True)
class Factor:
    """
    不可变因子。

    code：受限 DSL 表达式
    ast_depth：AST 嵌套深度（由 DSL 检查给出）
    parent_fids：父代因子 fid 列表（演化树用）
    sharpe：评估后的 Sharpe（初始 None）
    marginal_gain：组合边际增益（初始 None）
    family：机制族
    """
    fid: str
    family: str
    code: str
    ast_depth: int
    parent_fids: tuple = ()      # tuple 以支持 frozen=True
    sharpe: Optional[float] = None
    marginal_gain: Optional[float] = None

    def content_hash(self) -> str:
        """确定性内容哈希。只 hash 结构字段，不 hash 评估结果。"""
        payload = {
            "fid": self.fid,
            "family": self.family,
            "code": self.code,
            "ast_depth": self.ast_depth,
            "parent_fids": list(self.parent_fids),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def with_metrics(self, sharpe: Optional[float] = None,
                     marginal_gain: Optional[float] = None) -> "Factor":
        """返回带评估结果的新实例（不修改原实例）。"""
        return replace(
            self,
            sharpe=sharpe if sharpe is not None else self.sharpe,
            marginal_gain=(marginal_gain if marginal_gain is not None
                           else self.marginal_gain),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fid": self.fid,
            "family": self.family,
            "code": self.code,
            "ast_depth": self.ast_depth,
            "parent_fids": list(self.parent_fids),
            "sharpe": self.sharpe,
            "marginal_gain": self.marginal_gain,
            "content_hash": self.content_hash(),
        }

    def is_evaluated(self) -> bool:
        """是否已完成评估（用于决定能否进入 Pareto 排序）。"""
        return self.sharpe is not None and self.marginal_gain is not None