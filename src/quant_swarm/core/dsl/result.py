"""
DSL 检查结果数据结构。

关键设计：
1. 三层检查各有独立结果，门面组合成最终结果。
2. 失败时带 failure_type，供 L1 自愈分流。
3. 成功时带类型信息和字段使用信息，供下游层使用。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from main_tier.l1_favor.failure import FailureType


@dataclass
class ASTCheckResult:
    """AST 检查结果。"""
    ok: bool
    reason: Optional[str] = None
    max_depth: int = 0
    used_fields: Set[str] = field(default_factory=set)
    used_funcs: Set[str] = field(default_factory=set)

    def to_dict(self):
        return {
            "ok": self.ok,
            "reason": self.reason,
            "max_depth": self.max_depth,
            "used_fields": sorted(self.used_fields),
            "used_funcs": sorted(self.used_funcs),
        }


@dataclass
class SemanticCheckResult:
    """语义检查结果。"""
    ok: bool
    reason: Optional[str] = None
    violation_type: Optional[str] = None  # "depth" / "intermediate" / "scan"
    max_depth: int = 0
    estimated_intermediate: float = 0.0

    def to_dict(self):
        return {
            "ok": self.ok,
            "reason": self.reason,
            "violation_type": self.violation_type,
            "max_depth": self.max_depth,
            "estimated_intermediate": self.estimated_intermediate,
        }


@dataclass
class ContractCheckResult:
    """契约检查结果。"""
    ok: bool
    reason: Optional[str] = None
    inferred_type: Optional[str] = None
    field_types: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "ok": self.ok,
            "reason": self.reason,
            "inferred_type": self.inferred_type,
            "field_types": dict(self.field_types),
        }


@dataclass
class DSLCheckResult:
    """
    DSL 检查完整结果。

    三层结果组合。失败时 stage 标记哪一层失败。
    """
    ok: bool
    stage: str  # "ast" / "semantic" / "contract" / "ok"
    failure_type: Optional[FailureType] = None
    reason: Optional[str] = None
    ast_result: Optional[ASTCheckResult] = None
    semantic_result: Optional[SemanticCheckResult] = None
    contract_result: Optional[ContractCheckResult] = None
    inferred_type: Optional[str] = None

    def to_dict(self):
        return {
            "ok": self.ok,
            "stage": self.stage,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "reason": self.reason,
            "ast": self.ast_result.to_dict() if self.ast_result else None,
            "semantic": (self.semantic_result.to_dict()
                         if self.semantic_result else None),
            "contract": (self.contract_result.to_dict()
                         if self.contract_result else None),
            "inferred_type": self.inferred_type,
        }