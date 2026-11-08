"""
可测指标（v3.1 修正⑦：安全口径改为覆盖率 + 逃逸面通过率）。

两个指标：
1. api_surface_coverage：白名单函数/字段被测试集覆盖的比例。
2. escape_test_pass_rate：攻击/危险测试集的拦截率。

这两个指标是"测试时指标"，不是"运行时指标"——
L4 的安全性靠"跑过测试集"证明，而不是靠"宣称 100%"。
"""
from dataclasses import dataclass
from typing import List, Set

from main_tier.l4_sandbox.ast_layer import ALLOWED_FUNCS, ALLOWED_FIELDS
from main_tier.l4_sandbox.testset import (
    TestCase, AST_ATTACK_CASES, SEMANTIC_DANGER_CASES, LEGIT_CASES,
)


@dataclass
class SecurityMetrics:
    api_surface_coverage: float       # [0, 1]
    escape_test_pass_rate: float      # [0, 1]
    n_ast_attack_cases: int
    n_semantic_danger_cases: int
    n_legit_cases: int
    covered_funcs: Set[str]
    covered_fields: Set[str]
    uncovered_funcs: Set[str]
    uncovered_fields: Set[str]


def compute_metrics(sandbox) -> SecurityMetrics:
    """
    运行测试集，计算覆盖率与拦截率。
    sandbox 需实现 check(code) -> CheckResult（见 sandbox.py 门面）。
    """
    covered_funcs: Set[str] = set()
    covered_fields: Set[str] = set()

    # 1) 覆盖率：从合法用例中提取使用到的函数/字段
    for case in LEGIT_CASES:
        result = sandbox.check(case.code)
        if result.ast_used_funcs:
            covered_funcs |= result.ast_used_funcs
        if result.ast_used_fields:
            covered_fields |= result.ast_used_fields

    # 2) 拦截率：攻击用例应被对应层拦截
    ast_attacks_blocked = 0
    for case in AST_ATTACK_CASES:
        result = sandbox.check(case.code)
        if not result.ok and result.stage == "ast":
            ast_attacks_blocked += 1

    semantic_dangers_blocked = 0
    for case in SEMANTIC_DANGER_CASES:
        result = sandbox.check(case.code)
        # 语义危险用例可能被 AST 层先拦（如 cross_join 关键词）
        if not result.ok and result.stage in ("ast", "semantic"):
            semantic_dangers_blocked += 1

    total_attacks = len(AST_ATTACK_CASES) + len(SEMANTIC_DANGER_CASES)
    total_blocked = ast_attacks_blocked + semantic_dangers_blocked

    # 3) 合法用例应全部通过
    legit_passed = sum(1 for c in LEGIT_CASES if sandbox.check(c.code).ok)

    return SecurityMetrics(
        api_surface_coverage=(
            (len(covered_funcs) + len(covered_fields))
            / (len(ALLOWED_FUNCS) + len(ALLOWED_FIELDS))
        ),
        escape_test_pass_rate=total_blocked / total_attacks if total_attacks else 1.0,
        n_ast_attack_cases=len(AST_ATTACK_CASES),
        n_semantic_danger_cases=len(SEMANTIC_DANGER_CASES),
        n_legit_cases=len(LEGIT_CASES),
        covered_funcs=covered_funcs,
        covered_fields=covered_fields,
        uncovered_funcs=ALLOWED_FUNCS - covered_funcs,
        uncovered_fields=ALLOWED_FIELDS - covered_fields,
    )