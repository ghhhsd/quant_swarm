"""
可测指标（v3.1 修正⑦：拦截率 + 环境覆盖率）。

两个指标：
1. escape_test_pass_rate：攻击/危险/资源滥用测试集的拦截率。
2. env_coverage：白名单函数/字段被测试集覆盖的比例。

这两个指标是"测试时指标"，不是"运行时指标"——
受限执行的安全性靠"跑过测试集"证明，而不是靠"宣称 100%"。
"""
from dataclasses import dataclass
from typing import List, Set

from core.executor.testset import (
    AST_ATTACK_CASES, SEMANTIC_DANGER_CASES,
    RESOURCE_ABUSE_CASES, LEGIT_CASES,
)
from core.dsl.functions import ALLOWED_FUNCS
from core.data import DATA_CONTRACT


@dataclass
class ExecutorSecurityMetrics:
    """受限执行安全指标。"""
    escape_test_pass_rate: float
    env_coverage: float
    n_ast_attacks: int
    n_semantic_dangers: int
    n_resource_abuse: int
    n_legit: int
    uncovered_funcs: Set[str]
    uncovered_fields: Set[str]


def compute_metrics(dsl_checker, executor) -> ExecutorSecurityMetrics:
    """
    运行测试集，计算拦截率 + 覆盖率。

    dsl_checker：DSLChecker 实例
    executor：Executor 实例
    """
    covered_funcs: Set[str] = set()
    covered_fields: Set[str] = set()

    # 1) 覆盖率：从合法用例中提取使用到的函数/字段
    for case in LEGIT_CASES:
        result = dsl_checker.check(case.code)
        if result.ok and result.ast_result:
            covered_funcs |= result.ast_result.used_funcs
            covered_fields |= result.ast_result.used_fields

    # 2) 拦截率：攻击/危险/资源滥用用例应被对应层拦截
    n_blocked = 0
    total_cases = (AST_ATTACK_CASES + SEMANTIC_DANGER_CASES
                   + RESOURCE_ABUSE_CASES)
    for case in total_cases:
        # AST + 语义检查
        dsl_result = dsl_checker.check(case.code)
        if not dsl_result.ok:
            n_blocked += 1
            continue
        # 资源滥用：需要尝试执行（简化为跳过，真实部署应跑执行器）
        # 这里只统计 DSL 层拦截

    pass_rate = n_blocked / len(total_cases) if total_cases else 1.0

    # 3) 环境覆盖率
    env_coverage = (
        (len(covered_funcs) + len(covered_fields))
        / (len(ALLOWED_FUNCS) + len(DATA_CONTRACT))
    )

    return ExecutorSecurityMetrics(
        escape_test_pass_rate=pass_rate,
        env_coverage=env_coverage,
        n_ast_attacks=len(AST_ATTACK_CASES),
        n_semantic_dangers=len(SEMANTIC_DANGER_CASES),
        n_resource_abuse=len(RESOURCE_ABUSE_CASES),
        n_legit=len(LEGIT_CASES),
        uncovered_funcs=ALLOWED_FUNCS - covered_funcs,
        uncovered_fields=set(DATA_CONTRACT.keys()) - covered_fields,
    )