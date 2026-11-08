"""
逃逸面测试集（v3.1 修正⑦：可测指标）。

测试集分三类：
1. 已知攻击模式（应被 AST 层拦截）
2. 已知危险表达式（应被语义层拦截）
3. 已知资源滥用（应被受限执行拦截）

这些测试集不是"证明 100% 安全"，而是"评估拦截率"。
"""
from dataclasses import dataclass
from typing import List


@dataclass
class TestCase:
    name: str
    code: str
    expected_stage: str  # "ast" / "semantic" / "executor"
    description: str


# 已知攻击模式（应被 AST 层拦截）
AST_ATTACK_CASES: List[TestCase] = [
    TestCase("dunder_import", "__import__('os').system('ls')", "ast",
             "试图访问 __import__"),
    TestCase("dunder_class", "returns.__class__", "ast",
             "试图访问属性"),
    TestCase("lambda", "(lambda: 1)()", "ast",
             "lambda 不在白名单"),
    TestCase("list_comp", "[x for x in returns]", "ast",
             "列表推导不在白名单"),
    TestCase("subscript", "returns[0]", "ast",
             "下标访问不在白名单"),
    TestCase("undeclared_field", "unknown_field * 2", "ast",
             "未声明字段"),
    TestCase("unauthorized_func", "os_system('ls')", "ast",
             "未授权函数"),
    TestCase("exec_call", "exec('print(1)')", "ast",
             "exec 调用"),
    TestCase("open_call", "open('/etc/passwd')", "ast",
             "open 调用"),
]

# 已知危险表达式（合规但计算危险，应被语义层拦截）
SEMANTIC_DANGER_CASES: List[TestCase] = [
    TestCase("deep_nesting",
             "ts_mean(" * 20 + "returns" + ", 5)" * 20,
             "semantic", "嵌套深度 > 12"),
    TestCase("cartesian_hint", "cross_join(returns, returns)", "ast",
             "含 cross_join 关键词"),
    TestCase("rank_mul_rank", "rank(returns) * rank(pe_ratio)", "semantic",
             "截面函数相乘，疑似笛卡尔积"),
]

# 已知资源滥用（应被受限执行拦截）
RESOURCE_ABUSE_CASES: List[TestCase] = [
    TestCase("infinite_loop", "ts_mean(returns, 99999999)", "executor",
             "超大窗口导致超时/OOM"),
    TestCase("huge_allocation", "ts_std(returns, 1e18)", "executor",
             "超大数组导致 OOM"),
]

# 已知合法表达式（应通过所有层）
LEGIT_CASES: List[TestCase] = [
    TestCase("momentum", "ts_mean(returns, 20) - ts_mean(returns, 60)", "ok",
             "标准动量因子"),
    TestCase("value", "1.0 / pe_ratio", "ok", "标准价值因子"),
    TestCase("volatility", "ts_std(returns, 20)", "ok", "标准波动因子"),
    TestCase("liquidity", "ts_mean(turnover, 20)", "ok", "标准流动性因子"),
    TestCase("zscore_value", "zscore(1.0 / pe_ratio)", "ok", "zscore 变换"),
]


def all_test_cases() -> List[TestCase]:
    return (AST_ATTACK_CASES + SEMANTIC_DANGER_CASES
            + RESOURCE_ABUSE_CASES + LEGIT_CASES)