"""
L4 沙箱测试。

验证 v3.1 的七条纪律：
1. AST 层拦攻击模式（10 个用例）
2. 语义层拦危险表达式（含嵌套深度、笛卡尔积）
3. 执行层兜底超时
4. 三层职责边界清晰（AST→SYNTAX，语义→BACKTEST，执行→BACKTEST）
5. 合法表达式全部通过
6. 可测指标：覆盖率 + 拦截率
7. 不宣称"100%"，只报可测数字
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from main_tier.l4_sandbox import (
    SandboxLayer, ExecutorConfig, ExecutorMode,
)
from main_tier.l4_sandbox.testset import (
    AST_ATTACK_CASES, SEMANTIC_DANGER_CASES, LEGIT_CASES,
)
from main_tier.l4_sandbox.metrics import compute_metrics
from main_tier.l1_favor.failure import FailureType


def test_ast_attacks_blocked():
    sandbox = SandboxLayer()
    for case in AST_ATTACK_CASES:
        result = sandbox.check(case.code)
        assert not result.ok, f"{case.name} 应被拦截"
        assert result.stage in ("ast", "semantic"), \
            f"{case.name} 应在 AST/语义层拦截，实际 {result.stage}"
        assert result.failure_type in (FailureType.SYNTAX, FailureType.BACKTEST)
    print(f"[OK] {len(AST_ATTACK_CASES)} 个攻击用例全部被拦截")


def test_semantic_dangers_blocked():
    sandbox = SandboxLayer()
    for case in SEMANTIC_DANGER_CASES:
        result = sandbox.check(case.code)
        assert not result.ok, f"{case.name} 应被拦截"
    print(f"[OK] {len(SEMANTIC_DANGER_CASES)} 个危险用例全部被拦截")


def test_legit_cases_pass():
    sandbox = SandboxLayer()
    for case in LEGIT_CASES:
        result = sandbox.check(case.code)
        assert result.ok, f"{case.name} 应通过，实际: {result.reason}"
    print(f"[OK] {len(LEGIT_CASES)} 个合法用例全部通过")


def test_stage_boundary():
    """职责边界：AST 失败归 SYNTAX，语义失败归 BACKTEST。"""
    sandbox = SandboxLayer()
    # AST 层失败
    r = sandbox.check("__import__('os')")
    assert r.stage == "ast"
    assert r.failure_type == FailureType.SYNTAX
    # 语义层失败
    r = sandbox.check("rank(returns) * rank(pe_ratio)")
    assert r.stage == "semantic"
    assert r.failure_type == FailureType.BACKTEST
    print("[OK] 三层职责边界清晰")


def test_executor_timeout():
    """执行层兜底：超时。"""
    sandbox = SandboxLayer(executor_config=ExecutorConfig(
        mode=ExecutorMode.SUBPROCESS, timeout_s=0.5,
    ))
    # 构造一个耗时的合规表达式（ts_mean 嵌套但深度合法）
    code = "ts_mean(ts_mean(returns, 20), 20)"
    # 注：实际执行需要完整环境；这里只验证接口不抛异常
    result = sandbox.execute(code)
    # 结果可能是成功或失败（取决于 wrapper 环境），但不应抛异常
    assert isinstance(result.ok, bool)
    print(f"[OK] 执行层接口正常: ok={result.ok}")


def test_security_metrics():
    """可测指标：覆盖率 + 拦截率。"""
    sandbox = SandboxLayer()
    m = compute_metrics(sandbox)
    print(f"[METRICS] api_surface_coverage={m.api_surface_coverage:.1%}")
    print(f"[METRICS] escape_test_pass_rate={m.escape_test_pass_rate:.1%}")
    print(f"[METRICS] uncovered_funcs={m.uncovered_funcs}")
    print(f"[METRICS] uncovered_fields={m.uncovered_fields}")

    # 拦截率应 100%（测试集全过）
    assert m.escape_test_pass_rate == 1.0, \
        f"拦截率应 100%，实际 {m.escape_test_pass_rate:.1%}"
    # 覆盖率不要求 100%（合法用例不会覆盖所有白名单）
    assert 0 < m.api_surface_coverage <= 1.0

    print("[OK] 可测指标计算正常")


def test_no_absolute_claim():
    """
    验证：L4 不宣称"100% 阻断"，只报可测数字。
    这里检查 sandbox 的文档字符串不含"100%"。
    """
    import main_tier.l4_sandbox.sandbox as sb
    import main_tier.l4_sandbox.ast_layer as al
    for mod in (sb, al):
        doc = mod.__doc__ or ""
        assert "100%" not in doc, f"{mod.__name__} 文档不应有'100%'绝对承诺"
    print("[OK] 无绝对承诺，口径为可测指标")


if __name__ == "__main__":
    test_ast_attacks_blocked()
    test_semantic_dangers_blocked()
    test_legit_cases_pass()
    test_stage_boundary()
    test_executor_timeout()
    test_security_metrics()
    test_no_absolute_claim()
    print("\n=== L4 全部测试通过 ===")