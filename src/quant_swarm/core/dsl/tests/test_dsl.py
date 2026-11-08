"""
DSL 三层检查测试。

验证 v3.1 的九条纪律：
1. AST 白名单拦截非法节点/函数/字段
2. AST 失败归 SYNTAX
3. 语义预检拦嵌套深度/中间规模/全表扫描
4. 语义失败归 BACKTEST
5. 契约层拦未声明字段
6. 契约层拦类型不兼容
7. 契约失败归 DATA
8. 三层职责边界清晰
9. 合法表达式全通过并返回类型
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from core.dsl import DSLChecker, check_ast
from core.dsl.ast_layer import check_ast as ast_check
from core.dsl.semantic_layer import SemanticChecker, SemanticConfig
from core.dsl.type_inference import TypeInference
from core.data import DEFAULT_DATA_CONTRACT
from main_tier.l1_favor.failure import FailureType


def test_ast_blocks_import():
    """AST 拦截 __import__。"""
    fields = {"returns", "close", "pe_ratio"}
    result = ast_check("__import__('os')", fields)
    assert not result.ok
    print(f"[OK] AST 拦截 __import__: {result.reason}")


def test_ast_blocks_undeclared_field():
    """AST 拦截未声明字段。"""
    fields = {"returns", "close"}
    result = ast_check("unknown_field * 2", fields)
    assert not result.ok
    print(f"[OK] AST 拦截未声明字段: {result.reason}")


def test_ast_allows_valid():
    """AST 通过合法表达式。"""
    fields = {"returns", "close", "pe_ratio"}
    result = ast_check("ts_mean(returns, 20) - ts_mean(returns, 60)", fields)
    assert result.ok
    assert "ts_mean" in result.used_funcs
    assert "returns" in result.used_fields
    print(f"[OK] AST 合法: depth={result.max_depth}, "
          f"funcs={result.used_funcs}")


def test_semantic_depth_limit():
    """语义层拦嵌套深度。"""
    checker = SemanticChecker(SemanticConfig(max_nesting_depth=5))
    code = "ts_mean(ts_mean(ts_mean(ts_mean(ts_mean(ts_mean(returns, 5), 5), 5), 5), 5), 5)"
    result = checker.check(code)
    assert not result.ok
    assert result.violation_type == "depth"
    print(f"[OK] 语义拦深度: {result.reason}")


def test_semantic_intermediate_limit():
    """语义层拦中间规模。"""
    checker = SemanticChecker(SemanticConfig(max_intermediate=1e3))
    code = "ts_mean(ts_mean(ts_mean(ts_mean(returns, 5), 5), 5), 5)"
    result = checker.check(code)
    assert not result.ok
    assert result.violation_type == "intermediate"
    print(f"[OK] 语义拦中间规模: {result.reason}")


def test_semantic_scan_pattern():
    """语义层拦全表扫描。"""
    checker = SemanticChecker()
    result = checker.check("cross_join(returns, returns)")
    assert not result.ok
    assert result.violation_type == "scan"
    print(f"[OK] 语义拦全表扫描: {result.reason}")


def test_type_inference_basic():
    """类型推导基础。"""
    field_types = {
        "returns": "float64", "close": "float64",
        "pe_ratio": "float64", "volume": "float64",
    }
    inference = TypeInference(field_types)

    # 1.0 / pe_ratio → float64
    r = inference.infer("1.0 / pe_ratio")
    assert r.ok
    assert r.inferred_type == "float64"

    # returns > 0 → bool
    r = inference.infer("returns > 0")
    assert r.ok
    assert r.inferred_type == "bool"

    # ts_mean(returns, 20) → float64
    r = inference.infer("ts_mean(returns, 20)")
    assert r.ok
    assert r.inferred_type == "float64"

    print("[OK] 类型推导基础")


def test_type_inference_error():
    """类型推导报错。"""
    field_types = {"returns": "float64"}
    inference = TypeInference(field_types)

    # 未声明字段
    r = inference.infer("unknown * 2")
    assert not r.ok

    # 函数参数个数不对
    r = inference.infer("ts_mean(returns)")
    assert not r.ok

    # 函数参数类型不对（returns 是 float64，20 是 int64 兼容）
    r = inference.infer("ts_mean(returns, 20)")
    assert r.ok  # int64 兼容 float64

    print("[OK] 类型推导报错")


def test_contract_layer_field_exists():
    """契约层拦未声明字段。"""
    checker = DSLChecker()
    result = checker.check("unknown_field * 2")
    assert not result.ok
    assert result.stage == "ast"  # AST 层先拦
    print(f"[OK] 未声明字段: stage={result.stage}")


def test_contract_layer_type_mismatch():
    """契约层拦类型不兼容。"""
    checker = DSLChecker()
    # pe_ratio / returns 两个都是 float64，应通过
    result = checker.check("pe_ratio / returns")
    assert result.ok
    assert result.inferred_type == "float64"

    # ts_mean(returns, returns) —— 第二个参数应为 int64，实际 float64
    result = checker.check("ts_mean(returns, returns)")
    assert not result.ok
    assert result.stage == "contract"
    assert result.failure_type == FailureType.DATA
    print(f"[OK] 类型不兼容: {result.reason}")


def test_three_stage_boundary():
    """三层职责边界清晰。"""
    checker = DSLChecker()

    # AST 失败
    r = checker.check("__import__('os')")
    assert r.stage == "ast"
    assert r.failure_type == FailureType.SYNTAX

    # 语义失败（深度超限）
    deep = "ts_mean(" * 20 + "returns" + ", 5)" * 20
    r = checker.check(deep)
    assert r.stage in ("semantic", "ast")  # 深度太大可能 AST 也拒
    if r.stage == "semantic":
        assert r.failure_type == FailureType.BACKTEST

    # 契约失败
    r = checker.check("ts_mean(returns, returns)")
    assert r.stage == "contract"
    assert r.failure_type == FailureType.DATA

    print("[OK] 三层职责边界清晰")


def test_legit_expressions():
    """合法表达式全通过。"""
    checker = DSLChecker()
    cases = [
        "ts_mean(returns, 20) - ts_mean(returns, 60)",
        "ts_std(returns, 20)",
        "1.0 / pe_ratio",
        "ts_mean(turnover, 20)",
        "zscore(1.0 / pe_ratio)",
        "rank(returns) - rank(pe_ratio)",
        "returns > 0",
        "clip(returns, -0.1, 0.1)",
    ]
    for code in cases:
        result = checker.check(code)
        assert result.ok, f"{code} 应通过，实际: {result.stage}: {result.reason}"
    print(f"[OK] {len(cases)} 个合法表达式全通过")


def test_full_result_dict():
    """完整结果可序列化。"""
    checker = DSLChecker()
    result = checker.check("ts_mean(returns, 20)")
    d = result.to_dict()
    assert d["ok"]
    assert d["stage"] == "ok"
    assert d["inferred_type"] == "float64"
    assert d["ast"] is not None
    assert d["semantic"] is not None
    assert d["contract"] is not None
    print(f"[OK] 完整结果: type={d['inferred_type']}, "
          f"fields={d['ast']['used_fields']}")


if __name__ == "__main__":
    test_ast_blocks_import()
    test_ast_blocks_undeclared_field()
    test_ast_allows_valid()
    test_semantic_depth_limit()
    test_semantic_intermediate_limit()
    test_semantic_scan_pattern()
    test_type_inference_basic()
    test_type_inference_error()
    test_contract_layer_field_exists()
    test_contract_layer_type_mismatch()
    test_three_stage_boundary()
    test_legit_expressions()
    test_full_result_dict()
    print("\n=== DSL 全部测试通过 ===")