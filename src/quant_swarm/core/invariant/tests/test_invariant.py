"""
不变核心测试。

验证 v3.1 的九条纪律：
1. 深拷贝返回
2. 版本号 + 内容哈希双重锚定
3. 哈希覆盖嵌套结构（顺序无关）
4. 篡改检测（值 + 类型）
5. 逐键 diff（含列表）
6. 类型不匹配检测
7. 列表逐元素 diff
8. 访问可审计
9. 版本校验
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import copy
from core.invariant import (
    canonicalize, canonical_json, compute_hash,
    diff, DiffResult, InvariantViolation,
    InvariantVersion, create_version, InvariantRegistry,
)
from core.config import INVARIANT_CORE
from core.event_log import EventLog
import tempfile
from pathlib import Path


def make_registry(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    return InvariantRegistry(event_log=log), log


def test_deep_copy():
    """深拷贝返回。"""
    registry = InvariantRegistry()
    core1 = registry.get(log_access=False)
    core2 = registry.get(log_access=False)
    assert core1 is not core2
    assert core1 == core2
    # 修改 core1 不影响 core2
    core1["risk_breaker"]["max_drawdown"] = 0.99
    assert core2["risk_breaker"]["max_drawdown"] == 0.20
    print("[OK] 深拷贝")


def test_version_hash():
    """版本号 + 内容哈希。"""
    registry = InvariantRegistry()
    assert registry.get_version() == "ic-1.0.0"
    assert len(registry.get_hash()) == 64  # sha256
    version_info = registry.get_version_info()
    assert version_info.version == "ic-1.0.0"
    assert version_info.content_hash == registry.get_hash()
    print(f"[OK] 版本: {registry.get_version()}, "
          f"hash={registry.get_hash()[:12]}...")


def test_hash_nested_order():
    """哈希覆盖嵌套结构（顺序无关）。"""
    # 两个语义相同但嵌套顺序不同的 dict
    core_a = {
        "version": "ic-1.0.0",
        "risk_breaker": {"max_drawdown": 0.20, "max_leverage": 1.0},
        "position_sizing": {"method": "equal", "max_single": 0.1},
    }
    core_b = {
        "position_sizing": {"max_single": 0.1, "method": "equal"},
        "risk_breaker": {"max_leverage": 1.0, "max_drawdown": 0.20},
        "version": "ic-1.0.0",
    }
    h_a = compute_hash(core_a)
    h_b = compute_hash(core_b)
    assert h_a == h_b, f"嵌套顺序不应影响哈希: {h_a[:8]} vs {h_b[:8]}"
    print(f"[OK] 嵌套顺序无关: {h_a[:12]}...")


def test_hash_list_order():
    """列表顺序有意义（不排序）。"""
    core_a = {"funcs": ["ts_mean", "ts_std"]}
    core_b = {"funcs": ["ts_std", "ts_mean"]}
    assert compute_hash(core_a) != compute_hash(core_b)
    print("[OK] 列表顺序有意义")


def test_assert_unchanged_ok():
    """未篡改通过。"""
    registry = InvariantRegistry()
    candidate = registry.get(log_access=False)
    registry.assert_unchanged(candidate)  # 不抛
    print("[OK] 未篡改通过")


def test_assert_unchanged_value_tamper():
    """值篡改可检测。"""
    registry = InvariantRegistry()
    candidate = registry.get(log_access=False)
    candidate["risk_breaker"]["max_leverage"] = 10.0
    try:
        registry.assert_unchanged(candidate)
        assert False
    except InvariantViolation as e:
        assert "max_leverage" in str(e.diff_result)
        print(f"[OK] 值篡改: {e.message[:50]}...")


def test_assert_unchanged_type_tamper():
    """类型篡改可检测。"""
    registry = InvariantRegistry()
    candidate = registry.get(log_access=False)
    candidate["risk_breaker"]["max_drawdown"] = "0.20"  # 字符串
    try:
        registry.assert_unchanged(candidate)
        assert False
    except InvariantViolation as e:
        assert "type_mismatch" in str(e.diff_result.to_dict()) or \
            "max_drawdown" in str(e.diff_result)
        print(f"[OK] 类型篡改: {e.message[:50]}...")


def test_diff_list_elements():
    """列表逐元素 diff。"""
    expected = {"funcs": ["ts_mean", "ts_std", "rank"]}
    actual = {"funcs": ["ts_mean", "zscore"]}
    result = diff(expected, actual)
    # funcs 长度不同
    assert "funcs" in result.changed
    assert result.changed["funcs"]["reason"] == "length_mismatch"
    print(f"[OK] 列表 diff: {result.changed['funcs']['missing']} 缺失, "
          f"{result.changed['funcs']['extra']} 多余")


def test_diff_nested():
    """嵌套 dict diff。"""
    expected = {
        "risk_breaker": {"max_drawdown": 0.20, "max_leverage": 1.0},
    }
    actual = {
        "risk_breaker": {"max_drawdown": 0.20, "max_leverage": 10.0},
    }
    result = diff(expected, actual)
    assert "risk_breaker.max_leverage" in result.changed
    print(f"[OK] 嵌套 diff: {list(result.changed.keys())}")


def test_diff_missing_extra():
    """缺失/多余字段。"""
    expected = {"a": 1, "b": 2}
    actual = {"a": 1, "c": 3}
    result = diff(expected, actual)
    assert "b" in result.missing
    assert "c" in result.extra
    print(f"[OK] 缺失/多余: missing={result.missing}, extra={result.extra}")


def test_access_logged(tmp_path):
    """访问可审计。"""
    registry, log = make_registry(tmp_path)
    registry.get(governance_context={"requester": "L0", "round": 3})
    events = log.query("invariant_accessed")
    assert len(events) == 1
    assert events[0].payload["governance_context"]["requester"] == "L0"
    print(f"[OK] 访问审计: {events[0].payload['governance_context']}")


def test_violation_logged(tmp_path):
    """篡改事件可审计。"""
    registry, log = make_registry(tmp_path)
    candidate = registry.get(log_access=False)
    candidate["risk_breaker"]["max_leverage"] = 10.0
    try:
        registry.assert_unchanged(candidate)
    except InvariantViolation:
        pass
    events = log.query("invariant_violation")
    assert len(events) == 1
    assert events[0].payload["expected_hash"] != events[0].payload["actual_hash"]
    print(f"[OK] 篡改审计: {len(events)} 条")


def test_version_verify():
    """版本校验。"""
    registry = InvariantRegistry()
    assert registry.verify_version()
    # 篡改内部 core 后版本校验失败
    registry._core["risk_breaker"]["max_drawdown"] = 0.99
    assert not registry.verify_version()
    print("[OK] 版本校验")


def test_get_section():
    """获取某一部分。"""
    registry = InvariantRegistry()
    rb = registry.get_section("risk_breaker")
    assert "max_drawdown" in rb
    # 修改副本不影响原
    rb["max_drawdown"] = 0.99
    assert registry.get_section("risk_breaker")["max_drawdown"] == 0.20
    print("[OK] 获取 section")


def test_is_unchanged():
    """不抛异常的校验。"""
    registry = InvariantRegistry()
    candidate = registry.get(log_access=False)
    assert registry.is_unchanged(candidate)
    candidate["risk_breaker"]["max_leverage"] = 10.0
    assert not registry.is_unchanged(candidate)
    print("[OK] is_unchanged")


def test_event_logged_no_absolute():
    """不宣称绝对安全。"""
    import core.invariant.registry as reg
    doc = reg.__doc__ or ""
    assert "100%" not in doc
    print("[OK] 无绝对承诺")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_deep_copy()
        test_version_hash()
        test_hash_nested_order()
        test_hash_list_order()
        test_assert_unchanged_ok()
        test_assert_unchanged_value_tamper()
        test_assert_unchanged_type_tamper()
        test_diff_list_elements()
        test_diff_nested()
        test_diff_missing_extra()
        test_access_logged(tp)
        test_violation_logged(tp)
        test_version_verify()
        test_get_section()
        test_is_unchanged()
        test_event_logged_no_absolute()
        print("\n=== 不变核心全部测试通过 ===")