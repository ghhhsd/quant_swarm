"""
统计工具测试。

验证 v3.1 的九条纪律：
1. N 按批次记账
2. N 单调累积（同批次跨轮）
3. 新批次重置
4. DSR 的 N 敏感
5. DSR 考虑偏度/峰度
6. BH-FDR 批次内
7. p 值校验
8. 单次试验无需 FDR
9. 结果可序列化
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from core.stats import (
    SearchBatchCounter, deflated_sharpe, expected_max_sharpe,
    bh_fdr, bh_fdr_from_records, sharpe_to_pvalue, sharpes_to_pvalues,
    validate_pvalues, TrialRecord,
)
from core.event_log import EventLog
import tempfile
from pathlib import Path


def make_counter(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    return SearchBatchCounter(log), log


def test_trial_counter_batch():
    """N 按批次记账。"""
    counter = SearchBatchCounter()
    counter.record("batch_a", "f1", 0.5, 252)
    counter.record("batch_a", "f2", 0.3, 252)
    counter.record("batch_b", "f3", 0.4, 252)
    assert counter.get_n("batch_a") == 2
    assert counter.get_n("batch_b") == 1
    assert counter.get_n("batch_c") == 0
    print(f"[OK] N 按批次: {counter.summary()}")


def test_trial_counter_monotonic():
    """N 单调累积（同批次跨轮）。"""
    counter = SearchBatchCounter()
    # 第 1 轮
    counter.record("batch_a", "f1", 0.5, 252)
    counter.record("batch_a", "f2", 0.3, 252)
    assert counter.get_n("batch_a") == 2
    # 第 2 轮（同批次）
    counter.record("batch_a", "f3", 0.4, 252)
    assert counter.get_n("batch_a") == 3
    # 新批次重置
    counter.record("batch_b", "f4", 0.6, 252)
    assert counter.get_n("batch_a") == 3
    assert counter.get_n("batch_b") == 1
    print("[OK] N 单调累积")


def test_trial_counter_records():
    """记录完整信息。"""
    counter = SearchBatchCounter()
    counter.record("batch_a", "f1", 0.5, 252)
    counter.record("batch_a", "f2", 0.3, 252)
    records = counter.get_records("batch_a")
    assert len(records) == 2
    assert records[0].hid == "f1"
    assert records[0].trial_index == 1
    assert records[1].trial_index == 2
    assert counter.get_sharpes("batch_a") == [0.5, 0.3]
    print(f"[OK] 完整记录: {[r.to_dict() for r in records]}")


def test_dsr_n_sensitivity():
    """DSR 对 N 敏感。"""
    s = 0.5
    dsr_n10 = deflated_sharpe(s, 10, 252).dsr
    dsr_n100 = deflated_sharpe(s, 100, 252).dsr
    dsr_n1000 = deflated_sharpe(s, 1000, 252).dsr
    assert dsr_n10 > dsr_n100 > dsr_n1000, \
        f"N 越大 DSR 应越低: {dsr_n10:.3f} > {dsr_n100:.3f} > {dsr_n1000:.3f}"
    print(f"[OK] DSR 对 N 敏感: N=10→{dsr_n10:.3f}, "
          f"N=100→{dsr_n100:.3f}, N=1000→{dsr_n1000:.3f}")


def test_dsr_skew_kurt():
    """DSR 考虑偏度/峰度。"""
    rng = np.random.default_rng(42)
    # 正态收益
    normal_returns = rng.normal(0.001, 0.01, 252)
    # 肥尾收益（t 分布）
    fat_tail = rng.standard_t(df=3, size=252) * 0.01 + 0.001

    dsr_normal = deflated_sharpe(0.5, 10, 252, returns=normal_returns)
    dsr_fat = deflated_sharpe(0.5, 10, 252, returns=fat_tail)

    assert dsr_normal.skew_kurt_source == "empirical"
    assert dsr_fat.skew_kurt_source == "empirical"
    # 肥尾峰度应更高
    assert dsr_fat.kurt > dsr_normal.kurt
    print(f"[OK] 偏度/峰度: normal kurt={dsr_normal.kurt:.2f}, "
          f"fat kurt={dsr_fat.kurt:.2f}")


def test_dsr_no_returns_warning():
    """不传收益序列时标注警告。"""
    result = deflated_sharpe(0.5, 10, 252)
    assert result.skew_kurt_source == "normal_assumption"
    assert "未校正偏度/峰度" in result.note
    print(f"[OK] 警告: {result.note}")


def test_dsr_edge_cases():
    """DSR 边界情况。"""
    # N < 1
    r = deflated_sharpe(0.5, 0, 252)
    assert r.dsr == 0.0
    # n_obs < 10
    r = deflated_sharpe(0.5, 10, 5)
    assert r.dsr == 0.0
    print("[OK] 边界情况")


def test_expected_max_sharpe():
    """期望最大 Sharpe。"""
    assert expected_max_sharpe(1) == 0.0
    e10 = expected_max_sharpe(10)
    e100 = expected_max_sharpe(100)
    assert e100 > e10 > 0
    print(f"[OK] E[max]: N=10→{e10:.3f}, N=100→{e100:.3f}")


def test_bh_fdr_basic():
    """BH-FDR 基础。"""
    # 4 个 p 值，最小的两个应通过
    p_values = [0.001, 0.01, 0.5, 0.9]
    result = bh_fdr(p_values, alpha=0.10, batch_id="batch_a")
    assert result.flags[0] and result.flags[1]
    assert not result.flags[2] and not result.flags[3]
    assert result.n_significant == 2
    print(f"[OK] BH-FDR: {result.flags}")


def test_bh_fdr_batch_id():
    """BH-FDR 记录批次 ID。"""
    result = bh_fdr([0.01, 0.5], alpha=0.10, batch_id="batch_x")
    assert result.batch_id == "batch_x"
    print(f"[OK] 批次 ID: {result.batch_id}")


def test_bh_fdr_from_records():
    """从 TrialRecord 做 BH-FDR。"""
    counter = SearchBatchCounter()
    counter.record("batch_a", "f1", 1.5, 252)  # 高 Sharpe → 低 p 值
    counter.record("batch_a", "f2", 1.2, 252)
    counter.record("batch_a", "f3", 0.1, 252)  # 低 Sharpe → 高 p 值
    records = counter.get_records("batch_a")
    result = bh_fdr_from_records(records, alpha=0.10)
    assert result.batch_id == "batch_a"
    assert result.n_tests == 3
    print(f"[OK] 从记录: flags={result.flags}, "
          f"n_significant={result.n_significant}")


def test_bh_fdr_cross_batch_rejected():
    """跨批次记录被拒。"""
    r1 = TrialRecord("batch_a", "f1", 1, 0.5, 252, 0.0)
    r2 = TrialRecord("batch_b", "f2", 1, 0.3, 252, 0.0)
    try:
        bh_fdr_from_records([r1, r2])
        assert False, "应抛 ValueError"
    except ValueError as e:
        assert "不同批次" in str(e)
        print(f"[OK] 跨批次被拒: {e}")


def test_pvalue_validation():
    """p 值校验。"""
    validate_pvalues([0.0, 0.5, 1.0])  # 正常
    try:
        validate_pvalues([0.5, 1.5])  # 超出 [0,1]
        assert False
    except ValueError:
        print("[OK] p 值校验")


def test_sharpe_to_pvalue():
    """Sharpe → p 值。"""
    # 高 Sharpe → 低 p 值
    p_high = sharpe_to_pvalue(2.0, 252)
    p_low = sharpe_to_pvalue(0.1, 252)
    assert p_high < p_low
    assert 0 <= p_high <= 1
    assert 0 <= p_low <= 1
    print(f"[OK] Sharpe→p: sharpe=2.0→p={p_high:.4f}, "
          f"sharpe=0.1→p={p_low:.4f}")


def test_bh_fdr_single_test():
    """单次试验无需 FDR。"""
    result = bh_fdr([0.05], alpha=0.10, batch_id="batch_single")
    assert result.n_tests == 1
    assert result.flags == [True]  # 0.05 <= 0.10
    print(f"[OK] 单次试验: {result.flags}")


def test_result_serializable():
    """结果可序列化。"""
    dsr = deflated_sharpe(0.5, 10, 252)
    d = dsr.to_dict()
    assert "dsr" in d
    assert "n_trials" in d

    fdr = bh_fdr([0.01, 0.5], 0.10, "batch_a")
    d = fdr.to_dict()
    assert "flags" in d
    assert "batch_id" in d

    counter = SearchBatchCounter()
    counter.record("batch_a", "f1", 0.5, 252)
    r = counter.get_records("batch_a")[0]
    d = r.to_dict()
    assert "hid" in d
    assert "net_sharpe" in d
    print("[OK] 可序列化")


def test_event_logged(tmp_path):
    """事件流记录。"""
    counter, log = make_counter(tmp_path)
    counter.record("batch_a", "f1", 0.5, 252)
    events = log.query("trial_recorded")
    assert len(events) == 1
    assert events[0].payload["batch_id"] == "batch_a"
    print(f"[OK] 事件流: {len(events)} 条")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_trial_counter_batch()
        test_trial_counter_monotonic()
        test_trial_counter_records()
        test_dsr_n_sensitivity()
        test_dsr_skew_kurt()
        test_dsr_no_returns_warning()
        test_dsr_edge_cases()
        test_expected_max_sharpe()
        test_bh_fdr_basic()
        test_bh_fdr_batch_id()
        test_bh_fdr_from_records()
        test_bh_fdr_cross_batch_rejected()
        test_pvalue_validation()
        test_sharpe_to_pvalue()
        test_bh_fdr_single_test()
        test_result_serializable()
        test_event_logged(tp)
        print("\n=== 统计工具全部测试通过 ===")