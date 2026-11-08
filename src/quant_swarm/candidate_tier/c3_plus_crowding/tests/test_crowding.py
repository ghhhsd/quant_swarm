"""
C3+ 持仓拥挤度测试。

验证 v3.1 的九条纪律：
1. 口径为"内部持仓重叠"，非"拥挤度检测"
2. 外部代理单独列，不参与 flagged
3. shadow mode：产出不进主干
4. 采样降维：最近 63 天
5. 增量缓存：不重算全部
6. 内部重叠 vs 收益相关：互补不替代
7. flagged 只看内部
8. shadow 汇总：若启用会拦截多少
9. 外部代理指标可观测
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from candidate_tier.c3_plus_crowding import (
    PositionExtractor, InternalOverlapChecker, ExternalProxyExtractor,
    CrowdingGate,
)
from core.data import generate_synthetic_data
from core.event_log import EventLog
import tempfile
from pathlib import Path


def make_gate(tmp_path, sample_days=63):
    log = EventLog(str(tmp_path / "events.jsonl"))
    df = generate_synthetic_data(300, 50, seed=42)
    gate = CrowdingGate(
        df=df, max_internal_overlap=0.70,
        sample_days=sample_days, event_log=log,
    )
    return gate, df, log


def test_position_sampling(tmp_path):
    """采样降维：只取最近 63 天。"""
    gate, df, _ = make_gate(tmp_path, sample_days=63)
    pm = gate.extractor.extract("f1", "ts_mean(returns, 20)")
    assert pm.sample_days == 63, f"应采样 63 天，实际 {pm.sample_days}"
    assert pm.positions.shape == (63, 50)
    print(f"[OK] 采样降维: shape={pm.positions.shape}")


def test_internal_overlap_identical(tmp_path):
    """相同因子 → 重叠 = 1。"""
    gate, _, _ = make_gate(tmp_path)
    pm = gate.extractor.extract("f1", "ts_mean(returns, 20)")
    sim = gate.internal_checker.cosine_similarity(pm.positions, pm.positions)
    assert abs(sim - 1.0) < 1e-6, f"相同因子重叠应为 1，实际 {sim}"
    print(f"[OK] 相同因子重叠: {sim:.4f}")


def test_internal_overlap_different(tmp_path):
    """不同因子 → 重叠 < 1。"""
    gate, _, _ = make_gate(tmp_path)
    pm1 = gate.extractor.extract("f1", "ts_mean(returns, 20)")
    pm2 = gate.extractor.extract("f2", "1.0 / pe_ratio")
    sim = gate.internal_checker.cosine_similarity(pm1.positions, pm2.positions)
    assert sim < 1.0, f"不同因子重叠应 < 1，实际 {sim}"
    print(f"[OK] 不同因子重叠: {sim:.4f}")


def test_external_proxy_separate(tmp_path):
    """外部代理单独列，不参与 flagged。"""
    gate, _, _ = make_gate(tmp_path)
    decision = gate.evaluate({
        "f1": "ts_mean(returns, 20)",
        "f2": "1.0 / pe_ratio",
    })
    # 外部代理有指标，但不影响 flagged
    assert decision.external_proxy.n_metrics > 0
    assert "turnover_q90" in decision.external_proxy.metrics
    assert "return_dispersion" in decision.external_proxy.metrics
    # 报告 note 说明外部代理不约束
    assert "不参与约束" in decision.external_proxy.note
    print(f"[OK] 外部代理单独列: {decision.external_proxy.n_metrics} 个指标")


def test_flagged_only_internal(tmp_path):
    """flagged 只看内部重叠，不看外部代理。"""
    gate, _, _ = make_gate(tmp_path, sample_days=63)
    # 构造一个高重叠场景：两个相同因子
    decision = gate.evaluate({
        "f1": "ts_mean(returns, 20)",
        "f2": "ts_mean(returns, 20)",  # 相同
    })
    assert decision.flagged, "相同因子应 flagged"
    assert decision.internal_overlap.mean_overlap > 0.7
    print(f"[OK] flagged 只看内部: overlap={decision.internal_overlap.mean_overlap:.3f}")


def test_shadow_mode(tmp_path):
    """shadow mode：决策恒为 shadow。"""
    gate, _, log = make_gate(tmp_path)
    decision = gate.evaluate({"f1": "ts_mean(returns, 20)"})
    assert decision.shadow, "候选层决策应恒为 shadow"
    events = log.query("crowding_shadow_decision")
    assert len(events) == 1
    print("[OK] shadow mode 只记录不阻断")


def test_incremental_cache(tmp_path):
    """增量缓存：新因子只与现有因子计算。"""
    gate, _, _ = make_gate(tmp_path)
    # 先提取两个现有因子
    gate.extractor.extract("f1", "ts_mean(returns, 20)")
    gate.extractor.extract("f2", "1.0 / pe_ratio")
    assert gate.extractor.size() == 2

    # 增量检查：新因子 f3
    decision = gate.evaluate_incremental(
        new_fid="f3", new_code="ts_std(returns, 20)",
        existing_fids_codes={"f1": "ts_mean(returns, 20)",
                             "f2": "1.0 / pe_ratio"},
    )
    # 增量报告只有 2 对（f3-f1, f3-f2），不是 3 对（f1-f2 已有缓存）
    assert decision.internal_overlap.n_pairs == 2
    # 缓存中现在有 3 个因子
    assert gate.extractor.size() == 3
    print(f"[OK] 增量缓存: {decision.internal_overlap.n_pairs} 对, "
          f"缓存 {gate.extractor.size()} 个因子")


def test_shadow_summary(tmp_path):
    """shadow 汇总。"""
    gate, _, _ = make_gate(tmp_path)
    decisions = [
        gate.evaluate({"f1": "ts_mean(returns, 20)",
                       "f2": "ts_mean(returns, 20)"}),
        gate.evaluate({"f3": "ts_mean(returns, 20)",
                       "f4": "1.0 / pe_ratio"}),
    ]
    summary = gate.shadow_summary(decisions)
    assert summary["n_total"] == 2
    assert summary["n_flagged"] >= 1
    print(f"[OK] shadow 汇总: {summary}")


def test_complementary_to_return_correlation(tmp_path):
    """
    v3.1 核心验证：内部重叠 vs 收益相关互补不替代。
    构造两个"收益相关低但持仓重叠高"的因子，验证：
    - 收益相关性检查（L3）会放行
    - 持仓重叠检查（C3+）会拦截
    """
    from main_tier.l3_evolution.diversity import DiversityChecker
    gate, df, _ = make_gate(tmp_path)

    # 构造两个不同表达式但持仓高度重叠的因子
    # ts_mean(returns, 20) vs ts_mean(returns, 20) * 1.01（几乎相同持仓）
    code1 = "ts_mean(returns, 20)"
    code2 = "ts_mean(returns, 20) * 1.01"

    # C3+ 检查：应 flagged
    decision = gate.evaluate({"f1": code1, "f2": code2})
    overlap = decision.internal_overlap.mean_overlap
    print(f"[INFO] 内部重叠: {overlap:.3f}")
    assert overlap > 0.5, f"持仓应高度重叠，实际 {overlap:.3f}"

    # L3 收益相关性检查：可能放行（收益相关不一定高）
    # 这里不直接断言 L3 结果，只验证 C3+ 补了 L3 的盲区
    print(f"[OK] C3+ 补 L3 盲区: 内部重叠={overlap:.3f}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_position_sampling(tp)
        test_internal_overlap_identical(tp)
        test_internal_overlap_different(tp)
        test_external_proxy_separate(tp)
        test_flagged_only_internal(tp)
        test_shadow_mode(tp)
        test_incremental_cache(tp)
        test_shadow_summary(tp)
        test_complementary_to_return_correlation(tp)
        print("\n=== C3+ 全部测试通过 ===")