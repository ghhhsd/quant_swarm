"""
C5.5 实盘经济性测试。

验证 v3.1 的九条纪律：
1. 三档成本场景
2. 入库依据是 stress_100 net Sharpe，不是 base
3. 容量档位而非布尔
4. 容量档位映射到监控频率
5. 衰减监控触发 DECAYED 时写事件
6. 衰减事件回灌 Failure Trace
7. shadow mode：产出不进主干
8. 一阶修正的诚实标注
9. shadow 汇总
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from candidate_tier.c5_5_economic import (
    CostScenarioName, get_scenario, all_scenarios,
    CapacityEstimator, CapacityTier, CapacityCurve,
    DecayMonitor, DecayStatus,
    EconomicGate,
)
from core.data import generate_synthetic_data
from core.event_log import EventLog
import tempfile
from pathlib import Path


def make_gate(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    df = generate_synthetic_data(300, 50, seed=42)
    gate = EconomicGate(
        df=df, min_stress_sharpe=0.1,
        min_sharpe_threshold=0.3, event_log=log,
    )
    return gate, df, log


def test_three_scenarios():
    """三档成本场景。"""
    scenarios = all_scenarios()
    assert len(scenarios) == 3
    assert CostScenarioName.BASE.value in scenarios
    assert CostScenarioName.STRESS_50.value in scenarios
    assert CostScenarioName.STRESS_100.value in scenarios
    # 冲击上浮递增
    assert scenarios["base"].impact_stress == 1.0
    assert scenarios["stress_50"].impact_stress == 1.5
    assert scenarios["stress_100"].impact_stress == 2.0
    print(f"[OK] 三档场景: {list(scenarios.keys())}")


def test_capacity_curve():
    """容量曲线：Sharpe 随 AUM 衰减。"""
    curve = CapacityCurve(
        base_sharpe=0.8, capacity_half_life=5e7, decay_exponent=0.5)
    s_small = curve.sharpe_at(1e7)
    s_medium = curve.sharpe_at(1e8)
    s_large = curve.sharpe_at(1e9)
    assert s_small > s_medium > s_large, \
        f"Sharpe 应随 AUM 衰减: {s_small:.3f} > {s_medium:.3f} > {s_large:.3f}"
    print(f"[OK] 容量曲线: 1e7→{s_small:.3f}, 1e8→{s_medium:.3f}, 1e9→{s_large:.3f}")


def test_capacity_tier():
    """容量档位映射。"""
    est = CapacityEstimator(min_sharpe_threshold=0.3)
    # 高 Sharpe → LARGE
    r = est.estimate(base_sharpe=1.5, capacity_half_life=1e8)
    assert r.tier == CapacityTier.TIER_LARGE, f"高 Sharpe 应为 LARGE，实际 {r.tier}"
    # 低 Sharpe → REJECT
    r = est.estimate(base_sharpe=0.2)
    assert r.tier == CapacityTier.TIER_REJECT, f"低 Sharpe 应为 REJECT，实际 {r.tier}"
    print("[OK] 容量档位映射")


def test_monitor_frequency():
    """档位映射到监控频率。"""
    est = CapacityEstimator(min_sharpe_threshold=0.3)
    r_large = est.estimate(base_sharpe=1.5, capacity_half_life=1e8)
    r_small = est.estimate(base_sharpe=0.5)
    assert r_large.monitor_frequency_days > r_small.monitor_frequency_days
    print(f"[OK] 监控频率: LARGE={r_large.monitor_frequency_days}天, "
          f"SMALL={r_small.monitor_frequency_days}天")


def test_decay_monitor_healthy():
    """衰减监控：IC 正常。"""
    log = EventLog()
    monitor = DecayMonitor(ic_threshold=0.0, n_periods_below_zero=3,
                           event_log=log)
    ic = np.array([0.05, 0.04, 0.03, 0.02, 0.01])
    result = monitor.check("f1", ic, monitor_frequency_days=30)
    assert result.status == DecayStatus.HEALTHY
    print(f"[OK] IC 正常: {result.status.value}")


def test_decay_monitor_decayed():
    """衰减监控：IC 跌破阈值，写事件。"""
    log = EventLog()
    monitor = DecayMonitor(ic_threshold=0.0, n_periods_below_zero=3,
                           event_log=log)
    ic = np.array([-0.01, -0.02, -0.03, -0.01])
    result = monitor.check("f1", ic, monitor_frequency_days=30)
    assert result.status == DecayStatus.DECAYED
    # 应有 decay_event
    events = log.query("decay_event")
    assert len(events) == 1
    print(f"[OK] 衰减触发: {result.status.value}, 事件 {len(events)} 条")


def test_decay_no_monitor_for_reject():
    """REJECT 档位不监控。"""
    monitor = DecayMonitor()
    result = monitor.check("f1", np.array([0.05, 0.04]), monitor_frequency_days=0)
    assert result.status == DecayStatus.HEALTHY
    assert "REJECT" in result.reason
    print("[OK] REJECT 档位不监控")


def test_economic_gate_shadow(tmp_path):
    """shadow mode：决策恒为 shadow。"""
    gate, _, log = make_gate(tmp_path)
    decision = gate.evaluate("f1", "ts_mean(returns, 20)")
    assert decision.shadow, "候选层决策应恒为 shadow"
    events = log.query("economic_shadow_decision")
    assert len(events) == 1
    print("[OK] shadow mode 只记录不阻断")


def test_stress_100_judgment(tmp_path):
    """入库依据是 stress_100，不是 base。"""
    gate, _, _ = make_gate(tmp_path)
    decision = gate.evaluate("f1", "ts_mean(returns, 20)")
    # 三档 Sharpe 应有：base > stress_50 > stress_100
    s_base = decision.scenario_sharpes["base"].net_sharpe
    s_50 = decision.scenario_sharpes["stress_50"].net_sharpe
    s_100 = decision.scenario_sharpes["stress_100"].net_sharpe
    assert s_base >= s_50 >= s_100, \
        f"Sharpe 应随压力递增下降: {s_base:.3f} >= {s_50:.3f} >= {s_100:.3f}"
    # passed 判定用 s_100
    if s_100 < 0.1:
        assert not decision.passed
    print(f"[OK] stress_100 判定: base={s_base:.3f}, s50={s_50:.3f}, "
          f"s100={s_100:.3f}, passed={decision.passed}")


def test_residual_optimism_note(tmp_path):
    """一阶修正的诚实标注。"""
    gate, _, _ = make_gate(tmp_path)
    decision = gate.evaluate("f1", "ts_mean(returns, 20)")
    assert "一阶修正" in decision.residual_optimism_note
    assert "不消除" in decision.residual_optimism_note
    print(f"[OK] 诚实标注: {decision.residual_optimism_note[:50]}...")


def test_shadow_summary(tmp_path):
    """shadow 汇总。"""
    gate, _, _ = make_gate(tmp_path)
    decisions = [
        gate.evaluate("f1", "ts_mean(returns, 20)"),
        gate.evaluate("f2", "1.0 / pe_ratio"),
        gate.evaluate("f3", "ts_std(returns, 20)"),
    ]
    summary = gate.shadow_summary(decisions)
    assert summary["n_total"] == 3
    assert summary["n_passed"] >= 0
    assert "tier_breakdown" in summary
    print(f"[OK] shadow 汇总: {summary}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_three_scenarios()
        test_capacity_curve()
        test_capacity_tier()
        test_monitor_frequency()
        test_decay_monitor_healthy()
        test_decay_monitor_decayed()
        test_decay_no_monitor_for_reject()
        test_economic_gate_shadow(tp)
        test_stress_100_judgment(tp)
        test_residual_optimism_note(tp)
        test_shadow_summary(tp)
        print("\n=== C5.5 全部测试通过 ===")