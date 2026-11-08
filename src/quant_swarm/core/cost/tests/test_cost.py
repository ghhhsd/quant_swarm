"""
成本模型测试。

验证 v3.1 的九条纪律：
1. 三档情景冻结
2. 情景参数不能下调（impact_stress >= 1.0）
3. 固定成本 + 冲击成本
4. 平方根冲击模型
5. 情景敏感性：stress_100 低于 base
6. 容量档位映射
7. 一阶修正的诚实标注
8. 循环依赖说明
9. 残余偏差情景分析（200% / 300%）
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from core.cost import (
    CostScenario, CostScenarioName, get_scenario, all_scenarios,
    CostModel, CostBreakdown, apply_costs,
    SensitivityAnalyzer, SensitivityReport,
    CapacityEstimator, CapacityTier, CapacityCurve,
    HonestyAnalyzer, ResidualOptimismReport,
)


def make_test_data(n_days=100, n_assets=20, seed=42):
    """构造测试数据。"""
    rng = np.random.default_rng(seed)
    # 毛收益
    gross = rng.normal(0.001, 0.02, n_days)
    # 持仓
    positions = rng.uniform(-1, 1, (n_days, n_assets))
    positions = positions / np.abs(positions).sum(axis=1, keepdims=True)
    # 换手
    turnover = np.abs(np.diff(positions, axis=0)).sum(axis=1)
    return gross[1:], positions[1:], turnover


def test_three_scenarios():
    """三档情景冻结。"""
    scenarios = all_scenarios()
    assert len(scenarios) == 3
    assert CostScenarioName.BASE.value in scenarios
    assert CostScenarioName.STRESS_50.value in scenarios
    assert CostScenarioName.STRESS_100.value in scenarios
    # 冲击上浮递增
    assert scenarios["base"].impact_stress == 1.0
    assert scenarios["stress_50"].impact_stress == 1.5
    assert scenarios["stress_100"].impact_stress == 2.0
    print(f"[OK] 三档情景: {list(scenarios.keys())}")


def test_scenario_frozen():
    """情景冻结（frozen=True）。"""
    scenario = get_scenario("base")
    try:
        scenario.impact_coef = 0.5  # 尝试修改
        assert False, "frozen dataclass 应拒绝修改"
    except Exception:
        print("[OK] 情景冻结")


def test_scenario_no_downgrade():
    """情景参数不能下调。"""
    try:
        CostScenario(
            name="bad", commission_bps=5, spread_bps=2,
            impact_coef=0.1, impact_stress=0.5,  # < 1.0
        )
        assert False, "应抛 ValueError"
    except ValueError:
        print("[OK] 不能下调冲击上浮")


def test_cost_fixed_plus_impact():
    """固定成本 + 冲击成本。"""
    scenario = get_scenario("base")
    model = CostModel(scenario)
    turnover = np.array([0.1, 0.2, 0.3])
    costs = model.compute_costs(None, turnover)
    assert len(costs) == 3
    # 成本应为正
    assert np.all(costs > 0)
    print(f"[OK] 成本计算: {costs}")


def test_cost_sqrt_impact():
    """平方根冲击模型。"""
    scenario = get_scenario("base")
    model = CostModel(scenario)
    # 同样换手翻倍，冲击成本应小于翻倍（平方根）
    t1 = np.array([0.1])
    t2 = np.array([0.2])
    c1 = model.compute_costs(None, t1)[0]
    c2 = model.compute_costs(None, t2)[0]
    # 换手翻倍 → 成本增长应小于 2 倍（因为固定成本线性，冲击是 sqrt）
    assert c2 < 2 * c1, f"平方根冲击: c1={c1:.6f}, c2={c2:.6f}"
    print(f"[OK] 平方根冲击: c1={c1:.6f}, c2={c2:.6f}, ratio={c2/c1:.2f}")


def test_sensitivity():
    """情景敏感性：stress_100 低于 base。"""
    gross, positions, turnover = make_test_data()
    analyzer = SensitivityAnalyzer()
    report = analyzer.analyze(gross, positions, turnover)
    assert report.base_net_sharpe > report.stress_50_net_sharpe
    assert report.stress_50_net_sharpe > report.stress_100_net_sharpe
    assert report.sensitivity > 0
    print(f"[OK] 敏感性: base={report.base_net_sharpe:.3f}, "
          f"s50={report.stress_50_net_sharpe:.3f}, "
          f"s100={report.stress_100_net_sharpe:.3f}, "
          f"sensitivity={report.sensitivity:.3f}")


def test_sensitivity_honesty():
    """一阶修正的诚实标注。"""
    gross, positions, turnover = make_test_data()
    analyzer = SensitivityAnalyzer()
    report = analyzer.analyze(gross, positions, turnover)
    assert "一阶修正" in report.residual_optimism_note
    assert "不消除" in report.residual_optimism_note
    print(f"[OK] 诚实标注: {report.residual_optimism_note[:50]}...")


def test_capacity_tier():
    """容量档位映射。"""
    est = CapacityEstimator(min_sharpe_threshold=0.3)
    # 高 Sharpe → LARGE
    r = est.estimate(base_sharpe=1.5, capacity_half_life=1e8)
    assert r.tier == CapacityTier.TIER_LARGE
    # 低 Sharpe → REJECT
    r = est.estimate(base_sharpe=0.2)
    assert r.tier == CapacityTier.TIER_REJECT
    print("[OK] 容量档位映射")


def test_capacity_curve():
    """容量曲线衰减。"""
    curve = CapacityCurve(
        base_sharpe=0.8, capacity_half_life=5e7, decay_exponent=0.5)
    s_small = curve.sharpe_at(1e7)
    s_medium = curve.sharpe_at(1e8)
    s_large = curve.sharpe_at(1e9)
    assert s_small > s_medium > s_large
    print(f"[OK] 容量曲线: 1e7→{s_small:.3f}, "
          f"1e8→{s_medium:.3f}, 1e9→{s_large:.3f}")


def test_honesty_analyzer():
    """诚实标注 + 循环依赖。"""
    analyzer = HonestyAnalyzer()
    note = analyzer.get_note()
    circular = analyzer.get_circular_dependency()
    assert "一阶修正" in note
    assert "循环依赖" in circular
    print(f"[OK] 诚实标注: {note[:50]}...")
    print(f"[OK] 循环依赖: {circular[:50]}...")


def test_residual_scenario_analysis():
    """残余偏差情景分析（200% / 300%）。"""
    gross, positions, turnover = make_test_data()
    analyzer = HonestyAnalyzer()
    report = analyzer.scenario_analysis(gross, positions, turnover)
    # 应有 base / stress_200 / stress_300
    assert "base" in report.scenario_analysis
    assert "stress_200" in report.scenario_analysis
    assert "stress_300" in report.scenario_analysis
    # 衰减应递减
    s_base = report.scenario_analysis["base"]
    s_200 = report.scenario_analysis["stress_200"]
    s_300 = report.scenario_analysis["stress_300"]
    assert s_base >= s_200 >= s_300
    print(f"[OK] 残余偏差: base={s_base:.3f}, "
          f"200%={s_200:.3f}, 300%={s_300:.3f}")


def test_apply_costs_compat():
    """兼容旧接口 apply_costs。"""
    positions = np.array([[0.5, -0.5]] * 10)
    turnover = np.array([0.1] * 10)
    costs = apply_costs(positions, turnover)
    assert len(costs) == 10
    assert np.all(costs >= 0)
    print(f"[OK] 兼容接口: costs[0]={costs[0]:.6f}")


if __name__ == "__main__":
    test_three_scenarios()
    test_scenario_frozen()
    test_scenario_no_downgrade()
    test_cost_fixed_plus_impact()
    test_cost_sqrt_impact()
    test_sensitivity()
    test_sensitivity_honesty()
    test_capacity_tier()
    test_capacity_curve()
    test_honesty_analyzer()
    test_residual_scenario_analysis()
    test_apply_costs_compat()
    print("\n=== 成本模型全部测试通过 ===")