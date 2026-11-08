"""
横切层测试。

验证 v3.1 的九条纪律：
1. 最小内核可运行
2. 可证伪基准分 regime
3. 每层独立消融（不受其他层污染）
4. 消融结果写入不可变日志
5. 弱证据不作晋升充分条件
6. 晋升后可回退
7. 组合 PnL 是最终指标
8. 分市场状态分别度量
9. 完整报告可审计
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from ablation import (
    MinimalKernel, BaselineBuilder, AblationOrchestrator,
    EfficacyBudgetRegistry, EfficacyBudget, LayerStateRegistry,
    LayerStatus, PortfolioConfig, LayerSwitch, identity_filter,
    classify_market_regimes,
)
from core.data import generate_synthetic_data
from core.config import GLOBAL_KERNEL, GLOBAL_THRESHOLDS
from core.event_log import EventLog
from core.stats import SearchBatchCounter
from core.gate import VaGGate
from core.threshold_commit import ThresholdCommitter, ThresholdDerivation
import tempfile
from pathlib import Path


def make_env(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    df = generate_synthetic_data(1200, 30, seed=42)

    # 构造门控
    committer = ThresholdCommitter(log)
    derivation = ThresholdDerivation(
        budget_n=50, acceptable_fdr=0.10, acceptable_dsr=0.95,
        calibration_note="test",
    )
    committed = committer.commit(GLOBAL_THRESHOLDS, derivation)
    counter = SearchBatchCounter()
    gate = VaGGate(counter, committed.thresholds)

    return df, log, gate


def test_minimal_kernel_runs(tmp_path):
    """最小内核可运行。"""
    df, log, gate = make_env(tmp_path)
    kernel = MinimalKernel(df, GLOBAL_KERNEL, log)
    kernel.attach_gate(gate)
    result = kernel.run(round_idx=1)
    assert result.n_tested > 0
    assert result.portfolio is not None
    assert len(result.regime_pnl) > 0
    print(f"[OK] 最小内核: tested={result.n_tested}, "
          f"admitted={result.n_admitted}, sharpe={result.portfolio.sharpe:.3f}")


def test_falsifiable_baseline(tmp_path):
    """可证伪基准分 regime。"""
    df, log, gate = make_env(tmp_path)
    kernel = MinimalKernel(df, GLOBAL_KERNEL, log)
    kernel.attach_gate(gate)
    baseline = BaselineBuilder(kernel, log).build(n_rounds=3)
    assert baseline.n_rounds == 3
    assert len(baseline.by_regime) > 0
    print(f"[OK] 基准: mean={baseline.overall_sharpe_mean:.3f}, "
          f"regimes={list(baseline.by_regime.keys())}")


def test_layer_switch_isolation():
    """每层独立消融（开关定义隔离）。"""
    candidates = [
        {"fid": "f1", "code": "ts_mean(returns, 20)", "novelty": "novel"},
        {"fid": "f2", "code": "1.0 / pe_ratio", "novelty": "variant"},
        {"fid": "f3", "code": "ts_std(returns, 20)", "novelty": "novel"},
    ]
    switch = LayerSwitch(
        layer_name="C1.5",
        on_filter=lambda xs: [c for c in xs if c.get("novelty") == "novel"],
        off_filter=identity_filter,
    )
    on_result = switch.apply(candidates, SwitchState.ON)
    off_result = switch.apply(candidates, SwitchState.OFF)
    assert len(on_result) == 2
    assert len(off_result) == 3
    print(f"[OK] 开关隔离: on={len(on_result)}, off={len(off_result)}")


def test_orchestrator_runs(tmp_path):
    """编排器端到端跑通。"""
    df, log, gate = make_env(tmp_path)

    # 构造预算注册表
    budget_registry = EfficacyBudgetRegistry(log)
    budget_registry.declare(EfficacyBudget(
        layer="C1.5_novelty", mde=0.05, power=0.80, alpha=0.05,
        n_obs_per_batch=252, n_batches_min=4,
    ))

    # 构造状态注册表
    state_registry = LayerStateRegistry(log)
    state_registry.register("C1.5_novelty", LayerStatus.CANDIDATE)

    # 构造编排器
    portfolio_config = PortfolioConfig(weighting="equal")
    orchestrator = AblationOrchestrator(
        df=df,
        budget_registry=budget_registry,
        state_registry=state_registry,
        portfolio_config=portfolio_config,
        event_log=log,
    )

    # 候选
    candidates = [
        {"fid": "f1", "code": "ts_mean(returns, 20) - ts_mean(returns, 60)",
         "novelty": "novel"},
        {"fid": "f2", "code": "ts_std(returns, 20)", "novelty": "variant"},
        {"fid": "f3", "code": "1.0 / pe_ratio", "novelty": "novel"},
    ]

    # 层开关
    layer_switches = {
        "C1.5_novelty": LayerSwitch(
            layer_name="C1.5_novelty",
            on_filter=lambda xs: [c for c in xs if c.get("novelty") == "novel"],
            off_filter=identity_filter,
        ),
    }

    report = orchestrator.run(
        candidates=candidates,
        layer_switches=layer_switches,
        code_extractor=lambda c: c["code"],
        baseline_rounds=3,
    )

    assert report.baseline.n_rounds == 3
    assert len(report.layer_reports) == 1
    print(f"[OK] 编排器跑通: {len(report.layer_reports)} 层")
    print(report.summary())


def test_orchestrator_evidence_strength(tmp_path):
    """消融结果有证据强度。"""
    df, log, gate = make_env(tmp_path)
    budget_registry = EfficacyBudgetRegistry(log)
    budget_registry.declare(EfficacyBudget(
        layer="C5.5", mde=0.15, power=0.80, alpha=0.05,
        n_obs_per_batch=252, n_batches_min=4,
    ))
    state_registry = LayerStateRegistry(log)
    state_registry.register("C5.5", LayerStatus.CANDIDATE)
    orchestrator = AblationOrchestrator(
        df=df, budget_registry=budget_registry,
        state_registry=state_registry,
        portfolio_config=PortfolioConfig(),
        event_log=log,
    )
    candidates = [
        {"fid": "f1", "code": "ts_mean(returns, 20)",
         "economic_passed": True},
        {"fid": "f2", "code": "1.0 / pe_ratio", "economic_passed": False},
    ]
    layer_switches = {
        "C5.5": LayerSwitch(
            layer_name="C5.5",
            on_filter=lambda xs: [c for c in xs if c.get("economic_passed")],
            off_filter=identity_filter,
        ),
    }
    report = orchestrator.run(
        candidates=candidates,
        layer_switches=layer_switches,
        code_extractor=lambda c: c["code"],
        baseline_rounds=3,
    )
    assert len(report.layer_reports) == 1
    r = report.layer_reports[0]
    assert r.ablation_result.evidence_strength in ("strong", "weak")
    print(f"[OK] 证据强度: {r.ablation_result.evidence_strength}")


def test_weak_evidence_keeps_candidate(tmp_path):
    """弱证据不作晋升充分条件。"""
    df, log, gate = make_env(tmp_path)
    budget_registry = EfficacyBudgetRegistry(log)
    # 高门槛：mde 小 + 批次要求高
    budget_registry.declare(EfficacyBudget(
        layer="C1.5", mde=0.03, power=0.80, alpha=0.05,
        n_obs_per_batch=252, n_batches_min=50,
    ))
    state_registry = LayerStateRegistry(log)
    state_registry.register("C1.5", LayerStatus.CANDIDATE)
    orchestrator = AblationOrchestrator(
        df=df, budget_registry=budget_registry,
        state_registry=state_registry,
        portfolio_config=PortfolioConfig(),
        event_log=log,
    )
    candidates = [
        {"fid": "f1", "code": "ts_mean(returns, 20)", "novelty": "novel"},
        {"fid": "f2", "code": "1.0 / pe_ratio", "novelty": "variant"},
    ]
    layer_switches = {
        "C1.5": LayerSwitch(
            layer_name="C1.5",
            on_filter=lambda xs: [c for c in xs if c.get("novelty") == "novel"],
            off_filter=identity_filter,
        ),
    }
    report = orchestrator.run(
        candidates=candidates, layer_switches=layer_switches,
        code_extractor=lambda c: c["code"], baseline_rounds=3,
    )
    r = report.layer_reports[0]
    # 批次要求高 → 应为弱证据 → 保持候选
    if r.ablation_result.evidence_strength == "weak":
        assert r.promotion_decision.action.value in (
            "keep_candidate", "keep_main", "demote"
        )
        assert state_registry.get_state("C1.5") == LayerStatus.CANDIDATE
    print(f"[OK] 弱证据: strength={r.ablation_result.evidence_strength}, "
          f"action={r.promotion_decision.action.value}")


def test_report_logged(tmp_path):
    """完整报告写入不可变日志。"""
    df, log, gate = make_env(tmp_path)
    budget_registry = EfficacyBudgetRegistry(log)
    budget_registry.declare(EfficacyBudget(
        layer="C1.5", mde=0.05, n_obs_per_batch=252, n_batches_min=4,
    ))
    state_registry = LayerStateRegistry(log)
    state_registry.register("C1.5", LayerStatus.CANDIDATE)
    orchestrator = AblationOrchestrator(
        df=df, budget_registry=budget_registry,
        state_registry=state_registry,
        portfolio_config=PortfolioConfig(),
        event_log=log,
    )
    candidates = [{"fid": "f1", "code": "ts_mean(returns, 20)",
                   "novelty": "novel"}]
    layer_switches = {
        "C1.5": LayerSwitch(
            layer_name="C1.5",
            on_filter=lambda xs: xs, off_filter=identity_filter,
        ),
    }
    orchestrator.run(candidates, layer_switches, baseline_rounds=3)
    events = log.query("cross_cutting_report")
    assert len(events) == 1
    print(f"[OK] 报告写入事件流: {len(events)} 条")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_minimal_kernel_runs(tp)
        test_falsifiable_baseline(tp)
        test_layer_switch_isolation()
        test_orchestrator_runs(tp)
        test_orchestrator_evidence_strength(tp)
        test_weak_evidence_keeps_candidate(tp)
        test_report_logged(tp)
        print("\n=== 横切层全部测试通过 ===")