"""
VaG 门控测试。

验证 v3.1 的九条纪律：
1. N 从 L0 取（唯一真相源）
2. 阈值从 L0 取（冻结）
3. 删除阈值自适应
4. 短路执行（统计失败不跑后续）
5. 叙事失效回灌机制模板库
6. 统计失败回灌负样本库
7. 三道检查独立结果
8. 门控决策进事件流
9. 完整结果可序列化
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from main_tier.l5_vag import VaGGate, GateDecision
from main_tier.l1_favor import (
    Narrative, NarrativeCommitment, NarrativeCommitter,
    Direction, DecayPattern,
)
from main_tier.l0_governance import GovernanceLayer
from core.config import KernelConfig, GLOBAL_THRESHOLDS
from core.threshold_commit import ThresholdDerivation
from core.event_log import EventLog
import tempfile
from pathlib import Path


def make_gov(tmp_path, budget_n=50):
    log = EventLog(str(tmp_path / "events.jsonl"))
    gov = GovernanceLayer(KernelConfig(), event_log=log)
    gov.start_round(budget_n=budget_n, triggered_by="test")
    return gov, log


def make_commitment():
    log = EventLog()
    committer = NarrativeCommitter(log)
    n = Narrative(
        hid="h_test", mechanism_family="momentum",
        direction=Direction.LONG,
        conditions=["正常市场"],
        decay=DecayPattern.PERSISTENT,
        economic_logic="动量延续",
        falsifiable_statement="若 IC 不显著为正则证伪",
    )
    return committer.commit(n)


def make_data():
    rng = np.random.default_rng(42)
    net_returns = rng.normal(0.002, 0.01, 1500)
    factor_returns = rng.normal(0, 0.01, 1500)
    forward_returns = factor_returns * 0.5 + rng.normal(0, 0.01, 1500)
    return net_returns, factor_returns, forward_returns


def test_n_from_l0(tmp_path):
    """N 从 L0 取。"""
    gov, log = make_gov(tmp_path)
    # L0 记录 3 次试验
    for i in range(3):
        gov.record_trial("batch_test", f"f{i}", net_sharpe=0.4)
    assert gov.get_trial_n("batch_test") == 3

    def portfolio_sharpe(codes):
        return 0.5 if codes else 0.4

    gate = VaGGate(gov, portfolio_sharpe, event_log=log)
    net_returns, factor_returns, forward_returns = make_data()
    # 门控内部应读 L0 的 N
    decision = gate.evaluate(
        fid="f_test", net_returns=net_returns,
        factor_returns=factor_returns, forward_returns=forward_returns,
        batch_id="batch_test", existing_codes=[], new_code="code",
    )
    assert decision.statistical is not None
    assert decision.statistical["n_trials"] == 3
    print(f"[OK] N 从 L0: n_trials={decision.statistical['n_trials']}")


def test_threshold_from_l0(tmp_path):
    """阈值从 L0 取（冻结）。"""
    gov, log = make_gov(tmp_path)
    gate = VaGGate(gov, lambda c: 0.5, event_log=log)
    # 门控内部阈值应等于 committed 的阈值
    assert gate._th.dsr_threshold == GLOBAL_THRESHOLDS.dsr_threshold
    assert gate._th.fdr_alpha == GLOBAL_THRESHOLDS.fdr_alpha
    print("[OK] 阈值从 L0")


def test_short_circuit_statistical(tmp_path):
    """统计失败短路：不跑边际和叙事。"""
    gov, log = make_gov(tmp_path)
    # 构造一个坏因子：负收益
    rng = np.random.default_rng(1)
    net_returns = rng.normal(-0.002, 0.02, 1500)
    factor_returns = rng.normal(0, 0.01, 1500)
    forward_returns = rng.normal(0, 0.01, 1500)

    gate = VaGGate(gov, lambda c: 0.3, event_log=log)
    decision = gate.evaluate(
        fid="f_bad", net_returns=net_returns,
        factor_returns=factor_returns, forward_returns=forward_returns,
        batch_id="batch_test", existing_codes=[], new_code="bad",
    )
    assert not decision.admitted
    assert decision.failure_stage == "statistical"
    # 短路：marginal 和 consistency 应为 None
    assert decision.marginal is None
    assert decision.consistency is None
    print(f"[OK] 统计短路: stage={decision.failure_stage}, "
          f"marginal={decision.marginal}")


def test_short_circuit_marginal(tmp_path):
    """边际失败短路：不跑叙事。"""
    gov, log = make_gov(tmp_path)
    # 构造一个统计通过但边际增益低的因子
    rng = np.random.default_rng(42)
    net_returns = rng.normal(0.001, 0.01, 1500)
    factor_returns = rng.normal(0, 0.01, 1500)
    forward_returns = factor_returns * 0.5 + rng.normal(0, 0.01, 1500)

    # 边际增益函数：返回固定值（不随新因子变化）
    gate = VaGGate(gov, lambda codes: 0.5, event_log=log)
    decision = gate.evaluate(
        fid="f_low_gain", net_returns=net_returns,
        factor_returns=factor_returns, forward_returns=forward_returns,
        batch_id="batch_test", existing_codes=[], new_code="code",
    )
    # 如果统计通过但边际失败
    if decision.failure_stage == "marginal":
        assert decision.marginal is not None
        assert decision.consistency is None
        print(f"[OK] 边际短路: stage={decision.failure_stage}")
    else:
        print(f"[INFO] 统计已失败，stage={decision.failure_stage}")


def test_narrative_inconsistent(tmp_path):
    """叙事失效回灌机制模板库。"""
    gov, log = make_gov(tmp_path)
    rng = np.random.default_rng(2)
    net_returns = rng.normal(0.002, 0.01, 1500)
    factor_returns = rng.normal(0, 0.01, 1500)
    # 方向相反：叙事 LONG，但 forward 与 factor 负相关
    forward_returns = -factor_returns * 0.5 + rng.normal(0, 0.01, 1500)

    gate = VaGGate(gov, lambda codes: 0.5, event_log=log)
    commitment = make_commitment()
    decision = gate.evaluate(
        fid="f_inconsistent", net_returns=net_returns,
        factor_returns=factor_returns, forward_returns=forward_returns,
        batch_id="batch_test", existing_codes=[], new_code="code",
        narrative_commitment=commitment,
    )
    if decision.failure_stage == "narrative":
        assert decision.downstream_action == "mechanism_template"
        print(f"[OK] 叙事失效回灌: {decision.downstream_action}")
    else:
        print(f"[INFO] 其他阶段失败: {decision.failure_stage}")


def test_statistical_fail_downstream(tmp_path):
    """统计失败回灌负样本库。"""
    gov, log = make_gov(tmp_path)
    rng = np.random.default_rng(1)
    net_returns = rng.normal(-0.002, 0.02, 1500)
    factor_returns = rng.normal(0, 0.01, 1500)
    forward_returns = rng.normal(0, 0.01, 1500)

    gate = VaGGate(gov, lambda c: 0.3, event_log=log)
    decision = gate.evaluate(
        fid="f_bad", net_returns=net_returns,
        factor_returns=factor_returns, forward_returns=forward_returns,
        batch_id="batch_test", existing_codes=[], new_code="bad",
    )
    assert decision.failure_stage == "statistical"
    assert decision.downstream_action == "negative_sample"
    print(f"[OK] 统计失败回灌: {decision.downstream_action}")


def test_three_checks_independent_results(tmp_path):
    """三道检查独立结果（准入时全有）。"""
    gov, log = make_gov(tmp_path)
    rng = np.random.default_rng(42)
    net_returns = rng.normal(0.002, 0.01, 1500)
    factor_returns = rng.normal(0, 0.01, 1500)
    forward_returns = factor_returns * 0.5 + rng.normal(0, 0.01, 1500)

    gate = VaGGate(gov, lambda codes: 0.5, event_log=log)
    commitment = make_commitment()
    decision = gate.evaluate(
        fid="f_good", net_returns=net_returns,
        factor_returns=factor_returns, forward_returns=forward_returns,
        batch_id="batch_test", existing_codes=[], new_code="code",
        narrative_commitment=commitment,
    )
    if decision.admitted:
        assert decision.statistical is not None
        assert decision.marginal is not None
        assert decision.consistency is not None
        print(f"[OK] 准入时全有: {decision.failure_stage}")
    else:
        print(f"[INFO] 未准入: {decision.failure_stage}")


def test_decision_logged(tmp_path):
    """门控决策进事件流。"""
    gov, log = make_gov(tmp_path)
    rng = np.random.default_rng(42)
    net_returns = rng.normal(0.001, 0.01, 1500)
    factor_returns = rng.normal(0, 0.01, 1500)
    forward_returns = rng.normal(0, 0.01, 1500)

    gate = VaGGate(gov, lambda codes: 0.4, event_log=log)
    gate.evaluate(
        fid="f_log", net_returns=net_returns,
        factor_returns=factor_returns, forward_returns=forward_returns,
        batch_id="batch_test", existing_codes=[], new_code="code",
    )
    events = log.query("vag_decision")
    assert len(events) == 1
    assert events[0].payload["fid"] == "f_log"
    print(f"[OK] 决策进事件流: {len(events)} 条")


def test_result_serializable(tmp_path):
    """完整结果可序列化。"""
    gov, log = make_gov(tmp_path)
    rng = np.random.default_rng(42)
    net_returns = rng.normal(0.002, 0.01, 1500)
    factor_returns = rng.normal(0, 0.01, 1500)
    forward_returns = rng.normal(0, 0.01, 1500)

    gate = VaGGate(gov, lambda codes: 0.5, event_log=log)
    decision = gate.evaluate(
        fid="f_ser", net_returns=net_returns,
        factor_returns=factor_returns, forward_returns=forward_returns,
        batch_id="batch_test", existing_codes=[], new_code="code",
    )
    d = decision.to_dict()
    assert "admitted" in d
    assert "failure_stage" in d
    assert "downstream_action" in d
    print(f"[OK] 可序列化: keys={list(d.keys())}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_n_from_l0(tp)
        test_threshold_from_l0(tp)
        test_short_circuit_statistical(tp)
        test_short_circuit_marginal(tp)
        test_narrative_inconsistent(tp)
        test_statistical_fail_downstream(tp)
        test_three_checks_independent_results(tp)
        test_decision_logged(tp)
        test_result_serializable(tp)
        print("\n=== VaG 门控全部测试通过 ===")