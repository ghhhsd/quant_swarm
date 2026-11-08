# tests for slow_loop
"""
C6 慢环 + 盲样抽检测试。

验证 v3.1 的九条纪律：
1. 前向验证输出 CI
2. 双向盲样：独立种子 + 匿名 ID
3. 独立存储：审计日志与生成记录分离
4. 生成侧只能读聚合视图
5. 审批协议：风控先、PM 后
6. 审批判据：预设统计标准
7. 审批记录可审计
8. shadow mode：产出不进主干
9. shadow 汇总
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from candidate_tier.c6_slow_loop import (
    ForwardTester, DualBlindAuditor, ApprovalProtocol,
    ApprovalRole, ApprovalDecision, SlowLoop,
)
import tempfile
from pathlib import Path


def test_forward_test_ci():
    """前向验证输出 CI。"""
    tester = ForwardTester(confidence=0.95, min_obs=30)
    rng = np.random.default_rng(42)
    returns = rng.normal(0.002, 0.01, 252)
    result = tester.test("f1", returns, test_period="monthly")
    assert result.forward_sharpe > 0
    assert result.forward_ci_low < result.forward_sharpe < result.forward_ci_high
    assert result.n_obs == 252
    print(f"[OK] 前向验证: sharpe={result.forward_sharpe:.3f}, "
          f"CI=[{result.forward_ci_low:.3f}, {result.forward_ci_high:.3f}]")


def test_forward_test_insufficient_obs():
    """数据不足时不强行出结论。"""
    tester = ForwardTester(min_obs=30)
    returns = np.array([0.001] * 10)
    result = tester.test("f1", returns)
    assert result.n_obs == 10
    assert result.forward_sharpe == 0.0
    print("[OK] 数据不足时返回 0")


def test_dual_blind_anonymity(tmp_path):
    """双向盲样：fid 映射到匿名 ID。"""
    audit_path = str(tmp_path / "audit.jsonl")
    auditor = DualBlindAuditor(auditor_seed=999, audit_log_path=audit_path)
    fids = [f"f{i}" for i in range(10)]
    anon = auditor._anonymize(fids)
    for fid, aid in anon.items():
        assert aid != fid
        assert aid.startswith("audit_")
    # 同一组 fid 同样种子 → 同样匿名映射
    anon2 = auditor._anonymize(fids)
    assert anon == anon2
    print(f"[OK] 匿名映射可复现: {list(anon.values())[:3]}...")


def test_isolated_audit_log(tmp_path):
    """独立审计日志：与生成记录分离。"""
    audit_path = str(tmp_path / "audit.jsonl")
    auditor = DualBlindAuditor(
        audit_ratio=0.2, min_audit=2, auditor_seed=999,
        audit_log_path=audit_path,
    )
    forward = {f"f{i}": {"forward_sharpe": 0.5, "forward_ci_low": 0.4}
               for i in range(10)}
    result = auditor.audit_round(forward)
    # 审计日志文件应存在
    assert Path(audit_path).exists()
    # 聚合视图可读
    agg = auditor.get_aggregate()
    assert agg["n_total_records"] == 10
    assert agg["n_audited"] == 2
    print(f"[OK] 独立审计日志: {agg}")


def test_aggregate_does_not_expose_fids(tmp_path):
    """聚合视图不暴露具体 fid。"""
    audit_path = str(tmp_path / "audit.jsonl")
    auditor = DualBlindAuditor(
        audit_ratio=0.2, min_audit=2, audit_log_path=audit_path)
    forward = {f"f{i}": {"forward_sharpe": 0.5, "forward_ci_low": 0.4}
               for i in range(10)}
    auditor.audit_round(forward)
    agg = auditor.get_aggregate()
    # 聚合视图只有数字，无 fid 字符串
    for key, val in agg.items():
        if isinstance(val, str):
            assert not val.startswith("f"), f"聚合视图不应含 fid: {key}={val}"
    print("[OK] 聚合视图不暴露 fid")


def test_approval_protocol_approved(tmp_path):
    """审批协议：风控 + PM 都通过。"""
    approval_path = str(tmp_path / "approval.jsonl")
    protocol = ApprovalProtocol(
        forward_ci_threshold=0.3,
        marginal_gain_threshold=0.02,
        approval_log_path=approval_path,
    )
    result = protocol.submit(
        fid="f1", forward_ci_low=0.5, audit_passed=True,
        capacity_tier="tier_medium", marginal_gain=0.05,
    )
    assert result["status"] == "approved"
    assert len(result["records"]) == 2
    assert result["records"][0].role == ApprovalRole.RISK_OFFICER
    assert result["records"][1].role == ApprovalRole.PORTFOLIO_MANAGER
    print(f"[OK] 审批通过: {len(result['records'])} 条记录")


def test_approval_protocol_rejected_by_risk(tmp_path):
    """风控先拒绝，PM 不审批。"""
    approval_path = str(tmp_path / "approval.jsonl")
    protocol = ApprovalProtocol(approval_log_path=approval_path)
    result = protocol.submit(
        fid="f1", forward_ci_low=0.1,  # 低于阈值
        audit_passed=True,
        capacity_tier="tier_medium", marginal_gain=0.05,
    )
    assert result["status"] == "rejected"
    assert len(result["records"]) == 1  # 只有风控记录
    assert result["records"][0].role == ApprovalRole.RISK_OFFICER
    print(f"[OK] 风控拒绝: {result['records'][0].reason}")


def test_approval_protocol_rejected_by_pm(tmp_path):
    """风控通过，PM 拒绝。"""
    approval_path = str(tmp_path / "approval.jsonl")
    protocol = ApprovalProtocol(approval_log_path=approval_path)
    result = protocol.submit(
        fid="f1", forward_ci_low=0.5, audit_passed=True,
        capacity_tier="tier_medium",
        marginal_gain=0.01,  # 低于阈值
    )
    assert result["status"] == "rejected"
    assert len(result["records"]) == 2  # 风控 + PM
    assert result["records"][1].role == ApprovalRole.PORTFOLIO_MANAGER
    print(f"[OK] PM 拒绝: {result['records'][1].reason}")


def test_approval_log_auditable(tmp_path):
    """审批记录可审计。"""
    approval_path = str(tmp_path / "approval.jsonl")
    protocol = ApprovalProtocol(approval_log_path=approval_path)
    protocol.submit("f1", 0.5, True, "tier_medium", 0.05)
    history = protocol.get_history("f1")
    assert len(history) == 2
    assert all(r.fid == "f1" for r in history)
    assert Path(approval_path).exists()
    print(f"[OK] 审批记录可审计: {len(history)} 条")


def test_slow_loop_shadow(tmp_path):
    """慢环 shadow mode。"""
    audit_path = str(tmp_path / "audit.jsonl")
    approval_path = str(tmp_path / "approval.jsonl")
    slow = SlowLoop(
        forward_tester=ForwardTester(min_obs=30),
        auditor=DualBlindAuditor(
            audit_ratio=0.2, min_audit=2,
            audit_log_path=audit_path,
        ),
        approval=ApprovalProtocol(approval_log_path=approval_path),
    )

    rng = np.random.default_rng(42)
    forward_returns = {
        f"f{i}": rng.normal(0.002, 0.01, 252) for i in range(10)
    }
    capacity_tiers = {f"f{i}": "tier_medium" for i in range(10)}
    marginal_gains = {f"f{i}": 0.05 for i in range(10)}

    result = slow.run_round(forward_returns, capacity_tiers, marginal_gains)
    assert result.shadow
    assert result.n_forward_tested == 10
    assert result.n_audited == 2
    print(f"[OK] 慢环 shadow: forward={result.n_forward_tested}, "
          f"audited={result.n_audited}, approved={result.n_approved}")


def test_slow_loop_summary(tmp_path):
    """shadow 跨轮汇总。"""
    audit_path = str(tmp_path / "audit.jsonl")
    approval_path = str(tmp_path / "approval.jsonl")
    slow = SlowLoop(
        auditor=DualBlindAuditor(
            audit_ratio=0.2, min_audit=2, audit_log_path=audit_path),
        approval=ApprovalProtocol(approval_log_path=approval_path),
    )
    rng = np.random.default_rng(42)
    results = []
    for _ in range(2):
        forward_returns = {f"f{i}": rng.normal(0.002, 0.01, 252)
                          for i in range(5)}
        capacity_tiers = {f"f{i}": "tier_medium" for i in range(5)}
        marginal_gains = {f"f{i}": 0.05 for i in range(5)}
        results.append(slow.run_round(forward_returns, capacity_tiers,
                                      marginal_gains))
    summary = slow.shadow_summary(results)
    assert summary["n_rounds"] == 2
    print(f"[OK] shadow 汇总: {summary}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_forward_test_ci()
        test_forward_test_insufficient_obs()
        test_dual_blind_anonymity(tp)
        test_isolated_audit_log(tp)
        test_aggregate_does_not_expose_fids(tp)
        test_approval_protocol_approved(tp)
        test_approval_protocol_rejected_by_risk(tp)
        test_approval_protocol_rejected_by_pm(tp)
        test_approval_log_auditable(tp)
        test_slow_loop_shadow(tp)
        test_slow_loop_summary(tp)
        print("\n=== C6 全部测试通过 ===")