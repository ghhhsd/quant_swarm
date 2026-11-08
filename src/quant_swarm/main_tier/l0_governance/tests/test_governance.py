"""
L0 治理层测试。
验证 v3.1 的六条纪律：
1. 三层配额消耗 + 超额抛异常
2. 动态回收不虚增
3. 试验 N 按批次记账（跨轮累积，不按轮重置）
4. 阈值 commit 后禁止二次 commit
5. 不变核心篡改被拦截
6. 难度信号只调整搜索侧，不碰验收侧
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from main_tier.l0_governance import GovernanceLayer, BudgetExceeded
from main_tier.l0_governance.threshold_commit import ThresholdViolation
from main_tier.l0_governance.invariant import InvariantViolation
from main_tier.l1_favor import FailureType
from core.config import KernelConfig
from core.event_log import EventLog


def make_gov(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    return GovernanceLayer(KernelConfig(), event_log=log)


def test_quota_consume_and_exceed(tmp_path):
    gov = make_gov(tmp_path)
    q = gov.start_round()
    assert q.token_total == 200_000
    gov.consume_token(100_000)
    assert gov.quota.current.token_remaining == 100_000
    try:
        gov.consume_token(150_000)
        assert False, "应抛 BudgetExceeded"
    except BudgetExceeded as e:
        assert e.resource == "token"
    print("[OK] 配额消耗 + 超额拦截")


def test_reclaim_no_inflation(tmp_path):
    gov = make_gov(tmp_path)
    gov.start_round()
    gov.consume_token(100_000)
    gov.reclaim(token=50_000)
    assert gov.quota.current.token_remaining == 150_000
    # 回收超过总量不应虚增
    gov.reclaim(token=1_000_000)
    assert gov.quota.current.token_remaining == 200_000
    print("[OK] 动态回收不虚增")


def test_trial_n_by_batch(tmp_path):
    """N 按批次记账，跨轮累积（同批次），新批次重置。"""
    gov = make_gov(tmp_path)
    # 轮 1：batch_001 记 3 次
    gov.start_round()
    for i in range(3):
        gov.record_trial("batch_001", f"h_{i}")
    gov.end_round()
    assert gov.get_trial_n("batch_001") == 3

    # 轮 2：同批次继续累积
    gov.start_round()
    gov.record_trial("batch_001", "h_3")
    gov.end_round()
    assert gov.get_trial_n("batch_001") == 4, "同批次应跨轮累积"

    # 轮 3：新批次重置
    gov.start_round()
    gov.record_trial("batch_002", "h_4")
    gov.end_round()
    assert gov.get_trial_n("batch_002") == 1
    assert gov.get_trial_n("batch_001") == 4, "旧批次不受影响"
    print("[OK] 试验 N 按搜索批次记账")


def test_threshold_commit_once(tmp_path):
    gov = make_gov(tmp_path)
    ct = gov.commit_thresholds(budget_n=50, acceptable_fdr=0.10,
                               acceptable_dsr=0.95, note="首轮")
    assert ct.commit_hash
    assert gov.get_committed_thresholds().commit_hash == ct.commit_hash
    # 二次 commit 应抛异常
    try:
        gov.commit_thresholds(budget_n=100, acceptable_fdr=0.10,
                              acceptable_dsr=0.95)
        assert False, "应抛 ThresholdViolation"
    except ThresholdViolation:
        print("[OK] 阈值 commit 后禁止二次 commit")


def test_invariant_tamper_detected(tmp_path):
    gov = make_gov(tmp_path)
    core = gov.get_invariant(log_access=False)
    # 正常校验通过
    gov.assert_invariant_unchanged(core)
    # 篡改后校验失败
    tampered = dict(core)
    tampered["risk_breaker"] = dict(core["risk_breaker"])
    tampered["risk_breaker"]["max_leverage"] = 10.0
    try:
        gov.assert_invariant_unchanged(tampered)
        assert False, "应抛 InvariantViolation"
    except InvariantViolation as e:
        print(f"[OK] 不变核心篡改被拦截: {e}")


def test_difficulty_signal_search_side_only(tmp_path):
    gov = make_gov(tmp_path)
    # 连续 3 轮低增长
    adj = gov.evaluate_difficulty([0, 1, 0, 1])
    assert adj.exploration_ratio > 0.2, "低增长应提高探索配额"
    assert adj.escalate, "连续低增长应触发告警"
    # 验证：难度信号不返回任何"阈值调整"字段
    assert not hasattr(adj, "threshold") and not hasattr(adj, "dsr_threshold")
    print(f"[OK] 难度信号只调搜索侧: exploration={adj.exploration_ratio}, "
          f"escalate={adj.escalate}")


def test_failure_pruning(tmp_path):
    gov = make_gov(tmp_path)
    gov.start_round()
    # 语法错误重试 3 次
    for i in range(3):
        should_prune = gov.record_failure("h_syntax", FailureType.SYNTAX)
        assert not should_prune, f"第 {i+1} 次不应剪枝"
    assert gov.record_failure("h_syntax", FailureType.SYNTAX), "第 4 次应剪枝"
    assert gov.is_pruned("h_syntax")

    # 门控拒绝第一次就剪枝
    assert gov.record_failure("h_gate", FailureType.GATE_REJECTED), "门控拒绝立即剪枝"
    assert gov.is_pruned("h_gate")

    # 不同因子独立记账
    assert not gov.is_pruned("h_other"), "其他因子不受影响"
    print("[OK] 差异化失败剪枝")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_quota_consume_and_exceed(tp)
        test_reclaim_no_inflation(tp)
        test_trial_n_by_batch(tp)
        test_threshold_commit_once(tp)
        test_invariant_tamper_detected(tp)
        test_difficulty_signal_search_side_only(tp)
        test_failure_pruning(tp)
        print("\n=== L0 全部测试通过 ===")