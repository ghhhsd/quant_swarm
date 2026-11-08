"""
L6 测试。

验证 v3.1 的九条纪律：
1. 内容哈希含 LLM 调用记录（路 A）
2. 状态机合法转移
3. 状态机非法转移被拒
4. 状态机幂等键防重复
5. 事件溯源投影可从事件流重建
6. 投影与事件流一致性校验
7. 快环只产出 DRAFT，不做上线决策
8. 慢环前向验证 + 盲样抽检
9. 盲样双向：抽检者与生成侧隔离
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from main_tier.l6_skillstore import (
    SkillStore, SkillState, SkillStateMachine, SkillProjector,
    compute_skill_hash, llm_calls_hash,
    FastLoop, SlowLoop, BlindAuditor, SlowLoopResult,
)
from main_tier.l2_agents.llm_client import LLMCallRecord
from core.event_log import EventLog
import tempfile
from pathlib import Path


def make_event_log(tmp_path, name="events.jsonl"):
    return EventLog(str(tmp_path / name))


def make_llm_calls(n=2):
    return [
        LLMCallRecord(
            agent_role="analyst", model="stub", temperature=0.5,
            prompt=f"prompt_{i}", output=f"output_{i}",
            tokens_in=10, tokens_out=20, seed=42,
        ) for i in range(n)
    ]


def test_content_hash_includes_llm_calls():
    """路 A：LLM 调用记录纳入内容哈希。"""
    calls1 = make_llm_calls(2)
    calls2 = make_llm_calls(2)
    # 相同调用 → 相同哈希
    h1 = compute_skill_hash("ts_mean(returns, 20)", {"dsr": 0.95}, calls1)
    h2 = compute_skill_hash("ts_mean(returns, 20)", {"dsr": 0.95}, calls2)
    assert h1 == h2, "相同 LLM 调用应产生相同哈希"

    # 不同调用 → 不同哈希
    calls3 = [
        LLMCallRecord("analyst", "stub", 0.5, "different", "out",
                      10, 20, seed=42),
    ]
    h3 = compute_skill_hash("ts_mean(returns, 20)", {"dsr": 0.95}, calls3)
    assert h3 != h1, "不同 LLM 调用应产生不同哈希"
    print(f"[OK] 内容哈希含 LLM 调用记录")


def test_state_machine_valid():
    sm = SkillStateMachine()
    t = sm.transition("f1", SkillState.ACCEPTED, "test")
    assert t.from_state == SkillState.DRAFT
    assert t.to_state == SkillState.ACCEPTED
    assert sm.get_state("f1") == SkillState.ACCEPTED

    t = sm.transition("f1", SkillState.COMMITTED, "上线")
    assert sm.get_state("f1") == SkillState.COMMITTED
    print("[OK] 状态机合法转移")


def test_state_machine_invalid():
    sm = SkillStateMachine()
    sm.transition("f1", SkillState.ACCEPTED, "test")
    # COMMITTED 是终态，不能再转
    sm.transition("f1", SkillState.COMMITTED, "上线")
    try:
        sm.transition("f1", SkillState.REJECTED, "不应该")
        assert False, "COMMITTED 终态应禁止转移"
    except ValueError:
        print("[OK] 终态禁止转移")

    # 跳步：DRAFT 直接到 COMMITTED 应被拒
    sm2 = SkillStateMachine()
    try:
        sm2.transition("f2", SkillState.COMMITTED, "跳步")
        assert False, "应拒绝跳步"
    except ValueError:
        print("[OK] 跳步被拒")


def test_state_machine_idempotent():
    sm = SkillStateMachine()
    t1 = sm.transition("f1", SkillState.ACCEPTED, "test")
    t2 = sm.transition("f1", SkillState.ACCEPTED, "test")
    # 幂等键相同 → 返回同一转移
    assert t1.idempotency_key == t2.idempotency_key
    assert len(sm.get_history("f1")) == 1
    print("[OK] 幂等键防重复转移")


def test_event_sourcing_rebuild(tmp_path):
    log = make_event_log(tmp_path)
    store = SkillStore(event_log=log)
    calls = make_llm_calls(2)
    store.draft("f1", "ts_mean(returns, 20)", {"dsr": 0.95}, calls)
    store.accept("f1", "ok")

    # 重建投影
    projector = SkillProjector(log)
    projector.rebuild()
    p = projector.get("f1")
    assert p is not None
    assert p.state == SkillState.ACCEPTED
    assert p.code == "ts_mean(returns, 20)"
    print("[OK] 事件溯源投影可重建")


def test_projection_consistency(tmp_path):
    log = make_event_log(tmp_path)
    store = SkillStore(event_log=log)
    calls = make_llm_calls(2)
    store.draft("f1", "code1", {}, calls)
    store.draft("f2", "code2", {}, calls)
    store.accept("f1", "ok")
    store.reject("f2", "bad")

    errors = store.verify_consistency()
    assert not errors, f"一致性校验失败: {errors}"
    print("[OK] 投影与事件流一致")


def test_fast_loop_produces_draft_only(tmp_path):
    log = make_event_log(tmp_path)
    store = SkillStore(event_log=log)
    fast = FastLoop(store)

    # 构造 2 个准入因子
    from main_tier.l5_vag import GateDecision
    from main_tier.l3_evolution import Factor

    factors = [
        Factor("f1", "momentum", "ts_mean(returns, 20)", 2,
               sharpe=0.5, marginal_gain=0.1),
        Factor("f2", "value", "1/pe_ratio", 2,
               sharpe=0.4, marginal_gain=0.05),
    ]
    decisions = [
        GateDecision(admitted=True, fid="f1"),
        GateDecision(admitted=True, fid="f2"),
    ]
    calls = {"f1": make_llm_calls(2), "f2": make_llm_calls(2)}
    result = fast.run_round(factors, decisions, calls, {"dsr": 0.95})

    # 快环只产出 DRAFT
    assert len(result.drafted_fids) == 2
    drafts = store.projector.get_by_state(SkillState.DRAFT)
    assert len(drafts) == 2, "快环应只产出 DRAFT"
    # 不应有 ACCEPTED 或 COMMITTED
    assert not store.get_accepted()
    assert not store.get_committed()
    print(f"[OK] 快环只产出 DRAFT: {len(drafts)} 个")


def test_slow_loop_audit(tmp_path):
    log = make_event_log(tmp_path)
    store = SkillStore(event_log=log)
    calls = make_llm_calls(2)
    # 写 20 个 DRAFT
    for i in range(20):
        store.draft(f"f{i}", f"code_{i}", {}, calls)

    auditor = BlindAuditor(audit_ratio=0.10, min_audit=5, ci_threshold=0.3)
    slow = SlowLoop(store, auditor)

    # 构造前向收益：一半好一半坏
    rng = np.random.default_rng(42)
    forward_returns = {}
    for i in range(20):
        if i < 10:
            # 好的：正 Sharpe
            forward_returns[f"f{i}"] = rng.normal(0.002, 0.01, 252)
        else:
            # 坏的：负 Sharpe
            forward_returns[f"f{i}"] = rng.normal(-0.002, 0.01, 252)

    result = slow.run_round(forward_returns, use_audit=True)
    assert result.n_drafts == 20
    assert result.n_forward_tested == 20
    assert result.audit_result is not None
    # 抽检数 = max(2, 5) = 5
    assert result.audit_result.n_audited == 5
    print(f"[OK] 慢环抽检: 审计 {result.audit_result.n_audited}/20, "
          f"通过 {result.n_accepted}, 拒绝 {result.n_rejected}")


def test_blind_audit_anonymity():
    """盲样：fid 映射到匿名 ID，生成侧无法反推。"""
    auditor = BlindAuditor(auditor_seed=999)
    fids = [f"f{i}" for i in range(10)]
    anon = auditor._anonymize(fids)
    # 匿名 ID 与 fid 不同
    for fid, aid in anon.items():
        assert aid != fid
        assert aid.startswith("audit_")
    # 同一组 fid 同样种子 → 同样匿名映射（可复现）
    anon2 = auditor._anonymize(fids)
    assert anon == anon2
    print("[OK] 盲样匿名映射可复现")


def test_blind_audit_isolated_seed():
    """抽检种子与生成侧隔离。"""
    auditor1 = BlindAuditor(auditor_seed=999)
    auditor2 = BlindAuditor(auditor_seed=42)
    fids = [f"f{i}" for i in range(10)]
    anon1 = auditor1._anonymize(fids)
    anon2 = auditor2._anonymize(fids)
    assert anon1 != anon2, "不同种子应产生不同匿名映射"
    print("[OK] 抽检种子隔离")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_content_hash_includes_llm_calls()
        test_state_machine_valid()
        test_state_machine_invalid()
        test_state_machine_idempotent()
        test_event_sourcing_rebuild(tp)
        test_projection_consistency(tp)
        test_fast_loop_produces_draft_only(tp)
        test_slow_loop_audit(tp)
        test_blind_audit_anonymity()
        test_blind_audit_isolated_seed()
        print("\n=== L6 全部测试通过 ===")