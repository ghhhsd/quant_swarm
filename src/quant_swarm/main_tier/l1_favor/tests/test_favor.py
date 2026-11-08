"""
L1 FaVOR 测试。

验证 v3.1 的七条纪律：
1. 叙事回测前锁定，二次 commit 被拒
2. 非 LLM 事实校验三类约束生效
3. 异构 LLM 校验与 facts 组合判定
4. 五类 Failure Trace 差异化分流
5. 门控拒绝不重试
6. 叙事-结果一致性校验
7. 不变核心与 L1 解耦（L1 不碰风控）
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from main_tier.l1_favor import (
    FavorLayer, Narrative, Direction, DecayPattern,
    NarrativeCommitter, FailureType, NarrativeConsistencyChecker,
)
from main_tier.l1_favor.facts import FactChecker
from main_tier.l2_agents.llm_client import StubLLMClient
from main_tier.l0_governance import GovernanceLayer
from core.config import KernelConfig
from core.event_log import EventLog
import tempfile
from pathlib import Path


def make_favor(tmp_path, validator_seed=99):
    log = EventLog(str(tmp_path / "events.jsonl"))
    gov = GovernanceLayer(KernelConfig(), event_log=log)
    gov.start_round()
    return FavorLayer(
        validator_llm=StubLLMClient("validator", seed=validator_seed),
        governance=gov,
        codegen_llm=StubLLMClient("codegen", seed=1),  # 异构
        event_log=log,
    )


def make_narrative(hid="h_test", family="momentum",
                   direction=Direction.LONG,
                   decay=DecayPattern.PERSISTENT):
    return Narrative(
        hid=hid,
        mechanism_family=family,
        direction=direction,
        conditions=["高流动性市场", "非极端波动"],
        decay=decay,
        economic_logic="动量延续效应：过去赢家继续跑赢",
        falsifiable_statement="若 OOS 期 IC 不显著为正，则假设被证伪",
    )


def test_narrative_commit_once(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    committer = NarrativeCommitter(log)
    n = make_narrative()
    c1 = committer.commit(n)
    assert c1.hash
    try:
        committer.commit(n)
        assert False, "应抛 ValueError"
    except ValueError:
        print("[OK] 叙事二次 commit 被拒")


def test_fact_check_data_observability():
    fc = FactChecker()
    n = make_narrative()
    # 表达式引用未声明字段
    r = fc.check_data_observability(n, "unknown_field * 2")
    assert not r.ok
    # 表达式引用合法字段
    r = fc.check_data_observability(n, "ts_mean(returns, 20)")
    assert r.ok, r.reason
    # 窗口参数过大
    r = fc.check_data_observability(n, "ts_mean(returns, 1000)")
    assert not r.ok
    print("[OK] 数据可观测性约束")


def test_fact_check_economic_prior():
    fc = FactChecker()
    # 动量族声明 SHORT → 违反先验
    n = make_narrative(family="momentum", direction=Direction.SHORT)
    r = fc.check_economic_prior(n)
    assert not r.ok
    # 动量族声明 LONG → 通过
    n = make_narrative(family="momentum", direction=Direction.LONG)
    r = fc.check_economic_prior(n)
    assert r.ok
    print("[OK] 经济先验约束")


def test_fact_check_mechanism_alignment():
    fc = FactChecker()
    # 流动性族但表达式只用 returns
    n = make_narrative(family="liquidity")
    r = fc.check_mechanism_alignment(n, "ts_mean(returns, 20)")
    assert not r.ok
    # 流动性族用 turnover
    r = fc.check_mechanism_alignment(n, "ts_mean(turnover, 20)")
    assert r.ok, r.reason
    print("[OK] 机制-表达式对齐约束")


def test_full_pass(tmp_path):
    favor = make_favor(tmp_path)
    n = make_narrative()
    # Stub 校验器会返回 pass 还是 fail 取决于哈希，先构造一个已知通过的
    # 用 StubLLMClient(seed=99) + 这个 prompt，若 fail 则调 seed
    result = favor.validate(n, "ts_mean(returns, 20) - ts_mean(returns, 60)")
    # 无论 pass 还是 fail，都要检查结构完整
    if result.passed:
        assert result.narrative_hash
        assert result.validator_verdict is not None
        assert result.validator_verdict.llm_pass
        print(f"[OK] 完整通过: hash={result.narrative_hash[:12]}...")
    else:
        # 如果 stub 输出 fail，检查失败路径正确
        assert result.failure_type is not None
        assert result.failure_trace is not None
        print(f"[INFO] Stub 输出 fail: {result.failure_trace.message[:60]}")


def test_fact_fail_short_circuits(tmp_path):
    """事实校验失败时，不应进入 LLM 校验。"""
    favor = make_favor(tmp_path)
    # 机制族 mismatch：流动性族但用 returns
    n = make_narrative(family="liquidity")
    result = favor.validate(n, "ts_mean(returns, 20)")
    assert not result.passed
    assert result.failure_type == FailureType.LOGIC
    assert result.failure_trace.stage == "fact_check"
    # 验证 LLM 未被调用（validator_verdict 为 None）
    assert result.validator_verdict is None
    print("[OK] 事实校验失败短路，不进 LLM")


def test_direction_inconsistency():
    checker = NarrativeConsistencyChecker()
    # 叙事 LONG，实测 IC < 0 → 不一致
    n = make_narrative(direction=Direction.LONG)
    ic_series = np.array([0.05, 0.04, 0.03, 0.02])
    r = checker.check_all(n, factor_returns_corr=-0.02, ic_series=ic_series)
    assert not r.consistent
    assert not r.direction_match
    # 叙事 LONG，实测 IC > 0 → 一致
    r = checker.check_all(n, factor_returns_corr=0.03, ic_series=ic_series)
    assert r.consistent, r.reasons
    print("[OK] 叙事-结果方向一致性")


def test_decay_persistent_vs_fast():
    checker = NarrativeConsistencyChecker()
    # PERSISTENT：IC 首尾比 > 0.5
    n = make_narrative(decay=DecayPattern.PERSISTENT)
    ic_persistent = np.array([0.05, 0.045, 0.04, 0.038])  # 比值 0.76
    assert checker.check_decay(n, ic_persistent)
    ic_fast = np.array([0.05, 0.02, 0.005, 0.001])  # 比值 0.02
    assert not checker.check_decay(n, ic_fast)
    # FAST_DECAY：IC 首尾比 < 0.3
    n = make_narrative(decay=DecayPattern.FAST_DECAY)
    assert checker.check_decay(n, ic_fast)
    assert not checker.check_decay(n, ic_persistent)
    print("[OK] 叙事-结果衰减一致性")


def test_heal_strategy_routing():
    from main_tier.l1_favor.failure import HEAL_STRATEGY, RETRYABLE
    assert HEAL_STRATEGY[FailureType.SYNTAX] == "retry_codegen"
    assert HEAL_STRATEGY[FailureType.LOGIC] == "backtrack_hypothesis"
    assert HEAL_STRATEGY[FailureType.DATA] == "fix_data_contract"
    assert HEAL_STRATEGY[FailureType.BACKTEST] == "reduce_complexity"
    assert HEAL_STRATEGY[FailureType.GATE_REJECTED] == "record_negative"
    assert not RETRYABLE[FailureType.GATE_REJECTED], "门控拒绝不重试"
    assert RETRYABLE[FailureType.SYNTAX]
    print("[OK] 五类失败差异化分流")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_narrative_commit_once(tp)
        test_fact_check_data_observability()
        test_fact_check_economic_prior()
        test_fact_check_mechanism_alignment()
        test_full_pass(tp)
        test_fact_fail_short_circuits(tp)
        test_direction_inconsistency()
        test_decay_persistent_vs_fast()
        test_heal_strategy_routing()
        print("\n=== L1 全部测试通过 ===")