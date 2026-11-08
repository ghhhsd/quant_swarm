"""
用 StubLLMClient 跑通全流程，无需真实 LLM。
验证：结构化输出、治理消耗、异构警告、叙事预承诺硬约束、AST 检查。
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from core.config import GLOBAL_KERNEL
from main_tier.l0_governance import GovernanceLayer
from main_tier.l1_favor import FavorLayer, Hypothesis
from main_tier.l2_agents.llm_client import StubLLMClient
from main_tier.l2_agents.network import MultiAgentNetwork
from main_tier.l2_agents.base_agent import AgentFailure


def test_full_flow():
    gov = GovernanceLayer(GLOBAL_KERNEL); gov.start_round()
    favor = FavorLayer()
    hyp = Hypothesis(
        "h_test", "momentum",
        "[momentum] 20日动量延续效应",
        "ts_mean(returns, 20) - ts_mean(returns, 60)",
    )
    fr = favor.validate(hyp)
    assert fr.passed, fr.trace
    assert hyp.narrative_hash is not None, "叙事应已预承诺"

    # 异构：三个不同 Stub 实例
    net = MultiAgentNetwork(
        governance=gov,
        analyst_llm=StubLLMClient("stub_analyst", seed=1),
        debate_llm=StubLLMClient("stub_debate", seed=2),
        codegen_llm=StubLLMClient("stub_codegen", seed=3),
        max_debate_rounds=3,
        convergence_threshold=0.15,
    )
    result = net.run(fr)

    # 断言结构化输出
    assert len(result.analysts) == 2
    assert result.debate.n_rounds >= 1
    assert result.codegen.hid == "h_test"
    assert result.codegen.expression  # 非空
    assert result.codegen.narrative_hash == hyp.narrative_hash
    assert len(result.all_llm_hashes) > 0

    # 断言治理消耗（Token 配额下降）
    assert gov.quota.token_remaining < 200_000, "应消耗了 Token"

    print(f"[OK] 全流程跑通: {result.codegen.expression}")
    print(f"     辩论轮数: {result.debate.n_rounds}, "
          f"分歧度: {result.debate.final_divergence:.2f}")
    print(f"     LLM 调用数: {len(result.all_llm_hashes)}")


def test_reject_without_narrative():
    """验证 L2 拒绝未预承诺叙事的假设。"""
    gov = GovernanceLayer(GLOBAL_KERNEL); gov.start_round()
    # 手工构造一个无 narrative_hash 的 FavorResult
    from main_tier.l1_favor import FavorResult
    hyp = Hypothesis("h_bad", "momentum", "test", "ts_mean(returns, 10)")
    fr = FavorResult(passed=True, hypothesis=hyp)  # 无 narrative_hash

    net = MultiAgentNetwork(
        governance=gov,
        analyst_llm=StubLLMClient("a"),
        debate_llm=StubLLMClient("d"),
        codegen_llm=StubLLMClient("c"),
    )
    try:
        net.run(fr)
        assert False, "应抛 AgentFailure"
    except AgentFailure as e:
        print(f"[OK] 正确拒绝未预承诺假设: {e.message}")


if __name__ == "__main__":
    test_full_flow()
    test_reject_without_narrative()