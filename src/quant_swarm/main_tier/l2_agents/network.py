"""
L2 编排：analysts → debate → codegen。

与 v3.1 分层对齐：
- 接收 L1 的 FavorResult（已过逻辑校验 + 叙事预承诺）。
- 输出 CodegenReport，供 L4 沙箱 / L5 门控消费。
- 失败抛 AgentFailure，带 FailureType，由上层路由到 L1 自愈分流。
- 所有 LLM 调用记录累积到 self.all_calls，供 L6 事件溯源。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from main_tier.l2_agents.llm_client import LLMClient, LLMCallRecord
from main_tier.l2_agents.analyst import AnalystAgent, AnalystReport
from main_tier.l2_agents.debate import DebateAgent, DebateSession, DebateSummary
from main_tier.l2_agents.codegen import CodegenAgent, CodegenReport
from main_tier.l2_agents.base_agent import AgentFailure
from main_tier.l1_favor import FavorResult
from main_tier.l0_governance import GovernanceLayer


@dataclass
class NetworkResult:
    """L2 完整输出（结构化，可序列化进事件流）。"""
    hid: str
    analysts: List[AnalystReport]
    debate: DebateSummary
    codegen: CodegenReport
    all_llm_hashes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hid": self.hid,
            "analysts": [a.to_dict() for a in self.analysts],
            "debate": self.debate.to_dict(),
            "codegen": self.codegen.to_dict(),
            "all_llm_hashes": self.all_llm_hashes,
        }


class MultiAgentNetwork:
    """
    多 Agent 协作网络。

    异构设计（v3.1 L1 修正）：
    - 分析师 / 辩手 / 编码员各自可注入不同 LLMClient。
    - 编码员温度（0.3）显著低于辩手（0.8），分析师居中（0.5）。
    - 若外部传入同一个 client 实例，构造函数会发出警告（同源风险）。
    """

    def __init__(self,
                 governance: GovernanceLayer,
                 analyst_llm: LLMClient,
                 debate_llm: LLMClient,
                 codegen_llm: LLMClient,
                 max_debate_rounds: int = 3,
                 convergence_threshold: float = 0.15):
        self.gov = governance
        self.analyst_llm = analyst_llm
        self.debate_llm = debate_llm
        self.codegen_llm = codegen_llm
        self.max_debate_rounds = max_debate_rounds
        self.convergence_threshold = convergence_threshold
        self.all_calls: List[LLMCallRecord] = []

        # 异构检查：编码员与分析师/辩手不应是同一实例（弱警告，非硬失败）
        if codegen_llm is analyst_llm:
            print("[WARN] codegen 与 analyst 共用同一 LLMClient 实例，"
                  "存在同源自证风险（v3.1 L1 纪律：弱独立性）")

    def run(self, favor: FavorResult) -> NetworkResult:
        if not favor.passed:
            raise AgentFailure(
                favor.failure_type or "logic",
                f"L1 未通过：{favor.trace}",
            )
        hyp = favor.hypothesis
        self._reset_agents()

        # 1) 分析师（技术 + 基本面）
        tech = AnalystAgent(self.analyst_llm, self.gov, "technical")
        fund = AnalystAgent(self.analyst_llm, self.gov, "fundamental")
        analysts = [tech.run(hyp), fund.run(hyp)]

        # 2) 多空辩论
        bull = DebateAgent(self.debate_llm, self.gov, "bull")
        bear = DebateAgent(self.debate_llm, self.gov, "bear")
        session = DebateSession(bull, bear,
                                self.max_debate_rounds,
                                self.convergence_threshold)
        debate = session.run(hyp)

        # 3) 编码
        codegen = CodegenAgent(self.codegen_llm, self.gov)
        report = codegen.run(favor, debate)

        # 4) 累积所有 LLM 调用记录（供 L6 路 A）
        self.all_calls = (
            tech.calls + fund.calls + bull.calls + bear.calls + codegen.calls
        )
        all_hashes = [c.content_hash() for c in self.all_calls]

        return NetworkResult(
            hid=hyp.hid,
            analysts=analysts,
            debate=debate,
            codegen=report,
            all_llm_hashes=all_hashes,
        )

    def _reset_agents(self) -> None:
        self.all_calls.clear()