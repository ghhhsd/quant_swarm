"""
快环：候选池扩充（自动/廉价/高频）。

关键设计（v3.1 修正⑤）：
1. 快环只产出"候选池扩充"，不做上线决策。
2. 快环产出写入技能库的 DRAFT 状态，不是 COMMITTED。
3. 快环的目标是"高频跑"，所以不含前向模拟盘（那是慢环的事）。
4. 快环的"质量门槛"是 L5 门控——通过 L5 的因子进入候选池。
"""
from dataclasses import dataclass, field
from typing import List, Optional
from main_tier.l6_skillstore.skill_store import SkillStore
from main_tier.l6_skillstore.state_machine import SkillState
from main_tier.l3_evolution import Factor
from main_tier.l5_vag import GateDecision
from main_tier.l2_agents.llm_client import LLMCallRecord


@dataclass
class FastLoopResult:
    """快环单轮结果。"""
    round_idx: int
    n_candidates: int
    n_admitted: int
    drafted_fids: List[str] = field(default_factory=list)


class FastLoop:
    """
    快环：L1→L5→候选池。

    输入：通过 L5 门控的因子（已带评估结果）
    输出：技能库 DRAFT 条目
    """

    def __init__(self, store: SkillStore):
        self.store = store
        self.round_idx = 0

    def run_round(self,
                  admitted_factors: List[Factor],
                  gate_decisions: List[GateDecision],
                  llm_calls_per_factor: dict,
                  gate_params: dict) -> FastLoopResult:
        """
        admitted_factors：通过 L5 的因子列表
        gate_decisions：对应的门控决策
        llm_calls_per_factor：{fid: List[LLMCallRecord]}
        gate_params：L0 committed 的门控参数
        """
        self.round_idx += 1
        drafted = []
        for factor, decision in zip(admitted_factors, gate_decisions):
            if not decision.admitted:
                continue
            calls = llm_calls_per_factor.get(factor.fid, [])
            self.store.draft(
                fid=factor.fid,
                code=factor.code,
                gate_params=gate_params,
                llm_calls=calls,
            )
            drafted.append(factor.fid)
        return FastLoopResult(
            round_idx=self.round_idx,
            n_candidates=len(admitted_factors),
            n_admitted=len(drafted),
            drafted_fids=drafted,
        )