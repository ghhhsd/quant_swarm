"""
技能库（投影视图）。

关键设计：
1. 技能库不直接持有记录，而是持有事件流 + 投影。
2. 写入技能走状态机 + 事件流，投影增量更新。
3. 技能库的"内容"完全由事件流决定——drop 投影后 rebuild 可完全恢复。
"""
from typing import List, Optional, Dict, Any
from main_tier.l6_skillstore.state_machine import (
    SkillState, SkillStateMachine, StateTransition,
)
from main_tier.l6_skillstore.event_sourcing import (
    SkillProjector, SkillProjection,
)
from main_tier.l6_skillstore.content_hash import compute_skill_hash, llm_calls_hash
from main_tier.l2_agents.llm_client import LLMCallRecord
from core.event_log import EventLog, GLOBAL_EVENT_LOG


class SkillStore:
    """
    技能库（投影视图 + 状态机）。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self.state_machine = SkillStateMachine()
        self.projector = SkillProjector(self.event_log)
        # 初始化时重建投影（从已有事件流）
        self.projector.rebuild()

    # ---- 写入 -----------------------------------------------------------
    def draft(self, fid: str, code: str, gate_params: Dict[str, Any],
              llm_calls: List[LLMCallRecord]) -> SkillProjection:
        """写入草稿。"""
        # 计算内容哈希
        ch = compute_skill_hash(code, gate_params, llm_calls)
        lh = llm_calls_hash(llm_calls)
        # 写事件
        ev = self.event_log.append("skill_draft", {
            "fid": fid,
            "code": code,
            "content_hash": ch,
            "gate_params": gate_params,
            "llm_calls_hash": lh,
            "llm_calls_count": len(llm_calls),
        })
        # 增量应用
        self.projector.apply_event(ev)
        return self.projector.get(fid)

    def accept(self, fid: str, reason: str = "") -> StateTransition:
        """验收通过。"""
        t = self.state_machine.transition(fid, SkillState.ACCEPTED, reason)
        ev = self.event_log.append("skill_accepted", {
            "fid": fid, "reason": reason,
            "idempotency_key": t.idempotency_key,
        })
        self.projector.apply_event(ev)
        return t

    def commit(self, fid: str, reason: str = "") -> StateTransition:
        """正式提交。"""
        t = self.state_machine.transition(fid, SkillState.COMMITTED, reason)
        ev = self.event_log.append("skill_committed", {
            "fid": fid, "reason": reason,
            "idempotency_key": t.idempotency_key,
        })
        self.projector.apply_event(ev)
        return t

    def reject(self, fid: str, reason: str = "") -> StateTransition:
        """拒绝。"""
        t = self.state_machine.transition(fid, SkillState.REJECTED, reason)
        ev = self.event_log.append("skill_rejected", {
            "fid": fid, "reason": reason,
            "idempotency_key": t.idempotency_key,
        })
        self.projector.apply_event(ev)
        return t

    # ---- 查询 -----------------------------------------------------------
    def get(self, fid: str) -> Optional[SkillProjection]:
        return self.projector.get(fid)

    def get_committed(self) -> List[SkillProjection]:
        return self.projector.get_by_state(SkillState.COMMITTED)

    def get_accepted(self) -> List[SkillProjection]:
        return self.projector.get_by_state(SkillState.ACCEPTED)

    def size(self) -> int:
        return self.projector.size()

    def verify_consistency(self) -> List[str]:
        return self.projector.verify_consistency()