"""
技能写入状态机（v3.1 修正⑥）。

状态：
- DRAFT（草稿）：刚写入，不可用于决策
- ACCEPTED（验收通过）：快环产出 + 慢环前向验证通过
- COMMITTED（已提交）：正式进入技能库，可用于组合构建
- REJECTED（已拒绝）：被拒绝，不进入技能库

状态转移：
DRAFT → ACCEPTED（慢环验证通过）
DRAFT → REJECTED（慢环验证失败）
ACCEPTED → COMMITTED（人工上线决策）
ACCEPTED → REJECTED（人工上线决策拒绝）

关键纪律：
1. 状态转移必须走显式方法，不能直接改字段。
2. 每次转移进事件流（幂等键防重复）。
3. COMMITTED 是终态，不可回退（回退需人工干预 + 审计记录）。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import hashlib


class SkillState(str, Enum):
    DRAFT = "draft"
    ACCEPTED = "accepted"
    COMMITTED = "committed"
    REJECTED = "rejected"


# 合法状态转移
VALID_TRANSITIONS = {
    SkillState.DRAFT: {SkillState.ACCEPTED, SkillState.REJECTED},
    SkillState.ACCEPTED: {SkillState.COMMITTED, SkillState.REJECTED},
    SkillState.COMMITTED: set(),  # 终态
    SkillState.REJECTED: set(),   # 终态
}


@dataclass
class StateTransition:
    """状态转移记录。"""
    fid: str
    from_state: SkillState
    to_state: SkillState
    reason: str
    idempotency_key: str
    event_hash: str = ""


class SkillStateMachine:
    """
    技能状态机。
    持有每个 fid 的当前状态 + 转移历史。
    """

    def __init__(self):
        self.states: Dict[str, SkillState] = {}
        self.transitions: Dict[str, list] = {}

    def transition(self, fid: str, to_state: SkillState,
                   reason: str = "") -> StateTransition:
        """
        执行状态转移。非法转移抛异常。
        幂等键 = hash(fid + from_state + to_state + reason)，防重复转移。
        """
        from_state = self.states.get(fid, SkillState.DRAFT)
        # 首次写入视为 DRAFT → 目标状态
        if fid not in self.states:
            from_state = SkillState.DRAFT

        if to_state not in VALID_TRANSITIONS.get(from_state, set()):
            raise ValueError(
                f"非法状态转移: {fid} {from_state.value} → {to_state.value}"
            )

        idempotency_key = hashlib.sha256(
            f"{fid}|{from_state.value}|{to_state.value}|{reason}".encode()
        ).hexdigest()

        # 幂等检查
        existing = self.transitions.get(fid, [])
        for t in existing:
            if t.idempotency_key == idempotency_key:
                return t  # 已存在，幂等返回

        transition = StateTransition(
            fid=fid,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            idempotency_key=idempotency_key,
        )
        self.states[fid] = to_state
        self.transitions.setdefault(fid, []).append(transition)
        return transition

    def get_state(self, fid: str) -> Optional[SkillState]:
        return self.states.get(fid)

    def get_history(self, fid: str) -> list:
        return list(self.transitions.get(fid, []))