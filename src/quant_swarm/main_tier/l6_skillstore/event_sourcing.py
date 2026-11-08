"""
事件溯源投影（v3.1 修正⑥）。

关键设计：
1. 演化树是唯一事实源，技能库是投影视图。
2. 投影可以从事件流全量重建（rebuild）。
3. 投影支持增量更新（apply_event），避免每次全量重建。
4. 投影的"状态"与事件流一致——给定同一事件流，投影唯一。

事件类型：
- skill_draft：技能草稿写入
- skill_accepted：验收通过
- skill_committed：正式提交
- skill_rejected：拒绝
- llm_call_recorded：LLM 调用记录（路 A）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from main_tier.l6_skillstore.state_machine import SkillState
from core.event_log import EventLog, Event


@dataclass
class SkillProjection:
    """技能投影项。"""
    fid: str
    code: str
    content_hash: str
    state: SkillState
    gate_params: Dict[str, Any]
    llm_calls_hash: str
    created_at: float
    history: List[Dict[str, Any]] = field(default_factory=list)


class SkillProjector:
    """
    技能投影器。
    从事件流重建/增量更新技能投影。
    """

    def __init__(self, event_log: EventLog):
        self.event_log = event_log
        self.projection: Dict[str, SkillProjection] = {}

    def rebuild(self) -> Dict[str, SkillProjection]:
        """从事件流全量重建投影。"""
        self.projection = {}
        for event in self.event_log.events:
            self.apply_event(event)
        return self.projection

    def apply_event(self, event: Event) -> None:
        """增量应用单个事件。"""
        et = event.event_type
        p = event.payload

        if et == "skill_draft":
            fid = p["fid"]
            self.projection[fid] = SkillProjection(
                fid=fid,
                code=p["code"],
                content_hash=p["content_hash"],
                state=SkillState.DRAFT,
                gate_params=p.get("gate_params", {}),
                llm_calls_hash=p.get("llm_calls_hash", ""),
                created_at=event.timestamp,
            )
            self.projection[fid].history.append({"event": et, "ts": event.timestamp})

        elif et in ("skill_accepted", "skill_committed", "skill_rejected"):
            fid = p["fid"]
            if fid not in self.projection:
                # 事件流不完整：跳过（rebuild 时可能出现）
                return
            state_map = {
                "skill_accepted": SkillState.ACCEPTED,
                "skill_committed": SkillState.COMMITTED,
                "skill_rejected": SkillState.REJECTED,
            }
            self.projection[fid].state = state_map[et]
            self.projection[fid].history.append({"event": et, "ts": event.timestamp})

    def get(self, fid: str) -> Optional[SkillProjection]:
        return self.projection.get(fid)

    def get_by_state(self, state: SkillState) -> List[SkillProjection]:
        return [p for p in self.projection.values() if p.state == state]

    def size(self) -> int:
        return len(self.projection)

    def verify_consistency(self) -> List[str]:
        """
        校验投影与事件流一致。
        返回错误列表（空表示一致）。
        """
        errors = []
        # 重建一份对照投影
        reference = SkillProjector(self.event_log)
        reference.rebuild()
        # 比较
        if set(self.projection.keys()) != set(reference.projection.keys()):
            errors.append("投影 fid 集合不一致")
            return errors
        for fid in self.projection:
            a = self.projection[fid]
            b = reference.projection[fid]
            if a.state != b.state:
                errors.append(f"{fid} 状态不一致: {a.state} vs {b.state}")
            if a.content_hash != b.content_hash:
                errors.append(f"{fid} 内容哈希不一致")
        return errors