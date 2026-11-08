"""
层状态管理（v3.1 修正②：状态持久化 + 可回退）。

关键设计：
1. 每个层有 current_state（main / candidate）+ history（所有转移记录）。
2. 状态转移写入不可变日志（append-only），可审计。
3. "当前状态"来自"最近一次决策"，但历史保留所有决策。
4. 状态是"可回退"的——晋升后仍可降级，不是永久性。
5. 状态转移的幂等键 = hash(layer + from_state + to_state + reason)，
   防重复转移。
"""
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional
import hashlib
import time

from core.event_log import EventLog, GLOBAL_EVENT_LOG


class LayerStatus(str, Enum):
    """层状态。"""
    MAIN = "main"              # 主干层
    CANDIDATE = "candidate"    # 候选层


class TransitionAction(str, Enum):
    """状态转移动作。"""
    PROMOTE = "promote"        # 候选 → 主干
    DEMOTE = "demote"          # 主干 → 候选
    KEEP_CANDIDATE = "keep_candidate"  # 保持候选（不升不降）
    KEEP_MAIN = "keep_main"    # 保持主干（不升不降）


@dataclass
class StateTransition:
    """状态转移记录（不可变）。"""
    layer: str
    from_state: LayerStatus
    to_state: LayerStatus
    action: TransitionAction
    reason: str
    evidence_snapshot: Dict      # 决策依据快照
    idempotency_key: str
    timestamp: float
    event_hash: str = ""

    def to_dict(self) -> Dict:
        return {
            "layer": self.layer,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "action": self.action.value,
            "reason": self.reason,
            "evidence_snapshot": self.evidence_snapshot,
            "idempotency_key": self.idempotency_key,
            "timestamp": self.timestamp,
            "event_hash": self.event_hash,
        }


class LayerStateRegistry:
    """
    层状态注册表。

    - 持有每层的 current_state + history。
    - 状态转移写入不可变日志。
    - 幂等键防重复转移。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self.current_state: Dict[str, LayerStatus] = {}
        self.history: Dict[str, List[StateTransition]] = {}
        self._idempotency_keys: Dict[str, set] = {}

    def register(self, layer: str, initial_state: LayerStatus) -> None:
        """注册层。只注册一次，防覆盖。"""
        if layer in self.current_state:
            raise ValueError(f"层 {layer} 已注册（current={self.current_state[layer]}）")
        self.current_state[layer] = initial_state
        self.history[layer] = []
        self._idempotency_keys[layer] = set()
        self.event_log.append("layer_registered", {
            "layer": layer, "initial_state": initial_state.value,
        })

    def get_state(self, layer: str) -> Optional[LayerStatus]:
        return self.current_state.get(layer)

    def get_history(self, layer: str) -> List[StateTransition]:
        return list(self.history.get(layer, []))

    def transition(self,
                   layer: str,
                   to_state: LayerStatus,
                   action: TransitionAction,
                   reason: str,
                   evidence_snapshot: Dict) -> StateTransition:
        """
        执行状态转移。

        - 幂等键 = hash(layer + from_state + to_state + reason)
        - 已存在相同幂等键的转移 → 直接返回（幂等）
        - 否则写入事件流 + 更新状态
        """
        if layer not in self.current_state:
            raise ValueError(f"层 {layer} 未注册，先 register()")
        from_state = self.current_state[layer]

        # 幂等键
        idem_key = hashlib.sha256(
            f"{layer}|{from_state.value}|{to_state.value}|{reason}".encode()
        ).hexdigest()

        # 幂等检查
        if idem_key in self._idempotency_keys[layer]:
            for t in self.history[layer]:
                if t.idempotency_key == idem_key:
                    return t

        transition = StateTransition(
            layer=layer,
            from_state=from_state,
            to_state=to_state,
            action=action,
            reason=reason,
            evidence_snapshot=evidence_snapshot,
            idempotency_key=idem_key,
            timestamp=time.time(),
        )
        # 写事件流
        ev = self.event_log.append("layer_state_transition", transition.to_dict())
        transition.event_hash = ev.content_hash

        # 更新状态
        self.current_state[layer] = to_state
        self.history[layer].append(transition)
        self._idempotency_keys[layer].add(idem_key)
        return transition

    def get_all_states(self) -> Dict[str, LayerStatus]:
        return dict(self.current_state)

    def verify_history_integrity(self, layer: str) -> List[str]:
        """
        校验历史完整性：
        - 第一条 from_state 应等于注册时的初始状态
        - 后续每条 from_state 应等于前一条 to_state
        """
        errors = []
        history = self.history.get(layer, [])
        if not history:
            return errors
        for i, t in enumerate(history):
            if i == 0:
                continue
            prev = history[i - 1]
            if t.from_state != prev.to_state:
                errors.append(
                    f"{layer} 第 {i} 条转移 from_state={t.from_state.value} "
                    f"!= 前一条 to_state={prev.to_state.value}"
                )
        # 校验 current_state 与最后一条 to_state 一致
        if history:
            last = history[-1]
            if self.current_state[layer] != last.to_state:
                errors.append(
                    f"{layer} current_state={self.current_state[layer].value} "
                    f"!= 最后一条 to_state={last.to_state.value}"
                )
        return errors