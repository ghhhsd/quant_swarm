"""
晋升/降级协议（v3.1 修正②核心）。

关键设计：
1. 四象限决策：
   - 强证据 + 正增益 → PROMOTE
   - 强证据 + 负增益 → DEMOTE
   - 弱证据 → KEEP_CANDIDATE / KEEP_MAIN（不升不降）
   - 零增益 + 强证据 → KEEP_CANDIDATE（证据强但无增益）
2. 弱证据不因"负增益"而降级——证据不足时负增益可能是噪声。
3. 晋升后可回退：后续消融强证据 + 负增益 → DEMOTE。
4. 决策写入不可变日志（委托给 LayerStateRegistry）。
5. 决策依据（evidence_snapshot）必须记录，供审计。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

from ablation.layer_state import (
    LayerStatus, TransitionAction, StateTransition, LayerStateRegistry,
)
from ablation.comparator import AblationResult
from ablation.efficacy_budget import EfficacyBudget, EffectSize
from core.event_log import EventLog, GLOBAL_EVENT_LOG


@dataclass
class PromotionDecision:
    """晋升/降级决策。"""
    layer: str
    action: TransitionAction
    from_state: LayerStatus
    to_state: LayerStatus
    reason: str
    evidence_snapshot: Dict
    transition: Optional[StateTransition] = None

    def to_dict(self) -> Dict:
        return {
            "layer": self.layer,
            "action": self.action.value,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason,
            "evidence_snapshot": self.evidence_snapshot,
            "transition_hash": (
                self.transition.event_hash if self.transition else None
            ),
        }

    def summary(self) -> str:
        return (
            f"[{self.action.value.upper()}] {self.layer}: "
            f"{self.from_state.value} → {self.to_state.value} "
            f"({self.reason})"
        )


class PromotionProtocol:
    """
    晋升/降级协议。

    流程：
    1. 接收 AblationResult（来自开/关比较器）
    2. 四象限决策
    3. 写入状态注册表（不可变日志）
    4. 返回决策
    """

    def __init__(self,
                 state_registry: LayerStateRegistry,
                 event_log: Optional[EventLog] = None):
        self.state_registry = state_registry
        self.event_log = event_log or GLOBAL_EVENT_LOG

    def evaluate(self, result: AblationResult) -> PromotionDecision:
        """
        对消融结果做晋升/降级决策。

        四象限逻辑：
        1. evidence_strength == "strong" AND observed_effect > 0 → PROMOTE
        2. evidence_strength == "strong" AND observed_effect < 0 → DEMOTE
        3. evidence_strength == "strong" AND observed_effect == 0 → KEEP（零增益不晋升）
        4. evidence_strength == "weak" → KEEP（不升不降）
        """
        layer = result.layer
        current_state = self.state_registry.get_state(layer)
        if current_state is None:
            raise ValueError(
                f"层 {layer} 未注册，无法决策。先 register(layer, LayerStatus.CANDIDATE)"
            )

        evidence_strength = result.evidence_strength
        observed_effect = result.overall_power.observed_effect
        # 效应量阈值：用 mde 的 10% 作为"零增益"判定
        # 极小效应视为噪声，不触发晋升/降级
        zero_threshold = 0.01

        evidence_snapshot = {
            "evidence_strength": evidence_strength,
            "observed_effect": observed_effect,
            "observed_power": result.overall_power.observed_power,
            "target_power": result.overall_power.target_power,
            "available_batches": result.overall_power.available_batches,
            "required_batches": result.overall_power.required_batches,
            "by_regime": {
                k: {
                    "strength": v.evidence_strength,
                    "effect": v.observed_effect,
                } for k, v in result.by_regime.items()
            },
            "portfolio_config": result.portfolio_config,
            "note": result.note,
        }

        # 四象限决策
        if evidence_strength == "strong":
            if observed_effect > zero_threshold:
                # 强证据 + 正增益 → 晋升
                action = TransitionAction.PROMOTE
                to_state = LayerStatus.MAIN
                reason = (
                    f"强证据 + 正增益 (effect={observed_effect:.3f} > "
                    f"{zero_threshold}) → 晋升主干"
                )
            elif observed_effect < -zero_threshold:
                # 强证据 + 负增益 → 降级
                action = TransitionAction.DEMOTE
                to_state = LayerStatus.CANDIDATE
                reason = (
                    f"强证据 + 负增益 (effect={observed_effect:.3f} < "
                    f"{-zero_threshold}) → 降级候选"
                )
            else:
                # 强证据 + 零增益 → 保持现状（不晋升）
                action = TransitionAction.KEEP_CANDIDATE
                to_state = current_state
                reason = (
                    f"强证据但零增益 (|effect| <= {zero_threshold}) → "
                    f"保持现状"
                )
        else:
            # 弱证据 → 不升不降
            if current_state == LayerStatus.MAIN:
                action = TransitionAction.KEEP_MAIN
                reason = f"弱证据 (批次不足/功效不足)，保持主干: {result.note}"
            else:
                action = TransitionAction.KEEP_CANDIDATE
                reason = f"弱证据 (批次不足/功效不足)，保持候选: {result.note}"
            to_state = current_state

        # 写入状态注册表（幂等）
        transition = None
        if to_state != current_state:
            transition = self.state_registry.transition(
                layer=layer,
                to_state=to_state,
                action=action,
                reason=reason,
                evidence_snapshot=evidence_snapshot,
            )
        else:
            # 状态未变，但仍写事件流（记录决策）
            self.event_log.append("promotion_decision_no_change", {
                "layer": layer,
                "action": action.value,
                "reason": reason,
                "evidence_snapshot": evidence_snapshot,
            })

        decision = PromotionDecision(
            layer=layer,
            action=action,
            from_state=current_state,
            to_state=to_state,
            reason=reason,
            evidence_snapshot=evidence_snapshot,
            transition=transition,
        )
        return decision

    def evaluate_batch(self,
                       results: List[AblationResult]) -> List[PromotionDecision]:
        """批量决策。"""
        return [self.evaluate(r) for r in results]