"""
难度信号 → 搜索侧调整（v3.1 修正①核心）。

关键纪律：
- 难度信号只能调整"搜索侧"参数（探索配额、逻辑族切换、辩论预算）。
- 永远不能调整"验收侧"参数（阈值、DSR/FDR 门槛）。
- 搜索侧调整本身受预算约束：提高探索配额要消耗更多 Token，
  必须在 QuotaManager 的上限内。
- 若预算内调整仍不足，触发系统级告警（提示上游假设质量需换血），
  而非自动降门槛。
"""
from dataclasses import dataclass
from typing import Optional
from core.event_log import EventLog, GLOBAL_EVENT_LOG


@dataclass
class SearchSideAdjustment:
    """搜索侧调整建议。"""
    exploration_ratio: float       # 探索配额比例（默认 0.2，可上调）
    family_rotation: bool          # 是否切换逻辑族
    debate_budget_multiplier: float  # 辩论预算倍数（默认 1.0）
    reason: str
    escalate: bool = False         # 是否触发系统级告警


class DifficultySignal:
    """
    根据精英池补充速率调整搜索侧。

    输入：近 K 轮的精英池净增长。
    输出：SearchSideAdjustment（搜索侧调整建议，非验收侧）。
    """

    def __init__(self, event_log: Optional[EventLog] = None,
                 low_threshold: float = 1.0,   # 每轮平均净增长 < 1 为"低"
                 high_threshold: float = 3.0): # 每轮平均净增长 > 3 为"高"
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def evaluate(self, recent_growth: list[int]) -> SearchSideAdjustment:
        """
        recent_growth: 近 K 轮每轮精英池净增长。
        """
        if not recent_growth:
            return SearchSideAdjustment(0.2, False, 1.0, "无历史数据", False)

        avg = sum(recent_growth) / len(recent_growth)

        if avg < self.low_threshold:
            # 补充慢 → 提高探索配额 + 切换族 + 加大辩论预算
            adj = SearchSideAdjustment(
                exploration_ratio=0.35,
                family_rotation=True,
                debate_budget_multiplier=1.3,
                reason=f"近 {len(recent_growth)} 轮平均净增长 {avg:.1f} < {self.low_threshold}",
                escalate=False,
            )
        elif avg > self.high_threshold:
            # 补充快 → 收紧探索（节省预算给利用）
            adj = SearchSideAdjustment(
                exploration_ratio=0.15,
                family_rotation=False,
                debate_budget_multiplier=0.9,
                reason=f"近 {len(recent_growth)} 轮平均净增长 {avg:.1f} > {self.high_threshold}",
                escalate=False,
            )
        else:
            adj = SearchSideAdjustment(
                exploration_ratio=0.20,
                family_rotation=False,
                debate_budget_multiplier=1.0,
                reason=f"近 {len(recent_growth)} 轮平均净增长 {avg:.1f} 在正常区间",
                escalate=False,
            )

        self.event_log.append("difficulty_signal", {
            "recent_growth": recent_growth,
            "avg": avg,
            "exploration_ratio": adj.exploration_ratio,
            "family_rotation": adj.family_rotation,
            "debate_budget_multiplier": adj.debate_budget_multiplier,
            "escalate": adj.escalate,
        })
        return adj

    def escalate_if_persistent(self, recent_growth: list[int],
                               n_rounds: int = 3) -> SearchSideAdjustment:
        """
        若连续 n_rounds 轮平均增长都低于 low_threshold，
        触发系统级告警（提示假设生成端换血），而非降门槛。
        """
        adj = self.evaluate(recent_growth)
        if len(recent_growth) >= n_rounds:
            recent_n = recent_growth[-n_rounds:]
            if all(g < self.low_threshold for g in recent_n):
                adj.escalate = True
                adj.reason += f"；连续 {n_rounds} 轮低于阈值，触发系统告警"
                self.event_log.append("difficulty_escalation", {
                    "recent_growth": recent_n,
                    "message": "假设生成端质量不足，需上游换血（禁止降门槛）",
                })
        return adj