"""
慢环：前向验证 + 盲样抽检 + 上线决策（前向/昂贵/异步）。

关键设计（v3.1 修正⑤）：
1. 慢环输入：技能库 DRAFT 条目（来自快环）。
2. 慢环前向验证：候选因子在"前向模拟盘"运行，按周/月滚动出结果。
3. 慢环盲样抽检：BlindAuditor 按预设协议抽检。
4. 慢环输出：ACCEPTED / REJECTED 状态转移。
5. 慢环不做最终上线——上线是 ACCEPTED → COMMITTED，由人工走协议。

时序：
- 快环：每小时/每天，产出 DRAFT
- 慢环：每周/每月，前向验证 + 盲样抽检
- 上线：由人工定期（如每月）从 ACCEPTED 中按协议审批 COMMITTED
"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np
from scipy import stats as scipy_stats

from main_tier.l6_skillstore.skill_store import SkillStore
from main_tier.l6_skillstore.state_machine import SkillState
from main_tier.l6_skillstore.blind_audit import (
    BlindAuditor, SlowLoopResult, AuditRoundResult,
)


@dataclass
class SlowLoopRoundResult:
    """慢环单轮结果。"""
    round_idx: int
    n_drafts: int
    n_forward_tested: int
    n_accepted: int
    n_rejected: int
    audit_result: Optional[AuditRoundResult] = None


class SlowLoop:
    """
    慢环：前向验证 + 盲样抽检。
    """

    def __init__(self, store: SkillStore,
                 auditor: BlindAuditor,
                 forward_test_days: int = 252):
        self.store = store
        self.auditor = auditor
        self.forward_test_days = forward_test_days
        self.round_idx = 0

    def forward_test(self, fid: str,
                     forward_returns: np.ndarray) -> SlowLoopResult:
        """
        前向验证：计算前向 Sharpe 的 CI。
        forward_returns：前向模拟盘的日收益序列。
        """
        n = len(forward_returns)
        if n < 2:
            return SlowLoopResult(fid, 0.0, 0.0, 0.0, n)
        sharpe = float(np.mean(forward_returns) / (np.std(forward_returns) + 1e-9)
                       * np.sqrt(252))
        se = np.sqrt(252) / np.sqrt(n)
        z = scipy_stats.norm.ppf(1 - (1 - self.auditor.confidence) / 2)
        ci_low = sharpe - z * se
        ci_high = sharpe + z * se
        return SlowLoopResult(fid, sharpe, ci_low, ci_high, n)

    def run_round(self,
                  forward_returns_per_fid: dict,
                  use_audit: bool = True) -> SlowLoopRoundResult:
        """
        执行一轮慢环。
        forward_returns_per_fid：{fid: np.ndarray 前向收益}
        """
        self.round_idx += 1
        drafts = self.store.projector.get_by_state(SkillState.DRAFT)
        n_drafts = len(drafts)

        # 1) 前向验证
        results: List[SlowLoopResult] = []
        for draft in drafts:
            rets = forward_returns_per_fid.get(draft.fid)
            if rets is None or len(rets) < 2:
                continue
            results.append(self.forward_test(draft.fid, rets))

        # 2) 盲样抽检
        audit_result = None
        if use_audit and results:
            audit_result = self.auditor.audit_round(results)

        # 3) 状态转移：根据审计结果
        n_accepted = 0
        n_rejected = 0
        if audit_result:
            # 匿名 ID → fid 反查（仅限内部）
            anon_to_fid = {self.auditor._anonymize([r.fid for r in results])[r.fid]: r.fid
                           for r in results}
            for d in audit_result.decisions:
                if not d.audited:
                    continue
                fid = anon_to_fid.get(d.anonymous_id)
                if fid is None:
                    continue
                if d.decision == "accept":
                    self.store.accept(fid, reason=f"盲样通过: {d.reason}")
                    n_accepted += 1
                elif d.decision == "reject":
                    self.store.reject(fid, reason=f"盲样拒绝: {d.reason}")
                    n_rejected += 1

        return SlowLoopRoundResult(
            round_idx=self.round_idx,
            n_drafts=n_drafts,
            n_forward_tested=len(results),
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            audit_result=audit_result,
        )