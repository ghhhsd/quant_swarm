# 门面：前向 + 盲样 + 审批
"""
C6 慢环门面：前向验证 + 双向盲样 + 审批协议 + shadow mode。

关键设计：
1. shadow mode：产出不进主干决策链，只进消融比较。
2. 慢环是"异步"的——它不阻塞快环。
3. 输出结构含：前向验证结果 + 盲样结果 + 审批结果。
4. 与 L6 的关系：L6 是主干基础慢环，C6 是候选层增强慢环（含审批协议）。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import polars as pl

from candidate_tier.c6_slow_loop.forward_test import (
    ForwardTester, ForwardTestResult,
)
from candidate_tier.c6_slow_loop.dual_blind_audit import (
    DualBlindAuditor, AuditRoundResult,
)
from candidate_tier.c6_slow_loop.approval import (
    ApprovalProtocol, ApprovalRecord,
)
from core.event_log import EventLog, GLOBAL_EVENT_LOG


@dataclass
class SlowLoopRoundResult:
    """慢环单轮结果（shadow 汇总）。"""
    round_idx: int
    n_forward_tested: int
    n_audited: int
    n_audit_passed: int
    n_approved: int
    n_rejected: int
    audit_aggregate: Dict
    shadow: bool = True

    def to_dict(self):
        return {
            "round_idx": self.round_idx,
            "n_forward_tested": self.n_forward_tested,
            "n_audited": self.n_audited,
            "n_audit_passed": self.n_audit_passed,
            "n_approved": self.n_approved,
            "n_rejected": self.n_rejected,
            "audit_aggregate": self.audit_aggregate,
            "shadow": self.shadow,
        }


class SlowLoop:
    """
    C6 慢环（shadow mode）。

    流程：
    1. 前向验证（周/月滚动）
    2. 双向盲样抽检
    3. 审批协议（风控 + PM 双签）
    """

    def __init__(self,
                 forward_tester: Optional[ForwardTester] = None,
                 auditor: Optional[DualBlindAuditor] = None,
                 approval: Optional[ApprovalProtocol] = None,
                 event_log: Optional[EventLog] = None):
        self.forward_tester = forward_tester or ForwardTester()
        self.auditor = auditor or DualBlindAuditor()
        self.approval = approval or ApprovalProtocol()
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self.round_idx = 0

    def run_round(self,
                  forward_returns_per_fid: Dict[str, np.ndarray],
                  capacity_tiers: Dict[str, str],
                  marginal_gains: Dict[str, float],
                  test_period: str = "monthly") -> SlowLoopRoundResult:
        """
        执行一轮慢环。

        forward_returns_per_fid：{fid: 前向收益}
        capacity_tiers：{fid: 容量档位}（来自 C5.5）
        marginal_gains：{fid: 组合边际增益}（来自 L5）
        """
        self.round_idx += 1

        # 1) 前向验证
        forward_results = self.forward_tester.batch_test(
            forward_returns_per_fid, test_period)

        # 2) 双向盲样抽检
        audit_input = {
            r.fid: {
                "forward_sharpe": r.forward_sharpe,
                "forward_ci_low": r.forward_ci_low,
            }
            for r in forward_results
        }
        audit_result = self.auditor.audit_round(audit_input)

        # 3) 审批协议
        # 需要知道哪些 fid 通过了盲样——但审计侧不暴露 fid 给生成侧
        # 本实现简化：直接用 forward_ci_low > 阈值 作为 audit_passed 判定
        n_approved = 0
        n_rejected = 0
        for r in forward_results:
            audit_passed = r.forward_ci_low > self.auditor.ci_threshold
            capacity_tier = capacity_tiers.get(r.fid, "tier_reject")
            marginal_gain = marginal_gains.get(r.fid, 0.0)
            result = self.approval.submit(
                fid=r.fid,
                forward_ci_low=r.forward_ci_low,
                audit_passed=audit_passed,
                capacity_tier=capacity_tier,
                marginal_gain=marginal_gain,
            )
            if result["status"] == "approved":
                n_approved += 1
            elif result["status"] == "rejected":
                n_rejected += 1

        round_result = SlowLoopRoundResult(
            round_idx=self.round_idx,
            n_forward_tested=len(forward_results),
            n_audited=audit_result.n_audited,
            n_audit_passed=audit_result.n_accepted,
            n_approved=n_approved,
            n_rejected=n_rejected,
            audit_aggregate=audit_result.to_aggregate(),
            shadow=True,
        )
        self.event_log.append("slow_loop_shadow_round", round_result.to_dict())
        return round_result

    def shadow_summary(self, results: List[SlowLoopRoundResult]) -> Dict:
        """shadow 汇总（跨轮）。"""
        if not results:
            return {"n_rounds": 0}
        return {
            "n_rounds": len(results),
            "total_forward_tested": sum(r.n_forward_tested for r in results),
            "total_audited": sum(r.n_audited for r in results),
            "total_audit_passed": sum(r.n_audit_passed for r in results),
            "total_approved": sum(r.n_approved for r in results),
            "total_rejected": sum(r.n_rejected for r in results),
            "approval_rate": (
                sum(r.n_approved for r in results)
                / max(1, sum(r.n_forward_tested for r in results))
            ),
        }