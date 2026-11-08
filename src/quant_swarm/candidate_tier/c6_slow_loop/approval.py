# ACCEPTED → COMMITTED 审批协议
"""
审批协议（v3.1 修正⑤：ACCEPTED → COMMITTED 人工审批）。

关键设计：
1. 审批是"结构化流程"，不是"自由裁量"。
2. 双签：risk_officer（风控）+ portfolio_manager（PM）。
3. 审批顺序：先风控（看前向验证 + 盲样），后 PM（看容量档位 + 组合边际）。
4. 审批判据：预设统计标准。
5. 审批记录写入独立日志（approval_events.jsonl）。
6. 审批拒绝的理由必须结构化，供 L1 失败分流。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import json
import time
from pathlib import Path


class ApprovalRole(str, Enum):
    RISK_OFFICER = "risk_officer"
    PORTFOLIO_MANAGER = "portfolio_manager"


class ApprovalDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ApprovalRecord:
    """单次审批记录。"""
    fid: str
    role: ApprovalRole
    decision: ApprovalDecision
    reason: str
    evidence: Dict       # 审批依据（前向验证 + 盲样 + 容量档位 + 组合边际）
    reviewer: str        # 审批人标识
    timestamp: float
    sequence: int        # 审批顺序（1 = 风控先，2 = PM 后）

    def to_dict(self):
        return {
            "fid": self.fid,
            "role": self.role.value,
            "decision": self.decision.value,
            "reason": self.reason,
            "evidence": self.evidence,
            "reviewer": self.reviewer,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }


class IsolatedApprovalLog:
    """
    独立审批日志。

    与审计日志同理：写入独立文件，可审计。
    """

    def __init__(self, path: str = "approval_events.jsonl"):
        self.path = Path(path)
        self._records: List[ApprovalRecord] = []
        if self.path.exists():
            self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                d["role"] = ApprovalRole(d["role"])
                d["decision"] = ApprovalDecision(d["decision"])
                self._records.append(ApprovalRecord(**d))

    def append(self, record: ApprovalRecord) -> None:
        self._records.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def get_history(self, fid: str) -> List[ApprovalRecord]:
        return [r for r in self._records if r.fid == fid]

    def get_all(self) -> List[ApprovalRecord]:
        return list(self._records)


class ApprovalProtocol:
    """
    审批协议。

    流程：
    1. 风控审批（risk_officer）：
       - 判据：前向 CI 下界 > 阈值 + 盲样通过
       - 通过 → 进入 PM 审批
       - 拒绝 → 流程结束
    2. PM 审批（portfolio_manager）：
       - 判据：容量档位 != REJECT + 组合边际增益 > 阈值
       - 通过 → APPROVED
       - 拒绝 → REJECTED
    """

    def __init__(self,
                 forward_ci_threshold: float = 0.3,
                 marginal_gain_threshold: float = 0.02,
                 approval_log_path: str = "approval_events.jsonl"):
        self.forward_ci_threshold = forward_ci_threshold
        self.marginal_gain_threshold = marginal_gain_threshold
        self.approval_log = IsolatedApprovalLog(approval_log_path)
        self._pending: Dict[str, List[ApprovalRecord]] = {}

    def submit(self, fid: str,
               forward_ci_low: float,
               audit_passed: bool,
               capacity_tier: str,
               marginal_gain: float) -> Dict:
        """
        提交审批。
        返回 {"status": "approved" / "rejected" / "pending", "records": [...]}
        """
        records = []
        evidence = {
            "forward_ci_low": forward_ci_low,
            "audit_passed": audit_passed,
            "capacity_tier": capacity_tier,
            "marginal_gain": marginal_gain,
        }

        # 1) 风控审批
        risk_reasons = []
        if forward_ci_low <= self.forward_ci_threshold:
            risk_reasons.append(
                f"前向 CI 下界 {forward_ci_low:.3f} <= {self.forward_ci_threshold}"
            )
        if not audit_passed:
            risk_reasons.append("盲样未通过")

        risk_decision = ApprovalDecision.REJECTED if risk_reasons \
            else ApprovalDecision.APPROVED
        risk_record = ApprovalRecord(
            fid=fid, role=ApprovalRole.RISK_OFFICER,
            decision=risk_decision,
            reason="; ".join(risk_reasons) if risk_reasons else "风控通过",
            evidence=evidence,
            reviewer="risk_officer_auto",  # 自动审批（真实部署可人工）
            timestamp=time.time(),
            sequence=1,
        )
        self.approval_log.append(risk_record)
        records.append(risk_record)

        if risk_decision == ApprovalDecision.REJECTED:
            return {"status": "rejected", "records": records}

        # 2) PM 审批
        pm_reasons = []
        if capacity_tier == "tier_reject":
            pm_reasons.append(f"容量档位 REJECT")
        if marginal_gain <= self.marginal_gain_threshold:
            pm_reasons.append(
                f"组合边际增益 {marginal_gain:.4f} <= {self.marginal_gain_threshold}"
            )

        pm_decision = ApprovalDecision.REJECTED if pm_reasons \
            else ApprovalDecision.APPROVED
        pm_record = ApprovalRecord(
            fid=fid, role=ApprovalRole.PORTFOLIO_MANAGER,
            decision=pm_decision,
            reason="; ".join(pm_reasons) if pm_reasons else "PM 通过",
            evidence=evidence,
            reviewer="pm_auto",
            timestamp=time.time(),
            sequence=2,
        )
        self.approval_log.append(pm_record)
        records.append(pm_record)

        status = "approved" if pm_decision == ApprovalDecision.APPROVED \
            else "rejected"
        return {"status": status, "records": records}

    def get_history(self, fid: str) -> List[ApprovalRecord]:
        return self.approval_log.get_history(fid)