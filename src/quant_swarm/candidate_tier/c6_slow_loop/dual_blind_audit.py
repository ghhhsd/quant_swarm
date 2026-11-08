# 双向盲样抽检 + 独立存储
"""
双向盲样抽检（v3.1 修正⑤）。

关键设计：
1. 双向盲样：
   - 抽检者不知道因子来源（fid → 匿名 ID）
   - 生成侧不知道哪些因子被抽中（抽检结果独立存储）
2. 独立存储：
   - 抽检结果写入独立的审计日志（audit_events.jsonl）
   - 生成侧只能读聚合统计，不能读具体 fid
3. 抽检者与生成 Agent 隔离：
   - 独立随机种子（与生成侧不同源）
   - 独立存储（不与生成记录混在一起）
4. 统计判据：前向 CI 下界 > 阈值。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
import numpy as np
import hashlib
import json
from pathlib import Path
import time


@dataclass
class AuditRecord:
    """单次审计记录（写入独立审计日志）。"""
    anonymous_id: str
    audited: bool
    decision: str       # "accept" / "reject" / "skip"
    reason: str
    forward_sharpe: float
    forward_ci_low: float
    timestamp: float

    def to_dict(self):
        return {
            "anonymous_id": self.anonymous_id,
            "audited": self.audited,
            "decision": self.decision,
            "reason": self.reason,
            "forward_sharpe": self.forward_sharpe,
            "forward_ci_low": self.forward_ci_low,
            "timestamp": self.timestamp,
        }


@dataclass
class AuditRoundResult:
    """一轮审计结果（聚合视图，可暴露给生成侧）。"""
    round_idx: int
    n_total: int
    n_audited: int
    n_accepted: int
    n_rejected: int
    # 注意：不暴露具体 fid，只暴露匿名 ID
    decisions: List[AuditRecord] = field(default_factory=list)

    def to_aggregate(self) -> Dict:
        """聚合视图：生成侧可读。"""
        return {
            "round_idx": self.round_idx,
            "n_total": self.n_total,
            "n_audited": self.n_audited,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
        }


class IsolatedAuditLog:
    """
    独立审计日志。

    关键纪律：
    - 审计结果写入独立文件（audit_events.jsonl），不与生成记录混在一起。
    - 生成侧只能读聚合视图（to_aggregate），不能读具体记录。
    - 审计日志的读取权限在真实部署中应由权限系统控制。
    """

    def __init__(self, path: str = "audit_events.jsonl"):
        self.path = Path(path)
        self._records: List[AuditRecord] = []
        if self.path.exists():
            self._load()

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self._records.append(AuditRecord(**d))

    def append(self, record: AuditRecord) -> None:
        self._records.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def get_all(self) -> List[AuditRecord]:
        """仅供审计侧读取（真实部署应有权限控制）。"""
        return list(self._records)

    def get_aggregate(self) -> Dict:
        """聚合视图，生成侧可读。"""
        n_audited = sum(1 for r in self._records if r.audited)
        n_accepted = sum(1 for r in self._records
                         if r.audited and r.decision == "accept")
        n_rejected = sum(1 for r in self._records
                         if r.audited and r.decision == "reject")
        return {
            "n_total_records": len(self._records),
            "n_audited": n_audited,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
        }


class DualBlindAuditor:
    """
    双向盲样抽检器。

    与 L6 的 BlindAuditor 区别：
    - 抽检结果写入独立审计日志（隔离存储）
    - 生成侧只能读聚合视图
    """

    def __init__(self,
                 audit_ratio: float = 0.10,
                 min_audit: int = 5,
                 ci_threshold: float = 0.3,
                 auditor_seed: int = 999,
                 audit_log_path: str = "audit_events.jsonl"):
        self.audit_ratio = audit_ratio
        self.min_audit = min_audit
        self.ci_threshold = ci_threshold
        self.auditor_seed = auditor_seed
        self.audit_log = IsolatedAuditLog(audit_log_path)
        self._round_idx = 0
        self._anon_map: Dict[str, str] = {}  # fid → anonymous_id（仅审计侧）

    def _anonymize(self, fids: List[str]) -> Dict[str, str]:
        """
        fid → 匿名 ID。
        用独立种子，与生成侧不同源。
        匿名映射仅审计侧持有，不暴露给生成侧。
        """
        rng = np.random.default_rng(self.auditor_seed)
        perm = rng.permutation(len(fids))
        return {fids[i]: f"audit_{perm[i]:04d}" for i in range(len(fids))}

    def audit_round(self,
                    forward_results: Dict[str, Dict]) -> AuditRoundResult:
        """
        执行一轮双向盲样抽检。

        forward_results：{fid: {"forward_sharpe": ..., "forward_ci_low": ...}}
        返回聚合视图（生成侧可读）。
        """
        self._round_idx += 1
        fids = list(forward_results.keys())
        n = len(fids)
        if n == 0:
            return AuditRoundResult(
                round_idx=self._round_idx,
                n_total=0, n_audited=0, n_accepted=0, n_rejected=0,
            )

        n_audit = max(int(self.audit_ratio * n), min(self.min_audit, n))
        n_audit = min(n_audit, n)

        # 独立种子抽样
        rng = np.random.default_rng(self.auditor_seed + self._round_idx)
        indices = set(rng.choice(n, size=n_audit, replace=False).tolist())

        # 匿名化（审计侧持有映射）
        anon_map = self._anonymize(fids)
        self._anon_map.update(anon_map)

        records: List[AuditRecord] = []
        n_accepted = 0
        n_rejected = 0

        for i, fid in enumerate(fids):
            anon_id = anon_map[fid]
            fr = forward_results[fid]
            if i not in indices:
                rec = AuditRecord(
                    anonymous_id=anon_id, audited=False,
                    decision="skip", reason="未抽中",
                    forward_sharpe=fr.get("forward_sharpe", 0.0),
                    forward_ci_low=fr.get("forward_ci_low", 0.0),
                    timestamp=time.time(),
                )
                records.append(rec)
                self.audit_log.append(rec)
                continue

            # 盲样：只看 CI，不看来源
            ci_low = fr.get("forward_ci_low", 0.0)
            if ci_low > self.ci_threshold:
                rec = AuditRecord(
                    anonymous_id=anon_id, audited=True,
                    decision="accept",
                    reason=f"CI 下界 {ci_low:.3f} > {self.ci_threshold}",
                    forward_sharpe=fr.get("forward_sharpe", 0.0),
                    forward_ci_low=ci_low,
                    timestamp=time.time(),
                )
                n_accepted += 1
            else:
                rec = AuditRecord(
                    anonymous_id=anon_id, audited=True,
                    decision="reject",
                    reason=f"CI 下界 {ci_low:.3f} <= {self.ci_threshold}",
                    forward_sharpe=fr.get("forward_sharpe", 0.0),
                    forward_ci_low=ci_low,
                    timestamp=time.time(),
                )
                n_rejected += 1
            records.append(rec)
            self.audit_log.append(rec)

        return AuditRoundResult(
            round_idx=self._round_idx,
            n_total=n, n_audited=n_audit,
            n_accepted=n_accepted, n_rejected=n_rejected,
            decisions=records,  # 注意：此字段仅审计侧持有
        )

    def get_aggregate(self) -> Dict:
        """生成侧可读的聚合视图。"""
        return self.audit_log.get_aggregate()

    def get_anon_map(self) -> Dict[str, str]:
        """仅审计侧可读（真实部署应权限控制）。"""
        return dict(self._anon_map)