"""
结构化盲样抽检协议（v3.1 修正⑤）。

关键设计：
1. 抽检比例预设：max(audit_ratio * n, min_audit)。
2. 盲样双向：
   - 抽检者不知道因子来源（fid 映射到匿名 ID）
   - 生成 Agent 不知道哪些因子被抽中（抽检结果单独存储）
3. 统计判据：前向 Sharpe 的置信下界 > 阈值。
4. 抽检者与生成 Agent 隔离（独立随机种子）。
5. 抽检结果写入独立的审计日志，不与生成记录混在一起。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from scipy import stats as scipy_stats
import hashlib


@dataclass
class SlowLoopResult:
    """慢环前向验证结果。"""
    fid: str
    forward_sharpe: float
    forward_ci_low: float
    forward_ci_high: float
    n_obs: int


@dataclass
class AuditDecision:
    """单次盲样抽检决策。"""
    anonymous_id: str
    audited: bool
    decision: str  # "accept" / "reject" / "skip"
    reason: str


@dataclass
class AuditRoundResult:
    """一轮审计结果。"""
    round_idx: int
    n_total: int
    n_audited: int
    n_accepted: int
    n_rejected: int
    decisions: List[AuditDecision] = field(default_factory=list)


class BlindAuditor:
    """
    结构化盲样抽检器。

    关键纪律：
    - 抽检者与生成 Agent 隔离：用独立随机种子。
    - 盲样双向：fid 映射到匿名 ID，生成侧看不到抽检结果。
    - 统计判据：前向 CI 下界 > 阈值。
    """

    def __init__(self,
                 audit_ratio: float = 0.10,
                 min_audit: int = 5,
                 ci_threshold: float = 0.3,
                 auditor_seed: int = 999,
                 confidence: float = 0.95):
        self.audit_ratio = audit_ratio
        self.min_audit = min_audit
        self.ci_threshold = ci_threshold
        self.auditor_seed = auditor_seed
        self.confidence = confidence
        self._round_idx = 0

    def _anonymize(self, fids: List[str]) -> Dict[str, str]:
        """fid → 匿名 ID。用独立种子，生成侧无法反推。"""
        rng = np.random.default_rng(self.auditor_seed)
        perm = rng.permutation(len(fids))
        return {fids[i]: f"audit_{perm[i]:04d}" for i in range(len(fids))}

    def audit_round(self, results: List[SlowLoopResult]) -> AuditRoundResult:
        """
        执行一轮盲样抽检。
        results：慢环前向验证结果列表。
        """
        self._round_idx += 1
        n = len(results)
        n_audit = max(int(self.audit_ratio * n), min(self.min_audit, n))
        n_audit = min(n_audit, n)

        # 独立种子抽样
        rng = np.random.default_rng(self.auditor_seed + self._round_idx)
        indices = set(rng.choice(n, size=n_audit, replace=False).tolist())

        # 匿名化
        fids = [r.fid for r in results]
        anon_map = self._anonymize(fids)

        decisions = []
        n_accepted = 0
        n_rejected = 0
        for i, r in enumerate(results):
            anon_id = anon_map[r.fid]
            if i not in indices:
                decisions.append(AuditDecision(
                    anonymous_id=anon_id, audited=False,
                    decision="skip", reason="未抽中",
                ))
                continue
            # 盲样：只看 CI，不看来源/谱系/单因子绩效
            if r.forward_ci_low > self.ci_threshold:
                decisions.append(AuditDecision(
                    anonymous_id=anon_id, audited=True,
                    decision="accept",
                    reason=f"CI 下界 {r.forward_ci_low:.3f} > {self.ci_threshold}",
                ))
                n_accepted += 1
            else:
                decisions.append(AuditDecision(
                    anonymous_id=anon_id, audited=True,
                    decision="reject",
                    reason=f"CI 下界 {r.forward_ci_low:.3f} <= {self.ci_threshold}",
                ))
                n_rejected += 1

        return AuditRoundResult(
            round_idx=self._round_idx,
            n_total=n,
            n_audited=n_audit,
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            decisions=decisions,
        )