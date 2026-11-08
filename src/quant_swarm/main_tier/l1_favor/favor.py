"""
L1 门面：组合叙事预承诺 + 事实校验 + LLM 校验 + 失败分流。

对外接口：
- validate(hypothesis) -> FavorResult

流程（v3.1 严格时序）：
1. 假设 → 生成叙事（回测前）
2. 叙事哈希锁定（进事件流）
3. 非 LLM 事实校验（数据/先验/对齐）
4. 异构 LLM 校验（三阶段拆解）
5. 两者都通过 → FavorResult(passed=True)
6. 任一失败 → FavorResult(passed=False, failure_type, trace)

关键：步骤 1-2 必须在步骤 3-4 之前。叙事生成不能依赖校验结果
（否则又是事后合理化）。叙事由外部（L2 codegen 或独立假设生成器）
提供，L1 只负责锁性和校验。
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from main_tier.l1_favor.narrative import (
    Narrative, NarrativeCommitment, NarrativeCommitter,
)
from main_tier.l1_favor.facts import FactChecker, FactCheckResult
from main_tier.l1_favor.validator import (
    HeterogeneousValidator, ValidatorVerdict,
)
from main_tier.l1_favor.failure import (
    FailureType, FailureTrace, HEAL_STRATEGY, RETRYABLE,
)
from main_tier.l0_governance import GovernanceLayer
from main_tier.l2_agents.llm_client import LLMClient
from core.event_log import EventLog, GLOBAL_EVENT_LOG


@dataclass
class FavorResult:
    """
    L1 校验结果。

    passed=True 时：
    - hypothesis 带 narrative + narrative_hash + code
    - narrative_commitment 可用于 L5 事后一致性校验
    passed=False 时：
    - failure_type + failure_trace 供自愈分流
    """
    passed: bool
    hid: str
    narrative: Optional[Narrative] = None
    narrative_commitment: Optional[NarrativeCommitment] = None
    code: Optional[str] = None
    failure_type: Optional[FailureType] = None
    failure_trace: Optional[FailureTrace] = None
    fact_results: List[FactCheckResult] = field(default_factory=list)
    validator_verdict: Optional[ValidatorVerdict] = None

    @property
    def narrative_hash(self) -> Optional[str]:
        return self.narrative_commitment.hash if self.narrative_commitment else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "hid": self.hid,
            "narrative": self.narrative.to_dict() if self.narrative else None,
            "narrative_hash": self.narrative_hash,
            "code": self.code,
            "failure_type": self.failure_type.value if self.failure_type else None,
            "failure_trace": self.failure_trace.to_dict() if self.failure_trace else None,
            "fact_results": [
                {"ok": r.ok, "constraint": r.constraint, "reason": r.reason}
                for r in self.fact_results
            ],
            "validator_verdict": (
                {
                    "llm_pass": self.validator_verdict.llm_pass,
                    "mechanism_valid": self.validator_verdict.mechanism_valid,
                    "data_observable": self.validator_verdict.data_observable,
                    "counterexamples": self.validator_verdict.counterexamples,
                    "reason": self.validator_verdict.reason,
                } if self.validator_verdict else None
            ),
        }


class FavorLayer:
    """
    FaVOR 门面。

    依赖注入：
    - committer：叙事预承诺器
    - fact_checker：非 LLM 事实校验器
    - validator：异构 LLM 校验器（与 codegen 异构）
    - governance：L0 治理层（消耗 Token / 记录失败）
    """

    def __init__(self,
                 validator_llm: LLMClient,
                 governance: GovernanceLayer,
                 codegen_llm: Optional[LLMClient] = None,
                 event_log: Optional[EventLog] = None):
        log = event_log or GLOBAL_EVENT_LOG
        self.committer = NarrativeCommitter(log)
        self.fact_checker = FactChecker()
        self.validator = HeterogeneousValidator(
            validator_llm, governance, codegen_llm=codegen_llm,
        )
        self.gov = governance
        self.event_log = log

    def validate(self, narrative: Narrative, code: str) -> FavorResult:
        """
        校验一个已生成叙事 + 代码的假设。

        注意：叙事由调用方提供（L2 codegen 或独立假设生成器），
        L1 不负责"生成叙事"，只负责"锁定 + 校验"。
        这保证了叙事不是校验的产物。
        """
        hid = narrative.hid

        # 1) 叙事预承诺（回测前锁定）
        try:
            commitment = self.committer.commit(narrative)
        except ValueError as e:
            return self._fail(hid, FailureType.LOGIC, "narrative",
                              f"叙事提交失败: {e}")

        # 2) 非 LLM 事实校验
        fact_results = self.fact_checker.check_all(narrative, code)
        for r in fact_results:
            if not r.ok:
                # 数据约束失败归 DATA，其他归 LOGIC
                ftype = FailureType.DATA if r.constraint == "data_observability" else FailureType.LOGIC
                return self._fail(hid, ftype, "fact_check", r.reason,
                                  commitment=commitment, fact_results=fact_results)

        # 3) 异构 LLM 校验
        try:
            verdict = self.validator.run(narrative, code)
        except Exception as e:
            return self._fail(hid, FailureType.LOGIC, "llm_validate",
                              f"LLM 校验失败: {e}",
                              commitment=commitment, fact_results=fact_results)

        # 4) 组合判定：LLM 通过 AND 事实层通过 → passed
        if not verdict.llm_pass:
            ftype = FailureType.DATA if not verdict.data_observable else FailureType.LOGIC
            return self._fail(
                hid, ftype, "llm_validate",
                f"LLM 校验失败: {verdict.reason}; 反例: {verdict.counterexamples}",
                commitment=commitment, fact_results=fact_results,
                verdict=verdict,
            )

        # 全部通过
        self.event_log.append("favor_passed", {
            "hid": hid,
            "narrative_hash": commitment.hash,
            "fact_results": [
                {"constraint": r.constraint, "ok": r.ok} for r in fact_results
            ],
            "validator_llm_pass": verdict.llm_pass,
        })
        return FavorResult(
            passed=True,
            hid=hid,
            narrative=narrative,
            narrative_commitment=commitment,
            code=code,
            fact_results=fact_results,
            validator_verdict=verdict,
        )

    def _fail(self, hid: str, failure_type: FailureType, stage: str,
              message: str,
              commitment: Optional[NarrativeCommitment] = None,
              fact_results: Optional[List[FactCheckResult]] = None,
              verdict: Optional[ValidatorVerdict] = None) -> FavorResult:
        trace = FailureTrace(
            hid=hid,
            failure_type=failure_type,
            stage=stage,
            message=message,
            context={"narrative_hash": commitment.hash if commitment else None},
        )
        self.event_log.append("favor_failed", trace.to_dict())
        return FavorResult(
            passed=False,
            hid=hid,
            narrative=commitment.narrative if commitment else None,
            narrative_commitment=commitment,
            failure_type=failure_type,
            failure_trace=trace,
            fact_results=fact_results or [],
            validator_verdict=verdict,
        )