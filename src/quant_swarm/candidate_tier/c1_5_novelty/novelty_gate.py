"""
C1.5 假设新颖性门（v3.1 候选层，shadow mode）。

关键设计：
1. 三档判定：NOVEL / VARIANT / DUPLICATE。
2. 阈值事前冻结（来自 L0 的 CommittedThreshold 或独立 commit）。
3. shadow mode：产出不进主干决策链，只进消融比较。
4. 语义 + AST 双重近邻。
5. 机制声明开放集校验。
6. 异质来源标签记录到事件流。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from candidate_tier.c1_5_novelty.known_library import (
    KnownFactorLibrary, KnownFactor,
)
from candidate_tier.c1_5_novelty.semantic_distance import SemanticDistance
from candidate_tier.c1_5_novelty.ast_distance import ASTDistance
from candidate_tier.c1_5_novelty.mechanism import (
    MechanismDeclaration, MechanismValidator, KNOWN_MECHANISM_FAMILIES,
)
from candidate_tier.c1_5_novelty.heterogeneous import (
    HeterogeneousHypothesis, HypothesisSource,
)
from core.event_log import EventLog, GLOBAL_EVENT_LOG


class NoveltyVerdict(str, Enum):
    """三档新颖性判定。"""
    NOVEL = "novel"          # 新颖：可进 L2 辩论
    VARIANT = "variant"      # 变体：不进辩论，仅占低配额
    DUPLICATE = "duplicate"  # 重复：直接拒绝


@dataclass
class NoveltyDecision:
    """新颖性判定结果。"""
    hid: str
    verdict: NoveltyVerdict
    semantic_distance: float
    ast_distance: float
    nearest_semantic: Optional[str]
    nearest_ast: Optional[str]
    mechanism_ok: bool
    mechanism_reason: str
    source: str
    shadow: bool = True  # 恒为 True（候选层）

    def to_dict(self):
        return {
            "hid": self.hid,
            "verdict": self.verdict.value,
            "semantic_distance": self.semantic_distance,
            "ast_distance": self.ast_distance,
            "nearest_semantic": self.nearest_semantic,
            "nearest_ast": self.nearest_ast,
            "mechanism_ok": self.mechanism_ok,
            "mechanism_reason": self.mechanism_reason,
            "source": self.source,
            "shadow": self.shadow,
        }


@dataclass(frozen=True)
class NoveltyThresholds:
    """
    新颖性阈值（事前冻结）。

    关键纪律（v3.1 修正④）：
    - 阈值在运行前 commit，运行期禁止漂移。
    - 标定依据是"事前已知因子库的分布"，非运行中命中率。
    """
    semantic_distance_threshold: float
    ast_distance_threshold: float
    frozen: bool = True

    def __post_init__(self):
        if not self.frozen:
            raise ValueError("新颖性阈值必须事前冻结")


class NoveltyThresholdCommitter:
    """
    新颖性阈值 commit 器。

    与 L0 的 ThresholdCommitter 同等待遇：
    - 运行前 commit
    - 写入不可变日志
    - 禁止二次 commit
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self._committed: Optional[NoveltyThresholds] = None
        self._calibration: Dict[str, Any] = {}

    def calibrate_from_library(self, library: KnownFactorLibrary,
                               semantic: SemanticDistance,
                               ast_dist: ASTDistance,
                               percentile: float = 10.0) -> tuple:
        """
        从已知因子库分布标定阈值。

        阈值 = 库内因子两两距离的 percentile 分位。
        例如：10 分位意味着"库内最相似的 10% 因子对"的距离。
        新假设与库的距离低于此阈值 → 判定为"不够新颖"。
        """
        import numpy as np
        factors = library.get_all()
        n = len(factors)
        if n < 2:
            return (0.30, 0.30)  # 库太小，用默认值

        sem_dists = []
        ast_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                sem_dists.append(semantic.distance(
                    factors[i].code, factors[j].code))
                ast_dists.append(ast_dist.distance(
                    factors[i].code, factors[j].code))

        sem_threshold = float(np.percentile(sem_dists, percentile))
        ast_threshold = float(np.percentile(ast_dists, percentile))
        return (sem_threshold, ast_threshold)

    def commit(self, semantic_threshold: float,
               ast_threshold: float,
               calibration_note: str = "") -> NoveltyThresholds:
        """事前 commit 阈值。"""
        if self._committed is not None:
            raise ValueError("新颖性阈值已 commit，禁止二次 commit")
        th = NoveltyThresholds(
            semantic_distance_threshold=semantic_threshold,
            ast_distance_threshold=ast_threshold,
        )
        self._committed = th
        self._calibration = {
            "semantic_threshold": semantic_threshold,
            "ast_threshold": ast_threshold,
            "note": calibration_note,
        }
        self.event_log.append("novelty_thresholds_committed", self._calibration)
        return th

    def get(self) -> NoveltyThresholds:
        if self._committed is None:
            raise ValueError("新颖性阈值尚未 commit")
        return self._committed


class NoveltyGate:
    """
    C1.5 新颖性门（shadow mode）。

    三档判定逻辑：
    - 语义距离 >= sem_thr 且 AST 距离 >= ast_thr → NOVEL
    - 任一距离 < 阈值（但非两者都 <）→ VARIANT
    - 两者都 < 阈值 → DUPLICATE
    """

    def __init__(self,
                 library: KnownFactorLibrary,
                 thresholds: NoveltyThresholds,
                 semantic: Optional[SemanticDistance] = None,
                 ast_dist: Optional[ASTDistance] = None,
                 mechanism_validator: Optional[MechanismValidator] = None,
                 event_log: Optional[EventLog] = None):
        self.library = library
        self.th = thresholds
        assert self.th.frozen, "阈值必须事前冻结"
        self.semantic = semantic or SemanticDistance()
        self.ast_dist = ast_dist or ASTDistance(use_tree_edit=False)
        self.mechanism_validator = mechanism_validator or MechanismValidator(
            event_log)
        self.event_log = event_log or GLOBAL_EVENT_LOG

    def evaluate(self, hyp: HeterogeneousHypothesis) -> NoveltyDecision:
        """对单个假设做新颖性判定。"""
        # 1) 语义距离
        lib_codes = [f.code for f in self.library.get_all()]
        sem_d, sem_nearest, _ = self.semantic.nearest(hyp.code, lib_codes)

        # 2) AST 距离
        ast_d, ast_nearest, _ = self.ast_dist.nearest(hyp.code, lib_codes)

        # 3) 三档判定
        sem_novel = sem_d >= self.th.semantic_distance_threshold
        ast_novel = ast_d >= self.th.ast_distance_threshold

        if sem_novel and ast_novel:
            verdict = NoveltyVerdict.NOVEL
        elif sem_novel or ast_novel:
            verdict = NoveltyVerdict.VARIANT
        else:
            verdict = NoveltyVerdict.DUPLICATE

        # 4) 机制声明校验
        declaration = MechanismDeclaration(
            mechanism_family=hyp.mechanism_family,
            failure_precondition=hyp.failure_precondition,
            economic_rationale=hyp.economic_rationale,
        )
        mech_ok, mech_reason = self.mechanism_validator.validate(declaration)

        decision = NoveltyDecision(
            hid=hyp.hid,
            verdict=verdict,
            semantic_distance=sem_d,
            ast_distance=ast_d,
            nearest_semantic=sem_nearest,
            nearest_ast=ast_nearest,
            mechanism_ok=mech_ok,
            mechanism_reason=mech_reason,
            source=hyp.source.value,
            shadow=True,
        )
        # 进事件流（shadow 记录）
        self.event_log.append("novelty_shadow_decision", decision.to_dict())
        return decision

    def evaluate_batch(self,
                       hyps: List[HeterogeneousHypothesis]
                       ) -> List[NoveltyDecision]:
        return [self.evaluate(h) for h in hyps]

    def shadow_summary(self, decisions: List[NoveltyDecision]) -> Dict[str, Any]:
        """
        shadow 汇总：若启用会过滤掉多少。
        这是候选层"shadow mode"的核心输出——只记录，不阻断。
        """
        n_total = len(decisions)
        n_novel = sum(1 for d in decisions if d.verdict == NoveltyVerdict.NOVEL)
        n_variant = sum(1 for d in decisions if d.verdict == NoveltyVerdict.VARIANT)
        n_dup = sum(1 for d in decisions if d.verdict == NoveltyVerdict.DUPLICATE)
        n_mech_fail = sum(1 for d in decisions if not d.mechanism_ok)

        # 来源分布
        source_counts: Dict[str, int] = {}
        for d in decisions:
            source_counts[d.source] = source_counts.get(d.source, 0) + 1

        summary = {
            "n_total": n_total,
            "n_novel": n_novel,
            "n_variant": n_variant,
            "n_duplicate": n_dup,
            "n_mechanism_failed": n_mech_fail,
            "source_breakdown": source_counts,
            "would_filter": n_variant + n_dup,
            "filter_rate": (n_variant + n_dup) / n_total if n_total else 0.0,
        }
        self.event_log.append("novelty_shadow_summary", summary)
        return summary