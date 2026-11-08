"""
C3+ 门面：shadow mode + 增量缓存 + 内部/外部分离。

关键设计：
1. shadow mode：产出不进主干决策链，只进消融比较。
2. 内部重叠 + 外部代理分别输出，不混在一起。
3. 增量缓存：新因子入池时只与现有因子两两计算。
4. flagged 判定只看内部重叠，外部代理只记录。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import polars as pl

from candidate_tier.c3_plus_crowding.positions import (
    PositionExtractor, PositionMatrix,
)
from candidate_tier.c3_plus_crowding.internal_overlap import (
    InternalOverlapChecker, InternalOverlapReport, OverlapPair,
)
from candidate_tier.c3_plus_crowding.external_proxy import (
    ExternalProxyExtractor, ExternalProxyReport,
)
from core.event_log import EventLog, GLOBAL_EVENT_LOG


@dataclass
class CrowdingDecision:
    """
    C3+ 判定结果。

    注意字段命名：
    - internal_overlap：内部持仓重叠（能看见的）
    - external_proxy：外部拥挤代理（看不见的，单独列）
    - flagged：只看 internal_overlap
    """
    fid: str
    internal_overlap: InternalOverlapReport
    external_proxy: ExternalProxyReport
    flagged: bool
    reason: str
    shadow: bool = True

    def to_dict(self):
        return {
            "fid": self.fid,
            "internal_overlap": self.internal_overlap.to_dict(),
            "external_proxy": self.external_proxy.to_dict(),
            "flagged": self.flagged,
            "reason": self.reason,
            "shadow": self.shadow,
        }


class CrowdingGate:
    """
    C3+ 持仓拥挤度门（shadow mode）。

    三个职责：
    1. 内部持仓重叠计算（flagged 判定依据）
    2. 外部拥挤代理提取（单独记录）
    3. shadow mode（不阻断主干）
    """

    def __init__(self, df: pl.DataFrame,
                 max_internal_overlap: float = 0.70,
                 sample_days: int = 63,
                 event_log: Optional[EventLog] = None):
        self.df = df
        self.extractor = PositionExtractor(df, sample_days=sample_days)
        self.internal_checker = InternalOverlapChecker(max_internal_overlap)
        self.external_extractor = ExternalProxyExtractor()
        self.event_log = event_log or GLOBAL_EVENT_LOG
        # 外部代理只需提取一次（数据不变），缓存
        self._external_cache: Optional[ExternalProxyReport] = None

    def _get_external(self) -> ExternalProxyReport:
        if self._external_cache is None:
            self._external_cache = self.external_extractor.extract(self.df)
        return self._external_cache

    # ---- 全量检查 -------------------------------------------------------
    def evaluate(self,
                 fids_codes: Dict[str, str]) -> CrowdingDecision:
        """
        对一组因子做全量检查。

        fids_codes：{fid: code}。这里用 fid 作为决策对象，
        但内部重叠是全组的属性——所以只对第一个 fid 返回决策。
        """
        positions = self.extractor.extract_batch(fids_codes)
        internal = self.internal_checker.check(positions)
        external = self._get_external()
        flagged = self.internal_checker.would_flag(internal)
        reason = (f"内部重叠均值 {internal.mean_overlap:.3f} > "
                  f"{self.internal_checker.max_internal_overlap}"
                  if flagged else "ok")

        # 用第一个 fid 作为决策标识
        first_fid = next(iter(fids_codes)) if fids_codes else "group"
        decision = CrowdingDecision(
            fid=first_fid,
            internal_overlap=internal,
            external_proxy=external,
            flagged=flagged,
            reason=reason,
            shadow=True,
        )
        self.event_log.append("crowding_shadow_decision", decision.to_dict())
        return decision

    # ---- 增量检查 -------------------------------------------------------
    def evaluate_incremental(self,
                             new_fid: str,
                             new_code: str,
                             existing_fids_codes: Dict[str, str]
                             ) -> CrowdingDecision:
        """
        增量检查：新因子入池前，与现有因子两两计算重叠。
        不重算现有因子之间的重叠。
        """
        new_pos = self.extractor.extract(new_fid, new_code)
        existing_positions = {
            fid: self.extractor.extract(fid, code)
            for fid, code in existing_fids_codes.items()
        }
        pairs = self.internal_checker.incremental_check(
            new_fid, new_pos, existing_positions,
        )
        # 构造"伪"内部报告（只有新因子相关的对）
        sims = [p.cosine_similarity for p in pairs]
        internal = InternalOverlapReport(
            mean_overlap=float(np.mean(sims)) if sims else 0.0,
            max_overlap=float(np.max(sims)) if sims else 0.0,
            max_pair=max(pairs, key=lambda p: p.cosine_similarity) if pairs else None,
            n_factors=len(existing_positions) + 1,
            n_pairs=len(pairs),
            pairwise=pairs,
        )
        external = self._get_external()
        flagged = self.internal_checker.would_flag_incremental(pairs)
        reason = (f"新因子最大重叠 {internal.max_overlap:.3f} > "
                  f"{self.internal_checker.max_internal_overlap}"
                  if flagged else "ok")

        decision = CrowdingDecision(
            fid=new_fid,
            internal_overlap=internal,
            external_proxy=external,
            flagged=flagged,
            reason=reason,
            shadow=True,
        )
        self.event_log.append("crowding_shadow_decision", decision.to_dict())
        return decision

    # ---- shadow 汇总 ---------------------------------------------------
    def shadow_summary(self,
                       decisions: List[CrowdingDecision]) -> Dict:
        """
        shadow 汇总：若启用会拦截多少。
        """
        n_total = len(decisions)
        n_flagged = sum(1 for d in decisions if d.flagged)
        summary = {
            "n_total": n_total,
            "n_flagged": n_flagged,
            "flag_rate": n_flagged / n_total if n_total else 0.0,
            "internal_overlap_mean": float(np.mean(
                [d.internal_overlap.mean_overlap for d in decisions])
            ) if decisions else 0.0,
            "external_proxy_available": all(
                d.external_proxy.n_metrics > 0 for d in decisions
            ) if decisions else False,
        }
        self.event_log.append("crowding_shadow_summary", summary)
        return summary