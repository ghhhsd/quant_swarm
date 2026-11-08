"""
L3 门面：组合演化树 + 精英池 + 多样性 + 拥挤度 + 演化操作。

对外接口：
- propose(n) -> List[Factor]：生成 n 个候选
- admit(factor, returns) -> bool：入池（带多样性 + 拥挤度检查）
- pareto_front() -> List[Factor]
- update_exploration(ratio)：由 L0 难度信号驱动
- verify_integrity() -> List[str]：演化树完整性
"""
from typing import List, Optional, Dict
import numpy as np
import polars as pl

from main_tier.l3_evolution.factor import Factor
from main_tier.l3_evolution.tree import EvolutionTree
from main_tier.l3_evolution.elite_pool import ElitePool, DEFAULT_FAMILY_QUOTA
from main_tier.l3_evolution.diversity import DiversityChecker, DiversityVerdict
from main_tier.l3_evolution.crowding import CrowdingChecker, CrowdingReport
from main_tier.l3_evolution.evolution import EvolutionOperator
from core.event_log import EventLog, GLOBAL_EVENT_LOG


class EvolutionEngine:
    """
    L3 门面。
    """

    def __init__(self,
                 max_pool_size: int = 50,
                 min_structural_distance: float = 0.30,
                 max_return_correlation: float = 0.85,
                 max_internal_overlap: float = 0.70,
                 exploration_ratio: float = 0.20,
                 family_quota: Optional[Dict[str, int]] = None,
                 event_log: Optional[EventLog] = None,
                 seed: int = 42):
        log = event_log or GLOBAL_EVENT_LOG
        self.tree = EvolutionTree(log)
        self.pool = ElitePool(
            tree=self.tree, max_size=max_pool_size,
            family_quota=family_quota or dict(DEFAULT_FAMILY_QUOTA),
            event_log=log,
        )
        self.diversity = DiversityChecker(
            min_structural_distance=min_structural_distance,
            max_return_correlation=max_return_correlation,
            family_quota=family_quota or dict(DEFAULT_FAMILY_QUOTA),
        )
        self.crowding = CrowdingChecker(
            max_internal_overlap=max_internal_overlap,
        )
        import random
        self.evolution = EvolutionOperator(
            tree=self.tree, pool=self.pool,
            exploration_ratio=exploration_ratio,
            rng=random.Random(seed),
        )
        self.event_log = log

    # ---- 生成 -----------------------------------------------------------
    def propose(self, n: int) -> List[Factor]:
        return self.evolution.propose(n)

    # ---- 入池 -----------------------------------------------------------
    def admit(self, factor: Factor,
              df: Optional[pl.DataFrame] = None,
              factor_returns: Optional[np.ndarray] = None,
              pool_returns: Optional[Dict[str, np.ndarray]] = None,
              check_crowding: bool = True) -> bool:
        """
        入池。顺序：
        1. 多样性检查（结构距离 + 收益相关性）
        2. 拥挤度检查（内部持仓重叠）
        3. ElitePool.admit（族配额 + Pareto）
        """
        if not factor.is_evaluated():
            self.event_log.append("admit_rejected", {
                "fid": factor.fid, "reason": "not_evaluated",
            })
            return False

        # 1) 多样性
        dv = self.diversity.check(
            candidate=factor,
            pool_factors=self.pool.all_factors(),
            candidate_returns=factor_returns,
            pool_returns=pool_returns,
        )
        if not dv.ok:
            self.event_log.append("admit_rejected", {
                "fid": factor.fid, "reason": f"diversity:{dv.constraint}",
                "detail": dv.reason,
            })
            return False

        # 2) 拥挤度（只对内部重叠做约束）
        if check_crowding and df is not None:
            current_factors = self.pool.all_factors()
            test_set = current_factors + [factor]
            report = self.crowding.evaluate(df, test_set)
            if report.flagged:
                self.event_log.append("admit_rejected", {
                    "fid": factor.fid, "reason": "crowding",
                    "detail": report.reason,
                    "external_proxy": report.external_proxy,
                })
                return False

        # 3) ElitePool.admit
        return self.pool.admit(factor)

    # ---- 查询 -----------------------------------------------------------
    def pareto_front(self) -> List[Factor]:
        return self.pool.pareto_front()

    def update_exploration(self, ratio: float) -> None:
        """
        由 L0 难度信号驱动更新探索配额（不碰验收侧）。
        """
        self.evolution.exploration_ratio = max(0.05, min(0.50, ratio))
        self.event_log.append("exploration_updated", {
            "ratio": self.evolution.exploration_ratio,
        })

    def verify_integrity(self) -> List[str]:
        return self.tree.verify_integrity()

    def stats(self) -> Dict:
        return {
            "tree_size": self.tree.size(),
            "pool_size": self.pool.size(),
            "family_breakdown": self.pool.family_breakdown(),
            "front_size": len(self.pareto_front()),
            "exploration_ratio": self.evolution.exploration_ratio,
        }