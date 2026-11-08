"""
精英池：当前"够格"的因子集合。

关键设计（v3.1）：
1. 精英池是演化树的"投影视图"——池中的每个 fid 都必须在树中。
2. 池可增可减（淘汰），但淘汰不删除树节点。
3. 按 Pareto 前沿排序，front 上的因子是"非支配"的。
4. 淘汰规则：
   - 池满时，淘汰 Pareto 支配链底部的因子（被支配最多的）
   - 同族配额满时，淘汰该族内最弱的
5. 淘汰进事件流，可审计。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from main_tier.l3_evolution.factor import Factor
from main_tier.l3_evolution.tree import EvolutionTree
from core.event_log import EventLog, GLOBAL_EVENT_LOG


# 族配额（v2.0 冻结，v3.1 保留）
DEFAULT_FAMILY_QUOTA: Dict[str, int] = {
    "momentum": 12, "value": 12, "volatility": 12,
    "liquidity": 7, "other": 7,
}


class ElitePool:
    """
    精英池。

    注意：入池前的多样性/拥挤度约束由 DiversityChecker / CrowdingChecker
    在调用方（engine.py）完成；本类只负责"池内管理"和"Pareto 排序"。
    """

    def __init__(self, tree: EvolutionTree,
                 max_size: int = 50,
                 family_quota: Optional[Dict[str, int]] = None,
                 event_log: Optional[EventLog] = None):
        self.tree = tree
        self.max_size = max_size
        self.family_quota = family_quota or dict(DEFAULT_FAMILY_QUOTA)
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self.factors: Dict[str, Factor] = {}
        self.family_counts: Dict[str, int] = {}

    # ---- 入池 -----------------------------------------------------------
    def admit(self, factor: Factor) -> bool:
        """
        入池。返回是否成功。
        约束：因子必须在树中 + 已评估 + 池未满 + 族配额未超。
        """
        if factor.fid not in self.tree.nodes:
            return False
        if not factor.is_evaluated():
            return False
        if factor.fid in self.factors:
            # 已入池：更新指标（用于重评估）
            self.factors[factor.fid] = factor
            return True

        family = factor.family
        quota = self.family_quota.get(family, self.family_quota.get("other", 10))
        if self.family_counts.get(family, 0) >= quota:
            # 配额满：尝试淘汰同族最弱
            if not self._evict_weakest_in_family(family, factor):
                return False

        if len(self.factors) >= self.max_size:
            # 池满：尝试淘汰全局最弱
            if not self._evict_globally_weakest(factor):
                return False

        self.factors[factor.fid] = factor
        self.family_counts[family] = self.family_counts.get(family, 0) + 1
        self.event_log.append("elite_admitted", {
            "fid": factor.fid, "family": family,
            "sharpe": factor.sharpe, "marginal_gain": factor.marginal_gain,
            "pool_size": len(self.factors),
        })
        return True

    def _evict_weakest_in_family(self, family: str,
                                 newcomer: Factor) -> bool:
        """淘汰同族中最弱的，若新来者更强。"""
        same_family = [f for f in self.factors.values() if f.family == family]
        if not same_family:
            return False
        weakest = min(same_family, key=lambda f: (f.sharpe or 0) + (f.marginal_gain or 0))
        if (newcomer.sharpe or 0) + (newcomer.marginal_gain or 0) <= \
           (weakest.sharpe or 0) + (weakest.marginal_gain or 0):
            return False
        self._remove(weakest.fid, "family_quota_evict")
        return True

    def _evict_globally_weakest(self, newcomer: Factor) -> bool:
        """淘汰全局最弱的，若新来者更强。"""
        if not self.factors:
            return True
        weakest = min(self.factors.values(),
                      key=lambda f: (f.sharpe or 0) + (f.marginal_gain or 0))
        if (newcomer.sharpe or 0) + (newcomer.marginal_gain or 0) <= \
           (weakest.sharpe or 0) + (weakest.marginal_gain or 0):
            return False
        self._remove(weakest.fid, "global_evict")
        return True

    def _remove(self, fid: str, reason: str) -> None:
        f = self.factors.pop(fid, None)
        if f:
            self.family_counts[f.family] = max(
                0, self.family_counts.get(f.family, 0) - 1)
            self.event_log.append("elite_evicted", {
                "fid": fid, "family": f.family, "reason": reason,
            })

    # ---- Pareto 排序 ----------------------------------------------------
    def pareto_front(self) -> List[Factor]:
        """
        三目标非支配排序：
        - 最大化 sharpe
        - 最大化 marginal_gain
        - 最小化 ast_depth（硬约束）
        """
        evaluated = [f for f in self.factors.values() if f.is_evaluated()]
        if not evaluated:
            return []
        front = []
        for a in evaluated:
            dominated = False
            for b in evaluated:
                if a.fid == b.fid:
                    continue
                if (b.sharpe >= a.sharpe
                    and b.marginal_gain >= a.marginal_gain
                    and b.ast_depth <= a.ast_depth
                    and (b.sharpe > a.sharpe
                         or b.marginal_gain > a.marginal_gain
                         or b.ast_depth < a.ast_depth)):
                    dominated = True
                    break
            if not dominated:
                front.append(a)
        return front

    # ---- 查询 -----------------------------------------------------------
    def size(self) -> int:
        return len(self.factors)

    def get(self, fid: str) -> Optional[Factor]:
        return self.factors.get(fid)

    def all_factors(self) -> List[Factor]:
        return list(self.factors.values())

    def family_breakdown(self) -> Dict[str, int]:
        return dict(self.family_counts)