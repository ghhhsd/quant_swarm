"""
多样性约束（v2.0 冻结，v3.1 保留）。

四类：
1. 结构距离（AST 树编辑距离的简化：token Jaccard）
2. 收益相关性 ρ_max
3. 族配额（已在 ElitePool 实现，此处只做"预检"）
4. 探索配额（在 evolution.py 实现）

入池前调用 check()，返回是否可入池 + 原因。
"""
from dataclasses import dataclass
from typing import List, Optional, Dict
import re
import numpy as np

from main_tier.l3_evolution.factor import Factor


@dataclass
class DiversityVerdict:
    ok: bool
    reason: str
    constraint: str  # "structural" / "correlation" / "family_quota"


class DiversityChecker:
    """
    入池多样性检查。
    """

    def __init__(self,
                 min_structural_distance: float = 0.30,
                 max_return_correlation: float = 0.85,
                 family_quota: Optional[Dict[str, int]] = None):
        self.min_structural_distance = min_structural_distance
        self.max_return_correlation = max_return_correlation
        self.family_quota = family_quota or {
            "momentum": 12, "value": 12, "volatility": 12,
            "liquidity": 7, "other": 7,
        }

    # ---- 结构距离（token Jaccard）--------------------------------------
    @staticmethod
    def structural_distance(code_a: str, code_b: str) -> float:
        """
        简化版结构距离：token Jaccard 距离。
        真实实现应用 AST 树编辑距离，但 Jaccard 在受限 DSL 下近似度足够。
        """
        ta = set(re.findall(r"[a-zA-Z_]+|\d+", code_a))
        tb = set(re.findall(r"[a-zA-Z_]+|\d+", code_b))
        if not ta or not tb:
            return 1.0
        return 1.0 - len(ta & tb) / len(ta | tb)

    # ---- 收益相关性 -----------------------------------------------------
    @staticmethod
    def return_correlation(returns_a: np.ndarray,
                           returns_b: np.ndarray) -> float:
        """两个因子收益序列的 Pearson 相关。"""
        if len(returns_a) != len(returns_b) or len(returns_a) < 2:
            return 0.0
        a = np.asarray(returns_a, dtype=float)
        b = np.asarray(returns_b, dtype=float)
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    # ---- 全部检查 -------------------------------------------------------
    def check(self, candidate: Factor,
              pool_factors: List[Factor],
              candidate_returns: Optional[np.ndarray] = None,
              pool_returns: Optional[Dict[str, np.ndarray]] = None) -> DiversityVerdict:
        """
        入池前检查。
        candidate_returns / pool_returns 为 None 时跳过相关性检查。
        """
        # 1) 族配额预检
        family_count = sum(1 for f in pool_factors if f.family == candidate.family)
        quota = self.family_quota.get(candidate.family,
                                       self.family_quota.get("other", 10))
        if family_count >= quota:
            # 不直接拒绝——ElitePool.admit 会尝试淘汰同族最弱。
            # 这里只警告，返回 ok=True 让 ElitePool 决定。
            pass

        # 2) 结构距离：与池内每个因子比较，取最小距离
        if pool_factors:
            min_dist = 1.0
            nearest = None
            for ex in pool_factors:
                d = self.structural_distance(candidate.code, ex.code)
                if d < min_dist:
                    min_dist = d
                    nearest = ex
            if min_dist < self.min_structural_distance:
                # 绩效显著更优才破例（>1.3x）
                if (candidate.sharpe or 0) <= (nearest.sharpe or 0) * 1.3:
                    return DiversityVerdict(
                        False,
                        f"与 {nearest.fid} 结构距离 {min_dist:.2f} "
                        f"< {self.min_structural_distance}",
                        "structural",
                    )

        # 3) 收益相关性（若提供）
        if candidate_returns is not None and pool_returns:
            for fid, rets in pool_returns.items():
                corr = self.return_correlation(candidate_returns, rets)
                if abs(corr) > self.max_return_correlation:
                    return DiversityVerdict(
                        False,
                        f"与 {fid} 收益相关性 {corr:.2f} "
                        f"> {self.max_return_correlation}",
                        "correlation",
                    )

        return DiversityVerdict(True, "ok", "none")