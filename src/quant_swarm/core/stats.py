"""DSR / BH-FDR / 搜索批次 N。"""
from typing import Dict, List
import numpy as np
from scipy import stats as scipy_stats

class SearchBatchCounter:
    def __init__(self):
        self._counts: Dict[str, int] = {}
    def record(self, batch_id: str, n: int = 1):
        self._counts[batch_id] = self._counts.get(batch_id, 0) + n
    def get_n(self, batch_id: str) -> int:
        return self._counts.get(batch_id, 1)

def bh_fdr(p_values: List[float], alpha: float = 0.10) -> List[bool]:
    m = len(p_values)
    if m == 0: return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    thresh_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= rank / m * alpha:
            thresh_rank = rank
    flags = [False] * m
    for rank, (idx, _) in enumerate(indexed, start=1):
        if rank <= thresh_rank:
            flags[idx] = True
    return flags

def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    if n_obs < 10 or n_trials < 1:
        return 0.0
    e_max = (1 - np.euler_gamma) * scipy_stats.norm.ppf(1 - 1/n_trials) + \
            np.euler_gamma * scipy_stats.norm.ppf(1 - 1/(n_trials * np.e))
    se = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n_obs - 1))
    return float(scipy_stats.norm.cdf((sharpe - e_max) / (se + 1e-9)))

def sharpes_to_pvalues(sharpes: List[float], n_obs: int) -> List[float]:
    return [float(1 - scipy_stats.norm.cdf(s * np.sqrt(n_obs))) for s in sharpes]