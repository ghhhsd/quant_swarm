"""VaG 门控（阈值事前冻结，N 按搜索批次）。"""
from dataclasses import dataclass
from typing import List
from core.config import GLOBAL_THRESHOLDS, FrozenThresholds
from core.stats import SearchBatchCounter, bh_fdr, deflated_sharpe, sharpes_to_pvalues

@dataclass
class GateDecision:
    admitted: bool
    reasons: List[str]
    net_sharpe: float
    dsr: float
    p_value: float

class VaGGate:
    def __init__(self, counter: SearchBatchCounter,
                 thresholds: FrozenThresholds = GLOBAL_THRESHOLDS):
        self.counter = counter
        self.th = thresholds
        assert self.th.frozen

    def evaluate(self, net_sharpe: float, n_obs: int, batch_id: str,
                 all_sharpes_in_batch: List[float]) -> GateDecision:
        reasons = []
        n_trials = self.counter.get_n(batch_id)
        if net_sharpe < self.th.min_oos_sharpe:
            reasons.append(f"net_sharpe {net_sharpe:.3f} < {self.th.min_oos_sharpe}")
        dsr = deflated_sharpe(net_sharpe, n_trials, n_obs)
        if dsr < self.th.dsr_threshold:
            reasons.append(f"DSR {dsr:.3f} < {self.th.dsr_threshold}")
        pvals = sharpes_to_pvalues(all_sharpes_in_batch, n_obs)
        p = sharpes_to_pvalues([net_sharpe], n_obs)[0]
        flags = bh_fdr(pvals + [p], self.th.fdr_alpha)
        if not flags[-1]:
            reasons.append(f"未通过 BH-FDR (p={p:.4f})")
        return GateDecision(len(reasons) == 0, reasons, net_sharpe, dsr, p)