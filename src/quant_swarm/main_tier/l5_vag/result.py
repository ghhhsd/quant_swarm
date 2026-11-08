"""
门控决策数据结构（v3.1 修正③：失败分流 + 回灌目标）。

关键设计：
1. failure_stage 标记哪一道检查失败：
   - "statistical"：统计检查失败 → 回灌负样本库
   - "marginal"：边际增益失败 → 回灌负样本库
   - "narrative"：叙事失效 → 回灌机制模板库
2. 成功时带完整统计信息（DSR、FDR、边际增益、一致性）。
3. 可序列化（供事件溯源）。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from main_tier.l1_favor.failure import FailureType


@dataclass
class StatisticalResult:
    """统计检查结果（L5 内部）。"""
    mean_oos_sharpe: float
    sharpe_std: float
    sharpe_sign_stable: bool
    dsr: float
    p_value: float
    fdr_passed: bool
    n_trials: int
    reasons: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "mean_oos_sharpe": self.mean_oos_sharpe,
            "sharpe_std": self.sharpe_std,
            "sharpe_sign_stable": self.sharpe_sign_stable,
            "dsr": self.dsr,
            "p_value": self.p_value,
            "fdr_passed": self.fdr_passed,
            "n_trials": self.n_trials,
            "reasons": self.reasons,
        }


@dataclass
class MarginalResult:
    """边际增益结果。"""
    base_sharpe: float
    new_sharpe: float
    marginal_gain: float
    threshold: float
    passed: bool
    reason: str

    def to_dict(self):
        return {
            "base_sharpe": self.base_sharpe,
            "new_sharpe": self.new_sharpe,
            "marginal_gain": self.marginal_gain,
            "threshold": self.threshold,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass
class ConsistencyResult:
    """叙事一致性结果。"""
    consistent: bool
    direction_match: bool
    decay_match: bool
    condition_match: bool
    reasons: List[str]
    factor_returns_corr: float

    def to_dict(self):
        return {
            "consistent": self.consistent,
            "direction_match": self.direction_match,
            "decay_match": self.decay_match,
            "condition_match": self.condition_match,
            "reasons": self.reasons,
            "factor_returns_corr": self.factor_returns_corr,
        }


@dataclass
class GateDecision:
    """
    门控决策。

    - admitted：是否准入
    - failure_stage：哪道检查失败（None 表示准入）
    - failure_type：失败类型（GATE_REJECTED）
    - downstream_action：下游回灌目标
      - "negative_sample"：回灌负样本库（统计/边际失败）
      - "mechanism_template"：回灌机制模板库（叙事失效）
    """
    admitted: bool
    fid: str
    failure_stage: Optional[str] = None
    failure_type: Optional[FailureType] = None
    downstream_action: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    statistical: Optional[Dict[str, Any]] = None
    marginal: Optional[Dict[str, Any]] = None
    consistency: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {
            "admitted": self.admitted,
            "fid": self.fid,
            "failure_stage": self.failure_stage,
            "failure_type": (self.failure_type.value
                             if self.failure_type else None),
            "downstream_action": self.downstream_action,
            "reasons": self.reasons,
            "statistical": self.statistical,
            "marginal": self.marginal,
            "consistency": self.consistency,
        }