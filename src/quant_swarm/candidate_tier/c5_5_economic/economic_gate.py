"""
C5.5 门面：三档成本 + 容量档位 + 衰减监控 + shadow mode。

关键设计：
1. shadow mode：产出不进主干决策链，只进消融比较。
2. 入库依据是 stress_100 下的成本后 Sharpe。
3. 容量档位作为输出，不是布尔。
4. 衰减监控独立于门控（上线后在线跑，不是入库时跑）。
5. 一阶修正的诚实标注：成本模型是对回测乐观的一阶修正，不消除。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl

from candidate_tier.c5_5_economic.cost_scenarios import (
    CostScenario, CostScenarioName, get_scenario, all_scenarios,
)
from candidate_tier.c5_5_economic.capacity import (
    CapacityEstimator, CapacityReport, CapacityTier,
    CapacityCurve, TIER_MONITOR_FREQUENCY_DAYS,
)
from candidate_tier.c5_5_economic.decay_monitor import (
    DecayMonitor, DecayCheckResult, DecayStatus,
)
from core.cost_model import apply_costs
from core.backtest import evaluate_expression, _cross_zscore
from core.event_log import EventLog, GLOBAL_EVENT_LOG


@dataclass
class ScenarioSharpe:
    """单场景 Sharpe。"""
    scenario_name: str
    gross_sharpe: float
    net_sharpe: float
    total_cost: float
    turnover_mean: float

    def to_dict(self):
        return {
            "scenario_name": self.scenario_name,
            "gross_sharpe": self.gross_sharpe,
            "net_sharpe": self.net_sharpe,
            "total_cost": self.total_cost,
            "turnover_mean": self.turnover_mean,
        }


@dataclass
class EconomicDecision:
    """
    C5.5 判定结果。

    注意：
    - scenario_sharpes：三档成本场景下的 Sharpe
    - capacity_report：容量档位
    - shadow：恒为 True（候选层）
    - residual_optimism_note：一阶修正的诚实标注
    """
    fid: str
    scenario_sharpes: Dict[str, ScenarioSharpe]
    capacity_report: CapacityReport
    passed: bool
    reasons: List[str]
    shadow: bool = True
    residual_optimism_note: str = (
        "成本模型是对回测乐观的一阶修正，不消除残余偏差。"
        "若 stress_100 下 Sharpe 转负，说明 base 的乐观不可接受。"
    )

    def to_dict(self):
        return {
            "fid": self.fid,
            "scenario_sharpes": {
                k: v.to_dict() for k, v in self.scenario_sharpes.items()
            },
            "capacity_report": self.capacity_report.to_dict(),
            "passed": self.passed,
            "reasons": self.reasons,
            "shadow": self.shadow,
            "residual_optimism_note": self.residual_optimism_note,
        }


class EconomicGate:
    """
    C5.5 实盘经济性门（shadow mode）。

    流程：
    1. 三档成本回测 → 三档 net Sharpe
    2. 容量估计 → 容量档位
    3. 判定：stress_100 下 net Sharpe >= 阈值 且 容量档位 != REJECT
    4. shadow mode：只记录，不阻断
    """

    def __init__(self, df: pl.DataFrame,
                 min_stress_sharpe: float = 0.1,
                 min_sharpe_threshold: float = 0.3,
                 event_log: Optional[EventLog] = None):
        self.df = df
        self.min_stress_sharpe = min_stress_sharpe
        self.capacity_estimator = CapacityEstimator(
            min_sharpe_threshold=min_sharpe_threshold)
        self.event_log = event_log or GLOBAL_EVENT_LOG

    def _compute_returns(self, code: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算因子组合的毛收益和换手。
        返回 (gross_returns, turnover)。
        """
        fv = np.nan_to_num(evaluate_expression(self.df, code), nan=0.0)
        z = _cross_zscore(fv)
        positions = np.where(z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0))
        positions = positions / np.maximum(
            np.abs(positions).sum(axis=1, keepdims=True), 1)

        returns = self.df.pivot(values="returns", index="date",
                                on="asset").sort("date")
        ret_arr = np.array([returns[c].to_numpy()
                            for c in returns.columns[1:]]).T
        gross = (positions[:-1] * ret_arr[1:]).sum(axis=1)
        turnover = np.abs(np.diff(positions, axis=0)).sum(axis=1)
        return gross, turnover

    def _scenario_sharpe(self, code: str,
                         scenario: CostScenario,
                         gross: np.ndarray,
                         turnover: np.ndarray) -> ScenarioSharpe:
        """单场景 Sharpe。"""
        # 成本：用 core/cost_model.py 的 apply_costs，但替换场景参数
        # 这里手动计算，因为 core/cost_model.py 的 CostScenario 与
        # c5_5 的 CostScenario 字段兼容但类型不同
        tc = (scenario.commission_bps + scenario.spread_bps) / 1e4
        impact = (scenario.impact_coef * scenario.impact_stress
                  * np.sqrt(np.abs(turnover)) / 1e4)
        cost_per_unit = tc + impact
        cost = np.abs(turnover) * cost_per_unit
        net = gross - cost

        gross_sharpe = float(np.mean(gross) / (np.std(gross) + 1e-9) * np.sqrt(252))
        net_sharpe = float(np.mean(net) / (np.std(net) + 1e-9) * np.sqrt(252))

        return ScenarioSharpe(
            scenario_name=scenario.name,
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            total_cost=float(np.sum(cost)),
            turnover_mean=float(np.mean(turnover)),
        )

    def evaluate(self, fid: str, code: str) -> EconomicDecision:
        """
        对单个因子做经济性判定。
        """
        gross, turnover = self._compute_returns(code)

        # 三档场景
        scenario_sharpes: Dict[str, ScenarioSharpe] = {}
        for name in [CostScenarioName.BASE.value,
                     CostScenarioName.STRESS_50.value,
                     CostScenarioName.STRESS_100.value]:
            sc = get_scenario(name)
            scenario_sharpes[name] = self._scenario_sharpe(
                code, sc, gross, turnover)

        # 容量估计（用 base 场景的 net Sharpe 作为 base_sharpe）
        base_net_sharpe = scenario_sharpes[CostScenarioName.BASE.value].net_sharpe
        capacity_report = self.capacity_estimator.estimate(base_net_sharpe)

        # 判定：stress_100 下 net Sharpe >= 阈值 且 容量档位 != REJECT
        reasons = []
        stress_100 = scenario_sharpes[CostScenarioName.STRESS_100.value].net_sharpe
        if stress_100 < self.min_stress_sharpe:
            reasons.append(
                f"stress_100 net Sharpe {stress_100:.3f} < {self.min_stress_sharpe}"
            )
        if capacity_report.tier == CapacityTier.TIER_REJECT:
            reasons.append(f"容量档位 REJECT: {capacity_report.reason}")

        passed = len(reasons) == 0

        decision = EconomicDecision(
            fid=fid,
            scenario_sharpes=scenario_sharpes,
            capacity_report=capacity_report,
            passed=passed,
            reasons=reasons,
            shadow=True,
        )
        self.event_log.append("economic_shadow_decision", decision.to_dict())
        return decision

    def shadow_summary(self, decisions: List[EconomicDecision]) -> Dict:
        """shadow 汇总。"""
        n_total = len(decisions)
        n_passed = sum(1 for d in decisions if d.passed)
        tier_counts: Dict[str, int] = {}
        for d in decisions:
            t = d.capacity_report.tier.value
            tier_counts[t] = tier_counts.get(t, 0) + 1

        summary = {
            "n_total": n_total,
            "n_passed": n_passed,
            "pass_rate": n_passed / n_total if n_total else 0.0,
            "tier_breakdown": tier_counts,
            "mean_stress_100_sharpe": float(np.mean([
                d.scenario_sharpes[CostScenarioName.STRESS_100.value].net_sharpe
                for d in decisions
            ])) if decisions else 0.0,
        }
        self.event_log.append("economic_shadow_summary", summary)
        return summary