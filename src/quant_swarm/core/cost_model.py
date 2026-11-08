"""成本模型 + 情景敏感性（v3.1 修正：一阶修正，残余偏差显性化）。"""
from dataclasses import dataclass
import numpy as np

@dataclass
class CostScenario:
    name: str
    commission_bps: float = 5.0
    spread_bps: float = 2.0
    impact_coef: float = 0.1
    impact_stress: float = 1.0

SCENARIOS = {
    "base":       CostScenario("base", impact_stress=1.0),
    "stress_50":  CostScenario("stress_50", impact_stress=1.5),
    "stress_100": CostScenario("stress_100", impact_stress=2.0),
}

def apply_costs(positions: np.ndarray, turnover: np.ndarray,
                scenario: CostScenario = SCENARIOS["base"]) -> np.ndarray:
    tc = (scenario.commission_bps + scenario.spread_bps) / 1e4
    impact = scenario.impact_coef * scenario.impact_stress * np.sqrt(np.abs(turnover)) / 1e4
    return np.abs(turnover) * (tc + impact)