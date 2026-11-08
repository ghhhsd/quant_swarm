"""
消融编排器（v3.1 修正②：每层以最小内核上的边际增益定去留）。

关键设计：
1. 对每个候选层，独立跑"最小内核 + 该层"vs"最小内核（不含该层）"。
2. 用开/关比较器比较组合 PnL。
3. 用晋升/降级协议决定层去留。
4. 输出结构化报告：每层是否值得进主干。

与"完整系统 + 该层"的区别：
- 完整系统里其他层的存在会污染该层的贡献度量。
- 只有"最小内核 + 单层"才能隔离该层的边际贡献。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import polars as pl

from ablation.minimal_kernel import MinimalKernel, KernelResult
from ablation.falsifiable_baseline import FalsifiableBaseline, BaselineBuilder
from ablation.comparator import AblationComparator, AblationResult
from ablation.efficacy_budget import EfficacyBudget, EfficacyBudgetRegistry
from ablation.promotion import PromotionProtocol, PromotionDecision
from ablation.layer_state import LayerStateRegistry, LayerStatus
from ablation.layer_switch import LayerSwitch, SwitchState, identity_filter
from ablation.portfolio_config import PortfolioConfig
from ablation.regime_split import classify_market_regimes
from core.event_log import EventLog, GLOBAL_EVENT_LOG


@dataclass
class LayerAblationReport:
    """单层消融报告。"""
    layer: str
    ablation_result: AblationResult
    promotion_decision: PromotionDecision
    baseline_comparison: Dict[str, Any]  # 与基准的对比

    def to_dict(self):
        return {
            "layer": self.layer,
            "ablation": self.ablation_result.to_dict(),
            "promotion": self.promotion_decision.to_dict(),
            "baseline_comparison": self.baseline_comparison,
        }


@dataclass
class CrossCuttingReport:
    """横切层完整报告。"""
    baseline: FalsifiableBaseline
    layer_reports: List[LayerAblationReport]
    final_states: Dict[str, str]

    def to_dict(self):
        return {
            "baseline": self.baseline.to_dict(),
            "layer_reports": [r.to_dict() for r in self.layer_reports],
            "final_states": self.final_states,
        }

    def summary(self) -> str:
        lines = ["=== 横切层：消融与可证伪基准 ==="]
        lines.append(f"\n基准（最小内核 {self.baseline.n_rounds} 轮）:")
        lines.append(f"  整体 Sharpe: {self.baseline.overall_sharpe_mean:.3f} "
                     f"± {self.baseline.overall_sharpe_std:.3f}")
        for regime, rb in self.baseline.by_regime.items():
            lines.append(f"  [{regime}] Sharpe: {rb.sharpe_mean:.3f} "
                         f"± {rb.sharpe_std:.3f}")
        lines.append("\n层消融结果:")
        for r in self.layer_reports:
            lines.append(f"  {r.layer}:")
            lines.append(f"    证据强度: {r.ablation_result.evidence_strength}")
            lines.append(f"    效应量: {r.ablation_result.overall_power.observed_effect:.3f}")
            lines.append(f"    决策: {r.promotion_decision.action.value}")
            lines.append(f"    理由: {r.promotion_decision.reason}")
        lines.append("\n最终状态:")
        for layer, state in self.final_states.items():
            lines.append(f"  {layer}: {state}")
        return "\n".join(lines)


class AblationOrchestrator:
    """
    消融编排器。

    流程：
    1. 构建可证伪基准（最小内核 N 轮）
    2. 对每个候选层：
       a. 构造"最小内核 + 该层"的开关
       b. 跑开/关比较器
       c. 用晋升/降级协议决策
    3. 输出完整报告
    """

    def __init__(self,
                 df: pl.DataFrame,
                 budget_registry: EfficacyBudgetRegistry,
                 state_registry: LayerStateRegistry,
                 portfolio_config: PortfolioConfig,
                 event_log: Optional[EventLog] = None):
        self.df = df
        self.budget_registry = budget_registry
        self.state_registry = state_registry
        self.portfolio_config = portfolio_config
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self.comparator = AblationComparator(
            df=df, portfolio_config=portfolio_config,
        )
        self.promotion = PromotionProtocol(state_registry, event_log)

    def run(self,
            candidates: List[Dict[str, Any]],
            layer_switches: Dict[str, LayerSwitch],
            code_extractor: Callable[[Dict], str] = lambda c: c["code"],
            regimes: Optional[np.ndarray] = None,
            baseline_rounds: int = 5,
            ) -> CrossCuttingReport:
        """
        跑完整横切层。

        candidates：全量候选（带元数据，用于开/关筛选）
        layer_switches：每层的开关定义
        code_extractor：从候选提取因子代码
        regimes：市场状态标签
        baseline_rounds：基准跑几轮
        """
        # 1) 构建可证伪基准（最小内核 N 轮）
        # 这里简化：用同一 df 跑 N 轮最小内核
        from ablation.minimal_kernel import MinimalKernel
        from core.config import GLOBAL_KERNEL
        from core.stats import SearchBatchCounter
        from core.gate import VaGGate
        from core.threshold_commit import ThresholdCommitter, ThresholdDerivation
        from core.config import GLOBAL_THRESHOLDS

        # 构造门控（临时）
        committer = ThresholdCommitter(self.event_log)
        derivation = ThresholdDerivation(
            budget_n=50, acceptable_fdr=0.10, acceptable_dsr=0.95,
            calibration_note="baseline builder",
        )
        committed = committer.commit(GLOBAL_THRESHOLDS, derivation)
        counter = SearchBatchCounter()
        gate = VaGGate(counter, committed.thresholds)

        kernel = MinimalKernel(self.df, GLOBAL_KERNEL, self.event_log)
        kernel.attach_gate(gate)
        baseline = BaselineBuilder(kernel, self.event_log).build(baseline_rounds)

        # 2) 对每个候选层跑消融
        layer_reports = []
        for layer_name, switch in layer_switches.items():
            if not self.budget_registry.is_declared(layer_name):
                continue  # 未声明功效预算的层跳过
            budget = self.budget_registry.get(layer_name).budget

            # 用开/关比较器比较
            ablation_result = self.comparator.compare(
                budget=budget,
                layer_switch=switch,
                all_candidates=candidates,
                code_extractor=code_extractor,
                regimes=regimes,
            )

            # 与基准对比
            baseline_comparison = {
                "baseline_sharpe_mean": baseline.overall_sharpe_mean,
                "baseline_sharpe_std": baseline.overall_sharpe_std,
                "layer_sharpe_mean": ablation_result.overall_power.observed_effect,
                "delta": ablation_result.overall_power.observed_effect,
            }

            # 晋升/降级决策
            decision = self.promotion.evaluate(ablation_result)

            layer_reports.append(LayerAblationReport(
                layer=layer_name,
                ablation_result=ablation_result,
                promotion_decision=decision,
                baseline_comparison=baseline_comparison,
            ))

        # 3) 收集最终状态
        final_states = {
            layer: state.value
            for layer, state in self.state_registry.get_all_states().items()
        }

        report = CrossCuttingReport(
            baseline=baseline,
            layer_reports=layer_reports,
            final_states=final_states,
        )
        self.event_log.append("cross_cutting_report", report.to_dict())
        return report