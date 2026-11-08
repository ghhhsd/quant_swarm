from ablation.minimal_kernel import MinimalKernel, KernelResult
from ablation.falsifiable_baseline import (
    FalsifiableBaseline, RegimeBaseline, BaselineBuilder,
)
from ablation.orchestrator import (
    AblationOrchestrator, CrossCuttingReport, LayerAblationReport,
)

# 已有模块的导出
from ablation.efficacy_budget import (
    EfficacyBudget, CommittedBudget, EfficacyBudgetRegistry,
    BudgetViolation, EffectSize, MarketRegime, classify_effect_size,
)
from ablation.power_analysis import PowerAnalyzer, PowerResult
from ablation.regime_split import (
    classify_market_regimes, split_by_regime, split_batch_by_regime,
)
from ablation.portfolio_config import (
    PortfolioConfig, PortfolioBuilder, PortfolioPnLResult, WeightingScheme,
)
from ablation.oos_batch import OOSBatch, generate_oos_batches
from ablation.layer_switch import LayerSwitch, SwitchState, identity_filter
from ablation.comparator import (
    AblationComparator, AblationResult, BatchPnL,
)
from ablation.layer_state import (
    LayerStatus, TransitionAction, StateTransition, LayerStateRegistry,
)
from ablation.promotion import PromotionDecision, PromotionProtocol

__all__ = [
    # 最小内核 + 基准
    "MinimalKernel", "KernelResult",
    "FalsifiableBaseline", "RegimeBaseline", "BaselineBuilder",
    # 编排器
    "AblationOrchestrator", "CrossCuttingReport", "LayerAblationReport",
    # 功效预算
    "EfficacyBudget", "CommittedBudget", "EfficacyBudgetRegistry",
    "BudgetViolation", "EffectSize", "MarketRegime", "classify_effect_size",
    "PowerAnalyzer", "PowerResult",
    "classify_market_regimes", "split_by_regime", "split_batch_by_regime",
    # 组合层
    "PortfolioConfig", "PortfolioBuilder", "PortfolioPnLResult",
    "WeightingScheme",
    "OOSBatch", "generate_oos_batches",
    "LayerSwitch", "SwitchState", "identity_filter",
    # 比较器
    "AblationComparator", "AblationResult", "BatchPnL",
    # 晋升/降级
    "LayerStatus", "TransitionAction", "StateTransition",
    "LayerStateRegistry",
    "PromotionDecision", "PromotionProtocol",
]