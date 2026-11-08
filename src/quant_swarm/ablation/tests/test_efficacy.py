# 功效测试模块
"""
功效预算测试。

验证 v3.1 的九条纪律：
1. 预算事前声明，写入不可变日志
2. 二次声明被拒（防事后调参）
3. 效应量分三档（LARGE/MEDIUM/SMALL）
4. required_batches 按 mde/power/alpha 推导
5. 实际功效计算
6. 事前 vs 事后对照
7. 分市场状态切分
8. 弱证据不作晋升充分条件
9. 组合 PnL 为最终指标
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from ablation.efficacy_budget import (
    EfficacyBudget, EfficacyBudgetRegistry, BudgetViolation,
    EffectSize, MarketRegime, classify_effect_size,
)
from ablation.power_analysis import PowerAnalyzer
from ablation.regime_split import (
    classify_market_regimes, split_by_regime, split_batch_by_regime,
)
from ablation.comparator import AblationComparator
from core.event_log import EventLog
import tempfile
from pathlib import Path


def make_registry(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    return EfficacyBudgetRegistry(log), log


def make_budget(layer="test", mde=0.15):
    return EfficacyBudget(
        layer=layer, mde=mde, power=0.80, alpha=0.05,
        n_obs_per_batch=252, n_batches_min=8,
    )


def test_budget_declare_once(tmp_path):
    """预算事前声明，二次声明被拒。"""
    registry, log = make_registry(tmp_path)
    budget = make_budget()
    committed = registry.declare(budget)
    assert committed.commit_hash
    assert registry.is_declared("test")
    # 二次声明应被拒
    try:
        registry.declare(budget)
        assert False, "应抛 BudgetViolation"
    except BudgetViolation:
        print("[OK] 预算二次声明被拒")


def test_effect_size_classification():
    """效应量分三档。"""
    assert classify_effect_size(0.20) == EffectSize.LARGE
    assert classify_effect_size(0.15) == EffectSize.LARGE
    assert classify_effect_size(0.10) == EffectSize.MEDIUM
    assert classify_effect_size(0.05) == EffectSize.MEDIUM
    assert classify_effect_size(0.03) == EffectSize.SMALL
    print("[OK] 效应量三档分类")


def test_required_batches():
    """required_batches 按 mde/power/alpha 推导。"""
    large = make_budget(mde=0.15)
    medium = make_budget(mde=0.08)
    small = make_budget(mde=0.03)
    # 小效应需要更多批次
    assert small.required_batches() > medium.required_batches() > large.required_batches()
    print(f"[OK] required_batches: large={large.required_batches()}, "
          f"medium={medium.required_batches()}, small={small.required_batches()}")


def test_power_analyzer_strong():
    """大效应 + 足够样本 → 强证据。"""
    analyzer = PowerAnalyzer()
    budget = make_budget(mde=0.15)
    rng = np.random.default_rng(42)
    # on 显著高于 off
    on_sharpes = list(rng.normal(0.5, 0.1, 30))
    off_sharpes = list(rng.normal(0.3, 0.1, 30))
    result = analyzer.analyze(budget, on_sharpes, off_sharpes)
    assert result.observed_effect > 0.15
    assert result.evidence_strength == "strong", \
        f"应强证据，实际 {result.evidence_strength}: {result.note}"
    print(f"[OK] 强证据: effect={result.observed_effect:.3f}, "
          f"power={result.observed_power:.2f}")


def test_power_analyzer_weak():
    """小效应 + 样本不足 → 弱证据。"""
    analyzer = PowerAnalyzer()
    budget = make_budget(mde=0.15)
    rng = np.random.default_rng(42)
    # on 与 off 差异小
    on_sharpes = list(rng.normal(0.31, 0.1, 5))
    off_sharpes = list(rng.normal(0.30, 0.1, 5))
    result = analyzer.analyze(budget, on_sharpes, off_sharpes)
    assert result.evidence_strength == "weak"
    print(f"[OK] 弱证据: effect={result.observed_effect:.3f}, "
          f"power={result.observed_power:.2f}, note={result.note}")


def test_regime_classification():
    """市场状态分类。"""
    rng = np.random.default_rng(42)
    # 构造：前 100 天牛市，中间 100 天熊市，后 100 天震荡
    bull = rng.normal(0.002, 0.005, 100)
    bear = rng.normal(-0.002, 0.015, 100)
    sideways = rng.normal(0.0, 0.01, 100)
    market_returns = np.concatenate([bull, bear,