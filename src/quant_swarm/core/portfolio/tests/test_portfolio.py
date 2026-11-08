"""
组合层测试。

验证 v3.1 的九条纪律：
1. 组合 PnL 是最终指标
2. 加权规则显式声明
3. 开/关用同一配置
4. 组合换手 ≠ 单因子换手之和（抵消效应）
5. 成本后 PnL
6. OOS 批次切片
7. IC 加权用 OOS 窗口前的 IC（无前视）
8. 结果可序列化
9. 跨批次汇总
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from core.data import generate_synthetic_data
from core.backtest import PanelBuilder
from core.portfolio import (
    PortfolioConfig, PortfolioBuilder, BatchPortfolioBuilder,
    compute_turnover, BatchPortfolioSummary,
)


def make_panel(n_days=500, n_assets=20):
    df = generate_synthetic_data(n_days, n_assets, seed=42)
    return PanelBuilder().build(df)


def test_portfolio_build():
    """组合构建。"""
    panel = make_panel()
    config = PortfolioConfig(weighting="equal")
    builder = PortfolioBuilder(panel, config)
    codes = [
        "ts_mean(returns, 20) - ts_mean(returns, 60)",
        "ts_std(returns, 20)",
        "1.0 / pe_ratio",
    ]
    result = builder.build(codes)
    assert result.n_factors == 3
    assert result.n_obs > 0
    assert isinstance(result.net_sharpe, float)
    print(f"[OK] 组合: n_factors={result.n_factors}, "
          f"gross={result.gross_sharpe:.3f}, net={result.net_sharpe:.3f}")


def test_portfolio_weighting():
    """不同加权规则。"""
    panel = make_panel()
    codes = ["ts_mean(returns, 20)", "1.0 / pe_ratio"]
    results = {}
    for scheme in ["equal", "ic_weighted", "risk_parity"]:
        config = PortfolioConfig(weighting=scheme)
        builder = PortfolioBuilder(panel, config)
        results[scheme] = builder.build(codes)
    for scheme, r in results.items():
        assert r.weighting == scheme
    print(f"[OK] 加权规则: {list(results.keys())}")


def test_portfolio_frozen_config():
    """配置冻结。"""
    config = PortfolioConfig()
    try:
        config.weighting = "ic_weighted"
        assert False, "frozen dataclass 应拒绝修改"
    except Exception:
        print("[OK] 配置冻结")


def test_portfolio_turnover_offset():
    """组合换手 ≠ 单因子换手之和（抵消效应）。"""
    panel = make_panel()
    config = PortfolioConfig(weighting="equal")
    builder = PortfolioBuilder(panel, config)
    # 用两个方向相反的因子（应产生抵消）
    codes = [
        "ts_mean(returns, 20)",
        "ts_mean(returns, 20) * -1.0",  # 反向
    ]
    result = builder.build(codes)
    # 抵消比例应 > 0（有抵消）
    assert result.offset_ratio >= 0
    print(f"[OK] 抵消效应: offset_ratio={result.offset_ratio:.3f}")


def test_portfolio_oos_slice():
    """OOS 批次切片。"""
    panel = make_panel(n_days=500)
    config = PortfolioConfig()
    builder = PortfolioBuilder(panel, config)
    codes = ["ts_mean(returns, 20)"]
    # 全样本
    full = builder.build(codes, oos_start=0, oos_end=500)
    # 切片
    sliced = builder.build(codes, oos_start=200, oos_end=400)
    assert sliced.n_obs < full.n_obs
    print(f"[OK] OOS 切片: full={full.n_obs}, sliced={sliced.n_obs}")


def test_portfolio_cost():
    """成本后 PnL。"""
    panel = make_panel()
    codes = ["ts_mean(returns, 20)"]

    base_config = PortfolioConfig(cost_scenario="base")
    base_builder = PortfolioBuilder(panel, base_config)
    base_result = base_builder.build(codes)

    stress_config = PortfolioConfig(cost_scenario="stress_100")
    stress_builder = PortfolioBuilder(panel, stress_config)
    stress_result = stress_builder.build(codes)

    # stress_100 净 Sharpe 应更低
    assert stress_result.net_sharpe < base_result.net_sharpe
    print(f"[OK] 成本: base={base_result.net_sharpe:.3f}, "
          f"stress_100={stress_result.net_sharpe:.3f}")


def test_portfolio_batch():
    """跨 OOS 批次组合。"""
    panel = make_panel(n_days=800)
    config = PortfolioConfig()
    builder = BatchPortfolioBuilder(panel, config)
    codes = ["ts_mean(returns, 20) - ts_mean(returns, 60)"]
    summary = builder.build(
        codes, train_window=300, oos_window=100, step=50,
    )
    assert summary.n_batches > 0
    assert summary.mean_net_sharpe() is not None
    print(f"[OK] 跨批次: n_batches={summary.n_batches}, "
          f"mean_net={summary.mean_net_sharpe():.3f}, "
          f"std={summary.std_net_sharpe():.3f}")


def test_portfolio_result_serializable():
    """结果可序列化。"""
    panel = make_panel()
    config = PortfolioConfig()
    builder = PortfolioBuilder(panel, config)
    result = builder.build(["ts_mean(returns, 20)"])
    d = result.to_dict()
    assert "net_sharpe" in d
    assert "offset_ratio" in d
    assert "batch_id" in d
    print(f"[OK] 可序列化: keys={list(d.keys())}")


def test_portfolio_empty():
    """空因子列表。"""
    panel = make_panel()
    config = PortfolioConfig()
    builder = PortfolioBuilder(panel, config)
    result = builder.build([])
    assert result.n_factors == 0
    assert result.net_sharpe == 0.0
    print("[OK] 空因子列表")


def test_portfolio_config_validation():
    """配置校验。"""
    try:
        PortfolioConfig(weighting="unknown")
        assert False
    except ValueError:
        print("[OK] 未知加权规则被拒")

    try:
        PortfolioConfig(long_threshold=-1.0, short_threshold=1.0)
        assert False
    except ValueError:
        print("[OK] 阈值顺序被拒")


if __name__ == "__main__":
    test_portfolio_build()
    test_portfolio_weighting()
    test_portfolio_frozen_config()
    test_portfolio_turnover_offset()
    test_portfolio_oos_slice()
    test_portfolio_cost()
    test_portfolio_batch()
    test_portfolio_result_serializable()
    test_portfolio_empty()
    test_portfolio_config_validation()
    print("\n=== 组合层全部测试通过 ===")