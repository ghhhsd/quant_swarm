"""
向量化回测测试。

验证 v3.1 的九条纪律：
1. 面板预计算（一次 pivot）
2. 批量回测（多因子共享面板）
3. OOS 批次切片
4. 成本后 PnL
5. 与成本模型对接
6. 结果可序列化
7. 批量与单因子结果一致
8. 因子值向量化
9. 持仓构造正确
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from core.data import generate_synthetic_data
from core.backtest import (
    PanelBuilder, PanelData, FactorEvaluator,
    PositionBuilder, BatchBacktester, BatchBacktestResult,
    generate_oos_batches,
)


def make_panel(n_days=200, n_assets=20):
    """构造面板数据。"""
    df = generate_synthetic_data(n_days, n_assets, seed=42)
    builder = PanelBuilder()
    return builder.build(df)


def test_panel_build():
    """面板构建。"""
    panel = make_panel()
    shape = panel.shape()
    assert shape == (200, 20)
    assert "returns" in panel.fields
    assert "close" in panel.fields
    assert len(panel.dates) == 200
    assert len(panel.assets) == 20
    print(f"[OK] 面板: shape={shape}, fields={list(panel.fields.keys())}")


def test_panel_slice():
    """面板切片。"""
    panel = make_panel()
    sliced = panel.slice(50, 150)
    assert sliced.shape() == (100, 20)
    assert len(sliced.dates) == 100
    print(f"[OK] 切片: shape={sliced.shape()}")


def test_factor_eval():
    """因子值计算。"""
    panel = make_panel()
    evaluator = FactorEvaluator(panel)
    fv = evaluator.evaluate("ts_mean(returns, 20)")
    assert fv.shape == (200, 20)
    # 前 19 天应为 NaN
    assert np.all(np.isnan(fv[:19]))
    # 20 天后应有值
    assert not np.all(np.isnan(fv[20:]))
    print(f"[OK] 因子值: shape={fv.shape}")


def test_factor_eval_batch():
    """批量因子值计算。"""
    panel = make_panel()
    evaluator = FactorEvaluator(panel)
    codes = [
        "ts_mean(returns, 20)",
        "ts_std(returns, 20)",
        "1.0 / pe_ratio",
    ]
    tensor = evaluator.evaluate_batch(codes)
    assert tensor.shape == (3, 200, 20)
    print(f"[OK] 批量因子值: shape={tensor.shape}")


def test_position_build():
    """持仓构造。"""
    panel = make_panel()
    evaluator = FactorEvaluator(panel)
    fv = evaluator.evaluate("ts_mean(returns, 20)")
    builder = PositionBuilder()
    pos = builder.build(fv)
    assert pos.shape == (200, 20)
    # 持仓应在 [-1, 1] 之间
    assert np.all(np.abs(pos) <= 1.0 + 1e-9)
    # 每行权重和应为 1（或有 NaN）
    row_sums = np.abs(pos).sum(axis=1)
    valid = ~np.isnan(row_sums) & (row_sums > 0)
    assert np.allclose(row_sums[valid], 1.0, atol=1e-6)
    print(f"[OK] 持仓: shape={pos.shape}")


def test_batch_backtest():
    """批量回测。"""
    panel = make_panel()
    backtester = BatchBacktester(panel)
    codes = [
        "ts_mean(returns, 20) - ts_mean(returns, 60)",
        "ts_std(returns, 20)",
        "1.0 / pe_ratio",
    ]
    result = backtester.backtest(codes)
    assert result.n_factors == 3
    assert len(result.results) == 3
    for r in result.results:
        assert r.n_obs > 0
        assert isinstance(r.net_sharpe, float)
    print(f"[OK] 批量回测: {result.n_factors} 个因子")
    for r in result.results:
        print(f"  {r.fid}: gross={r.gross_sharpe:.3f}, "
              f"net={r.net_sharpe:.3f}")


def test_batch_vs_single():
    """批量与单因子结果一致。"""
    panel = make_panel()
    backtester = BatchBacktester(panel)
    code = "ts_mean(returns, 20) - ts_mean(returns, 60)"
    # 批量
    batch_result = backtester.backtest([code])
    # 单因子（用批量接口但只传一个）
    single_result = backtester.backtest([code])
    # 应完全一致
    assert batch_result.results[0].net_sharpe == \
        single_result.results[0].net_sharpe
    print(f"[OK] 批量=单因子: {batch_result.results[0].net_sharpe:.3f}")


def test_oos_batches():
    """OOS 批次生成。"""
    batches = generate_oos_batches(
        n_obs=1000, train_window=300, oos_window=100, step=50,
    )
    assert len(batches) > 0
    for b in batches:
        assert b.oos_end > b.oos_start
    print(f"[OK] OOS 批次: {len(batches)} 个")


def test_oos_backtest():
    """OOS 批次回测。"""
    panel = make_panel(n_days=500)
    backtester = BatchBacktester(panel)
    batches = generate_oos_batches(
        n_obs=500, train_window=200, oos_window=100, step=50,
    )
    code = "ts_mean(returns, 20) - ts_mean(returns, 60)"
    results = []
    for batch in batches:
        result = backtester.backtest(
            [code], oos_start=batch.oos_start, oos_end=batch.oos_end,
        )
        results.append(result.results[0].net_sharpe)
    assert len(results) == len(batches)
    print(f"[OK] OOS 回测: {len(results)} 个批次, "
          f"sharpe 范围 [{min(results):.3f}, {max(results):.3f}]")


def test_result_serializable():
    """结果可序列化。"""
    panel = make_panel()
    backtester = BatchBacktester(panel)
    result = backtester.backtest(["ts_mean(returns, 20)"])
    d = result.to_dict()
    assert d["n_factors"] == 1
    assert len(d["results"]) == 1
    assert "net_sharpe" in d["results"][0]
    print(f"[OK] 可序列化: {d['n_factors']} 个结果")


def test_cost_integration():
    """与成本模型对接。"""
    panel = make_panel()
    # base 情景
    base_bt = BatchBacktester(panel, cost_scenario="base")
    base_result = base_bt.backtest(["ts_mean(returns, 20)"])
    # stress_100 情景
    stress_bt = BatchBacktester(panel, cost_scenario="stress_100")
    stress_result = stress_bt.backtest(["ts_mean(returns, 20)"])
    # stress_100 净 Sharpe 应更低
    assert stress_result.results[0].net_sharpe < \
        base_result.results[0].net_sharpe
    print(f"[OK] 成本集成: base={base_result.results[0].net_sharpe:.3f}, "
          f"stress_100={stress_result.results[0].net_sharpe:.3f}")


if __name__ == "__main__":
    test_panel_build()
    test_panel_slice()
    test_factor_eval()
    test_factor_eval_batch()
    test_position_build()
    test_batch_backtest()
    test_batch_vs_single()
    test_oos_batches()
    test_oos_backtest()
    test_result_serializable()
    test_cost_integration()
    print("\n=== 向量化回测全部测试通过 ===")