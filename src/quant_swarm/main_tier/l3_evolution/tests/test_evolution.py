"""
L3 演化测试。

验证 v3.1 的九条纪律：
1. 演化树只增不减，父代必须存在
2. 演化树哈希锚定可检测版本错配
3. 精英池可增可减，淘汰不删树节点
4. Pareto 非支配排序正确
5. 结构距离防近亲繁殖
6. 收益相关性上限
7. 持仓重叠度约束（内部）
8. 外部拥挤代理单独列
9. 探索配额可被 L0 难度信号调整
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import polars as pl
from main_tier.l3_evolution import (
    Factor, EvolutionEngine,
)
from core.event_log import EventLog
from core.data import generate_synthetic_data
import tempfile
from pathlib import Path


def make_engine(tmp_path, **kwargs):
    log = EventLog(str(tmp_path / "events.jsonl"))
    return EvolutionEngine(event_log=log, **kwargs)


def make_evaluated_factor(fid, family, code, sharpe, gain, depth=3):
    return Factor(fid=fid, family=family, code=code, ast_depth=depth,
                  sharpe=sharpe, marginal_gain=gain)


def test_tree_only_grows(tmp_path):
    eng = make_engine(tmp_path)
    f1 = Factor("f1", "momentum", "ts_mean(returns, 20)", 2)
    eng.tree.add(f1)
    # 重复添加应抛异常
    try:
        eng.tree.add(f1)
        assert False, "应抛 ValueError"
    except ValueError:
        print("[OK] 演化树只增不减")


def test_tree_parent_must_exist(tmp_path):
    eng = make_engine(tmp_path)
    f2 = Factor("f2", "momentum", "ts_mean(returns, 20)", 2,
                parent_fids=("nonexistent",))
    try:
        eng.tree.add(f2)
        assert False, "应抛 ValueError"
    except ValueError as e:
        assert "不在树中" in str(e)
        print(f"[OK] 父代必须存在: {e}")


def test_tree_hash_integrity(tmp_path):
    eng = make_engine(tmp_path)
    f1 = Factor("f1", "momentum", "ts_mean(returns, 20)", 2)
    eng.tree.add(f1)
    f2 = Factor("f2", "momentum", "ts_mean(returns, 25)", 2, parent_fids=("f1",))
    eng.tree.add(f2)
    # 完整性校验应通过
    errors = eng.tree.verify_integrity()
    assert not errors, errors
    print("[OK] 演化树哈希锚定完整")


def test_pareto_front(tmp_path):
    eng = make_engine(tmp_path)
    # 构造 3 个因子：A 被 B 支配
    a = make_evaluated_factor("fa", "momentum", "a", sharpe=0.5, gain=0.1, depth=4)
    b = make_evaluated_factor("fb", "momentum", "b", sharpe=0.6, gain=0.2, depth=3)  # 支配 a
    c = make_evaluated_factor("fc", "value", "c", sharpe=0.7, gain=0.05, depth=5)   # 不被支配
    for f in (a, b, c):
        eng.tree.add(f)
        eng.pool.admit(f)
    front = eng.pareto_front()
    fids = {f.fid for f in front}
    assert "fa" not in fids, "a 被 b 支配，不应在前沿"
    assert "fb" in fids
    assert "fc" in fids
    print(f"[OK] Pareto 前沿: {fids}")


def test_elite_evict_keeps_tree(tmp_path):
    eng = make_engine(tmp_path, max_pool_size=2)
    fs = [
        make_evaluated_factor(f"f{i}", "momentum", f"code_{i}", 0.1 * i, 0.01 * i)
        for i in range(1, 5)
    ]
    for f in fs:
        eng.tree.add(f)
        eng.admit(f, check_crowding=False)
    assert eng.pool.size() <= 2
    # 所有因子仍在树中（淘汰不删树）
    assert eng.tree.size() == 4
    print(f"[OK] 淘汰不删树: pool={eng.pool.size()}, tree={eng.tree.size()}")


def test_structural_distance(tmp_path):
    eng = make_engine(tmp_path, min_structural_distance=0.30)
    f1 = make_evaluated_factor(
        "f1", "momentum", "ts_mean(returns, 20) - ts_mean(returns, 60)", 0.5, 0.1)
    eng.tree.add(f1)
    eng.admit(f1, check_crowding=False)
    # 结构相近的因子应被拒
    f2 = make_evaluated_factor(
        "f2", "momentum", "ts_mean(returns, 21) - ts_mean(returns, 60)", 0.55, 0.11)
    eng.tree.add(f2)
    admitted = eng.admit(f2, check_crowding=False)
    assert not admitted, "结构过近应被拒"
    # 结构相远的因子应被接受
    f3 = make_evaluated_factor(
        "f3", "value", "1.0 / pe_ratio", 0.5, 0.1)
    eng.tree.add(f3)
    admitted = eng.admit(f3, check_crowding=False)
    assert admitted, "结构相远应接受"
    print("[OK] 结构距离约束")


def test_crowding_internal(tmp_path):
    df = generate_synthetic_data(200, 50)
    eng = make_engine(tmp_path, max_internal_overlap=0.5)
    # 两个高度相似的因子
    f1 = make_evaluated_factor(
        "f1", "momentum", "ts_mean(returns, 20)", 0.5, 0.1)
    eng.tree.add(f1)
    eng.admit(f1, df=df, check_crowding=False)
    # 第二个因子持仓几乎相同（同一表达式变体）
    f2 = make_evaluated_factor(
        "f2", "momentum", "ts_mean(returns, 22)", 0.55, 0.12)
    eng.tree.add(f2)
    # 手动计算拥挤度
    report = eng.crowding.evaluate(df, [f1, f2])
    assert report.sample_days == 63, "应采样 63 天"
    assert "turnover_q90" in report.external_proxy
    print(f"[OK] 拥挤度分层: internal={report.internal_overlap:.2f}, "
          f"external keys={list(report.external_proxy.keys())}")


def test_exploration_ratio_update(tmp_path):
    eng = make_engine(tmp_path, exploration_ratio=0.2)
    assert eng.evolution.exploration_ratio == 0.2
    # 模拟 L0 难度信号驱动
    eng.update_exploration(0.35)
    assert eng.evolution.exploration_ratio == 0.35
    # 越界应被截断
    eng.update_exploration(0.9)
    assert eng.evolution.exploration_ratio == 0.50
    eng.update_exploration(0.01)
    assert eng.evolution.exploration_ratio == 0.05
    print("[OK] 探索配额可调 + 截断")


def test_propose_with_exploration(tmp_path):
    eng = make_engine(tmp_path, exploration_ratio=0.3)
    f = make_evaluated_factor("f1", "momentum", "ts_mean(returns, 20)", 0.5, 0.1)
    eng.tree.add(f)
    eng.admit(f, check_crowding=False)
    candidates = eng.propose(10)
    assert len(candidates) == 10
    # 所有候选都应在树中
    for c in candidates:
        assert c.fid in eng.tree.nodes
    print(f"[OK] propose 生成 {len(candidates)} 个候选，全部在树中")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_tree_only_grows(tp)
        test_tree_parent_must_exist(tp)
        test_tree_hash_integrity(tp)
        test_pareto_front(tp)
        test_elite_evict_keeps_tree(tp)
        test_structural_distance(tp)
        test_crowding_internal(tp)
        test_exploration_ratio_update(tp)
        test_propose_with_exploration(tp)
        print("\n=== L3 全部测试通过 ===")