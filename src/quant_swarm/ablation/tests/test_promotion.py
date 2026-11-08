"""
晋升/降级协议测试。

验证 v3.1 的九条纪律：
1. 强证据 + 正增益 → PROMOTE
2. 强证据 + 负增益 → DEMOTE
3. 弱证据 → KEEP（不升不降）
4. 强证据 + 零增益 → KEEP_CANDIDATE（不晋升）
5. 晋升后可回退（后续负增益 → DEMOTE）
6. 状态持久化到事件流
7. 幂等键防重复转移
8. 决策可解释（evidence_snapshot）
9. 历史完整性可校验
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from ablation import (
    EfficacyBudget, MarketRegime, PowerResult,
    AblationResult, BatchPnL,
    LayerStatus, TransitionAction, LayerStateRegistry, PromotionProtocol,
    PortfolioConfig, PortfolioPnLResult,
)
from core.event_log import EventLog
import tempfile
from pathlib import Path


def make_registry(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    return LayerStateRegistry(log), log


def make_budget(layer="test", mde=0.15):
    return EfficacyBudget(
        layer=layer, mde=mde, power=0.80, alpha=0.05,
        n_obs_per_batch=252, n_batches_min=4,
    )


def make_power_result(layer, effect, power, strength,
                      target=0.80, available=20, required=4):
    return PowerResult(
        layer=layer,
        observed_effect=effect,
        observed_power=power,
        target_power=target,
        n_obs_used=available * 252,
        available_batches=available,
        required_batches=required,
        evidence_strength=strength,
        note="test",
    )


def make_ablation_result(layer, effect, strength):
    """构造 AblationResult（简化版）。"""
    power = make_power_result(
        layer, effect, 0.85 if strength == "strong" else 0.5,
        strength,
    )
    by_regime = {
        MarketRegime.BULL.value: make_power_result(
            layer, effect * 0.5, 0.8 if strength == "strong" else 0.5,
            strength,
        ),
        MarketRegime.BEAR.value: make_power_result(
            layer, effect * 0.5, 0.8 if strength == "strong" else 0.5,
            strength,
        ),
        MarketRegime.SIDEWAYS.value: make_power_result(
            layer, 0.0, 0.5, "weak",
        ),
    }
    return AblationResult(
        layer=layer,
        batches=[],
        overall_power=power,
        by_regime=by_regime,
        evidence_strength=strength,
        note="test",
        portfolio_config={"weighting": "equal"},
    )


def test_promote_strong_positive(tmp_path):
    """强证据 + 正增益 → PROMOTE。"""
    registry, log = make_registry(tmp_path)
    registry.register("C5.5", LayerStatus.CANDIDATE)
    protocol = PromotionProtocol(registry, log)

    result = make_ablation_result("C5.5", effect=0.20, strength="strong")
    decision = protocol.evaluate(result)

    assert decision.action == TransitionAction.PROMOTE
    assert decision.from_state == LayerStatus.CANDIDATE
    assert decision.to_state == LayerStatus.MAIN
    assert registry.get_state("C5.5") == LayerStatus.MAIN
    print(f"[OK] 强证据+正增益 → {decision.action.value}: {decision.reason}")


def test_demote_strong_negative(tmp_path):
    """强证据 + 负增益 → DEMOTE。"""
    registry, log = make_registry(tmp_path)
    registry.register("L_old", LayerStatus.MAIN)  # 已在主干
    protocol = PromotionProtocol(registry, log)

    result = make_ablation_result("L_old", effect=-0.20, strength="strong")
    decision = protocol.evaluate(result)

    assert decision.action == TransitionAction.DEMOTE
    assert decision.from_state == LayerStatus.MAIN
    assert decision.to_state == LayerStatus.CANDIDATE
    assert registry.get_state("L_old") == LayerStatus.CANDIDATE
    print(f"[OK] 强证据+负增益 → {decision.action.value}: {decision.reason}")


def test_keep_candidate_weak(tmp_path):
    """弱证据 → 保持候选（不升不降）。"""
    registry, log = make_registry(tmp_path)
    registry.register("C1.5", LayerStatus.CANDIDATE)
    protocol = PromotionProtocol(registry, log)

    result = make_ablation_result("C1.5", effect=0.03, strength="weak")
    decision = protocol.evaluate(result)

    assert decision.action == TransitionAction.KEEP_CANDIDATE
    assert decision.from_state == LayerStatus.CANDIDATE
    assert decision.to_state == LayerStatus.CANDIDATE
    assert registry.get_state("C1.5") == LayerStatus.CANDIDATE
    print(f"[OK] 弱证据 → {decision.action.value}: {decision.reason}")


def test_keep_main_weak(tmp_path):
    """弱证据 + 当前主干 → 保持主干。"""
    registry, log = make_registry(tmp_path)
    registry.register("L1", LayerStatus.MAIN)
    protocol = PromotionProtocol(registry, log)

    result = make_ablation_result("L1", effect=0.0, strength="weak")
    decision = protocol.evaluate(result)

    assert decision.action == TransitionAction.KEEP_MAIN
    assert decision.to_state == LayerStatus.MAIN
    print(f"[OK] 弱证据+主干 → {decision.action.value}: {decision.reason}")


def test_keep_candidate_zero_gain(tmp_path):
    """强证据 + 零增益 → 保持候选（不晋升）。"""
    registry, log = make_registry(tmp_path)
    registry.register("C_zero", LayerStatus.CANDIDATE)
    protocol = PromotionProtocol(registry, log)

    result = make_ablation_result("C_zero", effect=0.005, strength="strong")
    decision = protocol.evaluate(result)

    assert decision.action == TransitionAction.KEEP_CANDIDATE
    assert decision.to_state == LayerStatus.CANDIDATE
    print(f"[OK] 强证据+零增益 → {decision.action.value}: {decision.reason}")


def test_promote_then_demote(tmp_path):
    """晋升后可回退（后续负增益 → DEMOTE）。"""
    registry, log = make_registry(tmp_path)
    registry.register("C5.5", LayerStatus.CANDIDATE)
    protocol = PromotionProtocol(registry, log)

    # 第一次：强证据 + 正增益 → 晋升
    result1 = make_ablation_result("C5.5", effect=0.20, strength="strong")
    decision1 = protocol.evaluate(result1)
    assert decision1.action == TransitionAction.PROMOTE
    assert registry.get_state("C5.5") == LayerStatus.MAIN

    # 第二次：强证据 + 负增益 → 降级
    result2 = make_ablation_result("C5.5", effect=-0.20, strength="strong")
    decision2 = protocol.evaluate(result2)
    assert decision2.action == TransitionAction.DEMOTE
    assert registry.get_state("C5.5") == LayerStatus.CANDIDATE

    # 历史应有 2 条转移
    history = registry.get_history("C5.5")
    assert len(history) == 2
    print(f"[OK] 晋升后回退: {len(history)} 条转移")


def test_state_persisted_to_event_log(tmp_path):
    """状态持久化到事件流。"""
    registry, log = make_registry(tmp_path)
    registry.register("C5.5", LayerStatus.CANDIDATE)
    protocol = PromotionProtocol(registry, log)

    result = make_ablation_result("C5.5", effect=0.20, strength="strong")
    protocol.evaluate(result)

    events = log.query("layer_state_transition")
    assert len(events) == 1
    assert events[0].payload["layer"] == "C5.5"
    assert events[0].payload["action"] == "promote"
    print(f"[OK] 事件流: {len(events)} 条 transition")


def test_idempotency_key(tmp_path):
    """幂等键防重复转移。"""
    registry, log = make_registry(tmp_path)
    registry.register("L1", LayerStatus.CANDIDATE)
    protocol = PromotionProtocol(registry, log)

    result = make_ablation_result("L1", effect=0.20, strength="strong")
    d1 = protocol.evaluate(result)
    # 再次评估同一结果 → 幂等（不重复转移）
    d2 = protocol.evaluate(result)
    # 第一次晋升后，第二次的 from_state 是 MAIN，结果应是 KEEP_MAIN
    # 但幂等键只在相同 from_state → to_state 时生效
    # 所以这里验证的是"同一状态转移不重复"
    history = registry.get_history("L1")
    # 第一次是 candidate→main，第二次是 main→main（无状态变化，不写 transition）
    assert len(history) == 1
    print(f"[OK] 幂等: history={len(history)} 条")


def test_evidence_snapshot(tmp_path):
    """决策可解释（evidence_snapshot）。"""
    registry, log = make_registry(tmp_path)
    registry.register("C5.5", LayerStatus.CANDIDATE)
    protocol = PromotionProtocol(registry, log)

    result = make_ablation_result("C5.5", effect=0.20, strength="strong")
    decision = protocol.evaluate(result)

    snap = decision.evidence_snapshot
    assert "evidence_strength" in snap
    assert "observed_effect" in snap
    assert "by_regime" in snap
    assert "portfolio_config" in snap
    assert snap["evidence_strength"] == "strong"
    assert snap["observed_effect"] == 0.20
    print(f"[OK] 证据快照: {list(snap.keys())}")


def test_history_integrity(tmp_path):
    """历史完整性可校验。"""
    registry, log = make_registry(tmp_path)
    registry.register("C5.5", LayerStatus.CANDIDATE)
    protocol = PromotionProtocol(registry, log)

    # 晋升
    protocol.evaluate(make_ablation_result("C5.5", 0.20, "strong"))
    # 降级
    protocol.evaluate(make_ablation_result("C5.5", -0.20, "strong"))
    # 再晋升
    protocol.evaluate(make_ablation_result("C5.5", 0.20, "strong"))

    errors = registry.verify_history_integrity("C5.5")
    assert not errors, f"历史完整性失败: {errors}"
    history = registry.get_history("C5.5")
    assert len(history) == 3
    # 校验状态链
    assert history[0].from_state == LayerStatus.CANDIDATE
    assert history[0].to_state == LayerStatus.MAIN
    assert history[1].from_state == LayerStatus.MAIN
    assert history[1].to_state == LayerStatus.CANDIDATE
    assert history[2].from_state == LayerStatus.CANDIDATE
    assert history[2].to_state == LayerStatus.MAIN
    print(f"[OK] 历史完整性: {len(history)} 条，状态链正确")


def test_unregistered_layer_rejected(tmp_path):
    """未注册的层被拒。"""
    registry, log = make_registry(tmp_path)
    protocol = PromotionProtocol(registry, log)

    result = make_ablation_result("unknown", 0.20, "strong")
    try:
        protocol.evaluate(result)
        assert False, "应抛 ValueError"
    except ValueError as e:
        print(f"[OK] 未注册层被拒: {e}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_promote_strong_positive(tp)
        test_demote_strong_negative(tp)
        test_keep_candidate_weak(tp)
        test_keep_main_weak(tp)
        test_keep_candidate_zero_gain(tp)
        test_promote_then_demote(tp)
        test_state_persisted_to_event_log(tp)
        test_idempotency_key(tp)
        test_evidence_snapshot(tp)
        test_history_integrity(tp)
        test_unregistered_layer_rejected(tp)
        print("\n=== 晋升/降级协议全部测试通过 ===")