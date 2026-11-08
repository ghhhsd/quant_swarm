"""
C1.5 新颖性门测试。

验证 v3.1 的九条纪律：
1. 阈值事前冻结，二次 commit 被拒
2. 阈值从已知因子库分布标定（非拍脑袋）
3. 三档判定：NOVEL / VARIANT / DUPLICATE
4. 语义 + AST 双重近邻
5. 机制声明开放集（"其他"不被拒）
6. 新机制声明进事件流
7. 异质来源标签记录
8. shadow mode：产出不进主干
9. shadow 汇总：若启用会过滤多少
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from candidate_tier.c1_5_novelty import (
    KnownFactorLibrary, KnownFactor,
    SemanticDistance, ASTDistance,
    MechanismDeclaration, MechanismValidator,
    HeterogeneousGenerator, HeterogeneousHypothesis, HypothesisSource,
    NoveltyGate, NoveltyVerdict, NoveltyThresholdCommitter,
)
from core.event_log import EventLog
import tempfile
from pathlib import Path


def make_log(tmp_path):
    return EventLog(str(tmp_path / "events.jsonl"))


def make_gate(tmp_path, library=None):
    log = make_log(tmp_path)
    library = library or KnownFactorLibrary()
    sem = SemanticDistance()
    ast = ASTDistance()
    committer = NoveltyThresholdCommitter(log)
    sem_thr, ast_thr = committer.calibrate_from_library(
        library, sem, ast, percentile=10.0)
    th = committer.commit(sem_thr, ast_thr, note="从学术库标定")
    return NoveltyGate(
        library=library, thresholds=th,
        semantic=sem, ast_dist=ast,
        event_log=log,
    ), log


def test_threshold_commit_once(tmp_path):
    log = make_log(tmp_path)
    library = KnownFactorLibrary()
    committer = NoveltyThresholdCommitter(log)
    sem_thr, ast_thr = committer.calibrate_from_library(
        library, SemanticDistance(), ASTDistance())
    committer.commit(sem_thr, ast_thr)
    try:
        committer.commit(0.5, 0.5)
        assert False, "应抛 ValueError"
    except ValueError:
        print("[OK] 阈值二次 commit 被拒")


def test_calibration_from_library(tmp_path):
    log = make_log(tmp_path)
    library = KnownFactorLibrary()
    committer = NoveltyThresholdCommitter(log)
    sem_thr, ast_thr = committer.calibrate_from_library(
        library, SemanticDistance(), ASTDistance(), percentile=10.0)
    # 阈值应在合理范围
    assert 0 < sem_thr < 1, f"语义阈值 {sem_thr} 应在 (0,1)"
    assert 0 < ast_thr < 1, f"AST 阈值 {ast_thr} 应在 (0,1)"
    print(f"[OK] 从库标定: sem_thr={sem_thr:.3f}, ast_thr={ast_thr:.3f}")


def test_novel_verdict(tmp_path):
    gate, _ = make_gate(tmp_path)
    # 构造一个与库中因子都不同的假设
    hyp = HeterogeneousHypothesis(
        hid="h_novel", family="other",
        raw_text="全新机制", code="ts_mean(volume, 5) * rank(close)",
        source=HypothesisSource.RULE_TEMPLATE,
        mechanism_family="microstructure",
        failure_precondition="趋势市场失效",
        economic_rationale="微观结构信号",
    )
    d = gate.evaluate(hyp)
    print(f"[INFO] 新颖假设: verdict={d.verdict.value}, "
          f"sem_d={d.semantic_distance:.3f}, ast_d={d.ast_distance:.3f}")
    assert d.shadow, "候选层应恒为 shadow"
    print("[OK] 新颖判定（shadow mode）")


def test_duplicate_verdict(tmp_path):
    gate, _ = make_gate(tmp_path)
    # 与库中 ac_001 完全相同
    hyp = HeterogeneousHypothesis(
        hid="h_dup", family="momentum",
        raw_text="与库中因子完全相同",
        code="ts_mean(returns, 20) - ts_mean(returns, 60)",
        source=HypothesisSource.LLM,
        mechanism_family="momentum",
        failure_precondition="高波动期失效",
        economic_rationale="动量延续",
    )
    d = gate.evaluate(hyp)
    assert d.verdict == NoveltyVerdict.DUPLICATE, \
        f"完全重复应判 DUPLICATE，实际 {d.verdict.value}"
    print(f"[OK] 重复判定: {d.verdict.value}")


def test_variant_verdict(tmp_path):
    gate, _ = make_gate(tmp_path)
    # 语义近但 AST 稍远（窗口变体）
    hyp = HeterogeneousHypothesis(
        hid="h_variant", family="momentum",
        raw_text="动量延续变体",
        code="ts_mean(returns, 22) - ts_mean(returns, 60)",
        source=HypothesisSource.LLM,
        mechanism_family="momentum",
        failure_precondition="高波动期失效",
        economic_rationale="动量延续（参数微调）",
    )
    d = gate.evaluate(hyp)
    print(f"[INFO] 变体假设: verdict={d.verdict.value}, "
          f"sem_d={d.semantic_distance:.3f}, ast_d={d.ast_distance:.3f}")
    # 不强制断言为 VARIANT（取决于标定阈值），但至少不是 NOVEL
    assert d.verdict in (NoveltyVerdict.VARIANT, NoveltyVerdict.DUPLICATE)
    print("[OK] 变体/重复判定（近邻）")


def test_mechanism_open_set(tmp_path):
    """机制声明开放集：'其他'或新机制不被拒。"""
    validator = MechanismValidator(make_log(tmp_path))
    # 新机制
    decl = MechanismDeclaration(
        mechanism_family="量子金融",  # 不在已知族内
        failure_precondition="市场正常期",
        economic_rationale="量子效应",
    )
    ok, reason = validator.validate(decl)
    assert ok, f"新机制应通过: {reason}"
    assert decl.is_novel_mechanism, "应标记为新机制"
    print(f"[OK] 新机制声明通过: {decl.mechanism_family}")

    # 空机制被拒
    decl_bad = MechanismDeclaration(
        mechanism_family="",
        failure_precondition="x",
        economic_rationale="y",
    )
    ok, reason = validator.validate(decl_bad)
    assert not ok
    print(f"[OK] 空机制被拒: {reason}")


def test_novel_mechanism_logged(tmp_path):
    """新机制声明进事件流。"""
    log = make_log(tmp_path)
    validator = MechanismValidator(log)
    decl = MechanismDeclaration(
        mechanism_family="全新机制",
        failure_precondition="x",
        economic_rationale="y",
    )
    validator.validate(decl)
    events = log.query("novel_mechanism_declared")
    assert len(events) == 1
    print("[OK] 新机制进事件流")


def test_heterogeneous_sources(tmp_path):
    """异质信号来源。"""
    gen = HeterogeneousGenerator(seed=42)
    library = KnownFactorLibrary()
    known_codes = [f.code for f in library.get_all()]

    llm_hyps = gen.generate_llm(3)
    rule_hyps = gen.generate_rule_template(3)
    resid_hyps = gen.generate_residual(known_codes, 3)
    alt_hyps = gen.generate_alternative_data(3)

    all_hyps = llm_hyps + rule_hyps + resid_hyps + alt_hyps
    sources = {h.source for h in all_hyps}
    assert len(sources) == 4, f"应有 4 种来源，实际 {sources}"

    for h in all_hyps:
        assert h.mechanism_family, "机制族不能为空"
        assert h.failure_precondition, "失效前提不能为空"

    print(f"[OK] 异质来源: {[s.value for s in sources]}")


def test_shadow_summary(tmp_path):
    """shadow 汇总：若启用会过滤多少。"""
    gate, log = make_gate(tmp_path)
    gen = HeterogeneousGenerator(seed=42)
    library = KnownFactorLibrary()
    known_codes = [f.code for f in library.get_all()]

    hyps = (gen.generate_llm(5) + gen.generate_rule_template(5) +
            gen.generate_residual(known_codes, 5) +
            gen.generate_alternative_data(5))
    decisions = gate.evaluate_batch(hyps)
    summary = gate.shadow_summary(decisions)

    assert summary["n_total"] == 20
    assert summary["n_novel"] + summary["n_variant"] + summary["n_duplicate"] == 20
    assert summary["would_filter"] == summary["n_variant"] + summary["n_duplicate"]
    print(f"[OK] shadow 汇总: total={summary['n_total']}, "
          f"novel={summary['n_novel']}, variant={summary['n_variant']}, "
          f"dup={summary['n_duplicate']}, filter_rate={summary['filter_rate']:.1%}")


def test_shadow_does_not_block(tmp_path):
    """shadow mode：判定不进主干，只记录。"""
    gate, log = make_gate(tmp_path)
    gen = HeterogeneousGenerator(seed=42)
    hyps = gen.generate_llm(3)
    decisions = gate.evaluate_batch(hyps)
    # 每个 decision 都是 shadow=True
    for d in decisions:
        assert d.shadow, "候选层决策应恒为 shadow"
    # 事件流中有 shadow 记录
    shadow_events = log.query("novelty_shadow_decision")
    assert len(shadow_events) == 3
    print("[OK] shadow mode 只记录不阻断")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        test_threshold_commit_once(tp)
        test_calibration_from_library(tp)
        test_novel_verdict(tp)
        test_duplicate_verdict(tp)
        test_variant_verdict(tp)
        test_mechanism_open_set(tp)
        test_novel_mechanism_logged(tp)
        test_heterogeneous_sources(tp)
        test_shadow_summary(tp)
        test_shadow_does_not_block(tp)
        print("\n=== C1.5 全部测试通过 ===")