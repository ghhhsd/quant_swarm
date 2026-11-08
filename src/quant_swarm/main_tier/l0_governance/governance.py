"""
L0 门面：组合以上模块，对外提供统一接口。

对外接口：
- start_round() / end_round()
- consume_token / consume_compute / consume_backtest
- record_trial（同时扣预算 + 记账 N）
- record_failure（路由到 FailureTracker）
- reclaim（动态回收）
- get_committed_thresholds
- get_invariant（只读）
- evaluate_difficulty（难度信号 → 搜索侧调整）

组合：
- quota（配额）
- failure_tracker（失败）
- trial_counter（试验 N）
- threshold_lifecycle（L0 对 core/threshold_commit 的薄包装）
- difficulty_signal（难度信号）

不变核心访问：直接使用 core/invariant.py 的 InvariantRegistry，
通过 governance_context 参数记录 L0 语义，不引入薄包装层。
"""
from typing import Optional, Dict, Any

from main_tier.l0_governance.quota import QuotaManager, RoundQuota, BudgetExceeded
from main_tier.l0_governance.failure_tracker import FailureTracker, FailurePolicy
from main_tier.l0_governance.trial_counter import TrialCounter, TrialRecord
from main_tier.l0_governance.threshold_lifecycle import L0ThresholdLifecycle
from main_tier.l0_governance.difficulty_signal import DifficultySignal, SearchSideAdjustment
from core.config import GLOBAL_THRESHOLDS, KernelConfig, FrozenThresholds
from core.threshold_commit import ThresholdDerivation, CommittedThreshold
from core.invariant import InvariantRegistry, InvariantViolation
from core.event_log import EventLog, GLOBAL_EVENT_LOG
from main_tier.l1_favor import FailureType


class GovernanceLayer:
    """
    L0 门面。所有层通过此对象访问治理能力。
    """

    def __init__(self,
                 config: KernelConfig = None,
                 thresholds: FrozenThresholds = None,
                 event_log: Optional[EventLog] = None):
        self.config = config or KernelConfig()
        self.thresholds = thresholds or GLOBAL_THRESHOLDS
        self.event_log = event_log or GLOBAL_EVENT_LOG

        # 子模块
        self.quota = QuotaManager(
            token_per_round=self.config.token_budget_per_round,
            compute_per_round=self.config.compute_budget_per_round,
            backtest_per_round=self.config.backtest_budget_per_round,
        )
        self.failures = FailureTracker()
        self.trials = TrialCounter(self.event_log)
        self.threshold_lifecycle = L0ThresholdLifecycle(self.event_log)
        # 直接使用 core 版本，通过 governance_context 记录 L0 语义
        self.invariants = InvariantRegistry(self.event_log)
        self.difficulty = DifficultySignal(self.event_log)
		# 轮次状态
        self.round_idx = 0

    # ---- 轮次 + 阈值 commit ---------------------------------------------
    def start_round(self, budget_n: int,
                    triggered_by: str = "l0_start_round",
                    **quota_overrides) -> RoundQuota:
        """
        开始新一轮：commit 阈值 + 启动配额。
        """
        self.round_idx += 1
        self.failures.reset()
        if not self.threshold_lifecycle.is_committed():
            derivation = ThresholdDerivation(
                budget_n=budget_n,
                acceptable_fdr=self.thresholds.fdr_alpha,
                acceptable_dsr=self.thresholds.dsr_threshold,
                calibration_note=f"round {self.round_idx}",
            )
            self.threshold_lifecycle.commit_for_round(
                round_idx=self.round_idx,
                thresholds=self.thresholds,
                derivation=derivation,
                triggered_by=triggered_by,
            )
        q = self.quota.start_round(**quota_overrides)
        self.event_log.append("round_started", {
            "round_idx": self.round_idx,
            "quota": {"token": q.token_total, "compute": q.compute_total,
                      "backtest": q.backtest_total},
        })
        return q

    def end_round(self) -> None:
        q = self.quota.current
        if q is not None:
            self.event_log.append("round_ended", {
                "round_idx": self.round_idx,
                "used": {"token": q.token_used, "compute": q.compute_used,
                         "backtest": q.backtest_used},
                "remaining": {"token": q.token_remaining,
                              "compute": q.compute_remaining,
                              "backtest": q.backtest_remaining},
            })
        self.quota.end_round()

    # ---- 消耗 -----------------------------------------------------------
    def consume_token(self, n: int) -> None:
        self.quota.consume_token(n)

    def consume_compute(self, sec: float) -> None:
        self.quota.consume_compute(sec)

    def consume_backtest(self, n: int = 1) -> None:
        self.quota.consume_backtest(n)

    # ---- 试验记账 -------------------------------------------------------
    def record_trial(self, batch_id: str, hid: str,
                     net_sharpe: Optional[float] = None,
                     consume_backtest: bool = True) -> TrialRecord:
        if consume_backtest:
            self.quota.consume_backtest(1)
        return self.trials.record_trial(batch_id, hid, net_sharpe)

    def get_trial_n(self, batch_id: str) -> int:
        return self.trials.get_n(batch_id)

    def get_batch_sharpes(self, batch_id: str) -> list:
        return self.trials.get_sharpes(batch_id)

    # ---- 失败记录 -------------------------------------------------------
    def record_failure(self, hid: str, failure_type: FailureType) -> bool:
        should_prune = self.failures.record(hid, failure_type)
        self.event_log.append("failure_recorded", {
            "hid": hid,
            "failure_type": failure_type.value if hasattr(failure_type, "value")
                            else str(failure_type),
            "should_prune": should_prune,
        })
        return should_prune

    def is_pruned(self, hid: str) -> bool:
        return self.failures.is_pruned(hid)

    # ---- 动态回收 -------------------------------------------------------
    def reclaim(self, token: int = 0, compute: float = 0.0,
                backtest: int = 0) -> None:
        self.quota.reclaim(token, compute, backtest)
        self.event_log.append("quota_reclaimed", {
            "token": token, "compute": compute, "backtest": backtest,
        })

    # ---- 阈值读取（L5 / C1.5 用）---------------------------------------
    def get_committed_thresholds(self) -> CommittedThreshold:
        return self.threshold_lifecycle.get_committed()

    # ---- 不变核心访问（L3 / L4 / L6 用）-------------------------------
    def get_invariant(self, purpose: str = "",
                      requester: str = "unknown") -> Dict[str, Any]:
        """L0 访问不变核心。通过 governance_context 记录 L0 语义。"""
        return self.invariants.get(
            log_access=True,
            governance_context={"requester": requester, "purpose": purpose},
        )

    def assert_invariant_unchanged(self, candidate: Dict[str, Any],
                                   requester: str = "unknown") -> None:
        """L0 校验候选产物未篡改。失败时记录治理上下文。"""
        self.invariants.assert_unchanged(
            candidate,
            governance_context={"requester": requester},
        )

    # ---- 难度信号 -------------------------------------------------------
    def evaluate_difficulty(self,
                            recent_growth: list[int]) -> SearchSideAdjustment:
        return self.difficulty.escalate_if_persistent(recent_growth)