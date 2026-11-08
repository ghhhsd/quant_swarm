"""
VaG 门控门面（v3.1 修正①③：短路执行 + 失败分流）。

关键设计：
1. N 来自 L0（唯一真相源），L5 不持有 N。
2. 阈值来自 L0 的 CommittedThreshold（冻结）。
3. 三道检查短路执行：
   - 统计失败 → 立即返回（不回灌机制模板库）
   - 边际失败 → 立即返回
   - 叙事失败 → 立即返回（回灌机制模板库）
4. failure_stage 标记哪道失败，downstream_action 标记回灌目标。
5. 门控决策进事件流。
"""
from typing import List, Optional, Callable, Dict, Any
import numpy as np

from main_tier.l5_vag.oos import walk_forward_split, rolling_oos_sharpes
from main_tier.l5_vag.statistical import StatisticalChecker
from main_tier.l5_vag.marginal import MarginalGainChecker
from main_tier.l5_vag.consistency import NarrativeConsistencyChecker
from main_tier.l5_vag.result import GateDecision
from main_tier.l0_governance import GovernanceLayer
from main_tier.l1_favor import NarrativeCommitment
from main_tier.l1_favor.failure import FailureType
from core.event_log import EventLog, GLOBAL_EVENT_LOG


class VaGGate:
    """
    VaG 门控。

    - N 从 L0 取。
    - 阈值从 L0 的 CommittedThreshold 取。
    - 三道检查短路执行。
    """

    def __init__(self,
                 governance: GovernanceLayer,
                 portfolio_sharpe_fn: Callable[[List[str]], float],
                 oos_train_window: int = 500,
                 oos_window: int = 252,
                 oos_step: int = 63,
                 event_log: Optional[EventLog] = None):
        self.gov = governance
        self.portfolio_sharpe_fn = portfolio_sharpe_fn
        self.oos_train_window = oos_train_window
        self.oos_window = oos_window
        self.oos_step = oos_step
        self.event_log = event_log or GLOBAL_EVENT_LOG

        # 从 L0 取已 commit 的阈值（运行期只读）
        self._committed = self.gov.get_committed_thresholds()
        self._th = self._committed.thresholds

        # 初始化检查器
        self.statistical = StatisticalChecker(self._th)
        self.marginal = MarginalGainChecker(
            threshold=self._th.marginal_gain_threshold,
        )
        self.consistency = NarrativeConsistencyChecker()

    def evaluate(self,
                 fid: str,
                 net_returns: np.ndarray,
                 factor_returns: np.ndarray,
                 forward_returns: np.ndarray,
                 batch_id: str,
                 existing_codes: List[str],
                 new_code: str,
                 narrative_commitment: Optional[NarrativeCommitment] = None,
                 regime_sharpes: Optional[dict] = None,
                 n_obs_per_window: Optional[int] = None) -> GateDecision:
        """
        执行三道检查（短路）。

        注意：N 从 L0 取，不由调用方传入。
        """
        # 从 L0 取 N 和批次记录
        n_trials = self.gov.get_trial_n(batch_id)
        batch_records = self.gov.trials.get_records(batch_id)

        # ---- 1) 统计检查 -----------------------------------------------
        windows = walk_forward_split(
            len(net_returns), self.oos_train_window,
            self.oos_window, self.oos_step,
        )
        oos_sharpes = rolling_oos_sharpes(net_returns, windows)
        n_obs = n_obs_per_window or self.oos_window

        stat_result = self.statistical.check(
            oos_sharpes, n_trials, batch_records, n_obs,
        )
        if stat_result.reasons:
            # 统计失败 → 立即返回（不回灌机制模板库）
            decision = GateDecision(
                admitted=False, fid=fid,
                failure_stage="statistical",
                failure_type=FailureType.GATE_REJECTED,
                downstream_action="negative_sample",
                reasons=stat_result.reasons,
                statistical=stat_result.to_dict(),
            )
            self.event_log.append("vag_decision", decision.to_dict())
            return decision

        # ---- 2) 边际增益检查 -------------------------------------------
        marg_result = self.marginal.check(
            existing_codes, new_code, self.portfolio_sharpe_fn,
        )
        if not marg_result.passed:
            # 边际失败 → 立即返回（不回灌机制模板库）
            decision = GateDecision(
                admitted=False, fid=fid,
                failure_stage="marginal",
                failure_type=FailureType.GATE_REJECTED,
                downstream_action="negative_sample",
                reasons=[marg_result.reason],
                statistical=stat_result.to_dict(),
                marginal=marg_result.to_dict(),
            )
            self.event_log.append("vag_decision", decision.to_dict())
            return decision

        # ---- 3) 叙事一致性检查 -----------------------------------------
        if narrative_commitment is not None:
            # 计算 factor-returns 与 forward-returns 的相关
            corr = 0.0
            if (len(factor_returns) >= 2
                    and len(forward_returns) == len(factor_returns)
                    and np.std(factor_returns) > 1e-9
                    and np.std(forward_returns) > 1e-9):
                corr = float(np.corrcoef(
                    factor_returns, forward_returns)[0, 1])
            ic_series = np.array([corr * 0.8, corr * 0.9, corr * 0.95, corr])

            cons_result = self.consistency.check_all(
                narrative_commitment.narrative, corr, ic_series,
                regime_sharpes,
            )
            if not cons_result.consistent:
                # 叙事失效 → 回灌机制模板库（不是负样本库）
                decision = GateDecision(
                    admitted=False, fid=fid,
                    failure_stage="narrative",
                    failure_type=FailureType.GATE_REJECTED,
                    downstream_action="mechanism_template",
                    reasons=cons_result.reasons,
                    statistical=stat_result.to_dict(),
                    marginal=marg_result.to_dict(),
                    consistency=cons_result.to_dict(),
                )
                self.event_log.append("vag_decision", decision.to_dict())
                return decision
        else:
            cons_result = None

        # ---- 全部通过 ---------------------------------------------------
        decision = GateDecision(
            admitted=True, fid=fid,
            statistical=stat_result.to_dict(),
            marginal=marg_result.to_dict(),
            consistency=(cons_result.to_dict() if cons_result else None),
        )
        self.event_log.append("vag_decision", decision.to_dict())
        return decision