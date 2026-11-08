"""
最小可证伪内核（v3.1 修正③：所有消融的基线）。

关键设计：
1. 最小内核只含四个组件：
   - 假设生成（LLM 或模板桩）
   - 成本后回测
   - DSR 门控
   - 组合 PnL 计算
2. 不含 L1.5 / L3 进化 / C3+ / C5.5 / C6——这些是候选层，不在基线里。
3. 最小内核是"可运行的"，不是"文档里的概念"——它产出一组因子 + 组合 PnL。
4. 最小内核的产出是"可证伪基准"的输入。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np
import polars as pl

from core.config import GLOBAL_KERNEL, KernelConfig, GLOBAL_THRESHOLDS
from core.event_log import EventLog, GLOBAL_EVENT_LOG
from core.stats import SearchBatchCounter, deflated_sharpe
from core.cost_model import apply_costs, SCENARIOS
from core.backtest import backtest_factor, evaluate_expression, _cross_zscore
from core.portfolio import build_portfolio, PortfolioResult
from core.gate import VaGGate, GateDecision


@dataclass
class KernelResult:
    """最小内核跑一轮的结果。"""
    round_idx: int
    candidates: List[dict]           # [{"fid": ..., "code": ..., "net_sharpe": ...}]
    admitted_codes: List[str]
    portfolio: PortfolioResult
    portfolio_stress: Dict[str, PortfolioResult]
    regime_pnl: Dict[str, PortfolioResult]  # 分市场状态的组合 PnL
    n_tested: int
    n_admitted: int

    def to_dict(self):
        return {
            "round_idx": self.round_idx,
            "n_tested": self.n_tested,
            "n_admitted": self.n_admitted,
            "admitted_codes": self.admitted_codes,
            "portfolio_sharpe": self.portfolio.sharpe,
            "portfolio_mdd": self.portfolio.max_drawdown,
            "portfolio_total": self.portfolio.total_return,
            "stress": {k: v.sharpe for k, v in self.portfolio_stress.items()},
            "regime": {k: v.sharpe for k, v in self.regime_pnl.items()},
        }


class MinimalKernel:
    """
    最小可证伪内核。

    流程：
    1. 假设生成（模板桩）
    2. 成本后回测
    3. DSR 门控
    4. 组合 PnL 计算
    """

    def __init__(self,
                 df: pl.DataFrame,
                 config: KernelConfig = None,
                 event_log: Optional[EventLog] = None):
        self.df = df
        self.config = config or GLOBAL_KERNEL
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self._counter = SearchBatchCounter()
        # 门控需要已 commit 的阈值（由 L0 提供）
        self._gate = None

    def attach_gate(self, gate: VaGGate) -> None:
        """由 L0 注入已配置的门控（阈值已 commit）。"""
        self._gate = gate

    def _generate_hypotheses(self, n: int) -> List[dict]:
        """假设生成（模板桩）。真实部署替换为 LLM 调用。"""
        templates = [
            ("momentum", "ts_mean(returns, 20) - ts_mean(returns, 60)"),
            ("momentum", "ts_mean(returns, 5) - ts_mean(returns, 20)"),
            ("volatility", "ts_std(returns, 20)"),
            ("volatility", "ts_std(returns, 60)"),
            ("value", "1.0 / pe_ratio"),
            ("liquidity", "ts_mean(turnover, 20)"),
            ("liquidity", "ts_mean(turnover, 60)"),
            ("other", "ts_mean(returns, 10)"),
        ]
        import random
        rng = random.Random(self.config.data_seed)
        return [
            {"fid": f"h_{i:03d}", "family": rng.choice(templates)[0],
             "code": rng.choice(templates)[1]}
            for i in range(n)
        ]

    def run(self, round_idx: int = 1) -> KernelResult:
        """
        跑一轮最小内核。
        """
        if self._gate is None:
            raise RuntimeError("MinimalKernel 需要先 attach_gate() 注入门控")

        # 1) 假设生成
        hypotheses = self._generate_hypotheses(self.config.max_candidates)

        # 2) 回测 + 门控
        candidates = []
        admitted_codes = []
        all_sharpes = []
        for hyp in hypotheses:
            try:
                bt = backtest_factor(self.df, hyp["code"], SCENARIOS["base"])
            except Exception:
                continue
            all_sharpes.append(bt.net_sharpe)
            self._counter.record(self.config.search_batch_id, 1)

            # DSR 门控
            decision = self._gate.evaluate(
                bt.net_sharpe, len(bt.net_returns),
                self.config.search_batch_id, all_sharpes,
            )
            candidates.append({
                "fid": hyp["fid"], "code": hyp["code"],
                "net_sharpe": bt.net_sharpe,
                "admitted": decision.admitted,
            })
            if decision.admitted:
                admitted_codes.append(hyp["code"])

        # 3) 组合 PnL（base + 压力 + 分 regime）
        portfolio = build_portfolio(
            self.df, admitted_codes,
            self.config.portfolio_weighting, SCENARIOS["base"],
        )
        portfolio_stress = {
            k: build_portfolio(self.df, admitted_codes,
                               self.config.portfolio_weighting, SCENARIOS[k])
            for k in SCENARIOS
        }
        regime_pnl = self._compute_regime_pnl(admitted_codes)

        result = KernelResult(
            round_idx=round_idx,
            candidates=candidates,
            admitted_codes=admitted_codes,
            portfolio=portfolio,
            portfolio_stress=portfolio_stress,
            regime_pnl=regime_pnl,
            n_tested=len(candidates),
            n_admitted=len(admitted_codes),
        )
        self.event_log.append("minimal_kernel_round", result.to_dict())
        return result

    def _compute_regime_pnl(self, codes: List[str]) -> Dict[str, PortfolioResult]:
        """分市场状态计算组合 PnL。"""
        from ablation.regime_split import classify_market_regimes
        # 市场收益
        returns = self.df.pivot(values="returns", index="date", on="asset").sort("date")
        ret_arr = np.array([returns[c].to_numpy() for c in returns.columns[1:]]).T
        market_returns = np.nanmean(ret_arr, axis=1)
        regimes = classify_market_regimes(market_returns, window=20)

        # 按 regime 切片计算组合 PnL（简化：用全样本组合，标注 regime 分布）
        # 真实实现应对每个 regime 单独构建组合
        portfolio = build_portfolio(
            self.df, codes, self.config.portfolio_weighting, SCENARIOS["base"],
        )
        # 用 regime 分布作为代理
        unique, counts = np.unique(regimes, return_counts=True)
        regime_dist = {str(u): int(c) for u, c in zip(unique, counts)}
        # 返回同一个组合（真实实现应分 regime 构建）
        return {r: portfolio for r in regime_dist}