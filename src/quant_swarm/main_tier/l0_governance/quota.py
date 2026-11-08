"""
三层配额与轮次调度。

关键设计：
1. 配额按轮次重置，不跨轮累积。
2. 动态回收：分支被剪枝时，其"已分配但未消耗"的配额回流。
   注意：回收的是"未消耗配额"，不是"已消耗配额"——已消耗的算沉没成本。
3. 任何超额消耗抛 BudgetExceeded，由上层路由到剪枝，而非静默截断。
"""
from dataclasses import dataclass
from typing import Optional, Dict


class BudgetExceeded(Exception):
    """配额耗尽。上层应捕获并剪枝该分支，而非重试。"""
    def __init__(self, resource: str, requested: float, remaining: float):
        super().__init__(
            f"{resource} 配额不足：请求 {requested}，剩余 {remaining}"
        )
        self.resource = resource
        self.requested = requested
        self.remaining = remaining


@dataclass
class RoundQuota:
    """单轮配额状态。"""
    token_total: int
    compute_total: float
    backtest_total: int
    token_remaining: int
    compute_remaining: float
    backtest_remaining: int

    @property
    def token_used(self) -> int:
        return self.token_total - self.token_remaining

    @property
    def compute_used(self) -> float:
        return self.compute_total - self.compute_remaining

    @property
    def backtest_used(self) -> int:
        return self.backtest_total - self.backtest_remaining


class QuotaManager:
    """
    三层配额管理器。

    配额来源：KernelConfig（全局配置），但每轮开始时可被覆盖
    （用于"搜索侧调整"——提高探索配额需要更多预算）。
    """

    def __init__(self, token_per_round: int, compute_per_round: float,
                 backtest_per_round: int):
        self.default_token = token_per_round
        self.default_compute = compute_per_round
        self.default_backtest = backtest_per_round
        self.round_idx = 0
        self.current: Optional[RoundQuota] = None

    def start_round(self,
                    token_override: Optional[int] = None,
                    compute_override: Optional[float] = None,
                    backtest_override: Optional[int] = None) -> RoundQuota:
        """
        开始新一轮。override 用于搜索侧调整（如提高探索配额），
        但受上限约束——不能无限扩大，见 `_cap`。
        """
        self.round_idx += 1
        t = self._cap(token_override or self.default_token, self.default_token * 1.5)
        c = self._cap(compute_override or self.default_compute, self.default_compute * 1.5)
        b = self._cap(backtest_override or self.default_backtest, self.default_backtest * 1.5)
        self.current = RoundQuota(
            token_total=int(t), compute_total=float(c), backtest_total=int(b),
            token_remaining=int(t), compute_remaining=float(c), backtest_remaining=int(b),
        )
        return self.current

    def end_round(self) -> None:
        self.current = None

    # ---- 消耗 -----------------------------------------------------------
    def consume_token(self, n: int) -> None:
        q = self._ensure()
        if q.token_remaining < n:
            raise BudgetExceeded("token", n, q.token_remaining)
        q.token_remaining -= n

    def consume_compute(self, sec: float) -> None:
        q = self._ensure()
        if q.compute_remaining < sec:
            raise BudgetExceeded("compute", sec, q.compute_remaining)
        q.compute_remaining -= sec

    def consume_backtest(self, n: int = 1) -> None:
        q = self._ensure()
        if q.backtest_remaining < n:
            raise BudgetExceeded("backtest", n, q.backtest_remaining)
        q.backtest_remaining -= n

    # ---- 动态回收 -------------------------------------------------------
    def reclaim(self, token: int = 0, compute: float = 0.0, backtest: int = 0) -> None:
        """
        回收未消耗配额。调用方应传入"该分支已分配但未用完"的量。
        回收不超本轮总量上限（防止重复回收导致虚增）。
        """
        q = self._ensure()
        q.token_remaining = min(q.token_total, q.token_remaining + token)
        q.compute_remaining = min(q.compute_total, q.compute_remaining + compute)
        q.backtest_remaining = min(q.backtest_total, q.backtest_remaining + backtest)

    def _ensure(self) -> RoundQuota:
        if self.current is None:
            raise RuntimeError("未在轮次内，先 start_round()")
        return self.current

    @staticmethod
    def _cap(value: float, max_value: float) -> float:
        """搜索侧调整的上限：最多为默认值的 1.5 倍。"""
        return min(value, max_value)