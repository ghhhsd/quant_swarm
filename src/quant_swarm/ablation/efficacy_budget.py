# 功效预算声明 + 不可变日志
"""
功效预算声明（v3.1 修正②核心）。

关键设计：
1. 每层在进入候选层时声明 EfficacyBudget。
2. 声明内容：mde（可容忍效应量）、power（目标功效）、alpha、每批次样本量、
   最小批次数、市场状态列表。
3. 声明写入不可变日志（commit），运行期禁止调整。
4. 效应量分三档：LARGE / MEDIUM / SMALL，档位由事前 mde 决定。
5. 分市场状态是必需项，不是可选项。

为什么"事前声明"重要：
- 如果只在跑完消融后说"功效不足"，等于事后找借口。
- 事前声明 = 承认"这层可能效应小，可能证不出来"，避免用"设计漂亮"替代"已证有效"。
"""
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Any
import hashlib
import json

from core.event_log import EventLog, GLOBAL_EVENT_LOG


class EffectSize(str, Enum):
    """效应量档位。"""
    LARGE = "large"      # mde >= 0.15
    MEDIUM = "medium"    # 0.05 <= mde < 0.15
    SMALL = "small"      # mde < 0.05


class MarketRegime(str, Enum):
    """市场状态。"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


def classify_effect_size(mde: float) -> EffectSize:
    """按事前 mde 分类效应量档位。"""
    if mde >= 0.15:
        return EffectSize.LARGE
    elif mde >= 0.05:
        return EffectSize.MEDIUM
    else:
        return EffectSize.SMALL


@dataclass(frozen=True)
class EfficacyBudget:
    """
    功效预算（事前声明，frozen=True）。

    字段：
    - layer：候选层名称（如 "C5.5_economic"）
    - mde：最小可检测效应量（minimum detectable effect）
    - power：目标统计功效（默认 0.80）
    - alpha：显著性水平（默认 0.05）
    - n_obs_per_batch：每批次 OOS 观测数
    - n_batches_min：最小批次数
    - market_regimes：必需的市场状态列表
    - effect_size：由 mde 自动推导的档位
    - note：声明理由（为什么预期这个效应量）
    """
    layer: str
    mde: float
    power: float = 0.80
    alpha: float = 0.05
    n_obs_per_batch: int = 252
    n_batches_min: int = 8
    market_regimes: tuple = (
        MarketRegime.BULL.value,
        MarketRegime.BEAR.value,
        MarketRegime.SIDEWAYS.value,
    )
    note: str = ""
    # 自动推导字段
    effect_size: str = ""

    def __post_init__(self):
        if not self.effect_size:
            object.__setattr__(
                self, "effect_size",
                classify_effect_size(self.mde).value,
            )
        # 校验
        if self.mde <= 0:
            raise ValueError(f"mde 必须 > 0，实际 {self.mde}")
        if not (0 < self.power < 1):
            raise ValueError(f"power 必须在 (0,1)，实际 {self.power}")
        if not (0 < self.alpha < 1):
            raise ValueError(f"alpha 必须在 (0,1)，实际 {self.alpha}")
        if self.n_obs_per_batch < 10:
            raise ValueError(f"n_obs_per_batch 太小: {self.n_obs_per_batch}")
        if self.n_batches_min < 1:
            raise ValueError(f"n_batches_min 必须 >= 1")
        if len(self.market_regimes) < 3:
            raise ValueError(
                f"market_regimes 必须至少含 bull/bear/sideways 三种，"
                f"实际 {self.market_regimes}"
            )

    def required_batches(self) -> int:
        """
        给定效应量、功效、alpha，估计所需 OOS 批次数。

        用两独立样本 t 检验的样本量公式：
        n_per_group = 2 * (z_alpha/2 + z_beta)^2 / mde^2
        所需批次数 = ceil(n_per_group / n_obs_per_batch)
        """
        import numpy as np
        from scipy import stats as scipy_stats
        z_alpha = scipy_stats.norm.ppf(1 - self.alpha / 2)
        z_beta = scipy_stats.norm.ppf(self.power)
        n_per_group = 2 * (z_alpha + z_beta) ** 2 / (self.mde ** 2)
        return max(1, int(np.ceil(n_per_group / self.n_obs_per_batch)))

    def is_feasible(self, available_batches: int) -> bool:
        """可用批次是否满足最小要求。"""
        return available_batches >= max(self.n_batches_min,
                                        self.required_batches())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["market_regimes"] = list(self.market_regimes)
        return d


@dataclass
class CommittedBudget:
    """已 commit 的功效预算。"""
    budget: EfficacyBudget
    commit_hash: str
    declared_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "budget": self.budget.to_dict(),
            "commit_hash": self.commit_hash,
            "declared_at": self.declared_at,
        }


class BudgetViolation(Exception):
    """运行期试图修改已 commit 的功效预算。"""
    pass


class EfficacyBudgetRegistry:
    """
    功效预算注册表。

    使用流程：
    1. 候选层进入时调用 declare()，传入 EfficacyBudget。
    2. declare() 写入不可变日志，返回 CommittedBudget。
    3. 运行期只读取，任何修改尝试抛 BudgetViolation。
    4. 消融时用 get() 获取已 commit 的预算。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self._committed: Dict[str, CommittedBudget] = {}

    def declare(self, budget: EfficacyBudget) -> CommittedBudget:
        """
        声明功效预算。写入不可变日志。
        同一层只能声明一次（防事后调整）。
        """
        if budget.layer in self._committed:
            raise BudgetViolation(
                f"层 {budget.layer} 的功效预算已声明，禁止二次声明"
                "（v3.1 修正②：防事后调参）"
            )
        # 确定性 commit hash
        payload = json.dumps(budget.to_dict(), sort_keys=True, ensure_ascii=False)
        commit_hash = hashlib.sha256(payload.encode()).hexdigest()

        committed = CommittedBudget(
            budget=budget,
            commit_hash=commit_hash,
            declared_at=time.time(),
        )
        self._committed[budget.layer] = committed
        self.event_log.append("efficacy_budget_declared", committed.to_dict())
        return committed

    def get(self, layer: str) -> CommittedBudget:
        if layer not in self._committed:
            raise BudgetViolation(f"层 {layer} 尚未声明功效预算")
        return self._committed[layer]

    def get_all(self) -> Dict[str, CommittedBudget]:
        return dict(self._committed)

    def is_declared(self, layer: str) -> bool:
        return layer in self._committed


# 需要 import time
import time