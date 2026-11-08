"""
core/config.py — 配置定义（唯一真相源）。

职责边界：
1. 本文件只定义"值"：
   - INVARIANT_CORE：不变核心的 dict
   - FrozenThresholds：冻结阈值的 dataclass
   - KernelConfig：全局配置的 dataclass
   - GLOBAL_THRESHOLDS / GLOBAL_KERNEL：全局单例
   - ConfigVersionManager：配置版本管理

2. 本文件不定义"访问"和"哈希"：
   - 不变核心的访问 / 哈希 / diff / 校验 → core.invariant
   - 冻结阈值的 commit / 校验 → core.threshold_commit
   - 全局配置的替换 / 版本管理 → 本文件的 ConfigVersionManager

3. 哈希计算的唯一真相源是 core.invariant.hash.compute_hash。
   本文件不重复实现哈希计算。
"""
from dataclasses import dataclass, asdict, replace
from typing import Dict, Any, Optional, List
import time

from core.event_log import EventLog, GLOBAL_EVENT_LOG
from core.invariant.hash import compute_hash


# =============================================================================
# 1. 不变核心（Invariant Core）—— 只定义值
# =============================================================================
# 绝对只读。风控断路器、仓位分配逻辑、AST 白名单。
# 版本号用于人工识别，内容哈希由 core.invariant.hash.compute_hash 计算。
# 任何演化产物只能引用，不能修改。

INVARIANT_CORE: Dict[str, Any] = {
    "version": "ic-1.0.0",
    "risk_breaker": {
        "max_drawdown": 0.20,
        "max_leverage": 1.0,
        "max_position_pct": 0.10,
        "daily_loss_limit": 0.05,
    },
    "position_sizing": {
        "method": "equal_weight_capped",
        "max_single_position": 0.10,
        "min_position": 0.0,
        "rebalance_frequency": "daily",
    },
    "ast_whitelist": {
        "version": "wl-1.0.0",
        "allowed_nodes": [
            "Expression", "BinOp", "UnaryOp", "Call", "Name", "Load",
            "Constant", "Add", "Sub", "Mult", "Div", "USub", "UAdd",
            "Compare", "Lt", "Gt", "LtE", "GtE", "Eq", "NotEq",
            "BoolOp", "And", "Or", "IfExp",
        ],
        "allowed_funcs": [
            "ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min", "ts_rank",
            "abs", "log", "sqrt", "sign", "clip", "rank", "zscore",
        ],
    },
}


# =============================================================================
# 2. 冻结阈值（Frozen Thresholds）—— 只定义值
# =============================================================================

@dataclass(frozen=True)
class FrozenThresholds:
    """
    冻结阈值。frozen=True，运行期禁止修改。

    运行前 commit 到不可变日志，二次 commit 被拒。
    commit 必须含推导依据（budget_n / acceptable_fdr / acceptable_dsr）。

    注意：本类只定义"值"，commit 逻辑在 core.threshold_commit。
    """
    # --- L5 VaG 门控 ---
    fdr_alpha: float = 0.10
    dsr_threshold: float = 0.95
    marginal_gain_threshold: float = 0.02
    min_oos_sharpe: float = 0.3

    # --- C1.5 新颖性 ---
    novelty_semantic_dist: float = 0.30
    novelty_ast_dist: float = 0.25

    # --- 消融功效预算 ---
    ablation_mde_large: float = 0.15
    ablation_mde_medium: float = 0.08
    ablation_mde_small: float = 0.05
    ablation_power: float = 0.80
    ablation_min_batches: int = 8
    ablation_alpha: float = 0.05

    # --- 冻结标记 ---
    frozen: bool = True

    def __post_init__(self):
        if not self.frozen:
            raise ValueError(
                "冻结阈值必须 frozen=True。"
                "若需修改阈值，应产生新的 FrozenThresholds 实例并重新 commit。"
            )
        if not (0 < self.fdr_alpha < 1):
            raise ValueError(f"fdr_alpha 必须在 (0,1)，实际 {self.fdr_alpha}")
        if not (0 < self.dsr_threshold < 1):
            raise ValueError(f"dsr_threshold 必须在 (0,1)，实际 {self.dsr_threshold}")
        if not (0 < self.marginal_gain_threshold < 1):
            raise ValueError("marginal_gain_threshold 必须在 (0,1)")
        if not (0 < self.novelty_semantic_dist < 1):
            raise ValueError("novelty_semantic_dist 必须在 (0,1)")
        if not (0 < self.novelty_ast_dist < 1):
            raise ValueError("novelty_ast_dist 必须在 (0,1)")
        if not (0 < self.ablation_power < 1):
            raise ValueError("ablation_power 必须在 (0,1)")
        if self.ablation_min_batches < 1:
            raise ValueError("ablation_min_batches 必须 >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        """
        内容哈希。

        委托给 core.invariant.hash.compute_hash（唯一真相源）。
        """
        return compute_hash(self.to_dict())


# =============================================================================
# 3. 全局配置（Kernel Config）—— 只定义值
# =============================================================================

@dataclass(frozen=True)
class KernelConfig:
    """
    全局配置。frozen=True，构造后不可改。

    允许事前修改：通过 with_overrides() 产生新实例，
    每次修改由调用方通过 ConfigVersionManager.register_update() 写入事件流。
    """
    # --- 数据 ---
    n_days: int = 1500
    n_assets: int = 200
    data_seed: int = 42

    # --- 轮次预算 ---
    max_evolution_rounds: int = 5
    token_budget_per_round: int = 200_000
    compute_budget_per_round: float = 600.0
    backtest_budget_per_round: int = 50

    # --- OOS / walk-forward ---
    oos_window: int = 252
    walk_forward_step: int = 63
    oos_train_window: int = 500

    # --- 搜索批次 ---
    search_batch_id: str = "batch_001"
    max_candidates: int = 50

    # --- 组合层 ---
    portfolio_weighting: str = "equal"
    portfolio_cost_scenario: str = "base"

    # --- 沙箱 ---
    backtest_timeout_s: float = 30.0

    def __post_init__(self):
        if self.n_days < 100:
            raise ValueError(f"n_days 太小: {self.n_days}")
        if self.n_assets < 10:
            raise ValueError(f"n_assets 太小: {self.n_assets}")
        if self.oos_window < 30:
            raise ValueError(f"oos_window 太小: {self.oos_window}")
        if self.walk_forward_step < 1:
            raise ValueError("walk_forward_step 必须 >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates 必须 >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        """内容哈希（委托给唯一真相源）。"""
        return compute_hash(self.to_dict())

    def with_overrides(self, **overrides) -> "KernelConfig":
        """
        产生修改后的新实例（不修改原实例）。

        调用方负责通过 ConfigVersionManager.register_update() 记录变更。
        """
        return replace(self, **overrides)


# =============================================================================
# 4. 全局单例
# =============================================================================

GLOBAL_THRESHOLDS = FrozenThresholds()
GLOBAL_KERNEL = KernelConfig()


# =============================================================================
# 5. 配置版本管理
# =============================================================================

@dataclass
class ConfigVersion:
    """配置版本记录。"""
    config_type: str          # "invariant_core" / "thresholds" / "kernel"
    version_hash: str
    config_dict: Dict[str, Any]
    timestamp: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_type": self.config_type,
            "version_hash": self.version_hash,
            "config_dict": self.config_dict,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


class ConfigVersionManager:
    """
    配置版本管理器。

    记录三类配置的所有版本：
    - invariant_core：不变核心（初始后不应更新）
    - thresholds：冻结阈值（每次 commit 时更新）
    - kernel：全局配置（每次 replace 时更新）

    每次注册写入事件流，可审计。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self._versions: Dict[str, List[ConfigVersion]] = {
            "invariant_core": [],
            "thresholds": [],
            "kernel": [],
        }

    def register_initial(self,
                         invariant_core: Dict[str, Any],
                         thresholds: FrozenThresholds,
                         kernel: KernelConfig) -> None:
        """注册初始版本（三类配置各一条）。"""
        # 不变核心
        v = ConfigVersion(
            config_type="invariant_core",
            version_hash=compute_hash(invariant_core),
            config_dict=invariant_core,
            timestamp=time.time(),
            reason="initial",
        )
        self._versions["invariant_core"].append(v)
        self.event_log.append("config_version_registered", v.to_dict())

        # 阈值
        v = ConfigVersion(
            config_type="thresholds",
            version_hash=thresholds.content_hash(),
            config_dict=thresholds.to_dict(),
            timestamp=time.time(),
            reason="initial",
        )
        self._versions["thresholds"].append(v)
        self.event_log.append("config_version_registered", v.to_dict())

        # 全局配置
        v = ConfigVersion(
            config_type="kernel",
            version_hash=kernel.content_hash(),
            config_dict=kernel.to_dict(),
            timestamp=time.time(),
            reason="initial",
        )
        self._versions["kernel"].append(v)
        self.event_log.append("config_version_registered", v.to_dict())

    def register_update(self, config_type: str,
                        new_config_dict: Dict[str, Any],
                        reason: str) -> ConfigVersion:
        """注册修改版本。"""
        if config_type not in self._versions:
            raise ValueError(f"未知配置类型: {config_type}")
        v = ConfigVersion(
            config_type=config_type,
            version_hash=compute_hash(new_config_dict),
            config_dict=new_config_dict,
            timestamp=time.time(),
            reason=reason,
        )
        self._versions[config_type].append(v)
        self.event_log.append("config_version_updated", v.to_dict())
        return v

    def get_latest(self, config_type: str) -> Optional[ConfigVersion]:
        versions = self._versions.get(config_type, [])
        return versions[-1] if versions else None

    def get_history(self, config_type: str) -> List[ConfigVersion]:
        return list(self._versions.get(config_type, []))

    def verify_no_invariant_change(self) -> List[str]:
        """
        校验不变核心从未被修改（除初始注册外）。
        返回错误列表（空表示从未修改）。
        """
        errors = []
        versions = self._versions.get("invariant_core", [])
        if len(versions) > 1:
            errors.append(
                f"不变核心被修改 {len(versions) - 1} 次（不应发生）"
            )
        if versions:
            current_hash = versions[-1].version_hash
            expected_hash = compute_hash(INVARIANT_CORE)
            if current_hash != expected_hash:
                errors.append(
                    f"不变核心哈希不匹配: 当前 {current_hash[:12]}, "
                    f"期望 {expected_hash[:12]}"
                )
        return errors