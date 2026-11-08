"""
合成数据生成器（v3.1 修正②：参数化 + 可预测信号）。

关键设计：
1. 参数化：信号强度、噪声水平、市场状态分布、缺失率可配置。
2. 可预测信号：内置动量、波动、价值三种信号，强度可调。
3. 可复现：固定种子。
4. 市场状态：牛/熊/震荡的分布可配置，用于消融的分 regime 测试。
5. 缺失值：按契约的 missing_policy 注入缺失。

为什么参数化重要：
- 消融时需要控制信号强度，验证"层在弱信号下是否仍然有效"。
- 固定信号强度会导致"所有层都有效"或"所有层都无效"，
  无法区分层的贡献。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import polars as pl

from core.data_contract import (
    DataContract, DEFAULT_DATA_CONTRACT,
    MISSING_POLICY_DROP, MISSING_POLICY_FFILL, MISSING_POLICY_ZERO,
    MISSING_POLICY_NAN, MISSING_POLICY_INTERPOLATE,
)


@dataclass
class SignalConfig:
    """
    信号强度配置。

    - momentum_coef：动量信号系数（正 → 动量延续）
    - volatility_coef：波动信号系数（负 → 低波动溢价）
    - value_coef：价值信号系数（正 → 价值回归）
    - noise_std：噪声标准差
    """
    momentum_coef: float = 0.02
    volatility_coef: float = -0.01
    value_coef: float = 0.005
    noise_std: float = 0.02

    def to_dict(self):
        return {
            "momentum_coef": self.momentum_coef,
            "volatility_coef": self.volatility_coef,
            "value_coef": self.value_coef,
            "noise_std": self.noise_std,
        }


@dataclass
class RegimeConfig:
    """
    市场状态配置。

    - bull_prob：牛市概率
    - bear_prob：熊市概率
    - sideways_prob：震荡市概率
    - regime_switch_prob：每期切换 regime 的概率
    """
    bull_prob: float = 0.4
    bear_prob: float = 0.2
    sideways_prob: float = 0.4
    regime_switch_prob: float = 0.05

    def __post_init__(self):
        total = self.bull_prob + self.bear_prob + self.sideways_prob
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"regime 概率之和必须为 1，实际 {total}"
            )

    def to_dict(self):
        return {
            "bull_prob": self.bull_prob,
            "bear_prob": self.bear_prob,
            "sideways_prob": self.sideways_prob,
            "regime_switch_prob": self.regime_switch_prob,
        }


@dataclass
class SyntheticDataConfig:
    """
    合成数据配置。
    """
    n_days: int = 1500
    n_assets: int = 200
    seed: int = 42
    signal: SignalConfig = field(default_factory=SignalConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    missing_rate: float = 0.0   # 缺失率（0 = 无缺失）
    start_date: str = "2018-01-01"
    asset_prefix: str = "A"

    def to_dict(self):
        return {
            "n_days": self.n_days,
            "n_assets": self.n_assets,
            "seed": self.seed,
            "signal": self.signal.to_dict(),
            "regime": self.regime.to_dict(),
            "missing_rate": self.missing_rate,
            "start_date": self.start_date,
            "asset_prefix": self.asset_prefix,
        }


class SyntheticDataGenerator:
    """
    合成数据生成器。

    流程：
    1. 生成市场状态序列（牛/熊/震荡）
    2. 生成因子暴露（动量、波动、价值）
    3. 按信号强度 + 噪声生成收益
    4. 生成价格、成交量、换手、PE、市值
    5. 按契约注入缺失值
    6. 返回 long-format DataFrame
    """

    def __init__(self,
                 config: Optional[SyntheticDataConfig] = None,
                 contract: Optional[DataContract] = None):
        self.config = config or SyntheticDataConfig()
        self.contract = contract or DEFAULT_DATA_CONTRACT

    def generate(self) -> pl.DataFrame:
        """生成合成数据。"""
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        # 1) 日期序列
        dates = pl.date_range(
            start=pl.date(2018, 1, 1), end=None,
            interval="1d", eager=True,
        ).head(cfg.n_days)

        # 2) 资产列表
        assets = [f"{cfg.asset_prefix}{i:03d}" for i in range(cfg.n_assets)]

        # 3) 市场状态序列
        regimes = self._generate_regimes(rng)

        # 4) 因子暴露
        mom = rng.normal(0, 1, (cfg.n_days, cfg.n_assets))
        vol = rng.normal(0, 1, (cfg.n_days, cfg.n_assets))
        val = rng.normal(0, 1, (cfg.n_days, cfg.n_assets))

        # 5) 滞后因子（用于计算下期收益）
        mom_lag = np.roll(mom, 1, axis=0); mom_lag[0] = 0
        vol_lag = np.roll(vol, 1, axis=0); vol_lag[0] = 0
        val_lag = np.roll(val, 1, axis=0); val_lag[0] = 0

        # 6) 收益 = 信号 + regime 调整 + 噪声
        sig = cfg.signal
        base_returns = (
            sig.momentum_coef * mom_lag
            + sig.volatility_coef * vol_lag
            + sig.value_coef * val_lag
        )
        # regime 调整：熊市整体收益下移
        regime_effect = np.zeros(cfg.n_days)
        regime_effect[regimes == "bear"] = -0.005
        regime_effect[regimes == "bull"] = 0.003
        returns = base_returns + regime_effect[:, None] + rng.normal(
            0, sig.noise_std, (cfg.n_days, cfg.n_assets)
        )

        # 7) 价格、成交量、换手、PE、市值
        close = np.cumprod(1 + returns, axis=0) * 100
        volume = np.abs(rng.normal(1e6, 2e5, (cfg.n_days, cfg.n_assets)))
        turnover = volume / 1e8
        pe = np.abs(rng.normal(20, 5, (cfg.n_days, cfg.n_assets)))
        mcap = np.abs(rng.normal(1e10, 2e9, (cfg.n_days, cfg.n_assets)))

        # 8) 按契约注入缺失值
        returns = self._inject_missing(returns, rng, "returns")
        close = self._inject_missing(close, rng, "close")
        volume = self._inject_missing(volume, rng, "volume")
        turnover = self._inject_missing(turnover, rng, "turnover")
        pe = self._inject_missing(pe, rng, "pe_ratio")
        mcap = self._inject_missing(mcap, rng, "market_cap")

        # 9) 构造 long-format DataFrame
        frames = []
        for j, asset in enumerate(assets):
            frames.append(pl.DataFrame({
                "date": dates,
                "asset": [asset] * cfg.n_days,
                "returns": returns[:, j],
                "close": close[:, j],
                "volume": volume[:, j],
                "turnover": turnover[:, j],
                "pe_ratio": pe[:, j],
                "market_cap": mcap[:, j],
            }))
        df = pl.concat(frames).sort(["date", "asset"])
        return df

    def _generate_regimes(self, rng: np.random.Generator) -> np.ndarray:
        """生成市场状态序列。"""
        cfg = self.config.regime
        regimes = np.empty(self.config.n_days, dtype=object)
        probs = np.array([cfg.bull_prob, cfg.bear_prob, cfg.sideways_prob])
        labels = np.array(["bull", "bear", "sideways"], dtype=object)
        current = rng.choice(labels, p=probs)
        for i in range(self.config.n_days):
            if rng.random() < cfg.regime_switch_prob:
                current = rng.choice(labels, p=probs)
            regimes[i] = current
        return regimes

    def _inject_missing(self, arr: np.ndarray, rng: np.random.Generator,
                        field_name: str) -> np.ndarray:
        """按契约的 missing_policy 注入缺失值。"""
        if self.config.missing_rate <= 0:
            return arr
        mask = rng.random(arr.shape) < self.config.missing_rate
        result = arr.copy()
        spec = self.contract.get_field(field_name)
        policy = spec.missing_policy if spec else MISSING_POLICY_NAN
        if policy == MISSING_POLICY_ZERO:
            result[mask] = 0.0
        elif policy == MISSING_POLICY_FFILL:
            # 前向填充：用前一个非缺失值
            for i in range(1, result.shape[0]):
                result[i][mask[i] & ~np.isnan(result[i - 1])] = \
                    result[i - 1][mask[i] & ~np.isnan(result[i - 1])]
        elif policy == MISSING_POLICY_NAN:
            result[mask] = np.nan
        # DROP 和 INTERPOLATE 在生成时不注入（下游处理）
        return result


# =============================================================================
# 便捷函数（兼容旧接口）
# =============================================================================

def generate_synthetic_data(n_days: int = 1500,
                            n_assets: int = 200,
                            seed: int = 42) -> pl.DataFrame:
    """便捷函数：生成默认合成数据（兼容旧接口）。"""
    config = SyntheticDataConfig(
        n_days=n_days, n_assets=n_assets, seed=seed,
    )
    return SyntheticDataGenerator(config).generate()