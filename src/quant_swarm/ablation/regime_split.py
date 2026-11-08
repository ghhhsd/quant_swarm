# 分市场状态切分
"""
分市场状态切分（v3.1 修正②：分市场状态分别度量）。

关键设计：
1. 单一静态基线会把"熊市有效的层"误判为无贡献。
2. 本模块把收益序列按市场状态切分，每个 regime 单独比较。
3. 市场状态判定：
   - BULL：市场收益 > 0 且波动 < 中位数
   - BEAR：市场收益 < 0
   - SIDEWAYS：其他
4. 消融时，任一 regime 显著才作强证据（不是所有 regime 都显著）。
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from ablation.efficacy_budget import MarketRegime


@dataclass
class RegimeSegment:
    """单个市场状态的片段。"""
    regime: str
    start_idx: int
    end_idx: int
    n_obs: int


def classify_market_regimes(market_returns: np.ndarray,
                            window: int = 20) -> np.ndarray:
    """
    按滚动窗口判定每个时点的市场状态。

    返回：与 market_returns 等长的字符串数组。
    """
    n = len(market_returns)
    regimes = np.array([MarketRegime.SIDEWAYS.value] * n, dtype=object)
    if n < window:
        return regimes

    # 滚动均值 + 滚动波动
    rolling_mean = np.convolve(market_returns, np.ones(window) / window,
                               mode="valid")
    rolling_std = np.array([
        np.std(market_returns[i:i + window])
        for i in range(n - window + 1)
    ])
    median_std = np.median(rolling_std) if len(rolling_std) > 0 else 1.0

    for i in range(window, n):
        mu = rolling_mean[i - window]
        sigma = rolling_std[i - window]
        if mu > 0 and sigma < median_std:
            regimes[i] = MarketRegime.BULL.value
        elif mu < 0:
            regimes[i] = MarketRegime.BEAR.value
        else:
            regimes[i] = MarketRegime.SIDEWAYS.value

    return regimes


def split_by_regime(returns: np.ndarray,
                    regimes: np.ndarray) -> Dict[str, np.ndarray]:
    """
    按市场状态切分收益序列。
    返回：{regime: 该 regime 下的收益子序列}
    """
    result: Dict[str, List[float]] = {}
    for r, ret in zip(regimes, returns):
        result.setdefault(r, []).append(float(ret))
    return {k: np.array(v) for k, v in result.items()}


def split_batch_by_regime(
    on_sharpes: List[float],
    off_sharpes: List[float],
    on_regimes: List[str],
    off_regimes: List[str],
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    按 regime 切分 on/off 的 Sharpe 序列。

    注意：这里假设 on_sharpes 和 off_sharpes 是"跨批次"的 Sharpe，
    on_regimes 和 off_regimes 是对应批次的市场状态标签。
    """
    on_by_regime: Dict[str, List[float]] = {}
    off_by_regime: Dict[str, List[float]] = {}

    for s, r in zip(on_sharpes, on_regimes):
        on_by_regime.setdefault(r, []).append(s)
    for s, r in zip(off_sharpes, off_regimes):
        off_by_regime.setdefault(r, []).append(s)

    return on_by_regime, off_by_regime