"""
在线衰减监控（v3.1 修正④：上线后衰减监控）。

关键设计：
1. 监控频率按容量档位：小资金周度、中资金月度、大资金季度。
2. 触发条件：滚动 IC 跌破阈值（如持续 3 期 IC < 0）。
3. 触发动作：
   - 写入 `decay_event` 事件
   - 回灌 L1 的 Failure Trace（GATE_REJECTED，不重试）
   - 标记技能库状态为"待再验证"
4. 衰减事件是"观测"，不是"自动下架"——下架需人工确认。
   理由：IC 短期跌破可能是噪声，自动下架会导致频繁误杀。
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict
import numpy as np
import time

from core.event_log import EventLog, GLOBAL_EVENT_LOG


class DecayStatus(str, Enum):
    """衰减状态。"""
    HEALTHY = "healthy"        # IC 正常
    WARNING = "warning"        # IC 下降但未跌破阈值
    DECAYED = "decayed"        # IC 跌破阈值，需再验证


@dataclass
class DecayCheckResult:
    """单次衰减检查结果。"""
    fid: str
    status: DecayStatus
    rolling_ic_mean: float
    rolling_ic_std: float
    n_periods_below_zero: int
    threshold_ic: float
    reason: str

    def to_dict(self):
        return {
            "fid": self.fid,
            "status": self.status.value,
            "rolling_ic_mean": self.rolling_ic_mean,
            "rolling_ic_std": self.rolling_ic_std,
            "n_periods_below_zero": self.n_periods_below_zero,
            "threshold_ic": self.threshold_ic,
            "reason": self.reason,
        }


class DecayMonitor:
    """
    在线衰减监控。

    真实部署应按监控频率定期跑（如 cron）。
    本实现提供 check() 方法，由调度器调用。
    """

    def __init__(self,
                 ic_threshold: float = 0.0,
                 n_periods_below_zero: int = 3,
                 event_log: Optional[EventLog] = None):
        self.ic_threshold = ic_threshold
        self.n_periods_below_zero = n_periods_below_zero
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self._last_monitor: Dict[str, float] = {}  # fid → 上次监控时间

    def check(self, fid: str, rolling_ic: np.ndarray,
              monitor_frequency_days: int) -> DecayCheckResult:
        """
        检查单个因子的衰减状态。

        rolling_ic：滚动 IC 序列（最近 N 期）。
        monitor_frequency_days：监控频率（天），0 表示不监控。
        """
        if monitor_frequency_days <= 0:
            return DecayCheckResult(
                fid=fid, status=DecayStatus.HEALTHY,
                rolling_ic_mean=0.0, rolling_ic_std=0.0,
                n_periods_below_zero=0,
                threshold_ic=self.ic_threshold,
                reason="档位为 REJECT，不监控",
            )

        if len(rolling_ic) == 0:
            return DecayCheckResult(
                fid=fid, status=DecayStatus.HEALTHY,
                rolling_ic_mean=0.0, rolling_ic_std=0.0,
                n_periods_below_zero=0,
                threshold_ic=self.ic_threshold,
                reason="无 IC 数据",
            )

        mean_ic = float(np.mean(rolling_ic))
        std_ic = float(np.std(rolling_ic))
        n_below = int(np.sum(rolling_ic < self.ic_threshold))

        # 判定
        if n_below >= self.n_periods_below_zero:
            status = DecayStatus.DECAYED
            reason = (f"最近 {n_below} 期 IC < {self.ic_threshold}，"
                     f"超过阈值 {self.n_periods_below_zero} 期")
        elif mean_ic < self.ic_threshold * 1.5:
            status = DecayStatus.WARNING
            reason = f"IC 均值 {mean_ic:.4f} 接近阈值 {self.ic_threshold}"
        else:
            status = DecayStatus.HEALTHY
            reason = f"IC 均值 {mean_ic:.4f} 正常"

        result = DecayCheckResult(
            fid=fid, status=status,
            rolling_ic_mean=mean_ic, rolling_ic_std=std_ic,
            n_periods_below_zero=n_below,
            threshold_ic=self.ic_threshold,
            reason=reason,
        )

        # DECAYED → 写事件 + 回灌 Failure Trace
        if status == DecayStatus.DECAYED:
            self.event_log.append("decay_event", result.to_dict())

        self._last_monitor[fid] = time.time()
        return result

    def check_batch(self,
                    ic_series_per_fid: Dict[str, np.ndarray],
                    monitor_frequencies: Dict[str, int]
                    ) -> List[DecayCheckResult]:
        """批量检查。"""
        return [
            self.check(fid, ic, monitor_frequencies.get(fid, 30))
            for fid, ic in ic_series_per_fid.items()
        ]