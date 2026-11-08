"""
试验次数 N 记账（v3.1 修正①核心）。

关键：N 按"搜索批次/演化谱系"记账，不是全局、不是单因子。

为什么不是全局：全局会把跨族、跨轮次的所有回测都算进同一 N，
DSR 校正过严，误杀合法因子。

为什么不是单因子：每个因子单独记账等于每次试验 N=1，DSR 退化为
普通 Sharpe，完全丧失多重检验校正。

正确粒度：同一假设族/同一搜索批次内的累计试验次数。L0 的 trial_counter
与 quota 的 backtest 计数器物理上分离——backtest 计数按轮重置，
trial N 跨轮累积（只要批次 ID 不变）。

与 L0 预算的关系：trial_counter 复用 quota 的回测消耗事件作为记账源，
但维护独立的累积状态。这样"预算扣减"和"统计 N 记账"是同一事件的两个视角，
不会出现"扣了预算但没记账"或反之。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from core.event_log import EventLog, GLOBAL_EVENT_LOG


@dataclass
class TrialRecord:
    """单次试验记录。"""
    batch_id: str
    hid: str
    trial_index: int          # 该批次内第几次试验（从 1 开始）
    net_sharpe: Optional[float] = None


class TrialCounter:
    """
    按搜索批次记账的试验计数器。

    - 每次回测消耗调用 `record_trial(batch_id, hid)`。
    - N 用于 DSR/FDR 校正，按批次查询。
    - 可选地记录 net_sharpe，用于同批次内的 BH-FDR。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        # batch_id -> List[TrialRecord]
        self._trials: Dict[str, List[TrialRecord]] = {}

    def record_trial(self, batch_id: str, hid: str,
                     net_sharpe: Optional[float] = None) -> TrialRecord:
        """记录一次试验，返回带 trial_index 的记录。"""
        records = self._trials.setdefault(batch_id, [])
        idx = len(records) + 1
        rec = TrialRecord(batch_id=batch_id, hid=hid,
                          trial_index=idx, net_sharpe=net_sharpe)
        records.append(rec)
        self.event_log.append("trial_recorded", {
            "batch_id": batch_id, "hid": hid,
            "trial_index": idx, "net_sharpe": net_sharpe,
        })
        return rec

    def get_n(self, batch_id: str) -> int:
        """该批次累计试验次数 N（用于 DSR）。"""
        return len(self._trials.get(batch_id, []))

    def get_sharpes(self, batch_id: str) -> List[float]:
        """该批次所有有效 net_sharpe（用于同批次 BH-FDR）。"""
        return [r.net_sharpe for r in self._trials.get(batch_id, [])
                if r.net_sharpe is not None]

    def get_batch_ids(self) -> List[str]:
        return list(self._trials.keys())

    def get_records(self, batch_id: str) -> List[TrialRecord]:
        return list(self._trials.get(batch_id, []))