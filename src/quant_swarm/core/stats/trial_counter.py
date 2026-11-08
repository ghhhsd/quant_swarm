"""
搜索批次计数器（v3.1 修正①：N 按批次记账）。

关键设计：
1. N 按"搜索批次/演化谱系"记账，不是全局、不是单因子。
2. 记录每次试验的完整信息（hid、trial_index、net_sharpe、n_obs）。
3. N 单调累积（同批次跨轮累积，新批次重置）。
4. 提供"按批次查询"接口：
   - get_n(batch_id)：该批次累计试验数
   - get_records(batch_id)：该批次所有试验记录
   - get_sharpes(batch_id)：该批次所有净 Sharpe
5. 写入事件流（供事件溯源）。

为什么记录完整信息：
- BH-FDR 需要同批次的 p 值，从 net_sharpe 转换。
- 审计需要"每次试验的完整信息"，不只是 count。
- 组合层需要"每批次的 Sharpe 分布"，用于消融比较。

为什么 N 单调累积：
- 同批次跨轮累积：第 1 轮 10 个试验，第 2 轮还是同批次再 10 个，N=20。
- 新批次重置：切换 batch_id 后，N 从 0 开始。
- 单调累积保证 DSR 的校正强度随试验次数增加而增强。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import time

from core.stats.result import TrialRecord
from core.event_log import EventLog, GLOBAL_EVENT_LOG


class SearchBatchCounter:
    """
    搜索批次计数器。

    - 记录每次试验的完整信息。
    - N 按批次查询。
    - 写入事件流。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        # batch_id → List[TrialRecord]
        self._trials: Dict[str, List[TrialRecord]] = {}

    def record(self, batch_id: str, hid: str,
               net_sharpe: float, n_obs: int) -> TrialRecord:
        """
        记录一次试验。

        batch_id：搜索批次 ID
        hid：因子 ID
        net_sharpe：净 Sharpe
        n_obs：观测数
        """
        records = self._trials.setdefault(batch_id, [])
        trial_index = len(records) + 1
        record = TrialRecord(
            batch_id=batch_id,
            hid=hid,
            trial_index=trial_index,
            net_sharpe=net_sharpe,
            n_obs=n_obs,
            timestamp=time.time(),
        )
        records.append(record)
        self.event_log.append("trial_recorded", record.to_dict())
        return record

    def get_n(self, batch_id: str) -> int:
        """该批次累计试验次数 N。"""
        return len(self._trials.get(batch_id, []))

    def get_records(self, batch_id: str) -> List[TrialRecord]:
        """该批次所有试验记录。"""
        return list(self._trials.get(batch_id, []))

    def get_sharpes(self, batch_id: str) -> List[float]:
        """该批次所有净 Sharpe。"""
        return [r.net_sharpe for r in self._trials.get(batch_id, [])]

    def get_batch_ids(self) -> List[str]:
        """所有批次 ID。"""
        return list(self._trials.keys())

    def get_total_trials(self) -> int:
        """全局试验总数（仅供参考，不用于 DSR）。"""
        return sum(len(r) for r in self._trials.values())

    def summary(self) -> Dict:
        """汇总（供审计）。"""
        return {
            "n_batches": len(self._trials),
            "total_trials": self.get_total_trials(),
            "by_batch": {
                b: len(r) for b, r in self._trials.items()
            },
        }