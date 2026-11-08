"""
叙事结构与哈希预承诺（v3.1 修正③核心）。

关键设计：
1. 叙事必须在回测前生成，含三要素：预测方向 / 生效条件 / 预期衰减特征。
2. 叙事哈希写入不可变日志，回测后无法修改。
3. 叙事是"承诺"，不是"解释"——它的作用是让 L5 能判"结果与承诺是否一致"。
4. 哈希锁定的是完整叙事结构（含三要素），不是自由文本。
"""
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Optional, Dict, Any
import hashlib
import json

from core.event_log import EventLog, GLOBAL_EVENT_LOG


class Direction(str, Enum):
    """预测方向。"""
    LONG = "long"       # 因子值高 → 未来收益高
    SHORT = "short"     # 因子值高 → 未来收益低
    NEUTRAL = "neutral" # 方向不确定（需额外论证）


class DecayPattern(str, Enum):
    """预期衰减特征。"""
    PERSISTENT = "persistent"   # 长期有效（如价值）
    FAST_DECAY = "fast_decay"   # 快速衰减（如短期反转）
    REGIME_DEPENDENT = "regime_dependent"  # 市场状态依赖
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Narrative:
    """
    叙事预承诺。frozen=True，生成后不可修改。

    三要素（v3.1 修正③）：
    - direction：预测方向
    - conditions：生效条件（列表，每条是可观测的市场/数据条件）
    - decay：预期衰减特征
    """
    hid: str
    mechanism_family: str          # 机制族（momentum/value/volatility/liquidity/other）
    direction: Direction
    conditions: List[str]          # 生效条件
    decay: DecayPattern
    economic_logic: str            # 经济学机制描述（自由文本，但需过非LLM事实校验）
    falsifiable_statement: str     # 可证伪陈述（"若 X 则假设被证伪"）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hid": self.hid,
            "mechanism_family": self.mechanism_family,
            "direction": self.direction.value,
            "conditions": list(self.conditions),
            "decay": self.decay.value,
            "economic_logic": self.economic_logic,
            "falsifiable_statement": self.falsifiable_statement,
        }

    def content_hash(self) -> str:
        """确定性哈希：排序键 + 枚举转值。"""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()


@dataclass
class NarrativeCommitment:
    """叙事提交凭据。"""
    narrative: Narrative
    hash: str
    commit_event_hash: str   # 事件流中对应事件的哈希（用于审计链）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative": self.narrative.to_dict(),
            "hash": self.hash,
            "commit_event_hash": self.commit_event_hash,
        }


class NarrativeCommitter:
    """
    叙事预承诺器。

    使用纪律：
    - 必须在回测前调用 commit()。
    - 一旦 commit，禁止对同一 hid 二次 commit（防止回测后改写叙事）。
    - commit 事件写入不可变日志，返回 event hash 供审计。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self._committed: Dict[str, NarrativeCommitment] = {}

    def commit(self, narrative: Narrative) -> NarrativeCommitment:
        if narrative.hid in self._committed:
            raise ValueError(
                f"叙事已 commit（hid={narrative.hid}），禁止二次 commit"
                "（v3.1 修正③：防止回测后改写叙事）"
            )
        n_hash = narrative.content_hash()
        ev = self.event_log.append("narrative_committed", {
            "hid": narrative.hid,
            "narrative_hash": n_hash,
            "narrative": narrative.to_dict(),
        })
        commitment = NarrativeCommitment(
            narrative=narrative,
            hash=n_hash,
            commit_event_hash=ev.content_hash,
        )
        self._committed[narrative.hid] = commitment
        return commitment

    def get(self, hid: str) -> Optional[NarrativeCommitment]:
        return self._committed.get(hid)

    def is_committed(self, hid: str) -> bool:
        return hid in self._committed