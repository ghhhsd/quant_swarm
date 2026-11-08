"""
五类 Failure Trace + 自愈分流（v2.0 冻结，v3.1 保留）。

五类：
- SYNTAX：语法错误（LLM 输出非合法 JSON/DSL）
- LOGIC：逻辑错误（伪逻辑、机制不成立）
- DATA：数据错误（字段不可观测、对齐失败）
- BACKTEST：回测异常（计算超限、数值溢出）
- GATE_REJECTED：门控拒绝（L5 未通过，不重试）

自愈策略差异化：
- SYNTAX：重试编码（换 prompt 或换温度）
- LOGIC：回退假设层重新拆解
- DATA：修正数据契约后重试（有次数上限）
- BACKTEST：降复杂度重试或剪枝
- GATE_REJECTED：记录负样本，不重试
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class FailureType(str, Enum):
    SYNTAX = "syntax"
    LOGIC = "logic"
    DATA = "data"
    BACKTEST = "backtest"
    GATE_REJECTED = "gate_rejected"


HEAL_STRATEGY: Dict[FailureType, str] = {
    FailureType.SYNTAX: "retry_codegen",
    FailureType.LOGIC: "backtrack_hypothesis",
    FailureType.DATA: "fix_data_contract",
    FailureType.BACKTEST: "reduce_complexity",
    FailureType.GATE_REJECTED: "record_negative",
}

# 是否重试（供上层路由）
RETRYABLE: Dict[FailureType, bool] = {
    FailureType.SYNTAX: True,
    FailureType.LOGIC: True,
    FailureType.DATA: True,
    FailureType.BACKTEST: True,
    FailureType.GATE_REJECTED: False,  # 门控拒绝不重试
}


@dataclass
class FailureTrace:
    """
    结构化失败轨迹。
    所有字段可序列化进事件流，供审计与自愈路由。
    """
    hid: str
    failure_type: FailureType
    stage: str              # "narrative" / "fact_check" / "llm_validate" / "codegen"
    message: str
    context: Optional[Dict[str, Any]] = None
    heal_strategy: str = ""
    retryable: bool = False

    def __post_init__(self):
        self.heal_strategy = HEAL_STRATEGY[self.failure_type]
        self.retryable = RETRYABLE[self.failure_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hid": self.hid,
            "failure_type": self.failure_type.value,
            "stage": self.stage,
            "message": self.message,
            "context": self.context or {},
            "heal_strategy": self.heal_strategy,
            "retryable": self.retryable,
        }