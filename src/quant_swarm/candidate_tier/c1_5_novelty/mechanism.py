"""
机制模板开放集声明（v3.1 精确化）。

关键设计：
1. 机制模板是"声明结构"，不是"机制白名单"。
2. LLM 必须声明 mechanism_family + failure_precondition，
   但 mechanism_family 可以是"其他"或自由文本。
3. 新机制声明会被记录到事件流，供人工评审是否纳入已知族。
4. 已枚举机制族仅用于"提示"，不用于"拒绝"——声明"其他"不会被拒。
"""
from dataclasses import dataclass, field
from typing import Optional, List
from core.event_log import EventLog, GLOBAL_EVENT_LOG


# 已知机制族（用于提示，不用于拒绝）
KNOWN_MECHANISM_FAMILIES = [
    "momentum",       # 动量延续
    "value",          # 价值回归
    "volatility",     # 低波动溢价
    "liquidity",      # 流动性溢价
    "behavioral",     # 行为偏差
    "microstructure", # 微观结构
    "cross_asset",    # 跨资产传导
]


@dataclass(frozen=True)
class MechanismDeclaration:
    """
    机制声明。frozen=True，声明后不可修改。

    字段：
    - mechanism_family：机制族（可以是已知族、"其他"、或自由文本）
    - failure_precondition：失效前提（机制在什么条件下失效）
    - economic_rationale：经济学依据
    """
    mechanism_family: str
    failure_precondition: str
    economic_rationale: str
    is_novel_mechanism: bool = False  # 由构造函数自动判定

    def __post_init__(self):
        # frozen=True 时，用 object.__setattr__ 绕过
        is_novel = self.mechanism_family not in KNOWN_MECHANISM_FAMILIES
        object.__setattr__(self, "is_novel_mechanism", is_novel)

    def to_dict(self):
        return {
            "mechanism_family": self.mechanism_family,
            "failure_precondition": self.failure_precondition,
            "economic_rationale": self.economic_rationale,
            "is_novel_mechanism": self.is_novel_mechanism,
        }


class MechanismValidator:
    """
    机制声明校验器。

    只校验"声明完整性"，不校验"机制是否在已知族内"。
    新机制声明会被记录，供人工评审。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG

    def validate(self, declaration: MechanismDeclaration) -> tuple:
        """
        返回 (ok, reason)。
        """
        if not declaration.mechanism_family or not declaration.mechanism_family.strip():
            return False, "mechanism_family 不能为空"
        if not declaration.failure_precondition or \
           not declaration.failure_precondition.strip():
            return False, "failure_precondition 不能为空（必须先声明失效前提）"
        if not declaration.economic_rationale or \
           not declaration.economic_rationale.strip():
            return False, "economic_rationale 不能为空"

        # 新机制声明进事件流（不拒绝，只记录）
        if declaration.is_novel_mechanism:
            self.event_log.append("novel_mechanism_declared", {
                "mechanism_family": declaration.mechanism_family,
                "failure_precondition": declaration.failure_precondition,
                "note": "新机制声明，待人工评审是否纳入已知族",
            })
        return True, "ok"

    @staticmethod
    def suggest_family(free_text: str) -> str:
        """
        根据自由文本建议机制族（供 LLM 参考，不强制）。
        """
        text = free_text.lower()
        mapping = {
            "momentum": ["动量", "趋势", "延续", "momentum"],
            "value": ["价值", "估值", "回归", "value"],
            "volatility": ["波动", "风险", "volatility"],
            "liquidity": ["流动性", "换手", "liquidity"],
            "behavioral": ["行为", "情绪", "过度反应", "behavioral"],
            "microstructure": ["微观", "订单", "价差", "microstructure"],
            "cross_asset": ["跨资产", "传导", "联动", "cross"],
        }
        for family, keywords in mapping.items():
            if any(kw in text for kw in keywords):
                return family
        return "其他"