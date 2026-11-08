"""
非 LLM 事实校验（v3.1 精确化核心）。

v3.1 说"校验器异构化是弱独立性，真正事实校验需引入非 LLM 约束"，
但没说清约束是什么。本模块落地三类硬约束：

1. 数据可观测性：假设引用的字段/窗口必须在数据契约内。
2. 经济先验：声明方向与机制族的先验方向一致性。
3. 机制-表达式对齐：叙事声明的机制族必须与表达式实际使用的函数/字段一致。

这三类约束是规则化、可枚举、不依赖 LLM 判断的，构成 L1 的"事实层"。
它们独立于 LLM 校验器，即使 LLM 校验通过，事实层不通过也判 L1 失败。
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
import ast
import re

from main_tier.l1_favor.narrative import Narrative, Direction
from core.data import DATA_CONTRACT
from core.dsl import ALLOWED_FUNCS


@dataclass
class FactCheckResult:
    ok: bool
    constraint: str       # "data_observability" / "economic_prior" / "mechanism_alignment"
    reason: str


# 经济先验：机制族 → 允许/禁止的方向
# 来源：金融经济学基本共识（动量正向、价值正向、波动负向、流动性负向）
ECONOMIC_PRIORS: Dict[str, Dict[str, any]] = {
    "momentum": {
        "allowed": {Direction.LONG, Direction.NEUTRAL},
        "forbidden": {Direction.SHORT},
        "note": "动量族通常正向下期收益，声明负向需额外论证",
    },
    "value": {
        "allowed": {Direction.LONG, Direction.NEUTRAL},
        "forbidden": {Direction.SHORT},
        "note": "价值族通常正向下期收益，声明负向需额外论证",
    },
    "volatility": {
        "allowed": {Direction.SHORT, Direction.NEUTRAL},
        "forbidden": {Direction.LONG},
        "note": "低波动溢价通常表现为波动负向，声明正向需额外论证",
    },
    "liquidity": {
        "allowed": {Direction.SHORT, Direction.NEUTRAL},
        "forbidden": {Direction.LONG},
        "note": "流动性溢价通常表现为换手负向，声明正向需额外论证",
    },
    "other": {
        "allowed": {Direction.LONG, Direction.SHORT, Direction.NEUTRAL},
        "forbidden": set(),
        "note": "未分类机制无先验约束",
    },
}


class FactChecker:
    """
    非 LLM 事实校验器。
    """

    def __init__(self, data_contract: Dict[str, str] = None,
                 allowed_funcs: Set[str] = None):
        self.data_contract = data_contract or DATA_CONTRACT
        self.allowed_funcs = allowed_funcs or ALLOWED_FUNCS

    # ---- 约束 1：数据可观测性 -------------------------------------------
    def check_data_observability(self, narrative: Narrative,
                                 code: str) -> FactCheckResult:
        """
        表达式引用的字段必须在 DATA_CONTRACT 中；
        表达式中的窗口参数不得超过数据覆盖的合理范围（这里用 500 作硬上限）。
        """
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError as e:
            return FactCheckResult(False, "data_observability", f"语法错误: {e}")

        used_fields = set()
        used_windows = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in self.data_contract:
                used_fields.add(node.id)
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                if node.value > 10:  # 粗判：>10 的整数视为窗口参数
                    used_windows.append(node.value)

        if not used_fields:
            return FactCheckResult(
                False, "data_observability",
                "表达式未引用任何数据契约内字段",
            )
        if used_windows and max(used_windows) > 500:
            return FactCheckResult(
                False, "data_observability",
                f"窗口参数 {max(used_windows)} 超过合理上限 500",
            )
        return FactCheckResult(True, "data_observability", "ok")

    # ---- 约束 2：经济先验 ----------------------------------------------
    def check_economic_prior(self, narrative: Narrative) -> FactCheckResult:
        family = narrative.mechanism_family
        prior = ECONOMIC_PRIORS.get(family)
        if prior is None:
            return FactCheckResult(False, "economic_prior",
                                   f"未知机制族: {family}")
        if narrative.direction in prior["forbidden"]:
            return FactCheckResult(
                False, "economic_prior",
                f"机制族 {family} 声明方向 {narrative.direction.value} "
                f"与先验冲突（{prior['note']}）",
            )
        return FactCheckResult(True, "economic_prior", "ok")

    # ---- 约束 3：机制-表达式对齐 ---------------------------------------
    def check_mechanism_alignment(self, narrative: Narrative,
                                  code: str) -> FactCheckResult:
        """
        叙事声明的机制族必须与表达式实际使用的函数/字段一致。
        例：声明 liquidity 但表达式只用 returns，判不一致。
        """
        # 机制族 → 必须出现的字段/函数关键词
        family_keywords = {
            "momentum": {"returns", "close"},
            "value": {"pe_ratio"},
            "volatility": {"returns", "close"},
            "liquidity": {"turnover", "volume"},
            "other": set(),  # 无约束
        }
        required = family_keywords.get(narrative.mechanism_family, set())
        if not required:
            return FactCheckResult(True, "mechanism_alignment", "无约束")
        # 提取表达式中的字段/函数 token
        tokens = set(re.findall(r"[a-zA-Z_]+", code))
        if not (tokens & required):
            return FactCheckResult(
                False, "mechanism_alignment",
                f"机制族 {narrative.mechanism_family} 要求使用 {required}，"
                f"但表达式未出现对应字段",
            )
        return FactCheckResult(True, "mechanism_alignment", "ok")

    # ---- 全部约束 ------------------------------------------------------
    def check_all(self, narrative: Narrative, code: str) -> List[FactCheckResult]:
        return [
            self.check_data_observability(narrative, code),
            self.check_economic_prior(narrative),
            self.check_mechanism_alignment(narrative, code),
        ]