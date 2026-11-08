"""
异质信号注入（v3.1 修正②）。

关键设计：
1. 假设生成有多个来源：LLM / 规则模板 / 残差驱动 / 另类数据字段。
2. 非 LLM 来源的假设与 LLM 假设并列输入 C1.5。
3. 来源标签记录到事件流，供消融比较"LLM vs 非 LLM"的边际贡献。
4. 异质信号的目的：对抗纯 LLM 假设的同质化坍缩。
"""
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import random
import re


class HypothesisSource(str, Enum):
    LLM = "llm"
    RULE_TEMPLATE = "rule_template"
    RESIDUAL_DRIVEN = "residual_driven"
    ALTERNATIVE_DATA = "alternative_data"


@dataclass
class HeterogeneousHypothesis:
    """异质假设。"""
    hid: str
    family: str
    raw_text: str
    code: str
    source: HypothesisSource
    mechanism_family: str
    failure_precondition: str
    economic_rationale: str

    def to_dict(self):
        return {
            "hid": self.hid, "family": self.family,
            "raw_text": self.raw_text, "code": self.code,
            "source": self.source.value,
            "mechanism_family": self.mechanism_family,
            "failure_precondition": self.failure_precondition,
            "economic_rationale": self.economic_rationale,
        }


class HeterogeneousGenerator:
    """
    异质假设生成器。

    真实部署应接入：
    - 规则模板库（人工编写的因子模板）
    - 残差驱动（对已知因子回归后取残差，残差驱动新因子）
    - 另类数据字段（如新闻情绪、卫星图像等，需数据源支持）

    本实现用桩模拟这些来源。
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._counter = 0

    def _next_hid(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:04d}"

    # ---- LLM 来源（桩：模拟 LLM 生成）---------------------------------
    def generate_llm(self, n: int) -> List[HeterogeneousHypothesis]:
        """LLM 来源（桩）。"""
        templates = [
            ("momentum", "ts_mean(returns, 20) - ts_mean(returns, 60)",
             "动量延续", "高波动期动量失效"),
            ("volatility", "ts_std(returns, 20)",
             "低波动溢价", "流动性枯竭期失效"),
            ("value", "1.0 / pe_ratio",
             "价值回归", "成长风格主导期失效"),
        ]
        out = []
        for _ in range(n):
            fam, code, rationale, precond = self.rng.choice(templates)
            out.append(HeterogeneousHypothesis(
                hid=self._next_hid("llm"),
                family=fam,
                raw_text=f"[LLM] {rationale}",
                code=code,
                source=HypothesisSource.LLM,
                mechanism_family=fam,
                failure_precondition=precond,
                economic_rationale=rationale,
            ))
        return out

    # ---- 规则模板来源（非 LLM）-----------------------------------------
    def generate_rule_template(self, n: int) -> List[HeterogeneousHypothesis]:
        """规则模板来源：人工编写的因子模板。"""
        templates = [
            ("momentum", "ts_mean(returns, 10) / ts_std(returns, 10)",
             "风险调整动量", "极端波动期失效"),
            ("liquidity", "ts_mean(turnover, 10) / ts_mean(turnover, 60)",
             "换手加速", "市场流动性枯竭期失效"),
            ("value", "zscore(1.0 / pe_ratio) + zscore(1.0 / market_cap)",
             "价值+规模复合", "小盘股崩盘期失效"),
        ]
        out = []
        for _ in range(n):
            fam, code, rationale, precond = self.rng.choice(templates)
            out.append(HeterogeneousHypothesis(
                hid=self._next_hid("rule"),
                family=fam,
                raw_text=f"[规则模板] {rationale}",
                code=code,
                source=HypothesisSource.RULE_TEMPLATE,
                mechanism_family=fam,
                failure_precondition=precond,
                economic_rationale=rationale,
            ))
        return out

    # ---- 残差驱动来源（非 LLM）-----------------------------------------
    def generate_residual(self, known_codes: List[str],
                          n: int) -> List[HeterogeneousHypothesis]:
        """
        残差驱动：对已知因子做"正交化"产生新因子。
        桩：简单拼接已知因子 + 反转。
        """
        if not known_codes:
            return []
        out = []
        for _ in range(n):
            base = self.rng.choice(known_codes)
            code = f"({base}) * -1.0"
            out.append(HeterogeneousHypothesis(
                hid=self._next_hid("resid"),
                family="other",
                raw_text=f"[残差驱动] 对 {base} 取反",
                code=code,
                source=HypothesisSource.RESIDUAL_DRIVEN,
                mechanism_family="其他",
                failure_precondition="原因子失效时同时失效",
                economic_rationale="对已知因子的残差暴露",
            ))
        return out

    # ---- 另类数据来源（非 LLM）-----------------------------------------
    def generate_alternative_data(self, n: int) -> List[HeterogeneousHypothesis]:
        """
        另类数据来源。
        桩：用现有字段模拟"另类数据"（真实部署应接入外部数据源）。
        """
        templates = [
            ("other", "ts_mean(volume, 5) / ts_mean(volume, 60)",
             "成交量加速", "流动性危机时失效"),
            ("microstructure", "rank(close) - rank(ts_mean(close, 20))",
             "价格偏离度", "趋势市场失效"),
        ]
        out = []
        for _ in range(n):
            fam, code, rationale, precond = self.rng.choice(templates)
            out.append(HeterogeneousHypothesis(
                hid=self._next_hid("alt"),
                family=fam,
                raw_text=f"[另类数据] {rationale}",
                code=code,
                source=HypothesisSource.ALTERNATIVE_DATA,
                mechanism_family=fam,
                failure_precondition=precond,
                economic_rationale=rationale,
            ))
        return out