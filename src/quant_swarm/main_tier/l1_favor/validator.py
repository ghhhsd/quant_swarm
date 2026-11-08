"""
异构 LLM 校验器（v3.1 修正③）。

三阶段拆解：
1. 机制是否成立（经济学上是否有依据）
2. 数据是否可观测（与 facts.py 有重叠，但 LLM 可从语义层面补充）
3. 是否存在已知反例（伪逻辑过滤）

异构纪律：
- 校验器与编码 Agent 必须不同源/不同温度。
- 校验器 temperature 应显著低于编码 Agent（编码 0.3，校验 0.1）。
- 同源时构造器发出警告，但不硬失败（弱独立性，必要不充分）。
- LLM 校验的结论必须与 facts.py 的非 LLM 校验组合使用——
  单独 LLM 校验通过不足以判定 L1 通过。
"""
import json
from dataclasses import dataclass
from typing import Optional, List

from main_tier.l1_favor.narrative import Narrative
from main_tier.l2_agents.llm_client import LLMClient, LLMCallRecord
from main_tier.l2_agents.base_agent import BaseAgent
from main_tier.l0_governance import GovernanceLayer


# 校验器输出 schema
VALIDATOR_SCHEMA = {
    "type": "object",
    "required": ["mechanism_valid", "data_observable", "counterexamples", "verdict"],
    "properties": {
        "mechanism_valid": {"type": "boolean"},
        "data_observable": {"type": "boolean"},
        "counterexamples": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "reason": {"type": "string"},
    },
}


@dataclass
class ValidatorVerdict:
    """异构 LLM 校验结果。"""
    llm_pass: bool
    mechanism_valid: bool
    data_observable: bool
    counterexamples: List[str]
    reason: str
    llm_call_hash: str


class HeterogeneousValidator(BaseAgent):
    """
    异构校验器 Agent。

    与 CodegenAgent 的异构性通过以下方式保证：
    1. 注入不同的 LLMClient 实例（构造时由 network 或外部注入）。
    2. temperature=0.1（显著低于 codegen 的 0.3）。
    3. role="validator"，与 codegen 的 role 不同。

    构造时若检测到与 codegen 同实例，发出警告（弱独立性）。
    """

    role = "validator"
    max_tokens = 768
    default_temperature = 0.1

    def __init__(self, llm: LLMClient, governance: GovernanceLayer,
                 codegen_llm: Optional[LLMClient] = None):
        super().__init__(llm, governance)
        if codegen_llm is not None and llm is codegen_llm:
            print("[WARN] validator 与 codegen 共用同一 LLMClient 实例，"
                  "存在同源自证风险（v3.1 修正③：弱独立性，必要不充分）")

    def run(self, narrative: Narrative, code: str) -> ValidatorVerdict:
        prompt = self._build_prompt(narrative, code)
        record = self._call_llm(prompt, temperature=self.default_temperature)
        obj = self._parse_json_output(record.output, VALIDATOR_SCHEMA)
        return ValidatorVerdict(
            llm_pass=(obj["verdict"] == "pass"),
            mechanism_valid=bool(obj["mechanism_valid"]),
            data_observable=bool(obj["data_observable"]),
            counterexamples=obj["counterexamples"],
            reason=obj.get("reason", ""),
            llm_call_hash=record.content_hash(),
        )

    def _build_prompt(self, narrative: Narrative, code: str) -> str:
        return (
            f"你是金融经济学逻辑校验器。请对以下 Alpha 假设做三阶段拆解：\n"
            f"机制族：{narrative.mechanism_family}\n"
            f"预测方向：{narrative.direction.value}\n"
            f"生效条件：{narrative.conditions}\n"
            f"预期衰减：{narrative.decay.value}\n"
            f"经济学机制：{narrative.economic_logic}\n"
            f"可证伪陈述：{narrative.falsifiable_statement}\n"
            f"表达式：{code}\n"
            f"请按以下三阶段判断：\n"
            f"1. mechanism_valid：经济学机制是否有依据？\n"
            f"2. data_observable：表达式是否只使用可观测数据？\n"
            f"3. counterexamples：是否存在已知反例（伪逻辑）？\n"
            f"输出 JSON：{{mechanism_valid, data_observable, counterexamples, "
            f"verdict, reason}}"
        )