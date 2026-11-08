"""
分析师 Agent：技术面 / 基本面。
每个分析师独立 LLM 调用，输出结构化报告。
"""
from dataclasses import dataclass
from typing import Dict, Any

from main_tier.l2_agents.base_agent import BaseAgent
from main_tier.l2_agents.schema import ANALYST_REPORT_SCHEMA
from main_tier.l1_favor import Hypothesis


@dataclass
class AnalystReport:
    analyst: str
    view: str
    confidence: float
    evidence: list
    llm_call_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analyst": self.analyst,
            "view": self.view,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


class AnalystAgent(BaseAgent):
    """分析师基类，technical / fundamental 共用。"""

    role = "analyst"
    max_tokens = 512

    def __init__(self, llm, governance, specialty: str):
        super().__init__(llm, governance)
        self.specialty = specialty  # "technical" / "fundamental"
        self.role = f"analyst_{specialty}"

    def run(self, hyp: Hypothesis) -> AnalystReport:
        prompt = self._build_prompt(hyp)
        record = self._call_llm(prompt, temperature=0.5)
        obj = self._parse_json_output(record.output, ANALYST_REPORT_SCHEMA)
        obj["analyst"] = self.specialty
        return AnalystReport(
            analyst=self.specialty,
            view=obj["view"],
            confidence=float(obj["confidence"]),
            evidence=obj["evidence"],
            llm_call_hash=record.content_hash(),
        )

    def _build_prompt(self, hyp: Hypothesis) -> str:
        return (
            f"你是{self.specialty}分析师。请评估以下 Alpha 假设：\n"
            f"假设：{hyp.raw_text}\n"
            f"机制：{hyp.economic_logic or '未提供'}\n"
            f"可证伪条件：{hyp.falsifiable_chain or '未提供'}\n"
            f"请输出 JSON：{{view, confidence, evidence}}"
        )