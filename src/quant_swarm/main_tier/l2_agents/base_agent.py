"""
Agent 基类：统一治理 + 调用记录 + 失败分类。

所有 Agent 继承此类，保证：
1. 每次 LLM 调用前走 governance.consume_token()。
2. 每次调用记录进 self.calls，最终进 L6 事件流。
3. 失败抛 AgentFailure，带 FailureType，供 L1 分流。
4. 输出必须通过 schema 校验。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

from main_tier.l2_agents.llm_client import LLMClient, LLMCallRecord
from main_tier.l2_agents.schema import validate_schema
from main_tier.l0_governance import GovernanceLayer, BudgetExceeded
from main_tier.l1_favor import FailureType


class AgentFailure(Exception):
    """Agent 失败异常，带 FailureType 供 L1 分流。"""

    def __init__(self, failure_type: FailureType, message: str):
        super().__init__(message)
        self.failure_type = failure_type
        self.message = message


class BaseAgent(ABC):
    """Agent 基类。"""

    role: str = "base"
    default_temperature: float = 0.7
    max_tokens: int = 1024

    def __init__(self, llm: LLMClient, governance: GovernanceLayer):
        self.llm = llm
        self.gov = governance
        self.calls: List[LLMCallRecord] = []

    # ---- LLM 调用包装（统一治理 + 记录）--------------------------------
    def _call_llm(self, prompt: str, *, temperature: Optional[float] = None,
                  seed: Optional[int] = None) -> LLMCallRecord:
        temp = temperature if temperature is not None else self.default_temperature
        # 估算 Token 消耗（保守上界：prompt 长度 / 2）
        estimated_tokens = max(1, len(prompt) // 2 + self.max_tokens)
        try:
            self.gov.consume_token(estimated_tokens)
        except BudgetExceeded as e:
            raise AgentFailure(FailureType.DATA, f"Token 配额不足: {e}")

        try:
            record = self.llm.complete(
                prompt, role=self.role, temperature=temp,
                max_tokens=self.max_tokens, seed=seed,
            )
        except Exception as e:
            # LLM 调用失败归为 LOGIC（叙事无法生成）
            raise AgentFailure(FailureType.LOGIC, f"LLM 调用失败: {e}")

        self.calls.append(record)
        return record

    # ---- 结构化输出解析 + schema 校验 -----------------------------------
    @staticmethod
    def _parse_json_output(output: str, schema: dict) -> Dict[str, Any]:
        try:
            obj = json.loads(output)
        except json.JSONDecodeError as e:
            raise AgentFailure(FailureType.SYNTAX, f"LLM 输出非合法 JSON: {e}")
        ok, reason = validate_schema(obj, schema)
        if not ok:
            raise AgentFailure(FailureType.SYNTAX, f"schema 校验失败: {reason}")
        return obj

    # ---- 调用记录哈希（供内容哈希）--------------------------------------
    def call_hashes(self) -> List[str]:
        return [c.content_hash() for c in self.calls]

    # ---- 重置（每轮复用 Agent 实例时调用）-------------------------------
    def reset_calls(self) -> None:
        self.calls.clear()

    @abstractmethod
    def run(self, *args, **kwargs):
        ...