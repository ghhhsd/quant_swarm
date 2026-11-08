"""
LLM 客户端抽象。

设计要点：
1. 所有 Agent 通过此接口调用 LLM，不直接依赖 SDK。
2. 异构校验靠"给不同 Agent 注入不同 client 实例"实现，而非硬编码模型名。
3. 每次调用返回完整记录（prompt/model/temperature/output/tokens），
   供 L6 事件溯源（路 A）使用。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import hashlib
import json


@dataclass
class LLMCallRecord:
    """单次 LLM 调用的完整记录（路 A：写入事件流，物化视图不重跑）。"""
    agent_role: str            # analyst_technical / analyst_fundamental / debate_bull / ...
    model: str
    temperature: float
    prompt: str
    output: str
    tokens_in: int
    tokens_out: int
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_role": self.agent_role,
            "model": self.model,
            "temperature": self.temperature,
            "prompt": self.prompt,
            "output": self.output,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "seed": self.seed,
        }

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()


class LLMClient(ABC):
    """LLM 客户端抽象。"""

    @abstractmethod
    def complete(self, prompt: str, *, role: str,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 seed: Optional[int] = None) -> LLMCallRecord:
        """返回完整调用记录。"""


class StubLLMClient(LLMClient):
    """
    零成本桩客户端：用确定性规则模拟 LLM 输出。
    测试与合成数据跑通时使用；真实部署替换为 OpenAIClient 等。
    """

    def __init__(self, model: str = "stub", seed: int = 42):
        self.model = model
        self._seed = seed
        self._counter = 0

    def complete(self, prompt: str, *, role: str,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 seed: Optional[int] = None) -> LLMCallRecord:
        self._counter += 1
        s = seed if seed is not None else self._seed
        h = int(hashlib.md5(f"{prompt}{role}{s}{self._counter}".encode()).hexdigest(), 16)

        # 按角色生成确定性输出（真实 LLM 的输出格式在此模拟）
        if role.startswith("analyst"):
            output = json.dumps({
                "view": "bullish" if h % 3 != 0 else "bearish",
                "confidence": round((h % 100) / 100, 2),
                "evidence": [f"evidence_{i}" for i in range(1 + h % 3)],
            }, ensure_ascii=False)
        elif role.startswith("debate"):
            side = "bull" if "bull" in role else "bear"
            output = json.dumps({
                "argument": f"{side} argument based on {h % 7}",
                "strength": round((h % 100) / 100, 2),
                "rebuttal_target": f"point_{h % 5}",
            }, ensure_ascii=False)
        elif role == "codegen":
            # 从 prompt 中提取 family 关键词，生成受限 DSL 表达式
            family = "momentum"
            for f in ("momentum", "volatility", "value", "liquidity"):
                if f in prompt:
                    family = f
                    break
            templates = {
                "momentum": ["ts_mean(returns, 20) - ts_mean(returns, 60)",
                             "ts_mean(returns, 5) - ts_mean(returns, 20)"],
                "volatility": ["ts_std(returns, 20)", "ts_std(returns, 60)"],
                "value": ["1.0 / pe_ratio", "zscore(1.0 / pe_ratio)"],
                "liquidity": ["ts_mean(turnover, 20)", "ts_mean(turnover, 60)"],
            }
            expr = templates[family][h % len(templates[family])]
            output = json.dumps({
                "expression": expr,
                "family": family,
                "rationale": f"{family} mechanism rationale",
            }, ensure_ascii=False)
        else:
            output = json.dumps({"raw": "unknown role"}, ensure_ascii=False)

        return LLMCallRecord(
            agent_role=role,
            model=self.model,
            temperature=temperature,
            prompt=prompt,
            output=output,
            tokens_in=max(1, len(prompt) // 4),
            tokens_out=max(1, len(output) // 4),
            seed=s,
        )


class OpenAIClient(LLMClient):
    """
    真实 OpenAI 客户端桩（接入时实现）。
    注意：异构校验要求此 client 与 validator 的 client 不同源/不同温度。
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str, *, role: str,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 seed: Optional[int] = None) -> LLMCallRecord:
        # TODO: 接入真实 SDK。以下为接口签名占位。
        raise NotImplementedError(
            "接入真实 LLM 时实现：调用 SDK，返回 LLMCallRecord"
        )