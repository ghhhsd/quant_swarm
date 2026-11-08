"""
多空辩论：轮数预算 + 共识收敛判据。

收敛判据：每轮计算 Bull/Bear 的 strength 之差作为分歧度；
分歧度 < convergence_threshold 时提前终止。
轮数上限由 L0 治理层配置提供。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List

from main_tier.l2_agents.base_agent import BaseAgent, AgentFailure
from main_tier.l2_agents.schema import DEBATE_TURN_SCHEMA, DEBATE_SUMMARY_SCHEMA
from main_tier.l1_favor import Hypothesis, FailureType


@dataclass
class DebateTurn:
    round_idx: int
    side: str
    argument: str
    strength: float
    rebuttal_target: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_idx": self.round_idx,
            "side": self.side,
            "argument": self.argument,
            "strength": self.strength,
            "rebuttal_target": self.rebuttal_target,
        }


@dataclass
class DebateSummary:
    n_rounds: int
    final_divergence: float
    converged: bool
    bull_points: List[str] = field(default_factory=list)
    bear_points: List[str] = field(default_factory=list)
    turns: List[DebateTurn] = field(default_factory=list)
    llm_call_hashes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rounds": self.n_rounds,
            "final_divergence": self.final_divergence,
            "converged": self.converged,
            "bull_points": self.bull_points,
            "bear_points": self.bear_points,
        }


class DebateAgent(BaseAgent):
    """
    单方辩手（bull 或 bear）。每轮独立调用。
    与另一个辩手共享历史，但各自 LLM 调用。
    """

    role = "debate"
    max_tokens = 512
    default_temperature = 0.7

    def __init__(self, llm, governance, side: str):
        super().__init__(llm, governance)
        self.side = side  # "bull" / "bear"
        self.role = f"debate_{side}"

    def run(self, hyp: Hypothesis, round_idx: int,
            history: List[DebateTurn]) -> DebateTurn:
        prompt = self._build_prompt(hyp, round_idx, history)
        # 辩手温度略高，鼓励不同角度
        record = self._call_llm(prompt, temperature=0.8)
        obj = self._parse_json_output(record.output, DEBATE_TURN_SCHEMA)
        obj["round_idx"] = round_idx
        obj["side"] = self.side
        return DebateTurn(
            round_idx=round_idx,
            side=self.side,
            argument=obj["argument"],
            strength=float(obj["strength"]),
            rebuttal_target=obj.get("rebuttal_target", ""),
        )

    def _build_prompt(self, hyp: Hypothesis, round_idx: int,
                      history: List[DebateTurn]) -> str:
        hist_text = "\n".join(
            f"[R{t.round_idx} {t.side}] {t.argument}" for t in history[-6:]
        ) or "（首轮）"
        return (
            f"你是{self.side}研究员。请针对以下假设给出第 {round_idx} 轮论点：\n"
            f"假设：{hyp.raw_text}\n"
            f"机制：{hyp.economic_logic or '未提供'}\n"
            f"历史辩论：\n{hist_text}\n"
            f"请输出 JSON：{{argument, strength, rebuttal_target}}"
        )


class DebateSession:
    """
    辩论会话：编排 Bull/Bear 多轮，带轮数预算 + 收敛判据。
    """

    def __init__(self, bull: DebateAgent, bear: DebateAgent,
                 max_rounds: int, convergence_threshold: float):
        self.bull = bull
        self.bear = bear
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold

    def run(self, hyp: Hypothesis) -> DebateSummary:
        turns: List[DebateTurn] = []
        bull_points: List[str] = []
        bear_points: List[str] = []
        final_divergence = 1.0
        converged = False

        for r in range(self.max_rounds):
            # Bull 发言
            b_turn = self.bull.run(hyp, r, turns)
            turns.append(b_turn)
            bull_points.append(b_turn.argument)
            # Bear 发言（看到 Bull 本轮论点后反驳）
            s_turn = self.bear.run(hyp, r, turns)
            turns.append(s_turn)
            bear_points.append(s_turn.argument)
            # 收敛判据：strength 之差
            final_divergence = abs(b_turn.strength - s_turn.strength)
            if final_divergence < self.convergence_threshold:
                converged = True
                break

        return DebateSummary(
            n_rounds=len(turns) // 2,
            final_divergence=final_divergence,
            converged=converged,
            bull_points=bull_points,
            bear_points=bear_points,
            turns=turns,
            llm_call_hashes=self.bull.call_hashes() + self.bear.call_hashes(),
        )