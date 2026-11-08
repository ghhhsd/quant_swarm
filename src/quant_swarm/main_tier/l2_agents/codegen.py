"""
策略研究员编码 Agent。

设计要点：
1. 接收已通过 L1 的 FavorResult（带 narrative_hash），拒绝未预承诺的假设。
2. LLM 输出必须是受限 DSL 表达式，经 AST 检查后才能进沙箱。
3. 与 L1 validator 异构：codegen client 与 validator client 应不同源/不同温度
   （由外部注入不同 LLMClient 实例实现，代码层面不硬编码）。
4. 输出结构化报告，含 narrative_hash + 所有 LLM 调用哈希，供 L6 内容哈希。
"""
import ast
from dataclasses import dataclass
from typing import Dict, Any, List

from main_tier.l2_agents.base_agent import BaseAgent, AgentFailure
from main_tier.l2_agents.schema import CODEGEN_REPORT_SCHEMA
from main_tier.l1_favor import Hypothesis, FavorResult, FailureType
from core.dsl import check_ast


@dataclass
class CodegenReport:
    hid: str
    expression: str
    family: str
    rationale: str
    ast_depth: int
    narrative_hash: str
    llm_call_hashes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hid": self.hid,
            "expression": self.expression,
            "family": self.family,
            "rationale": self.rationale,
            "ast_depth": self.ast_depth,
            "narrative_hash": self.narrative_hash,
            "llm_call_hashes": self.llm_call_hashes,
        }


class CodegenAgent(BaseAgent):
    """
    策略研究员：把假设 + 辩论结论编码成受限 DSL 表达式。
    温度略低（0.3），鼓励确定性输出；与辩手（0.8）形成对比。
    """

    role = "codegen"
    max_tokens = 512
    default_temperature = 0.3

    def run(self, favor: FavorResult,
            debate_summary) -> CodegenReport:
        # 硬约束 1：必须已有叙事预承诺
        hyp = favor.hypothesis
        if not hyp.narrative_hash:
            raise AgentFailure(
                FailureType.LOGIC,
                "叙事未预承诺：L2 拒绝处理未哈希锁定的假设（v3.1 L1 纪律）",
            )
        # 硬约束 2：辩论必须已收敛或达到轮数上限
        if debate_summary.n_rounds < 1:
            raise AgentFailure(FailureType.LOGIC, "辩论轮数为 0，无法编码")

        prompt = self._build_prompt(hyp, debate_summary)
        record = self._call_llm(prompt, temperature=self.default_temperature)
        obj = self._parse_json_output(record.output, CODEGEN_REPORT_SCHEMA)

        # 编码后必须过 AST 检查（防 LLM 输出非法 DSL）
        dsl_result = check_ast(obj["expression"])
        if not dsl_result.ok:
            raise AgentFailure(
                FailureType.SYNTAX,
                f"LLM 生成的表达式未通过 AST 白名单: {dsl_result.reason}",
            )

        return CodegenReport(
            hid=hyp.hid,
            expression=obj["expression"],
            family=obj.get("family", hyp.family),
            rationale=obj["rationale"],
            ast_depth=dsl_result.max_depth,
            narrative_hash=hyp.narrative_hash,
            llm_call_hashes=self.call_hashes(),
        )

    def _build_prompt(self, hyp: Hypothesis, debate_summary) -> str:
        bull = "\n".join(f"- {p}" for p in debate_summary.bull_points[-2:])
        bear = "\n".join(f"- {p}" for p in debate_summary.bear_points[-2:])
        return (
            f"你是量化策略研究员。基于以下信息生成一个受限 DSL 因子表达式：\n"
            f"假设：{hyp.raw_text}\n"
            f"机制族：{hyp.family}\n"
            f"可证伪条件：{hyp.falsifiable_chain}\n"
            f"辩论共识（分歧度 {debate_summary.final_divergence:.2f}）：\n"
            f"多头观点：\n{bull}\n"
            f"空头观点：\n{bear}\n"
            f"可用字段：returns/close/volume/turnover/pe_ratio/market_cap\n"
            f"可用函数：ts_mean/ts_std/ts_sum/ts_max/ts_min/ts_rank/"
            f"abs/log/sqrt/sign/clip/rank/zscore\n"
            f"请输出 JSON：{{expression, family, rationale}}"
        )