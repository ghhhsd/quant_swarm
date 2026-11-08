"""
内容哈希（v3.1 修正⑥ + 路 A）。

内容哈希 = hash(AST + 数据契约 + 门控参数 + LLM 调用记录 hash)。

关键设计：
1. LLM 调用记录 hash 是路 A 的核心——它让"重放"不需要重跑 LLM。
2. 哈希是确定性的：同样的 AST + 数据契约 + 门控参数 + LLM 记录，
   永远产生同一个 hash。
3. 门控参数从 L0 的 CommittedThreshold 取，不是硬编码。

为什么要把 LLM 调用记录纳入哈希：
- v3.0 的事件溯源假设"确定性投影"，但 LLM 输出非确定。
- 路 A 的解法：LLM 调用结果本身作为事件流的一部分（记录完整 prompt/output/seed），
  投影重放时只重放已记录的输出，不重新调 LLM。
- 这样事件流是确定的，代价是存储膨胀。
"""
import hashlib
import json
from typing import Dict, Any, List, Optional

from core.data import DATA_CONTRACT
from main_tier.l2_agents.llm_client import LLMCallRecord


def llm_calls_hash(calls: List[LLMCallRecord]) -> str:
    """
    计算 LLM 调用记录的哈希。
    按调用顺序拼接每个调用的 content_hash，再做整体 hash。
    """
    if not calls:
        return ""
    combined = "|".join(c.content_hash() for c in calls)
    return hashlib.sha256(combined.encode()).hexdigest()


def compute_skill_hash(code: str,
                       gate_params: Dict[str, Any],
                       llm_calls: List[LLMCallRecord],
                       data_contract: Optional[Dict[str, str]] = None) -> str:
    """
    技能内容哈希。

    参数：
    - code：AST 表达式
    - gate_params：门控参数（来自 L0 committed threshold 的 to_dict）
    - llm_calls：该技能的完整 LLM 调用记录（路 A）
    - data_contract：数据契约（默认用 DATA_CONTRACT）
    """
    dc = data_contract or DATA_CONTRACT
    payload = {
        "ast": code,
        "data_contract": dc,
        "gate_params": gate_params,
        "llm_calls_hash": llm_calls_hash(llm_calls),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()