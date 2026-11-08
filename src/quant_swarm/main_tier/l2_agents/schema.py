"""
所有 Agent 输出的 JSON Schema。
v3.1 要求全流程结构化报告，因此每个 Agent 输出必须通过 schema 校验。
"""
ANALYST_REPORT_SCHEMA = {
    "type": "object",
    "required": ["analyst", "view", "confidence", "evidence"],
    "properties": {
        "analyst": {"type": "string", "enum": ["technical", "fundamental"]},
        "view": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
}

DEBATE_TURN_SCHEMA = {
    "type": "object",
    "required": ["round_idx", "side", "argument", "strength"],
    "properties": {
        "round_idx": {"type": "integer", "minimum": 0},
        "side": {"type": "string", "enum": ["bull", "bear"]},
        "argument": {"type": "string"},
        "strength": {"type": "number", "minimum": 0, "maximum": 1},
        "rebuttal_target": {"type": "string"},
    },
}

DEBATE_SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["n_rounds", "final_divergence", "converged", "bull_points", "bear_points"],
    "properties": {
        "n_rounds": {"type": "integer", "minimum": 1},
        "final_divergence": {"type": "number", "minimum": 0, "maximum": 1},
        "converged": {"type": "boolean"},
        "bull_points": {"type": "array", "items": {"type": "string"}},
        "bear_points": {"type": "array", "items": {"type": "string"}},
    },
}

CODEGEN_REPORT_SCHEMA = {
    "type": "object",
    "required": ["hid", "expression", "family", "rationale", "ast_depth",
                 "narrative_hash", "llm_call_hashes"],
    "properties": {
        "hid": {"type": "string"},
        "expression": {"type": "string"},
        "family": {"type": "string"},
        "rationale": {"type": "string"},
        "ast_depth": {"type": "integer", "minimum": 1},
        "narrative_hash": {"type": "string"},
        "llm_call_hashes": {"type": "array", "items": {"type": "string"}},
    },
}


def validate_schema(obj: dict, schema: dict) -> tuple[bool, str]:
    """轻量 schema 校验（不引入 jsonschema 依赖，仅校验 required + type + enum）。"""
    for key in schema.get("required", []):
        if key not in obj:
            return False, f"缺少必填字段: {key}"
    for key, prop in schema.get("properties", {}).items():
        if key not in obj:
            continue
        val = obj[key]
        expected_type = prop.get("type")
        if expected_type == "string" and not isinstance(val, str):
            return False, f"{key} 应为 string"
        if expected_type == "number" and not isinstance(val, (int, float)):
            return False, f"{key} 应为 number"
        if expected_type == "integer" and not isinstance(val, int):
            return False, f"{key} 应为 integer"
        if expected_type == "boolean" and not isinstance(val, bool):
            return False, f"{key} 应为 boolean"
        if expected_type == "array" and not isinstance(val, list):
            return False, f"{key} 应为 array"
        if "enum" in prop and val not in prop["enum"]:
            return False, f"{key} 取值不在枚举内: {val}"
        if "minimum" in prop and val < prop["minimum"]:
            return False, f"{key} 小于最小值"
        if "maximum" in prop and val > prop["maximum"]:
            return False, f"{key} 大于最大值"
    return True, "ok"