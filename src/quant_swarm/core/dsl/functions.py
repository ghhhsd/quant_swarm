"""
函数签名（白名单 + 类型签名）。

关键设计：
1. 每个函数显式声明：参数类型、返回类型。
2. 类型推导依赖此签名表。
3. 新增函数时必须同时更新签名表 + AST 白名单。
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class FunctionSignature:
    """
    函数签名。

    - name：函数名
    - param_types：参数类型列表（"float64" / "int64" / "bool" / "any"）
    - return_type：返回类型
    - variadic：是否可变参数（如 clip(x, lo, hi) 支持多参数）
    - description：函数描述
    """
    name: str
    param_types: Tuple[str, ...]
    return_type: str
    variadic: bool = False
    description: str = ""


# 函数签名表（与 AST 白名单保持一致）
FUNCTION_SIGNATURES: Dict[str, FunctionSignature] = {
    # 时间序列函数（输入 float64 矩阵，输出 float64 矩阵）
    "ts_mean": FunctionSignature(
        "ts_mean", ("float64", "int64"), "float64",
        description="滚动均值",
    ),
    "ts_std": FunctionSignature(
        "ts_std", ("float64", "int64"), "float64",
        description="滚动标准差",
    ),
    "ts_sum": FunctionSignature(
        "ts_sum", ("float64", "int64"), "float64",
        description="滚动求和",
    ),
    "ts_max": FunctionSignature(
        "ts_max", ("float64", "int64"), "float64",
        description="滚动最大值",
    ),
    "ts_min": FunctionSignature(
        "ts_min", ("float64", "int64"), "float64",
        description="滚动最小值",
    ),
    "ts_rank": FunctionSignature(
        "ts_rank", ("float64", "int64"), "float64",
        description="滚动排名",
    ),
    # 一元数学函数
    "abs": FunctionSignature(
        "abs", ("float64",), "float64",
        description="绝对值",
    ),
    "log": FunctionSignature(
        "log", ("float64",), "float64",
        description="对数",
    ),
    "sqrt": FunctionSignature(
        "sqrt", ("float64",), "float64",
        description="平方根",
    ),
    "sign": FunctionSignature(
        "sign", ("float64",), "float64",
        description="符号函数",
    ),
    # 二元裁剪
    "clip": FunctionSignature(
        "clip", ("float64", "float64", "float64"), "float64",
        description="裁剪到 [lo, hi]",
    ),
    # 截面函数
    "rank": FunctionSignature(
        "rank", ("float64",), "float64",
        description="截面排名",
    ),
    "zscore": FunctionSignature(
        "zscore", ("float64",), "float64",
        description="截面 zscore",
    ),
}


# AST 白名单（与签名表保持一致）
ALLOWED_FUNCS = set(FUNCTION_SIGNATURES.keys())


# AST 节点白名单
ALLOWED_NODES = {
    "Expression", "BinOp", "UnaryOp", "Call", "Name", "Load",
    "Constant", "Add", "Sub", "Mult", "Div", "USub", "UAdd",
    "Compare", "Lt", "Gt", "LtE", "GtE", "Eq", "NotEq",
    "BoolOp", "And", "Or", "IfExp",
}


# 二元操作的类型规则
BINOP_TYPE_RULES = {
    "Add": {
        ("float64", "float64"): "float64",
        ("float64", "int64"): "float64",
        ("int64", "float64"): "float64",
        ("int64", "int64"): "int64",
    },
    "Sub": {
        ("float64", "float64"): "float64",
        ("float64", "int64"): "float64",
        ("int64", "float64"): "float64",
        ("int64", "int64"): "int64",
    },
    "Mult": {
        ("float64", "float64"): "float64",
        ("float64", "int64"): "float64",
        ("int64", "float64"): "float64",
        ("int64", "int64"): "int64",
    },
    "Div": {
        ("float64", "float64"): "float64",
        ("float64", "int64"): "float64",
        ("int64", "float64"): "float64",
        ("int64", "int64"): "float64",  # 整数除法返回 float
    },
}


# 比较操作的类型规则
COMPARE_TYPE_RULES = {
    "Lt": {("float64", "float64"): "bool", ("int64", "int64"): "bool",
           ("float64", "int64"): "bool", ("int64", "float64"): "bool"},
    "Gt": {("float64", "float64"): "bool", ("int64", "int64"): "bool",
           ("float64", "int64"): "bool", ("int64", "float64"): "bool"},
    "LtE": {("float64", "float64"): "bool", ("int64", "int64"): "bool",
            ("float64", "int64"): "bool", ("int64", "float64"): "bool"},
    "GtE": {("float64", "float64"): "bool", ("int64", "int64"): "bool",
            ("float64", "int64"): "bool", ("int64", "float64"): "bool"},
    "Eq": {("float64", "float64"): "bool", ("int64", "int64"): "bool",
           ("float64", "int64"): "bool", ("int64", "float64"): "bool"},
    "NotEq": {("float64", "float64"): "bool", ("int64", "int64"): "bool",
              ("float64", "int64"): "bool", ("int64", "float64"): "bool"},
}