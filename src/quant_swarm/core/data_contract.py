"""
数据契约（v3.1 修正⑥：唯一真相源）。

关键设计：
1. 数据契约是"结构化对象"，不是"普通 dict"。
2. 显式声明：
   - 字段名 + 类型
   - 时间索引字段名
   - 资产标识字段名
   - 缺失值处理规则（每字段）
   - 对齐规则（时间索引对齐、资产对齐）
3. 契约本身带版本号 + 内容哈希，用于事件溯源锚定。
4. 所有下游层（DSL 校验、回测、成本模型、组合层）引用此契约，
   不重复定义字段名。

为什么需要结构化：
- 之前的 Dict[str, str] 只说了"字段名 → 类型"，没说"缺失值怎么处理"。
- 静默填充会导致回测偏差，必须显式声明。
"""
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import hashlib
import json


# 缺失值处理策略
MISSING_POLICY_DROP = "drop"           # 删除该行
MISSING_POLICY_FFILL = "ffill"         # 前向填充
MISSING_POLICY_ZERO = "zero"           # 填零
MISSING_POLICY_NAN = "nan"             # 保留 NaN（下游处理）
MISSING_POLICY_INTERPOLATE = "interpolate"  # 线性插值


@dataclass(frozen=True)
class FieldSpec:
    """
    单字段规格。

    - name：字段名
    - dtype：类型（"float64" / "int64" / "date" / "string"）
    - missing_policy：缺失值处理策略
    - min_value / max_value：值域（可选，用于校验）
    - description：字段描述
    """
    name: str
    dtype: str
    missing_policy: str = MISSING_POLICY_NAN
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""

    def to_dict(self):
        return asdict(self)

    def validate_value(self, value: float) -> bool:
        """校验单个值是否在值域内。"""
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


@dataclass(frozen=True)
class DataContract:
    """
    数据契约。

    显式声明：
    - 时间索引字段
    - 资产标识字段
    - 字段列表（每个字段的完整规格）
    - 对齐规则
    """
    contract_id: str
    version: str
    time_index_field: str         # 时间索引字段名（如 "date"）
    asset_id_field: str           # 资产标识字段名（如 "asset"）
    fields: tuple                 # Tuple[FieldSpec, ...]（frozen 要求 tuple）
    alignment_rule: str = "time_asset"  # 对齐规则："time_asset" / "time_only"
    description: str = ""

    def __post_init__(self):
        # 校验：time_index_field 和 asset_id_field 必须在 fields 中
        field_names = {f.name for f in self.fields}
        if self.time_index_field not in field_names:
            raise ValueError(
                f"time_index_field '{self.time_index_field}' 不在 fields 中"
            )
        if self.asset_id_field not in field_names:
            raise ValueError(
                f"asset_id_field '{self.asset_id_field}' 不在 fields 中"
            )
        # 校验：不能有重复字段名
        if len(field_names) != len(self.fields):
            raise ValueError("fields 中有重复字段名")

    def get_field(self, name: str) -> Optional[FieldSpec]:
        for f in self.fields:
            if f.name == name:
                return f
        return None

    def get_field_names(self) -> List[str]:
        return [f.name for f in self.fields]

    def get_data_fields(self) -> List[FieldSpec]:
        """返回非索引字段（即数据字段）。"""
        return [
            f for f in self.fields
            if f.name not in (self.time_index_field, self.asset_id_field)
        ]

    def content_hash(self) -> str:
        """契约内容哈希（用于事件溯源锚定）。"""
        payload = {
            "contract_id": self.contract_id,
            "version": self.version,
            "time_index_field": self.time_index_field,
            "asset_id_field": self.asset_id_field,
            "fields": [f.to_dict() for f in self.fields],
            "alignment_rule": self.alignment_rule,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "contract_id": self.contract_id,
            "version": self.version,
            "time_index_field": self.time_index_field,
            "asset_id_field": self.asset_id_field,
            "fields": [f.to_dict() for f in self.fields],
            "alignment_rule": self.alignment_rule,
            "description": self.description,
            "content_hash": self.content_hash(),
        }


# =============================================================================
# 默认数据契约（系统内置）
# =============================================================================

DEFAULT_DATA_CONTRACT = DataContract(
    contract_id="default_equity_daily",
    version="dc-1.0.0",
    time_index_field="date",
    asset_id_field="asset",
    fields=(
        FieldSpec(
            name="date", dtype="date",
            description="交易日（时间索引）",
        ),
        FieldSpec(
            name="asset", dtype="string",
            description="资产标识（如 A001）",
        ),
        FieldSpec(
            name="returns", dtype="float64",
            missing_policy=MISSING_POLICY_NAN,
            description="日收益率（小数，如 0.01 表示 1%）",
        ),
        FieldSpec(
            name="close", dtype="float64",
            missing_policy=MISSING_POLICY_FFILL,
            min_value=0.0,
            description="收盘价",
        ),
        FieldSpec(
            name="volume", dtype="float64",
            missing_policy=MISSING_POLICY_ZERO,
            min_value=0.0,
            description="成交量（股）",
        ),
        FieldSpec(
            name="turnover", dtype="float64",
            missing_policy=MISSING_POLICY_ZERO,
            min_value=0.0,
            description="换手率（小数）",
        ),
        FieldSpec(
            name="pe_ratio", dtype="float64",
            missing_policy=MISSING_POLICY_NAN,
            description="市盈率（可为负）",
        ),
        FieldSpec(
            name="market_cap", dtype="float64",
            missing_policy=MISSING_POLICY_NAN,
            min_value=0.0,
            description="市值（元）",
        ),
    ),
    alignment_rule="time_asset",
    description="默认股票日频数据契约",
)


# 便捷访问：字段名 → 类型（兼容旧接口）
def get_field_types(contract: DataContract = DEFAULT_DATA_CONTRACT) -> Dict[str, str]:
    """返回 {字段名: 类型}，兼容旧 DATA_CONTRACT 接口。"""
    return {f.name: f.dtype for f in contract.fields}


# 兼容旧代码的 DATA_CONTRACT 常量
DATA_CONTRACT = get_field_types(DEFAULT_DATA_CONTRACT)