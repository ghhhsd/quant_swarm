"""
core/data.py — 数据层统一入口（重写）。

re-export 数据契约、合成数据、真实数据源的所有公开接口。
所有下游层通过此文件访问数据层，不直接 import 子模块。
"""
from core.data_contract import (
    DataContract, FieldSpec,
    DEFAULT_DATA_CONTRACT, DATA_CONTRACT, get_field_types,
    MISSING_POLICY_DROP, MISSING_POLICY_FFILL, MISSING_POLICY_ZERO,
    MISSING_POLICY_NAN, MISSING_POLICY_INTERPOLATE,
)
from core.synthetic_data import (
    SyntheticDataGenerator, SyntheticDataConfig,
    SignalConfig, RegimeConfig,
    generate_synthetic_data,
)
from core.data_source import (
    DataSource, DataSourceError, DataValidationError,
    DataVersion, DataSourceRegistry,
    CSVDataSource, APIDataSource,
)

__all__ = [
    # 契约
    "DataContract", "FieldSpec",
    "DEFAULT_DATA_CONTRACT", "DATA_CONTRACT", "get_field_types",
    "MISSING_POLICY_DROP", "MISSING_POLICY_FFILL", "MISSING_POLICY_ZERO",
    "MISSING_POLICY_NAN", "MISSING_POLICY_INTERPOLATE",
    # 合成数据
    "SyntheticDataGenerator", "SyntheticDataConfig",
    "SignalConfig", "RegimeConfig",
    "generate_synthetic_data",
    # 真实数据源
    "DataSource", "DataSourceError", "DataValidationError",
    "DataVersion", "DataSourceRegistry",
    "CSVDataSource", "APIDataSource",
]