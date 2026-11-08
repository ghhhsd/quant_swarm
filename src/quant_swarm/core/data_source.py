"""
真实数据接口桩（v3.1 修正⑥：数据版本管理 + 抽象基类）。

关键设计：
1. 抽象基类 DataSource 显式声明接口：
   - load()：加载数据
   - validate()：按契约校验
   - get_version()：数据版本（用于事件溯源）
   - cache()：缓存管理
2. 给出"接入真实数据时需要实现什么"的清单。
3. 数据版本管理：每次加载产生版本号，用于事件溯源锚定。

真实数据接入清单（实现 DataSource 时需要做）：
- [ ] load()：从数据源（CSV / Parquet / 数据库 / API）读取
- [ ] validate()：按 DataContract 校验字段名、类型、值域、缺失策略
- [ ] get_version()：产生数据版本号（如文件哈希 / API 时间戳 / 数据库快照 ID）
- [ ] cache()：缓存策略（本地缓存 / 内存缓存 / 无缓存）
- [ ] 时间索引对齐：确保 time_index_field 是排序的日期序列
- [ ] 资产对齐：确保每个日期有相同的资产集合（或显式声明不对称）
- [ ] 缺失值处理：按契约的 missing_policy 处理
- [ ] 异常处理：数据源不可用时的降级策略
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import hashlib
import json
import polars as pl

from core.data_contract import (
    DataContract, DEFAULT_DATA_CONTRACT, FieldSpec,
)
from core.event_log import EventLog, GLOBAL_EVENT_LOG


class DataSourceError(Exception):
    """数据源异常。"""
    pass


class DataValidationError(DataSourceError):
    """数据校验异常。"""
    pass


@dataclass
class DataVersion:
    """数据版本。"""
    source_id: str
    version_hash: str
    row_count: int
    asset_count: int
    date_range: tuple
    contract_hash: str
    loaded_at: float

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "version_hash": self.version_hash,
            "row_count": self.row_count,
            "asset_count": self.asset_count,
            "date_range": list(self.date_range),
            "contract_hash": self.contract_hash,
            "loaded_at": self.loaded_at,
        }


class DataSource(ABC):
    """
    数据源抽象基类。

    真实数据接入时，继承此类并实现 4 个抽象方法。
    """

    def __init__(self, contract: Optional[DataContract] = None,
                 event_log: Optional[EventLog] = None):
        self.contract = contract or DEFAULT_DATA_CONTRACT
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self._last_version: Optional[DataVersion] = None

    @abstractmethod
    def load(self) -> pl.DataFrame:
        """加载数据。返回 long-format DataFrame。"""
        ...

    @abstractmethod
    def get_version_hash(self, df: pl.DataFrame) -> str:
        """
        计算数据版本哈希。
        实现方式：文件哈希 / API 时间戳 / 数据库快照 ID。
        """
        ...

    @abstractmethod
    def source_id(self) -> str:
        """数据源标识（如 "csv_local" / "api_tushare"）。"""
        ...

    def validate(self, df: pl.DataFrame) -> None:
        """
        按契约校验数据。

        默认实现：
        - 字段名匹配
        - 时间索引排序
        - 资产集合一致
        - 值域校验
        """
        # 1) 字段名校验
        contract_fields = set(self.contract.get_field_names())
        df_fields = set(df.columns)
        missing = contract_fields - df_fields
        extra = df_fields - contract_fields
        if missing:
            raise DataValidationError(f"缺少字段: {missing}")
        if extra:
            # 额外字段不报错，只警告（在事件流记录）
            self.event_log.append("data_extra_fields", {
                "source_id": self.source_id(),
                "extra_fields": list(extra),
            })

        # 2) 时间索引排序
        time_field = self.contract.time_index_field
        if not df[time_field].is_sorted():
            raise DataValidationError(
                f"时间索引 {time_field} 未排序"
            )

        # 3) 资产集合一致（每个日期的资产数应相同）
        asset_field = self.contract.asset_id_field
        asset_counts = df.group_by(time_field).agg(
            pl.col(asset_field).count().alias("n")
        )
        unique_counts = asset_counts["n"].unique().to_list()
        if len(unique_counts) > 1:
            raise DataValidationError(
                f"每个日期的资产数不一致: {unique_counts[:5]}..."
            )

        # 4) 值域校验
        for spec in self.contract.get_data_fields():
            if spec.min_value is None and spec.max_value is None:
                continue
            col = df[spec.name]
            if spec.min_value is not None:
                min_val = col.min()
                if min_val is not None and min_val < spec.min_value:
                    raise DataValidationError(
                        f"字段 {spec.name} 最小值 {min_val} < {spec.min_value}"
                    )
            if spec.max_value is not None:
                max_val = col.max()
                if max_val is not None and max_val > spec.max_value:
                    raise DataValidationError(
                        f"字段 {spec.name} 最大值 {max_val} > {spec.max_value}"
                    )

    def load_and_validate(self) -> pl.DataFrame:
        """
        加载 + 校验 + 记录版本。
        """
        import time
        df = self.load()
        self.validate(df)
        version_hash = self.get_version_hash(df)
        # 记录版本
        time_field = self.contract.time_index_field
        asset_field = self.contract.asset_id_field
        date_range = (
            str(df[time_field].min()),
            str(df[time_field].max()),
        )
        self._last_version = DataVersion(
            source_id=self.source_id(),
            version_hash=version_hash,
            row_count=df.height,
            asset_count=df[asset_field].n_unique(),
            date_range=date_range,
            contract_hash=self.contract.content_hash(),
            loaded_at=time.time(),
        )
        self.event_log.append("data_loaded", self._last_version.to_dict())
        return df

    def get_last_version(self) -> Optional[DataVersion]:
        return self._last_version


# =============================================================================
# 实现桩：CSV 本地数据源
# =============================================================================

class CSVDataSource(DataSource):
    """
    CSV 本地数据源（实现桩）。

    真实部署时：
    - 修改 path 为真实 CSV 路径
    - 修改 get_version_hash() 为文件哈希
    """

    def __init__(self, path: str,
                 contract: Optional[DataContract] = None,
                 event_log: Optional[EventLog] = None):
        super().__init__(contract, event_log)
        self.path = path

    def load(self) -> pl.DataFrame:
        # TODO: 真实实现：pl.read_csv(self.path)
        raise NotImplementedError(
            "CSVDataSource.load() 需实现：pl.read_csv(self.path)，"
            "并按契约处理字段类型和缺失值"
        )

    def get_version_hash(self, df: pl.DataFrame) -> str:
        # TODO: 真实实现：文件内容哈希
        # 临时实现：用 df 的行数 + 首尾日期作为版本
        payload = f"{self.path}|{df.height}|{df[self.contract.time_index_field].min()}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def source_id(self) -> str:
        return "csv_local"


# =============================================================================
# 实现桩：API 数据源
# =============================================================================

class APIDataSource(DataSource):
    """
    API 数据源（实现桩）。

    真实部署时：
    - 修改 api_endpoint / api_key
    - 实现 load() 的 API 调用
    - 实现 get_version_hash() 的 API 时间戳
    """

    def __init__(self, api_endpoint: str, api_key: str,
                 contract: Optional[DataContract] = None,
                 event_log: Optional[EventLog] = None):
        super().__init__(contract, event_log)
        self.api_endpoint = api_endpoint
        self.api_key = api_key

    def load(self) -> pl.DataFrame:
        # TODO: 真实实现：调用 API，解析 JSON，构造 DataFrame
        raise NotImplementedError(
            "APIDataSource.load() 需实现：调用 API，解析 JSON，"
            "构造 long-format DataFrame，并按契约处理字段类型"
        )

    def get_version_hash(self, df: pl.DataFrame) -> str:
        # TODO: 真实实现：API 响应时间戳
        payload = f"{self.api_endpoint}|{df.height}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def source_id(self) -> str:
        return "api_remote"


# =============================================================================
# 数据源注册表
# =============================================================================

class DataSourceRegistry:
    """
    数据源注册表。

    记录所有数据源及其版本，用于事件溯源。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self._sources: Dict[str, DataSource] = {}
        self._versions: Dict[str, List[DataVersion]] = {}

    def register(self, name: str, source: DataSource) -> None:
        """注册数据源。"""
        if name in self._sources:
            raise ValueError(f"数据源 {name} 已注册")
        self._sources[name] = source
        self._versions[name] = []
        self.event_log.append("data_source_registered", {
            "name": name,
            "source_id": source.source_id(),
            "contract_hash": source.contract.content_hash(),
        })

    def load(self, name: str) -> pl.DataFrame:
        """加载指定数据源。"""
        if name not in self._sources:
            raise DataSourceError(f"数据源 {name} 未注册")
        source = self._sources[name]
        df = source.load_and_validate()
        version = source.get_last_version()
        if version:
            self._versions[name].append(version)
        return df

    def get_versions(self, name: str) -> List[DataVersion]:
        return list(self._versions.get(name, []))

    def get_latest_version(self, name: str) -> Optional[DataVersion]:
        versions = self._versions.get(name, [])
        return versions[-1] if versions else None