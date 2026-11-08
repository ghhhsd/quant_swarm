"""真实数据源接口桩测试。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import polars as pl
from core.data import (
    DataSource, DataSourceRegistry, CSVDataSource, APIDataSource,
    DataValidationError, DataSourceError,
)
from core.data import generate_synthetic_data


class MockDataSource(DataSource):
    """测试用数据源：包装合成数据。"""

    def __init__(self, df):
        super().__init__()
        self._df = df

    def load(self):
        return self._df.clone()

    def get_version_hash(self, df):
        return f"mock_{df.height}"

    def source_id(self):
        return "mock"


def test_load_and_validate():
    """加载 + 校验 + 版本记录。"""
    df = generate_synthetic_data(300, 20, seed=42)
    source = MockDataSource(df)
    loaded = source.load_and_validate()
    assert loaded.height == df.height
    version = source.get_last_version()
    assert version is not None
    assert version.row_count == df.height
    assert version.asset_count == 20
    print(f"[OK] 加载+校验: version={version.version_hash}")


def test_validation_missing_field():
    """缺少字段被拒。"""
    df = generate_synthetic_data(100, 10, seed=42).drop("returns")
    source = MockDataSource(df)
    try:
        source.load_and_validate()
        assert False, "应抛 DataValidationError"
    except DataValidationError as e:
        assert "returns" in str(e)
        print(f"[OK] 缺少字段被拒: {e}")


def test_validation_unsorted():
    """时间索引未排序被拒。"""
    df = generate_synthetic_data(100, 10, seed=42).sort(
        ["date", "asset"], descending=[True, False])
    source = MockDataSource(df)
    try:
        source.load_and_validate()
        assert False, "应抛 DataValidationError"
    except DataValidationError as e:
        assert "未排序" in str(e)
        print(f"[OK] 未排序被拒: {e}")


def test_registry():
    """数据源注册表。"""
    df = generate_synthetic_data(200, 10, seed=42)
    registry = DataSourceRegistry()
    registry.register("mock", MockDataSource(df))
    loaded = registry.load("mock")
    assert loaded.height == 200
    version = registry.get_latest_version("mock")
    assert version is not None
    print(f"[OK] 注册表: {version.to_dict()}")


def test_csv_stub():
    """CSV 数据源桩。"""
    source = CSVDataSource("/tmp/test.csv")
    try:
        source.load()
        assert False
    except NotImplementedError as e:
        print(f"[OK] CSV 桩: {str(e)[:50]}...")


def test_api_stub():
    """API 数据源桩。"""
    source = APIDataSource("https://api.example.com", "key123")
    try:
        source.load()
        assert False
    except NotImplementedError as e:
        print(f"[OK] API 桩: {str(e)[:50]}...")


if __name__ == "__main__":
    test_load_and_validate()
    test_validation_missing_field()
    test_validation_unsorted()
    test_registry()
    test_csv_stub()
    test_api_stub()
    print("\n=== 数据源全部测试通过 ===")