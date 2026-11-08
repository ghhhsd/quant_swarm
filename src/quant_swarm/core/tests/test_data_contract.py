"""数据契约测试。"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from core.data import (
    DataContract, FieldSpec, DEFAULT_DATA_CONTRACT, DATA_CONTRACT,
    get_field_types,
)


def test_contract_fields():
    """契约字段完整。"""
    names = DEFAULT_DATA_CONTRACT.get_field_names()
    assert "date" in names
    assert "asset" in names
    assert "returns" in names
    assert "close" in names
    print(f"[OK] 契约字段: {names}")


def test_contract_index_validation():
    """契约索引字段必须在 fields 中。"""
    try:
        DataContract(
            contract_id="test", version="1.0",
            time_index_field="nonexistent",
            asset_id_field="asset",
            fields=(FieldSpec("asset", "string"),),
        )
        assert False, "应抛 ValueError"
    except ValueError:
        print("[OK] 索引字段校验")


def test_contract_content_hash():
    """契约内容哈希。"""
    h1 = DEFAULT_DATA_CONTRACT.content_hash()
    h2 = DEFAULT_DATA_CONTRACT.content_hash()
    assert h1 == h2
    print(f"[OK] 契约哈希: {h1[:12]}...")


def test_field_spec_validate():
    """字段值域校验。"""
    spec = FieldSpec("price", "float64", min_value=0.0)
    assert spec.validate_value(10.0)
    assert not spec.validate_value(-1.0)
    print("[OK] 值域校验")


def test_compat_data_contract():
    """兼容旧 DATA_CONTRACT 接口。"""
    assert "returns" in DATA_CONTRACT
    assert DATA_CONTRACT["returns"] == "float64"
    types = get_field_types()
    assert types == DATA_CONTRACT
    print("[OK] 兼容旧接口")


if __name__ == "__main__":
    test_contract_fields()
    test_contract_index_validation()
    test_contract_content_hash()
    test_field_spec_validate()
    test_compat_data_contract()
    print("\n=== 数据契约全部测试通过 ===")