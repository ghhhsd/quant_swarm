"""
面板预计算（v3.1 修正③：一次 pivot，多次复用）。

关键设计：
1. 一次 pivot 所有字段，缓存到 PanelData。
2. 所有因子共享面板，避免 N 次 pivot。
3. 面板形状：(T, A)，T = 天数，A = 资产数。
4. 提供 OOS 批次切片接口（按索引切片，不重新加载）。

为什么需要面板预计算：
- 之前的 evaluate_expression 每次 pivot，对 N 个因子是 N 次。
- 组合层需要 N 个因子同时计算，pivot 成为瓶颈。
- 面板预计算把 pivot 从 O(N × T × A) 降为 O(T × A)。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import polars as pl

from core.data import DEFAULT_DATA_CONTRACT


@dataclass
class PanelData:
    """
    面板数据。

    - fields：{字段名: 面板矩阵 (T, A)}
    - dates：日期列表（长度 T）
    - assets：资产列表（长度 A）
    - contract_hash：数据契约哈希（用于事件溯源）
    """
    fields: Dict[str, np.ndarray]
    dates: List[str]
    assets: List[str]
    contract_hash: str

    def get(self, field_name: str) -> np.ndarray:
        """获取字段面板。"""
        if field_name not in self.fields:
            raise KeyError(f"字段 {field_name} 不在面板中")
        return self.fields[field_name]

    def shape(self) -> Tuple[int, int]:
        """面板形状 (T, A)。"""
        if not self.fields:
            return (0, 0)
        first = next(iter(self.fields.values()))
        return first.shape

    def slice(self, start: int, end: int) -> "PanelData":
        """
        OOS 批次切片。

        start / end：索引范围 [start, end)
        返回：新的 PanelData（字段切片）
        """
        sliced_fields = {
            k: v[start:end] for k, v in self.fields.items()
        }
        return PanelData(
            fields=sliced_fields,
            dates=self.dates[start:end],
            assets=self.assets,
            contract_hash=self.contract_hash,
        )

    def to_dict(self):
        """元信息（不包含实际数据，供事件溯源）。"""
        return {
            "shape": list(self.shape()),
            "n_fields": len(self.fields),
            "date_range": (self.dates[0] if self.dates else None,
                           self.dates[-1] if self.dates else None),
            "contract_hash": self.contract_hash,
        }


class PanelBuilder:
    """
    面板构建器。

    从 long-format DataFrame 构建 PanelData。
    """

    def __init__(self, contract=None):
        self.contract = contract or DEFAULT_DATA_CONTRACT

    def build(self, df: pl.DataFrame) -> PanelData:
        """
        构建面板。

        df：long-format DataFrame，字段由契约定义。
        """
        time_field = self.contract.time_index_field
        asset_field = self.contract.asset_id_field

        # 1) 提取日期和资产列表
        dates = sorted(df[time_field].unique().to_list())
        assets = sorted(df[asset_field].unique().to_list())

        # 2) 对每个字段 pivot
        fields = {}
        for spec in self.contract.get_data_fields():
            field_name = spec.name
            pivot = df.pivot(
                values=field_name, index=time_field, on=asset_field,
            ).sort(time_field)
            # 提取数据列（去掉 time_field）
            data_cols = [c for c in pivot.columns if c != time_field]
            # 按资产列表排序
            data_cols = sorted(data_cols)
            arr = np.array([
                pivot[c].to_numpy() for c in data_cols
            ]).T  # (T, A)
            fields[field_name] = arr

        return PanelData(
            fields=fields,
            dates=[str(d) for d in dates],
            assets=assets,
            contract_hash=self.contract.content_hash(),
        )