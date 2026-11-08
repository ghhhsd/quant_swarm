"""
已知因子库（v3.1 修正④：去重阈值的标定依据）。

关键设计：
1. 已知因子库是"事前"的——学术因子 + 已入库精英 + 公开 Alpha 集。
2. 去重阈值的标定依据是"事前已知因子库的分布"，而非运行中命中率。
3. 已知因子库的"分布"用于计算阈值：
   - 语义距离阈值 = 已知因子库内部分布的 10 分位（即库内因子之间的
     最小距离的 10 分位，作为"什么是足够不同"的基准）
   - AST 距离阈值 = 同理
4. 这样阈值完全由事前库决定，运行期不漂移。
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json

from core.data import DATA_CONTRACT


@dataclass(frozen=True)
class KnownFactor:
    """已知因子条目。"""
    kid: str
    family: str
    code: str
    source: str          # "academic" / "elite" / "public_alpha"
    description: str = ""

    def to_dict(self):
        return {
            "kid": self.kid, "family": self.family,
            "code": self.code, "source": self.source,
            "description": self.description,
        }


# 内置学术因子库（示例，真实部署应扩展）
DEFAULT_ACADEMIC_FACTORS: List[KnownFactor] = [
    KnownFactor("ac_001", "momentum", "ts_mean(returns, 20) - ts_mean(returns, 60)",
                "academic", "12-1 动量"),
    KnownFactor("ac_002", "momentum", "ts_mean(returns, 5) - ts_mean(returns, 20)",
                "academic", "短期反转"),
    KnownFactor("ac_003", "volatility", "ts_std(returns, 20)",
                "academic", "20 日波动率"),
    KnownFactor("ac_004", "volatility", "ts_std(returns, 60)",
                "academic", "60 日波动率"),
    KnownFactor("ac_005", "value", "1.0 / pe_ratio",
                "academic", "EP 因子"),
    KnownFactor("ac_006", "value", "1.0 / market_cap",
                "academic", "规模因子"),
    KnownFactor("ac_007", "liquidity", "ts_mean(turnover, 20)",
                "academic", "20 日换手"),
    KnownFactor("ac_008", "liquidity", "ts_mean(turnover, 60)",
                "academic", "60 日换手"),
]


class KnownFactorLibrary:
    """
    已知因子库。
    """

    def __init__(self, factors: Optional[List[KnownFactor]] = None):
        self.factors: List[KnownFactor] = list(factors or DEFAULT_ACADEMIC_FACTORS)
        # 可动态扩展（从精英池、公开 Alpha 集注入）
        self._by_family: Dict[str, List[KnownFactor]] = {}
        self._index()

    def _index(self):
        self._by_family = {}
        for f in self.factors:
            self._by_family.setdefault(f.family, []).append(f)

    def add(self, factor: KnownFactor) -> None:
        if any(f.kid == factor.kid for f in self.factors):
            return
        self.factors.append(factor)
        self._by_family.setdefault(factor.family, []).append(factor)

    def get_all(self) -> List[KnownFactor]:
        return list(self.factors)

    def get_by_family(self, family: str) -> List[KnownFactor]:
        return list(self._by_family.get(family, []))

    def size(self) -> int:
        return len(self.factors)