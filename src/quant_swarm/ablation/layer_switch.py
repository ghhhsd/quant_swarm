"""
开/关定义（v3.1 修正③：显式化层开关语义）。

关键设计：
1. "开"和"关"必须是"同一候选池定义 + 层是否额外筛一道"。
2. "关"不是"没有候选池"，而是"候选池更大"。
3. 每层的开关语义显式声明：
   - 开：层的判定参与筛选
   - 关：层的判定不参与筛选（所有假设都进下一步）
4. 开关不改变组合构建规则——开/关用同一 PortfolioConfig。

为什么这很重要：
- 如果"关"被定义为"没有候选池"，那比较的是"有组合 vs 空组合"，
  而不是"层的贡献"。
- 如果开/关用不同组合规则，比较的是"规则 + 层"，不是"层"。
"""
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Any, Optional


class SwitchState(str, Enum):
    ON = "on"
    OFF = "off"


@dataclass
class LayerSwitch:
    """
    层开关定义。

    layer_name：层名（如 "C1.5_novelty"）
    on_filter：开启时的筛选函数（输入候选，输出通过的候选）
    off_filter：关闭时的筛选函数（通常是恒等函数，所有候选通过）
    """
    layer_name: str
    on_filter: Callable[[List[Any]], List[Any]]
    off_filter: Callable[[List[Any]], List[Any]]

    def apply(self, candidates: List[Any],
              state: SwitchState) -> List[Any]:
        """按状态应用筛选。"""
        if state == SwitchState.ON:
            return self.on_filter(candidates)
        else:
            return self.off_filter(candidates)


# 常用 off_filter：恒等（所有候选通过）
def identity_filter(candidates: List[Any]) -> List[Any]:
    return list(candidates)