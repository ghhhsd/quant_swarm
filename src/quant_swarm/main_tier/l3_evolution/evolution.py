"""
演化操作：突变 / 交叉 / 探索配额。

关键设计：
1. 突变基于精英池 Pareto 前沿因子（定向）。
2. 交叉基于前沿上两个因子的组合。
3. 探索配额（v2.0 冻结 + v3.1 强化）：每轮强制保留一定比例配额
   给"非精英邻域随机突变"，配额比例由 L0 难度信号驱动。
4. 所有生成的新因子都写入演化树（唯一事实源）。
"""
import random
import re
from typing import List, Optional
from main_tier.l3_evolution.factor import Factor
from main_tier.l3_evolution.elite_pool import ElitePool
from main_tier.l3_evolution.tree import EvolutionTree


class EvolutionOperator:
    """
    演化操作器。
    """

    def __init__(self, tree: EvolutionTree, pool: ElitePool,
                 exploration_ratio: float = 0.20,
                 rng: Optional[random.Random] = None):
        self.tree = tree
        self.pool = pool
        self.exploration_ratio = exploration_ratio
        self.rng = rng or random.Random(42)
        self._counter = 0

    def _next_fid(self, family: str) -> str:
        self._counter += 1
        return f"f_{family}_{self._counter:05d}"

    # ---- 突变 ----------------------------------------------------------
    def mutate(self, base: Factor) -> Factor:
        """
        基于 base 的 AST 定向突变。
        本实现：替换表达式中的窗口参数（5/10/20/60）。
        真实实现应解析 AST 做节点替换。
        """
        new_code = re.sub(
            r"\b(5|10|20|60)\b",
            lambda m: str(int(m.group(1)) + self.rng.choice([-5, 5, 10])),
            base.code,
        )
        if new_code == base.code:
            new_code = base.code + " * 1.0"  # 保证有变化
        fid = self._next_fid(base.family)
        new_factor = Factor(
            fid=fid, family=base.family, code=new_code,
            ast_depth=base.ast_depth, parent_fids=(base.fid,),
        )
        self.tree.add(new_factor)
        return new_factor

    # ---- 交叉 ----------------------------------------------------------
    def crossover(self, a: Factor, b: Factor) -> Factor:
        """
        基于两个父代的表达式拼接。
        真实实现应做 AST 子树交换，而非字符串拼接。
        """
        new_code = f"({a.code}) + ({b.code})"
        fid = self._next_fid(a.family)
        new_factor = Factor(
            fid=fid, family=a.family, code=new_code,
            ast_depth=max(a.ast_depth, b.ast_depth) + 1,
            parent_fids=(a.fid, b.fid),
        )
        self.tree.add(new_factor)
        return new_factor

    # ---- 探索（非精英邻域随机突变）-------------------------------------
    def explore(self) -> Factor:
        """
        探索配额：从池内随机因子出发，做"远距离"突变。
        与 mutate() 的区别：mutate 做定向小突变，explore 做随机大扰动。
        """
        all_factors = self.pool.all_factors()
        if not all_factors:
            # 冷启动：随机生成
            fid = self._next_fid("other")
            f = Factor(fid=fid, family="other",
                       code="ts_mean(returns, 10)", ast_depth=2)
            self.tree.add(f)
            return f
        base = self.rng.choice(all_factors)
        # 远距离扰动：拼接一个随机表达式
        perturbations = [
            " * zscore(ts_std(returns, 20))",
            " - ts_mean(turnover, 10)",
            " + rank(1.0 / pe_ratio)",
        ]
        perturb = self.rng.choice(perturbations)
        new_code = base.code + perturb
        fid = self._next_fid(base.family)
        f = Factor(fid=fid, family=base.family, code=new_code,
                   ast_depth=base.ast_depth + 2, parent_fids=(base.fid,))
        self.tree.add(f)
        return f

    # ---- 批量生成 ------------------------------------------------------
    def propose(self, n: int) -> List[Factor]:
        """
        生成 n 个候选，按探索配额分流。
        - 利用配额（1 - exploration_ratio）：从前沿突变/交叉
        - 探索配额（exploration_ratio）：非精英邻域随机突变
        """
        front = self.pool.pareto_front()
        if not front:
            front = self.pool.all_factors()

        n_explore = max(1, int(n * self.exploration_ratio)) if n > 1 else 0
        n_exploit = n - n_explore

        candidates: List[Factor] = []

        # 利用：突变 / 交叉
        for _ in range(n_exploit):
            if not front:
                candidates.append(self.explore())
                continue
            if len(front) >= 2 and self.rng.random() < 0.5:
                a, b = self.rng.sample(front, 2)
                candidates.append(self.crossover(a, b))
            else:
                base = self.rng.choice(front)
                candidates.append(self.mutate(base))

        # 探索：非精英邻域随机突变
        for _ in range(n_explore):
            candidates.append(self.explore())

        return candidates