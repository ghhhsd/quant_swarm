"""
演化树：所有生成过的因子的谱系图。

关键纪律（v3.1 修正⑥）：
1. 演化树是唯一事实源，精英池是它的"投影视图"。
2. 树只增不减——即使因子被精英池淘汰，谱系信息不丢失。
3. 每个节点引用父代 fid + 父代 content_hash（防止版本错配）。
4. 树的写入进事件流，可重建。

v3.1 修正⑥的事件溯源：演化树节点写入时，同时记录
"父代 content_hash"，这样若父代被修改（理论上不应发生，但可检测），
可发现版本错配。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from main_tier.l3_evolution.factor import Factor
from core.event_log import EventLog, GLOBAL_EVENT_LOG


@dataclass
class TreeNode:
    """演化树节点。"""
    factor: Factor
    parent_hashes: List[str]   # 父代 content_hash 列表（版本锚定）
    generation: int            # 代数（根为 0）
    event_hash: str = ""       # 写入事件流的哈希

    def to_dict(self):
        return {
            "fid": self.factor.fid,
            "code": self.factor.code,
            "family": self.factor.family,
            "ast_depth": self.factor.ast_depth,
            "parent_fids": list(self.factor.parent_fids),
            "parent_hashes": self.parent_hashes,
            "generation": self.generation,
            "content_hash": self.factor.content_hash(),
        }


class EvolutionTree:
    """
    演化树。只增不减。

    - add()：新增节点，父代必须在树中（除根）。
    - get_lineage()：回溯谱系。
    - verify_integrity()：校验所有父代哈希锚定正确（防版本错配）。
    """

    def __init__(self, event_log: Optional[EventLog] = None):
        self.event_log = event_log or GLOBAL_EVENT_LOG
        self.nodes: Dict[str, TreeNode] = {}

    def add(self, factor: Factor) -> TreeNode:
        """新增节点。父代必须在树中（根除外）。"""
        if factor.fid in self.nodes:
            raise ValueError(f"因子 {factor.fid} 已在树中（演化树只增不减）")

        # 计算代数 + 父代哈希锚定
        if not factor.parent_fids:
            generation = 0
            parent_hashes = []
        else:
            parent_hashes = []
            for pfid in factor.parent_fids:
                if pfid not in self.nodes:
                    raise ValueError(f"父代 {pfid} 不在树中，无法添加 {factor.fid}")
                parent_hashes.append(self.nodes[pfid].factor.content_hash())
            generation = max(self.nodes[pfid].generation
                             for pfid in factor.parent_fids) + 1

        node = TreeNode(
            factor=factor,
            parent_hashes=parent_hashes,
            generation=generation,
        )
        ev = self.event_log.append("tree_node_added", node.to_dict())
        node.event_hash = ev.content_hash
        self.nodes[factor.fid] = node
        return node

    def get(self, fid: str) -> Optional[TreeNode]:
        return self.nodes.get(fid)

    def get_lineage(self, fid: str) -> List[str]:
        """从根到当前节点的谱系（fid 列表）。"""
        if fid not in self.nodes:
            return []
        chain = []
        cur = fid
        while cur:
            chain.append(cur)
            parents = self.nodes[cur].factor.parent_fids
            cur = parents[0] if parents else None
        return list(reversed(chain))

    def size(self) -> int:
        return len(self.nodes)

    def verify_integrity(self) -> List[str]:
        """
        校验所有父代哈希锚定正确。
        返回错误列表（空表示全部正确）。
        """
        errors = []
        for fid, node in self.nodes.items():
            for pfid, phash in zip(node.factor.parent_fids, node.parent_hashes):
                if pfid not in self.nodes:
                    errors.append(f"{fid} 的父代 {pfid} 不在树中")
                    continue
                actual_hash = self.nodes[pfid].factor.content_hash()
                if actual_hash != phash:
                    errors.append(
                        f"{fid} 的父代 {pfid} 哈希错配："
                        f"锚定 {phash[:12]}，实际 {actual_hash[:12]}"
                    )
        return errors