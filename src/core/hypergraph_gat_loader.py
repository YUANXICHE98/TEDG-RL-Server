"""
GAT友好的超图加载器
支持PyTorch Geometric格式的图数据加载
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class HypergraphGATLoader:
    """GAT友好的超图加载器"""
    
    def __init__(self, graph_file: str):
        """
        Args:
            graph_file: GAT结构的超图文件路径
        """
        with open(graph_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.nodes = self.data['nodes']
        self.edges = self.data['edges']
        self.meta = self.data['meta']
        
        # 构建节点类型映射
        self.node_type_map = {
            'condition': 0,
            'operator': 1,
            'effect': 2,
        }
        
        # 构建边类型映射
        self.edge_type_map = {
            'satisfies': 0,
            'context_of': 1,
            'leads_to': 2,
        }
        
        # 构建快速查找索引
        self._build_indices()
        
        print(f"[HypergraphGATLoader] 加载完成:")
        print(f"  - 节点数: {len(self.nodes)}")
        print(f"  - 边数: {len(self.edges)}")
        print(f"  - Condition节点: {self.meta['node_types']['condition']}")
        print(f"  - Operator节点: {self.meta['node_types']['operator']}")
        print(f"  - Effect节点: {self.meta['node_types']['effect']}")
    
    def _build_indices(self):
        """构建快速查找索引"""
        # 标签到节点的映射
        self.label_to_node = {node['label']: node for node in self.nodes}
        
        # 按类型分组的节点
        self.nodes_by_type = {
            'condition': [],
            'operator': [],
            'effect': [],
        }
        for node in self.nodes:
            self.nodes_by_type[node['type']].append(node)
        
        # 邻接表（用于快速查询邻居）
        self.adjacency = {i: {'in': [], 'out': []} for i in range(len(self.nodes))}
        for edge in self.edges:
            src, tgt, rel = edge[0], edge[1], edge[2]
            self.adjacency[src]['out'].append((tgt, rel))
            self.adjacency[tgt]['in'].append((src, rel))
    
    def get_pyg_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        构建PyTorch Geometric Data对象
        
        Returns:
            edge_index: [2, num_edges] 的边索引张量
            edge_attr: [num_edges] 的边类型张量
            node_types: [num_nodes] 的节点类型张量
            num_nodes: 节点总数
        """
        # 1. 边索引 (Edge Index)
        src = [e[0] for e in self.edges]
        dst = [e[1] for e in self.edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # 2. 边类型 (Edge Attributes)
        edge_attr = torch.tensor(
            [self.edge_type_map[e[2]] for e in self.edges],
            dtype=torch.long
        )
        
        # 3. 节点类型 (Node Types)
        node_types = torch.tensor(
            [self.node_type_map[n['type']] for n in self.nodes],
            dtype=torch.long
        )
        
        return edge_index, edge_attr, node_types, len(self.nodes)
    
    def get_node_by_label(self, label: str) -> Optional[Dict]:
        """根据标签查找节点"""
        return self.label_to_node.get(label)
    
    def get_node_by_id(self, node_id: int) -> Optional[Dict]:
        """根据ID查找节点"""
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None
    
    def get_operator_nodes(self) -> List[Dict]:
        """获取所有操作符节点"""
        return self.nodes_by_type['operator']
    
    def get_condition_nodes(self) -> List[Dict]:
        """获取所有条件节点"""
        return self.nodes_by_type['condition']
    
    def get_effect_nodes(self) -> List[Dict]:
        """获取所有效果节点"""
        return self.nodes_by_type['effect']
    
    def get_neighbors(self, node_id: int, direction: str = 'out') -> List[Tuple[int, str]]:
        """
        获取节点的邻居
        
        Args:
            node_id: 节点ID
            direction: 'in' (入边) 或 'out' (出边)
        
        Returns:
            [(neighbor_id, relation_type), ...]
        """
        return self.adjacency[node_id][direction]
    
    def get_active_operators(self, active_conditions: List[str]) -> List[Dict]:
        """
        根据激活的条件节点，获取可能激活的操作符节点
        
        Args:
            active_conditions: 激活的条件节点标签列表
        
        Returns:
            可能激活的操作符节点列表
        """
        active_operators = set()
        
        for cond_label in active_conditions:
            cond_node = self.get_node_by_label(cond_label)
            if cond_node:
                # 获取该条件节点指向的所有操作符
                neighbors = self.get_neighbors(cond_node['id'], 'out')
                for neighbor_id, rel in neighbors:
                    neighbor = self.get_node_by_id(neighbor_id)
                    if neighbor and neighbor['type'] == 'operator':
                        active_operators.add(neighbor_id)
        
        return [self.get_node_by_id(op_id) for op_id in active_operators]
    
    def get_operator_effects(self, operator_label: str) -> List[Dict]:
        """
        获取操作符的所有效果节点
        
        Args:
            operator_label: 操作符标签
        
        Returns:
            效果节点列表
        """
        op_node = self.get_node_by_label(operator_label)
        if not op_node:
            return []
        
        effects = []
        neighbors = self.get_neighbors(op_node['id'], 'out')
        for neighbor_id, rel in neighbors:
            if rel == 'leads_to':
                neighbor = self.get_node_by_id(neighbor_id)
                if neighbor and neighbor['type'] == 'effect':
                    effects.append(neighbor)
        
        return effects
    
    def get_operator_conditions(self, operator_label: str) -> Dict[str, List[Dict]]:
        """
        获取操作符的所有前置条件
        
        Args:
            operator_label: 操作符标签
        
        Returns:
            {'satisfies': [...], 'context_of': [...]}
        """
        op_node = self.get_node_by_label(operator_label)
        if not op_node:
            return {'satisfies': [], 'context_of': []}
        
        conditions = {'satisfies': [], 'context_of': []}
        neighbors = self.get_neighbors(op_node['id'], 'in')
        
        for neighbor_id, rel in neighbors:
            neighbor = self.get_node_by_id(neighbor_id)
            if neighbor and neighbor['type'] == 'condition':
                if rel in conditions:
                    conditions[rel].append(neighbor)
        
        return conditions
    
    def visualize_subgraph(self, center_node_label: str, depth: int = 1) -> str:
        """
        生成以某个节点为中心的子图的文本可视化
        
        Args:
            center_node_label: 中心节点标签
            depth: 扩展深度
        
        Returns:
            文本格式的子图描述
        """
        center = self.get_node_by_label(center_node_label)
        if not center:
            return f"节点 '{center_node_label}' 不存在"
        
        lines = [f"\n=== 子图中心: {center['label']} ({center['type']}) ===\n"]
        
        # 入边
        in_neighbors = self.get_neighbors(center['id'], 'in')
        if in_neighbors:
            lines.append("输入 (In):")
            for neighbor_id, rel in in_neighbors[:10]:  # 限制显示数量
                neighbor = self.get_node_by_id(neighbor_id)
                lines.append(f"  ← [{rel:12s}] {neighbor['label']} ({neighbor['type']})")
        
        # 出边
        out_neighbors = self.get_neighbors(center['id'], 'out')
        if out_neighbors:
            lines.append("\n输出 (Out):")
            for neighbor_id, rel in out_neighbors[:10]:
                neighbor = self.get_node_by_id(neighbor_id)
                lines.append(f"  → [{rel:12s}] {neighbor['label']} ({neighbor['type']})")
        
        return "\n".join(lines)


# 使用示例
if __name__ == "__main__":
    loader = HypergraphGATLoader("data/hypergraph/hypergraph_gat_structure.json")
    
    # 获取PyG格式数据
    edge_index, edge_attr, node_types, num_nodes = loader.get_pyg_data()
    
    print(f"\nPyG数据格式:")
    print(f"  edge_index shape: {edge_index.shape}")
    print(f"  edge_attr shape: {edge_attr.shape}")
    print(f"  node_types shape: {node_types.shape}")
    print(f"  num_nodes: {num_nodes}")
    
    # 查询示例
    print(f"\n节点统计:")
    print(f"  操作符节点数: {len(loader.get_operator_nodes())}")
    print(f"  条件节点数: {len(loader.get_condition_nodes())}")
    print(f"  效果节点数: {len(loader.get_effect_nodes())}")
    
    # 可视化示例
    print(loader.visualize_subgraph("move_ac717ec4"))
    
    # 测试激活查询
    print("\n\n=== 激活测试 ===")
    active_conditions = ["hp_full", "has_gold"]
    active_ops = loader.get_active_operators(active_conditions)
    print(f"激活条件: {active_conditions}")
    print(f"可能激活的操作符数量: {len(active_ops)}")
    print(f"前5个操作符: {[op['label'] for op in active_ops[:5]]}")
