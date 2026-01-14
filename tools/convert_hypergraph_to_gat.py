#!/usr/bin/env python3
"""
超图结构转换脚本：从扁平结构转换为GAT友好的拓扑结构

输入: data/hypergraph/hypergraph_complete_real.json (旧格式)
输出: data/hypergraph/hypergraph_gat_structure.json (新格式)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any


def convert_to_gat_structure(input_file: str, output_file: str):
    """
    将扁平超图转换为GAT友好的拓扑结构
    
    Args:
        input_file: 输入的旧格式超图文件路径
        output_file: 输出的新格式超图文件路径
    """
    print(f"正在读取: {input_file} ...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    
    # 提取元数据
    old_meta = old_data.get('meta', {})
    hyperedges = old_data.get('hyperedges', [])
    
    print(f"  - 原始超边数量: {len(hyperedges)}")
    print(f"  - 原始操作符数量: {old_meta.get('total_operators', 'N/A')}")
    
    # === 核心数据结构 ===
    node_registry: Dict[str, Dict[str, Any]] = {}  # 节点去重注册表
    gat_nodes: List[Dict[str, Any]] = []  # 节点列表
    gat_edges: List[List[Any]] = []  # 边列表 [source_id, target_id, relation_type]
    
    next_node_id = 0
    
    # 统计信息
    stats = {
        'condition_nodes': 0,
        'operator_nodes': 0,
        'effect_nodes': 0,
        'satisfies_edges': 0,
        'context_of_edges': 0,
        'leads_to_edges': 0,
    }
    
    def get_or_create_node(name: str, node_type: str) -> int:
        """获取或创建节点，返回节点ID"""
        nonlocal next_node_id
        
        # 使用 type:name 作为唯一键
        unique_key = f"{node_type}:{name}"
        
        if unique_key not in node_registry:
            node_info = {
                "id": next_node_id,
                "label": name,
                "type": node_type,
            }
            
            node_registry[unique_key] = node_info
            gat_nodes.append(node_info)
            
            # 更新统计
            if node_type == "condition":
                stats['condition_nodes'] += 1
            elif node_type == "operator":
                stats['operator_nodes'] += 1
            elif node_type == "effect":
                stats['effect_nodes'] += 1
            
            next_node_id += 1
        
        return node_registry[unique_key]["id"]
    
    # === 处理每条超边 ===
    print("\n正在转换超边...")
    
    for idx, he in enumerate(hyperedges):
        if (idx + 1) % 50 == 0:
            print(f"  处理进度: {idx + 1}/{len(hyperedges)}")
        
        # --- A. 创建中心算子节点 (Operator Node) ---
        operator = he.get('operator', 'unknown')
        variant = he.get('variant', '')
        
        # 使用 operator_variant 作为唯一标识
        op_name = f"{operator}_{variant}" if variant else operator
        op_id = get_or_create_node(op_name, "operator")
        
        # --- B. 处理输入条件 (Conditions) ---
        # 1. 前置条件 (Pre-nodes)
        for pre in he.get('pre_nodes', []):
            pre_id = get_or_create_node(pre, "condition")
            # 建立边: Condition -> Operator (满足关系)
            gat_edges.append([pre_id, op_id, "satisfies"])
            stats['satisfies_edges'] += 1
        
        # 2. 场景原子 (Scene Atoms)
        for scene in he.get('scene_atoms', []):
            scene_id = get_or_create_node(scene, "condition")
            # 建立边: Condition -> Operator (场景关系)
            gat_edges.append([scene_id, op_id, "context_of"])
            stats['context_of_edges'] += 1
        
        # --- C. 处理输出效果 (Effects) ---
        for eff in he.get('eff_nodes', []):
            eff_id = get_or_create_node(eff, "effect")
            # 建立边: Operator -> Effect (导致关系)
            gat_edges.append([op_id, eff_id, "leads_to"])
            stats['leads_to_edges'] += 1
    
    # === 构建输出数据结构 ===
    print("\n构建输出数据...")
    
    output_data = {
        "meta": {
            "version": "3.0_GAT",
            "description": "Graph structure for GAT processing. Nodes are shared. Edges represent logic flow.",
            "source_version": old_meta.get('version', 'unknown'),
            "total_nodes": len(gat_nodes),
            "total_edges": len(gat_edges),
            "node_types": {
                "condition": stats['condition_nodes'],
                "operator": stats['operator_nodes'],
                "effect": stats['effect_nodes'],
            },
            "edge_types": {
                "satisfies": stats['satisfies_edges'],
                "context_of": stats['context_of_edges'],
                "leads_to": stats['leads_to_edges'],
            },
            "original_hyperedges": len(hyperedges),
            "data_sources": old_meta.get('data_sources', {}),
        },
        "nodes": gat_nodes,  # 节点特征表 (Feature Matrix X 的基础)
        "edges": gat_edges,  # 邻接表 (Adjacency Matrix 的基础)
    }
    
    # === 保存输出文件 ===
    print(f"\n保存到: {output_file} ...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # === 打印统计信息 ===
    print("\n" + "="*60)
    print("转换完成！统计信息：")
    print("="*60)
    print(f"总节点数: {len(gat_nodes)}")
    print(f"  - Condition 节点 (状态): {stats['condition_nodes']}")
    print(f"  - Operator 节点 (动作): {stats['operator_nodes']}")
    print(f"  - Effect 节点 (效果): {stats['effect_nodes']}")
    print(f"\n总连边数: {len(gat_edges)}")
    print(f"  - satisfies (条件→动作): {stats['satisfies_edges']}")
    print(f"  - context_of (场景→动作): {stats['context_of_edges']}")
    print(f"  - leads_to (动作→效果): {stats['leads_to_edges']}")
    print(f"\n原始超边数: {len(hyperedges)}")
    print(f"节点复用率: {len(gat_edges) / len(gat_nodes):.2f}x (平均每个节点被{len(gat_edges) / len(gat_nodes):.1f}条边使用)")
    print("="*60)
    
    # === 生成示例可视化代码 ===
    print("\n生成示例节点和边（用于验证）：")
    print("\n前10个节点:")
    for node in gat_nodes[:10]:
        print(f"  ID {node['id']:3d} | {node['type']:10s} | {node['label']}")
    
    print("\n前10条边:")
    for edge in gat_edges[:10]:
        src_node = gat_nodes[edge[0]]
        tgt_node = gat_nodes[edge[1]]
        print(f"  {src_node['label']:20s} --[{edge[2]:12s}]--> {tgt_node['label']}")
    
    return output_data


def generate_pyg_loader_example(output_file: str):
    """生成PyG数据加载器的示例代码"""
    
    example_code = '''
# PyTorch Geometric 数据加载示例
# 将此代码保存为 src/core/hypergraph_gat_loader.py

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple

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
        
        print(f"[HypergraphGATLoader] 加载完成:")
        print(f"  - 节点数: {len(self.nodes)}")
        print(f"  - 边数: {len(self.edges)}")
    
    def get_pyg_data(self):
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
    
    def get_node_by_label(self, label: str) -> Dict:
        """根据标签查找节点"""
        for node in self.nodes:
            if node['label'] == label:
                return node
        return None
    
    def get_operator_nodes(self) -> List[Dict]:
        """获取所有操作符节点"""
        return [n for n in self.nodes if n['type'] == 'operator']
    
    def get_condition_nodes(self) -> List[Dict]:
        """获取所有条件节点"""
        return [n for n in self.nodes if n['type'] == 'condition']
    
    def get_effect_nodes(self) -> List[Dict]:
        """获取所有效果节点"""
        return [n for n in self.nodes if n['type'] == 'effect']


# 使用示例
if __name__ == "__main__":
    loader = HypergraphGATLoader("data/hypergraph/hypergraph_gat_structure.json")
    
    # 获取PyG格式数据
    edge_index, edge_attr, node_types, num_nodes = loader.get_pyg_data()
    
    print(f"\\nPyG数据格式:")
    print(f"  edge_index shape: {edge_index.shape}")
    print(f"  edge_attr shape: {edge_attr.shape}")
    print(f"  node_types shape: {node_types.shape}")
    print(f"  num_nodes: {num_nodes}")
    
    # 查询示例
    print(f"\\n操作符节点数: {len(loader.get_operator_nodes())}")
    print(f"条件节点数: {len(loader.get_condition_nodes())}")
    print(f"效果节点数: {len(loader.get_effect_nodes())}")
'''
    
    loader_file = Path(output_file).parent.parent / "src" / "core" / "hypergraph_gat_loader.py"
    
    print(f"\n生成PyG加载器示例代码:")
    print(f"  建议保存路径: {loader_file}")
    print("\n示例代码:")
    print(example_code)
    
    return example_code


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data" / "hypergraph" / "hypergraph_complete_real.json"
    output_file = project_root / "data" / "hypergraph" / "hypergraph_gat_structure.json"
    
    # 检查输入文件
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 执行转换
    try:
        output_data = convert_to_gat_structure(str(input_file), str(output_file))
        
        # 生成加载器示例
        generate_pyg_loader_example(str(output_file))
        
        print(f"\n✓ 转换成功！")
        print(f"  输入: {input_file}")
        print(f"  输出: {output_file}")
        
    except Exception as e:
        print(f"\n✗ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
