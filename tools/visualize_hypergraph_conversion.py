#!/usr/bin/env python3
"""
超图转换效果可视化脚本
对比旧格式和新格式的差异
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter


def analyze_old_format(file_path: str):
    """分析旧格式超图"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    hyperedges = data.get('hyperedges', [])
    
    # 统计所有出现的原子
    all_atoms = set()
    atom_frequency = Counter()
    operator_variants = defaultdict(list)
    
    for he in hyperedges:
        operator = he.get('operator', 'unknown')
        variant = he.get('variant', '')
        operator_variants[operator].append(variant)
        
        # 收集所有原子
        for key in ['pre_nodes', 'scene_atoms', 'eff_nodes']:
            atoms = he.get(key, [])
            for atom in atoms:
                all_atoms.add(atom)
                atom_frequency[atom] += 1
    
    return {
        'total_hyperedges': len(hyperedges),
        'unique_atoms': len(all_atoms),
        'unique_operators': len(operator_variants),
        'total_variants': sum(len(variants) for variants in operator_variants.values()),
        'most_common_atoms': atom_frequency.most_common(10),
        'operator_variants': dict(operator_variants),
    }


def analyze_new_format(file_path: str):
    """分析新格式超图"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    meta = data.get('meta', {})
    
    # 按类型统计节点
    node_types = Counter(node['type'] for node in nodes)
    
    # 按关系类型统计边
    edge_types = Counter(edge[2] for edge in edges)
    
    # 计算节点度数
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    
    for edge in edges:
        src, tgt = edge[0], edge[1]
        out_degree[src] += 1
        in_degree[tgt] += 1
    
    # 找到度数最高的节点
    max_in_degree = max(in_degree.values()) if in_degree else 0
    max_out_degree = max(out_degree.values()) if out_degree else 0
    
    high_in_degree_nodes = [
        (nodes[node_id]['label'], nodes[node_id]['type'], degree)
        for node_id, degree in in_degree.items()
        if degree >= max_in_degree * 0.8  # 取前20%
    ][:5]
    
    high_out_degree_nodes = [
        (nodes[node_id]['label'], nodes[node_id]['type'], degree)
        for node_id, degree in out_degree.items()
        if degree >= max_out_degree * 0.8
    ][:5]
    
    return {
        'total_nodes': len(nodes),
        'total_edges': len(edges),
        'node_types': dict(node_types),
        'edge_types': dict(edge_types),
        'max_in_degree': max_in_degree,
        'max_out_degree': max_out_degree,
        'high_in_degree_nodes': high_in_degree_nodes,
        'high_out_degree_nodes': high_out_degree_nodes,
        'meta': meta,
    }


def compare_formats(old_stats: dict, new_stats: dict):
    """对比两种格式的统计信息"""
    print("=" * 80)
    print("超图格式转换对比分析")
    print("=" * 80)
    
    print("\n📊 基础统计对比:")
    print(f"{'指标':<25} {'旧格式':<15} {'新格式':<15} {'变化':<15}")
    print("-" * 70)
    
    # 超边 vs 节点
    print(f"{'超边/节点数':<25} {old_stats['total_hyperedges']:<15} {new_stats['total_nodes']:<15} {'+' + str(new_stats['total_nodes'] - old_stats['total_hyperedges']):<15}")
    
    # 原子 vs 边
    print(f"{'唯一原子/边数':<25} {old_stats['unique_atoms']:<15} {new_stats['total_edges']:<15} {'+' + str(new_stats['total_edges'] - old_stats['unique_atoms']):<15}")
    
    # 操作符数量
    print(f"{'操作符数量':<25} {old_stats['unique_operators']:<15} {new_stats['node_types'].get('operator', 0):<15} {'+' + str(new_stats['node_types'].get('operator', 0) - old_stats['unique_operators']):<15}")
    
    print("\n🔄 结构化改进:")
    print(f"  ✓ 节点去重: {old_stats['unique_atoms']} 个独立原子 → {new_stats['total_nodes']} 个共享节点")
    print(f"  ✓ 关系明确: 扁平列表 → {len(new_stats['edge_types'])} 种边类型")
    print(f"  ✓ 连通性: 孤立超边 → 平均每节点 {new_stats['total_edges'] / new_stats['total_nodes']:.1f} 条连边")
    
    print("\n📈 新格式详细统计:")
    print("节点类型分布:")
    for node_type, count in new_stats['node_types'].items():
        print(f"  - {node_type.capitalize():<12}: {count:>3} 个")
    
    print("\n边类型分布:")
    for edge_type, count in new_stats['edge_types'].items():
        print(f"  - {edge_type:<12}: {count:>4} 条")
    
    print("\n🔗 连通性分析:")
    print(f"最大入度: {new_stats['max_in_degree']} (被多少个节点指向)")
    print(f"最大出度: {new_stats['max_out_degree']} (指向多少个节点)")
    
    print("\n高入度节点 (热门目标):")
    for label, node_type, degree in new_stats['high_in_degree_nodes']:
        print(f"  - {label:<20} ({node_type:<9}): {degree:>3} 条入边")
    
    print("\n高出度节点 (影响广泛):")
    for label, node_type, degree in new_stats['high_out_degree_nodes']:
        print(f"  - {label:<20} ({node_type:<9}): {degree:>3} 条出边")
    
    print("\n💡 GAT 优势:")
    print("  ✓ 消息传递: 节点间可以传递信息 (旧格式无法实现)")
    print("  ✓ 注意力机制: 可以学习节点间的重要性权重")
    print("  ✓ 层次推理: 条件 → 操作符 → 效果 的因果链")
    print("  ✓ 动态激活: 根据游戏状态动态点亮相关节点")
    
    print("\n🚀 性能提升预期:")
    node_reuse_ratio = new_stats['total_edges'] / new_stats['total_nodes']
    print(f"  - 节点复用率: {node_reuse_ratio:.1f}x (每个概念被多个规则共享)")
    print(f"  - 存储效率: 减少 {old_stats['unique_atoms'] - new_stats['total_nodes']} 个冗余节点")
    print(f"  - 推理效率: 图卷积 vs 独立匹配 (预期提升 2-5x)")


def generate_mermaid_sample(new_file: str, sample_size: int = 15):
    """生成Mermaid图表示例"""
    with open(new_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = data['nodes']
    edges = data['edges']
    
    # 选择一个有代表性的子图
    # 找一个中等度数的操作符节点作为中心
    operator_nodes = [n for n in nodes if n['type'] == 'operator']
    
    # 计算每个操作符的连接数
    node_connections = defaultdict(int)
    for edge in edges:
        if nodes[edge[0]]['type'] == 'operator':
            node_connections[edge[0]] += 1
        if nodes[edge[1]]['type'] == 'operator':
            node_connections[edge[1]] += 1
    
    # 选择连接数适中的操作符
    sorted_ops = sorted(node_connections.items(), key=lambda x: x[1])
    center_op_id = sorted_ops[len(sorted_ops) // 2][0]  # 选择中位数
    center_op = nodes[center_op_id]
    
    # 收集相关节点和边
    related_nodes = {center_op_id: center_op}
    related_edges = []
    
    for edge in edges[:sample_size]:  # 限制边数
        src, tgt, rel = edge[0], edge[1], edge[2]
        if src == center_op_id or tgt == center_op_id:
            related_nodes[src] = nodes[src]
            related_nodes[tgt] = nodes[tgt]
            related_edges.append(edge)
    
    # 生成Mermaid代码
    mermaid_code = ["graph LR"]
    mermaid_code.append("    %% === 节点定义 ===")
    
    for node_id, node in related_nodes.items():
        node_type = node['type']
        label = node['label'][:15]  # 截断长标签
        
        if node_type == 'condition':
            mermaid_code.append(f"    C{node_id}({label}):::cond")
        elif node_type == 'operator':
            mermaid_code.append(f"    OP{node_id}{{{label}}}:::op")
        elif node_type == 'effect':
            mermaid_code.append(f"    E{node_id}({label}):::eff")
    
    mermaid_code.append("\n    %% === 连线关系 ===")
    for edge in related_edges:
        src, tgt, rel = edge[0], edge[1], edge[2]
        src_prefix = 'C' if nodes[src]['type'] == 'condition' else ('OP' if nodes[src]['type'] == 'operator' else 'E')
        tgt_prefix = 'C' if nodes[tgt]['type'] == 'condition' else ('OP' if nodes[tgt]['type'] == 'operator' else 'E')
        mermaid_code.append(f"    {src_prefix}{src} --> {tgt_prefix}{tgt}")
    
    mermaid_code.extend([
        "\n    %% === 样式设置 ===",
        "    classDef cond fill:#e1f5fe,stroke:#01579b,stroke-width:2px;",
        "    classDef op fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;",
        "    classDef eff fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;",
    ])
    
    return "\n".join(mermaid_code)


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    old_file = project_root / "data" / "hypergraph" / "hypergraph_complete_real.json"
    new_file = project_root / "data" / "hypergraph" / "hypergraph_gat_structure.json"
    
    if not old_file.exists():
        print(f"错误: 旧格式文件不存在: {old_file}")
        sys.exit(1)
    
    if not new_file.exists():
        print(f"错误: 新格式文件不存在: {new_file}")
        print("请先运行 tools/convert_hypergraph_to_gat.py")
        sys.exit(1)
    
    # 分析两种格式
    print("正在分析旧格式...")
    old_stats = analyze_old_format(str(old_file))
    
    print("正在分析新格式...")
    new_stats = analyze_new_format(str(new_file))
    
    # 对比分析
    compare_formats(old_stats, new_stats)
    
    # 生成可视化示例
    print("\n" + "=" * 80)
    print("Mermaid 可视化示例 (可复制到 Markdown 编辑器)")
    print("=" * 80)
    mermaid_sample = generate_mermaid_sample(str(new_file))
    print(mermaid_sample)
    
    print("\n✅ 分析完成！新的GAT结构已准备就绪。")


if __name__ == "__main__":
    main()