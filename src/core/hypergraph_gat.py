"""
超图GAT推理层
用于NetHack的因果推理和意图提取

核心功能:
1. 根据当前游戏状态动态激活超图节点
2. 通过GAT消息传递进行因果推理
3. 提取Operator节点的激活分数和全局Intent Vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.core.hypergraph_gat_loader import HypergraphGATLoader


class HypergraphGAT(nn.Module):
    """
    超图GAT推理层
    
    设计思想:
    1. 节点初始化: 根据类型和激活状态初始化节点特征
    2. 消息传递: 2层GAT，信息从Condition流向Operator流向Effect
    3. 操作符聚合: 提取Operator节点的嵌入和注意力分数
    4. 意图提取: 全局Readout得到Intent Vector
    
    Args:
        hypergraph_path: 超图结构文件路径
        num_nodes: 节点总数 (默认527)
        hidden_dim: 隐藏层维度 (默认256)
        num_heads: GAT注意力头数 (默认4)
        dropout: Dropout率 (默认0.1)
    """
    
    def __init__(self,
                 hypergraph_path: str = "data/hypergraph/hypergraph_gat_structure.json",
                 num_nodes: int = 527,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 加载超图结构
        self.loader = HypergraphGATLoader(hypergraph_path)
        
        # 获取PyG格式数据
        self.edge_index, self.edge_attr, self.node_types, _ = self.loader.get_pyg_data()
        
        # 注册为buffer (不需要梯度，但需要随模型移动到GPU)
        self.register_buffer('_edge_index', self.edge_index)
        self.register_buffer('_edge_attr', self.edge_attr)
        self.register_buffer('_node_types', self.node_types)
        
        # 节点类型嵌入 (condition=0, operator=1, effect=2)
        self.node_type_embedding = nn.Embedding(3, hidden_dim)
        
        # 激活状态嵌入 (0=未激活, 1=激活)
        self.activation_embedding = nn.Embedding(2, hidden_dim)
        
        # 2层GAT
        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=False,  # 不拼接多头，而是平均
            add_self_loops=True
        )
        
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=False,
            add_self_loops=True
        )
        
        # LayerNorm (稳定训练)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # 全局Readout (提取Intent Vector)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Operator节点索引 (用于快速提取)
        self._build_operator_indices()
        
        print(f"[HypergraphGAT] 初始化完成:")
        print(f"  - 节点数: {num_nodes}")
        print(f"  - 边数: {self.edge_index.size(1)}")
        print(f"  - Operator节点数: {len(self.operator_indices)}")
        print(f"  - 隐藏维度: {hidden_dim}")
        print(f"  - 注意力头数: {num_heads}")
    
    def _build_operator_indices(self):
        """构建Operator节点的索引列表"""
        operator_nodes = self.loader.get_operator_nodes()
        self.operator_indices = torch.tensor(
            [node['id'] for node in operator_nodes],
            dtype=torch.long
        )
        self.register_buffer('_operator_indices', self.operator_indices)
    
    def activate_nodes_from_atoms(self, atoms: Dict[str, List[str]]) -> torch.Tensor:
        """
        根据当前游戏状态的atoms激活超图节点
        
        Args:
            atoms: {"pre_nodes": [...], "scene_atoms": [...], ...}
        
        Returns:
            active_mask: (num_nodes,) 0/1向量，1表示激活
        """
        active_mask = torch.zeros(self.num_nodes, device=self.node_types.device)
        
        # 收集所有atoms
        all_atoms = []
        for key in ['pre_nodes', 'scene_atoms']:
            if key in atoms:
                all_atoms.extend(atoms[key])
        
        # 激活对应的Condition节点
        for atom in all_atoms:
            node = self.loader.get_node_by_label(atom)
            if node and node['type'] == 'condition':
                active_mask[node['id']] = 1.0
        
        return active_mask
    
    def forward(self, 
                active_mask: Optional[torch.Tensor] = None,
                atoms: Optional[Dict[str, List[str]]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            active_mask: (num_nodes,) 节点激活mask，优先使用
            atoms: 如果active_mask为None，则从atoms构造
        
        Returns:
            h_logic: (hidden_dim,) 全局Intent Vector
            operator_scores: (num_operators,) Operator节点激活分数
            attention_weights: GAT第2层的注意力权重 (用于可视化)
        """
        # 1. 构造激活mask
        if active_mask is None:
            if atoms is None:
                raise ValueError("必须提供active_mask或atoms之一")
            active_mask = self.activate_nodes_from_atoms(atoms)
        
        # 确保在正确的设备上
        device = self.node_types.device
        if active_mask.device != device:
            active_mask = active_mask.to(device)
        
        # 2. 初始化节点特征
        # 节点特征 = 类型嵌入 + 激活嵌入
        node_type_emb = self.node_type_embedding(self.node_types)  # (num_nodes, hidden_dim)
        activation_emb = self.activation_embedding(active_mask.long())  # (num_nodes, hidden_dim)
        x = node_type_emb + activation_emb  # (num_nodes, hidden_dim)
        
        # 3. 第一层GAT
        x1 = self.gat1(x, self.edge_index)  # (num_nodes, hidden_dim)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)
        x1 = x + x1  # 残差连接
        
        # 4. 第二层GAT
        x2, attention_weights = self.gat2(x1, self.edge_index, return_attention_weights=True)
        x2 = self.ln2(x2)
        x2 = F.relu(x2)
        x2 = x1 + x2  # 残差连接
        
        # 5. 提取Operator节点的嵌入
        operator_embeddings = x2[self.operator_indices]  # (num_operators, hidden_dim)
        
        # 6. 计算Operator激活分数 (简单的L2范数)
        operator_scores = torch.norm(operator_embeddings, dim=-1)  # (num_operators,)
        
        # 7. 全局Readout (加权平均 + MLP)
        # 使用激活mask加权
        active_nodes_emb = x2 * active_mask.unsqueeze(-1)  # (num_nodes, hidden_dim)
        global_emb = active_nodes_emb.sum(dim=0) / (active_mask.sum() + 1e-8)  # (hidden_dim,)
        h_logic = self.readout(global_emb)  # (hidden_dim,)
        
        return h_logic, operator_scores, attention_weights
    
    def get_active_operators(self, 
                            operator_scores: torch.Tensor,
                            top_k: int = 10,
                            threshold: float = 0.5) -> List[Dict]:
        """
        获取激活分数最高的Operator节点
        
        Args:
            operator_scores: (num_operators,) Operator激活分数
            top_k: 返回前K个
            threshold: 最低激活阈值
        
        Returns:
            激活的Operator节点列表 (包含label和score)
        """
        # 过滤低于阈值的
        valid_mask = operator_scores > threshold
        valid_indices = torch.where(valid_mask)[0]
        valid_scores = operator_scores[valid_mask]
        
        if len(valid_indices) == 0:
            return []
        
        # Top-K
        k = min(top_k, len(valid_indices))
        topk_values, topk_indices = torch.topk(valid_scores, k)
        
        # 转换为节点信息
        result = []
        for i in range(k):
            op_idx = valid_indices[topk_indices[i]].item()
            node_id = self.operator_indices[op_idx].item()
            node = self.loader.get_node_by_id(node_id)
            result.append({
                'label': node['label'],
                'score': topk_values[i].item(),
                'node_id': node_id
            })
        
        return result


class BatchedHypergraphGAT(nn.Module):
    """
    批处理版本的HypergraphGAT
    用于训练时处理batch数据
    
    注意: 由于超图结构是固定的，批处理主要体现在active_mask上
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gat = HypergraphGAT(*args, **kwargs)
    
    def forward(self, 
                active_masks: torch.Tensor,
                atoms_list: Optional[List[Dict]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批处理前向传播
        
        Args:
            active_masks: (batch, num_nodes) 批量激活mask
            atoms_list: 如果提供，用于构造active_masks
        
        Returns:
            h_logic_batch: (batch, hidden_dim) 批量Intent Vector
            operator_scores_batch: (batch, num_operators) 批量Operator分数
        """
        batch_size = active_masks.size(0)
        
        h_logic_list = []
        operator_scores_list = []
        
        for i in range(batch_size):
            h_logic, operator_scores, _ = self.gat(active_mask=active_masks[i])
            h_logic_list.append(h_logic)
            operator_scores_list.append(operator_scores)
        
        h_logic_batch = torch.stack(h_logic_list, dim=0)  # (batch, hidden_dim)
        operator_scores_batch = torch.stack(operator_scores_list, dim=0)  # (batch, num_operators)
        
        return h_logic_batch, operator_scores_batch


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*60)
    print("测试 HypergraphGAT")
    print("="*60 + "\n")
    
    # 初始化
    gat = HypergraphGAT(
        hypergraph_path="data/hypergraph/hypergraph_gat_structure.json",
        hidden_dim=256,
        num_heads=4
    )
    
    # 测试1: 使用atoms激活
    print("\n[测试1] 使用atoms激活节点")
    test_atoms = {
        "pre_nodes": ["hp_full", "has_gold"],
        "scene_atoms": ["dlvl_1", "monsters_present"]
    }
    
    h_logic, operator_scores, attn = gat(atoms=test_atoms)
    print(f"  Intent Vector shape: {h_logic.shape}")
    print(f"  Operator scores shape: {operator_scores.shape}")
    print(f"  Intent Vector norm: {torch.norm(h_logic).item():.4f}")
    print(f"  Operator scores range: [{operator_scores.min().item():.4f}, {operator_scores.max().item():.4f}]")
    
    # 测试2: 获取激活的Operator
    print("\n[测试2] 获取Top-5激活的Operator")
    active_ops = gat.get_active_operators(operator_scores, top_k=5, threshold=0.0)
    for i, op in enumerate(active_ops, 1):
        print(f"  {i}. {op['label']}: {op['score']:.4f}")
    
    # 测试3: 批处理
    print("\n[测试3] 批处理测试")
    batched_gat = BatchedHypergraphGAT(hidden_dim=256)
    batch_size = 4
    active_masks = torch.rand(batch_size, gat.num_nodes) > 0.8  # 随机激活20%节点
    h_logic_batch, scores_batch = batched_gat(active_masks.float())
    print(f"  Batch Intent Vector shape: {h_logic_batch.shape}")
    print(f"  Batch Operator scores shape: {scores_batch.shape}")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60 + "\n")
