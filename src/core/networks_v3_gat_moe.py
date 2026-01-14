"""
V3: GAT引导的混合专家策略网络
Causal-Graph Guided Hierarchical MoE

核心创新:
1. 双流编码: Visual Stream (数值特征) + Logic Stream (GAT推理)
2. 因果路由: Sparsemax路由，GAT提供因果偏置
3. 语义专家: Survival/Combat/Exploration/General
4. 可解释性: GAT注意力 + 专家选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from src.core.hypergraph_gat import HypergraphGAT


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax激活函数
    
    相比Softmax:
    - 可以产生稀疏输出 (某些维度为0)
    - 软中带硬，避免平均主义
    
    Args:
        logits: 输入logits
        dim: 归一化维度
    
    Returns:
        稀疏概率分布
    """
    # 简化实现: 使用top-k + softmax近似
    # 完整实现需要排序和阈值计算，这里用简化版
    k = max(2, logits.size(dim) // 2)  # 保留一半
    topk_values, topk_indices = torch.topk(logits, k, dim=dim)
    
    # 对top-k做softmax
    topk_probs = F.softmax(topk_values, dim=dim)
    
    # 构造稀疏输出
    output = torch.zeros_like(logits)
    output.scatter_(dim, topk_indices, topk_probs)
    
    return output


class SemanticExpert(nn.Module):
    """
    语义专家 - 每个专家关注特定游戏意图
    
    专家定义:
    - Survival: 生存相关 (吃喝、回血、逃跑)
    - Combat: 战斗相关 (攻击、走位、使用武器)
    - Exploration: 探索相关 (开图、搜索、捡东西)
    - General: 通用/兜底
    """
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 action_dim: int = 23):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # [修改] 增大初始化增益，从 0.01 改为 0.5
        # 0.01 会导致初始梯度极小，专家学不动
        nn.init.orthogonal_(self.network[-1].weight, gain=0.5)
        nn.init.constant_(self.network[-1].bias, 0)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) 输入特征
        
        Returns:
            logits: (batch, action_dim) 动作logits
        """
        return self.network(h)


class CausalRouter(nn.Module):
    """
    因果路由器 - 使用Sparsemax实现软中带硬的路由
    
    输入: h_vis + h_logic (拼接)
    输出: α权重 (Sparsemax归一化)
    
    关键: h_logic来自GAT，提供因果偏置
    """
    
    def __init__(self, 
                 input_dim: int = 512,  # h_vis(256) + h_logic(256)
                 hidden_dim: int = 128,
                 num_experts: int = 4):
        super().__init__()
        
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # [修改] Router 最后一层初始化稍微大一点，让 Router 大胆选择
        nn.init.orthogonal_(self.router[-1].weight, gain=0.1)
    
    def forward(self, z: torch.Tensor, use_sparsemax: bool = True) -> torch.Tensor:
        """
        Args:
            z: (batch, input_dim) 拼接的特征
            use_sparsemax: 是否使用Sparsemax (训练时可以warmup)
        
        Returns:
            alpha: (batch, num_experts) 专家权重
        """
        logits = self.router(z)
        
        # [修改] 移除过于激进的 clamp，只保留数值稳定性处理
        # 之前的 clamp(-20, 20) 限制太死，如果 logits 本来就很小，clamp 没意义
        logits = torch.nan_to_num(logits, nan=0.0)
        
        if use_sparsemax:
            alpha = sparsemax(logits, dim=-1)
        else:
            alpha = F.softmax(logits, dim=-1)
        
        return alpha


class GATGuidedMoEPolicy(nn.Module):
    """
    V3: GAT引导的混合专家策略网络
    
    架构流程:
    1. 双流编码: Visual (blstats) + Logic (GAT)
    2. 因果路由: Sparsemax选择专家
    3. 专家融合: 加权组合
    4. 价值估计: Critic网络
    
    Args:
        hypergraph_path: 超图结构文件路径
        state_dim: 输入状态维度 (默认115, blstats等)
        hidden_dim: 隐藏层维度 (默认256)
        action_dim: 动作空间大小 (默认23)
        num_experts: 专家数量 (默认4)
        use_sparsemax: 是否使用Sparsemax路由 (默认True)
    """
    
    def __init__(self,
                 hypergraph_path: str = "data/hypergraph/hypergraph_gat_structure.json",
                 state_dim: int = 115,
                 hidden_dim: int = 256,
                 action_dim: int = 23,
                 num_experts: int = 4,
                 use_sparsemax: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.use_sparsemax = use_sparsemax
        
        # 1. 超图GAT (Logic Stream)
        self.gat = HypergraphGAT(
            hypergraph_path=hypergraph_path,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # 2. 视觉编码器 (Visual Stream)
        # 输入: blstats等数值特征
        self.visual_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 3. 因果路由器
        self.router = CausalRouter(
            input_dim=hidden_dim * 2,  # h_vis + h_logic
            hidden_dim=128,
            num_experts=num_experts
        )
        
        # 4. 四个语义专家
        self.expert_names = ['Survival', 'Combat', 'Exploration', 'General']
        self.experts = nn.ModuleList([
            SemanticExpert(hidden_dim, 128, action_dim)
            for _ in range(num_experts)
        ])
        
        # 5. Critic网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        
        print(f"[GATGuidedMoEPolicy] 初始化完成:")
        print(f"  - 状态维度: {state_dim}")
        print(f"  - 隐藏维度: {hidden_dim}")
        print(f"  - 动作维度: {action_dim}")
        print(f"  - 专家数量: {num_experts}")
        print(f"  - 路由方式: {'Sparsemax' if use_sparsemax else 'Softmax'}")
        print(f"  - 专家: {', '.join(self.expert_names)}")
    
    def forward(self, 
                state: torch.Tensor,
                atoms: Optional[Dict[str, List[str]]] = None,
                active_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            state: (batch, state_dim) 数值状态
            atoms: 当前激活的atoms字典 (用于GAT)
            active_mask: (batch, num_nodes) 预计算的激活mask
        
        Returns:
            logits: (batch, action_dim) 动作logits
            alpha: (batch, num_experts) 专家权重
            value: (batch, 1) 状态价值
            aux_info: 辅助信息 (用于可视化和分析)
        """
        # 处理单个样本
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = state.size(0)
        
        # 1. Visual Stream: 编码数值特征
        h_vis = self.visual_encoder(state)  # (batch, hidden_dim)
        
        # 2. Logic Stream: GAT推理
        # 注意: 当前实现假设batch内所有样本使用相同的atoms
        # 实际训练时可能需要改进为per-sample atoms
        if atoms is not None or active_mask is not None:
            if batch_size == 1:
                # 单样本: 直接调用GAT
                h_logic, operator_scores, gat_attention = self.gat(
                    active_mask=active_mask[0] if active_mask is not None else None,
                    atoms=atoms
                )
                h_logic = h_logic.unsqueeze(0)  # (1, hidden_dim)
                operator_scores = operator_scores.unsqueeze(0)  # (1, num_operators)
            else:
                # 批处理: 逐个处理 (可以优化)
                h_logic_list = []
                operator_scores_list = []
                for i in range(batch_size):
                    mask_i = active_mask[i] if active_mask is not None else None
                    h_i, scores_i, _ = self.gat(active_mask=mask_i, atoms=atoms)
                    h_logic_list.append(h_i)
                    operator_scores_list.append(scores_i)
                h_logic = torch.stack(h_logic_list, dim=0)
                operator_scores = torch.stack(operator_scores_list, dim=0)
                gat_attention = None  # 批处理时不返回attention
        else:
            # 没有atoms: 使用零向量 (降级模式)
            h_logic = torch.zeros(batch_size, self.hidden_dim, device=state.device)
            operator_scores = None
            gat_attention = None
        
        # 3. 拼接双流特征
        z = torch.cat([h_vis, h_logic], dim=-1)  # (batch, hidden_dim*2)
        
        # 4. 因果路由
        alpha = self.router(z, use_sparsemax=self.use_sparsemax)  # (batch, num_experts)
        
        # 5. 专家输出
        expert_logits = torch.stack([
            expert(h_vis) for expert in self.experts
        ], dim=1)  # (batch, num_experts, action_dim)
        
        # 6. 融合
        alpha_expanded = alpha.unsqueeze(2)  # (batch, num_experts, 1)
        fused_logits = (alpha_expanded * expert_logits).sum(dim=1)  # (batch, action_dim)
        
        # 7. 价值估计
        value = self.critic(z)  # (batch, 1)
        
        # 8. 数值稳定性
        fused_logits = torch.nan_to_num(fused_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # 9. 辅助信息
        aux_info = {
            'h_vis': h_vis,
            'h_logic': h_logic,
            'operator_scores': operator_scores,
            'gat_attention': gat_attention,
            'expert_logits': expert_logits,
        }
        
        if squeeze_output:
            fused_logits = fused_logits.squeeze(0)
            alpha = alpha.squeeze(0)
            value = value.squeeze(0)
        
        return fused_logits, alpha, value, aux_info
    
    def get_action_distribution(self, 
                               state: torch.Tensor,
                               atoms: Optional[Dict] = None) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """获取动作分布和价值 (用于PPO采样)"""
        logits, _, value, _ = self.forward(state, atoms=atoms)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value
    
    def get_expert_usage_stats(self, alpha_history: List[torch.Tensor]) -> Dict:
        """
        分析专家使用统计
        
        Args:
            alpha_history: 历史α权重列表
        
        Returns:
            统计信息字典
        """
        if not alpha_history:
            return {}
        
        alpha_tensor = torch.stack(alpha_history, dim=0)  # (num_steps, num_experts)
        
        stats = {
            'mean_alpha': alpha_tensor.mean(dim=0).tolist(),
            'std_alpha': alpha_tensor.std(dim=0).tolist(),
            'max_alpha': alpha_tensor.max(dim=0)[0].tolist(),
            'min_alpha': alpha_tensor.min(dim=0)[0].tolist(),
            'expert_names': self.expert_names,
        }
        
        # 计算每个专家被"主要使用"的次数 (α > 0.5)
        dominant_counts = (alpha_tensor > 0.5).sum(dim=0).tolist()
        stats['dominant_counts'] = dominant_counts
        
        return stats


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*60)
    print("测试 GATGuidedMoEPolicy")
    print("="*60 + "\n")
    
    # 初始化
    policy = GATGuidedMoEPolicy(
        state_dim=115,
        hidden_dim=256,
        action_dim=23,
        num_experts=4,
        use_sparsemax=True
    )
    
    # 测试1: 单样本前向传播
    print("\n[测试1] 单样本前向传播")
    state = torch.randn(115)
    test_atoms = {
        "pre_nodes": ["hp_full", "has_gold"],
        "scene_atoms": ["dlvl_1", "monsters_present"]
    }
    
    logits, alpha, value, aux_info = policy(state, atoms=test_atoms)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Alpha shape: {alpha.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Alpha values: {alpha.detach().numpy()}")
    print(f"  Expert usage: {dict(zip(policy.expert_names, alpha.detach().numpy()))}")
    
    # 测试2: 批处理
    print("\n[测试2] 批处理前向传播")
    batch_state = torch.randn(4, 115)
    logits_batch, alpha_batch, value_batch, _ = policy(batch_state, atoms=test_atoms)
    print(f"  Batch logits shape: {logits_batch.shape}")
    print(f"  Batch alpha shape: {alpha_batch.shape}")
    print(f"  Batch value shape: {value_batch.shape}")
    
    # 测试3: 动作分布
    print("\n[测试3] 获取动作分布")
    dist, value = policy.get_action_distribution(state, atoms=test_atoms)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    print(f"  Sampled action: {action.item()}")
    print(f"  Log prob: {log_prob.item():.4f}")
    print(f"  Value: {value.item():.4f}")
    
    # 测试4: 专家使用统计
    print("\n[测试4] 专家使用统计")
    alpha_history = [alpha_batch[i] for i in range(4)]
    stats = policy.get_expert_usage_stats(alpha_history)
    print(f"  Mean alpha: {stats['mean_alpha']}")
    print(f"  Dominant counts: {dict(zip(stats['expert_names'], stats['dominant_counts']))}")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60 + "\n")
