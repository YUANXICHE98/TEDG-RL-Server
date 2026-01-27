"""
V4: Cross-Attention Guided Hierarchical MoE
Causal-Gated Cross-Attention for Modal Balance

核心创新 (相比V3):
1. **Cross-Attention融合**: 替代简单concat，让符号信息主动查询视觉信息
2. **Sparse Attention Gate**: 只关注相关的视觉特征，过滤噪声
3. **Context Vector**: 生成紧凑的256维上下文表示
4. **模态平衡**: 缓解V3中可能存在的模态主导问题

V3 → V4的关键变化:
- V3: z = concat(h_vis, h_logic)  # 512维
- V4: c = CrossAttn(Q=h_logic, K=h_vis, V=h_vis)  # 256维
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math

from src.core.hypergraph_gat import HypergraphGAT


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax激活函数 (沿用V3)
    
    相比Softmax:
    - 可以产生稀疏输出 (某些维度为0)
    - 软中带硬，避免平均主义
    """
    k = max(2, logits.size(dim) // 2)
    topk_values, topk_indices = torch.topk(logits, k, dim=dim)
    topk_probs = F.softmax(topk_values, dim=dim)
    output = torch.zeros_like(logits)
    output.scatter_(dim, topk_indices, topk_probs)
    return output


class CausalGatedCrossAttention(nn.Module):
    """
    因果门控交叉注意力 (The Semantic Prism)
    
    核心思想:
    - Query: 来自GAT的高层意图 (h_logic) - "我想做什么"
    - Key/Value: 来自CNN的视觉特征 (h_vis) - "环境中有什么"
    - 让符号信息主动查询相关的视觉特征
    
    流程:
    1. 投影: Q ← h_logic, K,V ← h_vis
    2. 稀疏注意力: Attention = Softmax(Q@K^T / √d_k) [稀疏化]
    3. 语义过滤: c = Σ(Attention ⊙ V)
    
    Args:
        hidden_dim: 隐藏层维度 (默认256)
        num_heads: 注意力头数 (默认4)
        dropout: Dropout率 (默认0.1)
        sparse_topk: 稀疏化保留的top-k比例 (默认0.3)
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 sparse_topk: float = 0.3):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sparse_topk = sparse_topk
        
        # Query投影 (from h_logic)
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        
        # Key/Value投影 (from h_vis)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.xavier_uniform_(self.W_O.weight)
    
    def forward(self, 
                h_logic: torch.Tensor, 
                h_vis: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            h_logic: (batch, hidden_dim) GAT输出的高层意图
            h_vis: (batch, hidden_dim) CNN输出的视觉特征
            return_attention: 是否返回注意力权重（用于可视化）
        
        Returns:
            c: (batch, hidden_dim) Context Vector
            attention_weights: (batch, num_heads, 1, 1) 注意力权重（可选）
        """
        batch_size = h_logic.size(0)
        
        # 1. 投影到Q, K, V
        Q = self.W_Q(h_logic)  # (batch, hidden_dim)
        K = self.W_K(h_vis)    # (batch, hidden_dim)
        V = self.W_V(h_vis)    # (batch, hidden_dim)
        
        # 2. 分头 (Multi-Head)
        # (batch, hidden_dim) → (batch, num_heads, head_dim)
        Q = Q.view(batch_size, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.num_heads, self.head_dim)
        
        # 3. 计算注意力分数
        # (batch, num_heads, head_dim) @ (batch, num_heads, head_dim)^T
        # → (batch, num_heads, 1, 1) [因为Q和K都是单个向量]
        scores = torch.einsum('bhd,bhd->bh', Q, K) / math.sqrt(self.head_dim)
        scores = scores.unsqueeze(-1).unsqueeze(-1)  # (batch, num_heads, 1, 1)
        
        # 4. Sparse Attention Gate (稀疏化)
        # 只保留top-k的注意力头
        if self.sparse_topk < 1.0:
            k = max(1, int(self.num_heads * self.sparse_topk))
            topk_values, topk_indices = torch.topk(scores.squeeze(-1).squeeze(-1), k, dim=-1)
            
            # 创建稀疏mask
            sparse_mask = torch.zeros_like(scores.squeeze(-1).squeeze(-1))
            sparse_mask.scatter_(-1, topk_indices, 1.0)
            sparse_mask = sparse_mask.unsqueeze(-1).unsqueeze(-1)
            
            # 应用mask
            scores = scores * sparse_mask + (1 - sparse_mask) * (-1e9)
        
        # 5. Softmax归一化
        attention_weights = F.softmax(scores, dim=1)  # (batch, num_heads, 1, 1)
        attention_weights = self.dropout(attention_weights)
        
        # 6. 加权求和
        # (batch, num_heads, 1, 1) * (batch, num_heads, head_dim)
        # → (batch, num_heads, head_dim)
        context = attention_weights.squeeze(-1).squeeze(-1).unsqueeze(-1) * V
        
        # 7. 合并多头
        # (batch, num_heads, head_dim) → (batch, hidden_dim)
        context = context.view(batch_size, self.hidden_dim)
        
        # 8. 输出投影
        c = self.W_O(context)
        
        # 9. 残差连接 + LayerNorm
        # 以h_logic为基准做残差
        c = self.layer_norm(c + h_logic)
        
        if return_attention:
            return c, attention_weights
        else:
            return c, None


class SemanticExpert(nn.Module):
    """
    语义专家 (沿用V3)
    
    专家定义:
    - Survival: 生存相关 (吃喝、回血、逃跑)
    - Combat: 战斗相关 (攻击、走位、使用武器)
    - Exploration: 探索相关 (开图、搜索、捡东西)
    - General: 通用/兜底
    """
    
    def __init__(self, 
                 input_dim: int = 256,  # V4: 输入是Context Vector (256维)
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
        
        nn.init.orthogonal_(self.network[-1].weight, gain=0.5)
        nn.init.constant_(self.network[-1].bias, 0)
    
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c: (batch, input_dim) Context Vector
        
        Returns:
            logits: (batch, action_dim) 动作logits
        """
        return self.network(c)


class CausalRouter(nn.Module):
    """
    因果路由器 (修改为接受Context Vector)
    
    V3: 输入 z = concat(h_vis, h_logic) (512维)
    V4: 输入 c = CrossAttn(h_logic, h_vis) (256维)
    """
    
    def __init__(self, 
                 input_dim: int = 256,  # V4: Context Vector维度
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
        
        nn.init.orthogonal_(self.router[-1].weight, gain=0.1)
    
    def forward(self, c: torch.Tensor, use_sparsemax: bool = True) -> torch.Tensor:
        """
        Args:
            c: (batch, input_dim) Context Vector
            use_sparsemax: 是否使用Sparsemax
        
        Returns:
            alpha: (batch, num_experts) 专家权重
        """
        logits = self.router(c)
        logits = torch.nan_to_num(logits, nan=0.0)
        
        if use_sparsemax:
            alpha = sparsemax(logits, dim=-1)
        else:
            alpha = F.softmax(logits, dim=-1)
        
        return alpha


class CrossAttentionMoEPolicy(nn.Module):
    """
    V4: Cross-Attention引导的混合专家策略网络
    
    架构流程:
    1. 双流编码: Visual (blstats) + Logic (GAT)
    2. **Cross-Attention融合**: c = CrossAttn(Q=h_logic, K=h_vis, V=h_vis)
    3. 因果路由: Sparsemax选择专家 (输入c)
    4. 专家融合: 加权组合 (输入c)
    5. 价值估计: Critic网络
    
    Args:
        hypergraph_path: 超图结构文件路径
        state_dim: 输入状态维度 (默认115)
        hidden_dim: 隐藏层维度 (默认256)
        action_dim: 动作空间大小 (默认23)
        num_experts: 专家数量 (默认4)
        use_sparsemax: 是否使用Sparsemax路由 (默认True)
        cross_attn_heads: Cross-Attention头数 (默认4)
        sparse_topk: 稀疏注意力保留比例 (默认0.3)
    """
    
    def __init__(self,
                 hypergraph_path: str = "data/hypergraph/hypergraph_gat_structure.json",
                 state_dim: int = 115,
                 hidden_dim: int = 256,
                 action_dim: int = 23,
                 num_experts: int = 4,
                 use_sparsemax: bool = True,
                 cross_attn_heads: int = 4,
                 sparse_topk: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.use_sparsemax = use_sparsemax
        
        # 1. 超图GAT (Logic Stream) - 沿用V3
        self.gat = HypergraphGAT(
            hypergraph_path=hypergraph_path,
            hidden_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # 2. 视觉编码器 (Visual Stream) - 沿用V3
        self.visual_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 3. **Cross-Attention融合层** (NEW!)
        self.cross_attention = CausalGatedCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=cross_attn_heads,
            dropout=0.1,
            sparse_topk=sparse_topk
        )
        
        # 4. 因果路由器 (修改为接受Context Vector)
        self.router = CausalRouter(
            input_dim=hidden_dim,  # V4: 256维 (vs V3的512维)
            hidden_dim=128,
            num_experts=num_experts
        )
        
        # 5. 语义专家 (修改为接受Context Vector)
        self.experts = nn.ModuleList([
            SemanticExpert(
                input_dim=hidden_dim,  # V4: 256维
                hidden_dim=128,
                action_dim=action_dim
            )
            for _ in range(num_experts)
        ])
        
        # 6. Critic网络 (沿用V3)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )
    
    def forward(self, 
                state: torch.Tensor,
                atoms: Dict[str, List[str]],
                return_attention: bool = False,
                return_expert_logits: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: (batch, state_dim) 状态向量
            atoms: 超图atoms字典
            return_attention: 是否返回注意力权重
            return_expert_logits: 是否返回专家logits
        
        Returns:
            outputs: {
                'policy_logits': (batch, action_dim) 最终策略logits
                'value': (batch, 1) 状态价值
                'alpha': (batch, num_experts) 专家权重
                'context_vector': (batch, hidden_dim) Context Vector
                'attention_weights': (batch, num_heads, 1, 1) 注意力权重 (可选)
                'expert_logits': (batch, num_experts, action_dim) 专家logits (可选)
                'h_logic': (batch, hidden_dim) GAT输出
                'h_vis': (batch, hidden_dim) CNN输出
                'operator_scores': (batch, num_operators) GAT的operator激活分数
            }
        """
        # 处理单个样本
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = state.size(0)
        
        # 1. 双流编码
        # Visual Stream
        h_vis = self.visual_encoder(state)  # (batch, 256)
        
        # Logic Stream (GAT)
        # 注意: 当前实现假设batch内所有样本使用相同的atoms
        if batch_size == 1:
            # 单样本: 直接调用GAT
            h_logic, operator_scores, gat_attention = self.gat(atoms=atoms)
            h_logic = h_logic.unsqueeze(0)  # (1, 256)
            operator_scores = operator_scores.unsqueeze(0)  # (1, num_operators)
        else:
            # 批处理: 逐个处理 (与V3保持一致)
            h_logic_list = []
            operator_scores_list = []
            for i in range(batch_size):
                h_i, scores_i, _ = self.gat(atoms=atoms)
                h_logic_list.append(h_i)
                operator_scores_list.append(scores_i)
            h_logic = torch.stack(h_logic_list, dim=0)
            operator_scores = torch.stack(operator_scores_list, dim=0)
            gat_attention = None  # 批处理时不返回attention
        
        # 2. **Cross-Attention融合** (V4核心改动)
        c, attention_weights = self.cross_attention(
            h_logic=h_logic,
            h_vis=h_vis,
            return_attention=return_attention
        )
        # c: (batch, 256) Context Vector
        
        # 3. 因果路由
        alpha = self.router(c, use_sparsemax=self.use_sparsemax)
        # alpha: (batch, num_experts)
        
        # 4. 专家推理
        expert_logits_list = []
        for expert in self.experts:
            expert_logits = expert(c)  # 每个专家都接收Context Vector
            expert_logits_list.append(expert_logits)
        
        expert_logits = torch.stack(expert_logits_list, dim=1)
        # expert_logits: (batch, num_experts, action_dim)
        
        # 5. 加权融合
        # (batch, num_experts, 1) * (batch, num_experts, action_dim)
        # → (batch, num_experts, action_dim) → sum → (batch, action_dim)
        policy_logits = torch.sum(
            alpha.unsqueeze(-1) * expert_logits,
            dim=1
        )
        
        # 6. 价值估计
        value = self.critic(c)
        
        # 7. 构造输出
        outputs = {
            'policy_logits': policy_logits,
            'value': value,
            'alpha': alpha,
            'context_vector': c,
            'h_logic': h_logic,
            'h_vis': h_vis,
            'operator_scores': operator_scores,
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        if return_expert_logits:
            outputs['expert_logits'] = expert_logits
        
        return outputs
    
    def get_value(self, state: torch.Tensor, atoms: Dict[str, List[str]]) -> torch.Tensor:
        """
        获取状态价值 (用于PPO)
        
        Args:
            state: (batch, state_dim)
            atoms: 超图atoms
        
        Returns:
            value: (batch, 1)
        """
        with torch.no_grad():
            outputs = self.forward(state, atoms)
            return outputs['value']
    
    def get_action_and_value(self,
                            state: torch.Tensor,
                            atoms: Dict[str, List[str]],
                            action: Optional[torch.Tensor] = None,
                            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作和价值 (用于PPO)
        
        Args:
            state: (batch, state_dim)
            atoms: 超图atoms
            action: (batch,) 已有动作 (用于计算log_prob)
            deterministic: 是否确定性采样
        
        Returns:
            action: (batch,) 采样的动作
            log_prob: (batch,) 动作的log概率
            entropy: (batch,) 策略熵
            value: (batch, 1) 状态价值
        """
        outputs = self.forward(state, atoms)
        policy_logits = outputs['policy_logits']
        value = outputs['value']
        
        # 创建分布
        probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # 采样动作
        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()
        
        # 计算log_prob和熵
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


# 为了兼容性，创建别名
GATGuidedMoEPolicy = CrossAttentionMoEPolicy
