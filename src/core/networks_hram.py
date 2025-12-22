"""
H-RAM (Hypergraph-Retrieval Augmented MoE) 网络架构
基于检索增强的混合专家模型，替代手工特征工程

版本历史:
- V1: 手工特征 + 软融合 (MultiChannelPolicyNet)
- V2: Gumbel-Softmax + Sparse MoE
- V3: H-RAM 端到端学习 (失败，策略崩溃)
- V3.2: H-RAM 改进版 - Top-K检索 + 降维注意力 (当前版本)

核心设计思想:
1. 检索流(3072维): 用于在超图知识库中查找相关知识
2. 决策流(256维): 用于RL策略网络的实际决策
3. 知识适配器: 将高维知识压缩到决策空间
4. 交叉注意力: 让Agent学会从多条知识中挑选最有用的
5. 残差融合: 保证即使知识无用，也能用原始状态决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


class HypergraphMemory(nn.Module):
    """
    超图记忆模块 - Key-Value 存储库
    
    设计思想:
    - 超图知识库包含数百条超边，每条超边已经用LLM预编码为3072维向量
    - 给定一个查询向量，返回Top-K条最相关的知识
    - 不做内部投影，保持原始3072维，由下游Adapter降维
    
    为什么这样设计:
    - Top-K硬检索 vs 软加权: 硬检索可以让模型学会“挑选”而不是“平均”
    - 保持原始嵌入: 让Adapter学习如何压缩，而不是在这里压缩
    """
    
    def __init__(self, 
                 hypergraph_path: str,
                 embedding_index_path: str = "data/cache/hypergraph_embedding_index_minsup5.pkl",
                 embed_dim: int = 3072):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # ========== 加载超图结构 ==========
        # hypergraph_complete_real.json 包含超边的元数据（前置条件、场景、效果等）
        with open(hypergraph_path, 'r') as f:
            import json
            self.hypergraph = json.load(f)
        
        # ========== 加载预计算的超边嵌入 ==========
        # embedding_index 包含:
        #   - "embeddings": (N, 3072) 每条超边的LLM嵌入向量
        #   - "meta": 每条超边的元数据（用于可解释性）
        with open(embedding_index_path, 'rb') as f:
            embedding_index = pickle.load(f)
        
        # ========== 注册为buffer ==========
        # 使用register_buffer而不是nn.Parameter，因为这些嵌入是预计算的，不需要梯度
        # buffer会随模型自动迁移到GPU
        keys = torch.FloatTensor(embedding_index["embeddings"])  # (num_edges, 3072)
        self.register_buffer('keys', keys)
        # 检索稳定性：使用归一化后的keys做相似度（cosine），避免dot-product数值爆炸
        # keys_norm 是 keys 的派生量，不需要写入 checkpoint（否则旧 checkpoint 会缺 key）
        self.register_buffer('keys_norm', F.normalize(keys, dim=-1), persistent=False)
        self.edge_meta = embedding_index["meta"]  # 元数据列表，用于调试和可解释性
        
        print(f"[HypergraphMemory] 加载 {len(self.keys)} 条超边嵌入")
        
    def forward(self, query: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Top-K硬检索，返回原始嵌入
        
        流程:
        1. 计算query与所有超边的相似度分数
        2. 选取Top-K个最相似的超边
        3. 返回这K条超边的原始3072维嵌入
        
        Args:
            query: (batch, 3072) 状态编码后的查询向量
            k: 检索多少条知识，默认10
            
        Returns:
            top_k_embeddings: (batch, k, 3072) Top-K个超边的原始嵌入
            top_edges: 检索到的超边元数据列表（用于可解释性）
        """
        batch_size = query.size(0)
        
        # ========== Step 1: 计算相似度 ==========
        # query: (B, 3072), keys: (N, 3072)
        # scores: (B, N) - 每个样本对所有超边的相似度
        query = torch.nan_to_num(query, nan=0.0, posinf=1.0, neginf=-1.0)
        query_norm = F.normalize(query, dim=-1)
        scores = torch.matmul(query_norm, self.keys_norm.T)
        
        # ========== Step 2: Top-K硬检索 ==========
        # 不用softmax软加权，直接选出最相关的K条
        # topk_indices: (B, K) - 每个样本Top-K超边的索引
        _, topk_indices = torch.topk(scores, k=k, dim=-1)
        
        # ========== Step 3: 提取嵌入 ==========
        # 使用高级索引: self.keys[topk_indices] 自动广播
        # 结果: (B, K, 3072)
        top_k_embeddings = self.keys[topk_indices]
        
        # ========== Step 4: 收集元数据(可选，用于调试) ==========
        top_edges = []
        for i in range(batch_size):
            edges = [self.edge_meta[idx.item()] for idx in topk_indices[i]]
            top_edges.append(edges)
        
        return top_k_embeddings, top_edges


class KnowledgeAdapter(nn.Module):
    """
    知识适配器 - 降维打击
    
    设计思想:
    - LLM嵌入是3072维（文本空间），对RL来说太大了
    - RL喜欢在低维空间(256维)工作，梯度更稳定
    - 这个模块就是“翻译官”，把文本知识翻译成RL能理解的语言
    
    结构: 3072 -> 1024 -> 256 + LayerNorm
    - 两层MLP逐步压缩，不会丢失太多信息
    - LayerNorm稳定梯度，防止训练崩溃
    """
    def __init__(self, input_dim: int = 3072, output_dim: int = 256):
        super().__init__()
        
        # ========== 两层MLP + LayerNorm ==========
        # 为什么用LayerNorm:
        #   - 稳定训练，防止梯度爆炸/消失
        #   - 对batch size不敏感（不像BatchNorm）
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 1024),   # 3072 -> 1024: 第一次压缩
            nn.ReLU(),                     # 非线性激活
            nn.Linear(1024, output_dim),  # 1024 -> 256: 第二次压缩
            nn.LayerNorm(output_dim)       # 归一化，稳定梯度
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将高维知识压缩到低维决策空间
        
        注意: 这个模块支持任意形状的输入
        - (batch, 3072) -> (batch, 256)
        - (batch, k, 3072) -> (batch, k, 256)  ← 常用，对K条知识分别压缩
        
        Args:
            x: 输入张量，最后一维是3072
            
        Returns:
            压缩后的张量，最后一维是256
        """
        return self.projector(x)


class StateEncoder(nn.Module):
    """
    状态编码器 - 将原始观测编码为查询向量
    
    设计思想:
    - 原始状态只有115维，需要扩展到3072维才能和超图嵌入做相似度计算
    - 用深层MLP学习怎么把状态变成“查字典的查询词”
    - 这个编码器只用于检索，不用于决策（决策有单独的编码器）
    
    结构: 115 -> 512 -> 512 -> 3072 + LayerNorm
    - 逐步扩展维度，让网络学习如何构造查询向量
    """
    
    def __init__(self, 
                 raw_obs_dim: int = 115,  # 原始状态维度 (NetHack blstats等)
                 embed_dim: int = 3072,   # 目标嵌入维度 (与LLM嵌入一致)
                 hidden_dim: int = 512):  # 隐藏层维度
        super().__init__()
        
        # ========== 深层MLP编码器 ==========
        # 为什么用深层MLP:
        #   - 状态->3072维是一个复杂的映射，需要足够的表达能力
        #   - 两层512维隐藏层提供足够的非线性变换
        self.encoder = nn.Sequential(
            nn.Linear(raw_obs_dim, hidden_dim),  # 115 -> 512
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),    # 512 -> 512
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),     # 512 -> 3072
            nn.LayerNorm(embed_dim)                # 归一化，方便做相似度计算
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        编码状态为查询向量
        
        流程: state(115维) -> MLP -> query(3072维)
        这个query将用于在超图知识库中搜索相关知识
        
        Args:
            state: (batch, 115) 原始状态向量
            
        Returns:
            query: (batch, 3072) 查询向量，用于检索
        """
        query = self.encoder(state)
        return query


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块 - 让Agent学会“翻书”
    
    设计思想:
    - Agent有一个“问题”(state)，手上有K本“参考书”(knowledge)
    - 用注意力机制让Agent学会：“这K本书里，哪本对我当前的问题最有用？”
    - 这就是“检索增强”的核心: 不是死记硬背，而是根据情况查资料
    
    数学本质:
    - Query(Q): 当前状态 (1个向量)
    - Key(K): K条知识 (K个向量)
    - Value(V): K条知识 (同K)
    - 输出: softmax(Q·K^T) · V -> 加权平均的知识
    
    为什么这样比V3好:
    - V3是1对1的伪注意力，softmax恒等于1，没有选择
    - V3.2是1对K的真注意力，模型可以学会“第1条重要，第2条不重要”
    """
    
    def __init__(self, 
                 d_model: int = 256,      # 决策空间维度
                 num_heads: int = 4,      # 注意力头数
                 action_dim: int = 23):   # 动作空间大小 (NetHack有23个动作)
        super().__init__()
        
        # ========== 多头注意力 ==========
        # 为什么4个头: 让不同的头关注不同方面的知识
        # batch_first=True: 输入格式是(batch, seq, dim)而不是(seq, batch, dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # ========== 动作决策头 ==========
        # 输入: state(256) + context(256) = 512维
        # 为什么拼接:
        #   - 残差连接思想: 即使知识无用，也能用原始状态决策
        #   - 防止知识检索失败导致完全崩溃
        self.action_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # 512 -> 256
            nn.ReLU(),
            nn.Linear(d_model, action_dim)     # 256 -> 23
        )
        
        # ========== 初始化 ==========
        # 最后一层用小增益正交初始化，防止初始logits太大
        nn.init.orthogonal_(self.action_head[-1].weight, gain=0.01)
        nn.init.constant_(self.action_head[-1].bias, 0)
        
    def forward(self, 
                state_query: torch.Tensor, 
                knowledge_bank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        交叉注意力融合
        
        流程:
        1. state作为Query，K条知识作为Key/Value
        2. 注意力计算：哪条知识最相关？
        3. 加权求和得到context
        4. 拼接state+context，输出动作
        
        Args:
            state_query: (batch, 1, 256) 当前状态向量
            knowledge_bank: (batch, K, 256) K条知识向量
            
        Returns:
            logits: (batch, 23) 动作概率分布(unnormalized)
            attn_weights: (batch, 1, K) 注意力权重，可以看到模型关注了哪条知识
        """
        # ========== Step 1: 注意力计算 ==========
        # attn_output: (batch, 1, 256) - 加权平均后的知识
        # attn_weights: (batch, 1, K) - 每条知识的权重（和为1）
        # 例如: [0.7, 0.1, 0.1, 0.1] 表示第1条知识最重要
        attn_output, attn_weights = self.mha(
            query=state_query,
            key=knowledge_bank,
            value=knowledge_bank
        )
        
        # ========== Step 2: 残差融合 ==========
        # 拼接原始状态 + 注意力输出
        # 这样即使知识检索完全失败，state仍然可用
        combined = torch.cat([
            state_query.squeeze(1),      # (batch, 256) - 原始状态
            attn_output.squeeze(1)       # (batch, 256) - 知识上下文
        ], dim=-1)  # (batch, 512)
        
        # ========== Step 3: 生成动作 ==========
        logits = self.action_head(combined)
        
        return logits, attn_weights


class HRAMPolicyNet(nn.Module):
    """
    H-RAM V3.2 策略网络 - 真正的检索增强架构
    
    ============================================================
    整体架构图:
    ============================================================
    
    输入: state (115维, 包含HP/Gold/Depth等游戏状态)
           │
           ├──────────────────────┬──────────────────────┤
           ↓                      ↓                      ↓
    [检索编码器]           [决策编码器]            [Critic]
    115 -> 3072            115 -> 256             115 -> value
           │                      │
           ↓                      │
    [超图记忆库]                 │
    检索Top-K条知识            │
    (K, 3072)                  │
           │                      │
           ↓                      │
    [知识适配器]                 │
    3072 -> 256                │
    (K, 256)                   │
           │                      │
           └───────────┬───────────┘
                       ↓
              [交叉注意力融合]
              Q: state(1, 256)
              K/V: knowledge(K, 256)
                       │
                       ↓
              [拼接融合] state + context
                       │
                       ↓
              [动作头] -> logits(23)
    
    ============================================================
    为什么这样设计:
    ============================================================
    1. 双流分离: 检索流(3072)和决策流(256)分开，防止梯度打架
    2. 降维瓶颈: RL在256维空间工作，梯度更稳定
    3. Top-K检索: 让模型学会“挑选”而不是“平均”
    4. 残差融合: 即使知识无用，也能用原始状态决策
    """
    
    def __init__(self,
                 state_dim: int = 115,        # 原始状态维度 (NetHack blstats)
                 embed_dim: int = 3072,       # 检索空间维度 (与LLM嵌入一致)
                 internal_dim: int = 256,     # 决策空间维度 (RL友好)
                 top_k: int = 10,             # 检索多少条知识
                 action_dim: int = 23,        # 动作空间大小 (NetHack)
                 hypergraph_path: str = "data/hypergraph/hypergraph_complete_real.json",
                 embedding_index_path: str = "data/cache/hypergraph_embedding_index_minsup5.pkl"):
        super().__init__()
        
        # 保存配置
        self.top_k = top_k
        self.internal_dim = internal_dim
        self.action_dim = action_dim
        
        # ========== 组件 1: 检索编码器 ==========
        # 作用: 把状态变成“查字典的查询词”
        # 维度: 115 -> 3072 (必须和LLM嵌入一致才能计算相似度)
        # 注意: 这个编码器只用于检索，不用于决策
        self.retrieval_encoder = StateEncoder(
            raw_obs_dim=state_dim,
            embed_dim=embed_dim
        )
        
        # ========== 组件 2: 决策编码器 ==========
        # 作用: 把状态变成RL能理解的低维表示
        # 维度: 115 -> 256 (RL喜欢低维空间)
        # 为什么分开: 检索和决策是两个不同的任务，让它们学不同的表示
        self.decision_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),        # 115 -> 256
            nn.ReLU(),
            nn.Linear(256, internal_dim),     # 256 -> 256
            nn.LayerNorm(internal_dim)         # 归一化，稳定梯度
        )
        
        # ========== 组件 3: 超图记忆库 ==========
        # 作用: 存储所有超边的LLM嵌入，提供Top-K检索
        # 输入: 查询向量 (batch, 3072)
        # 输出: Top-K条知识 (batch, K, 3072)
        self.memory = HypergraphMemory(
            hypergraph_path=hypergraph_path,
            embedding_index_path=embedding_index_path,
            embed_dim=embed_dim
        )
        
        # ========== 组件 4: 知识适配器 ==========
        # 作用: 把高维知识压缩到低维决策空间
        # 维度: 3072 -> 256
        # 为什么需要: LLM空间(3072)太大，RL在小空间(256)训练更稳定
        self.adapter = KnowledgeAdapter(
            input_dim=embed_dim,
            output_dim=internal_dim
        )
        
        # ========== 组件 5: 交叉注意力融合 ==========
        # 作用: 让Agent学会从K条知识中挑选最有用的
        # 输入: state(1, 256) + knowledge(K, 256)
        # 输出: logits(23) + 注意力权重(K)
        self.fusion = CrossAttentionFusion(
            d_model=internal_dim,
            num_heads=4,                       # 4个注意力头
            action_dim=action_dim
        )
        
        # ========== 组件 6: Critic网络 ==========
        # 作用: 估计当前状态的价值 (用于PPO的优势计算)
        # 注意: Critic直接从原始状态计算，不用知识
        #       这样设计是因为价值估计不需要知识检索
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),         # 115 -> 256
            nn.ReLU(),
            nn.Linear(256, 256),               # 256 -> 256
            nn.ReLU(),
            nn.Linear(256, 1)                  # 256 -> 1 (状态价值)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播 - 整个H-RAM的核心流程
        
        流程图:
        state(115) ─┬─> retrieval_encoder ─> query(3072) ─> memory ─> raw_knowledge(K,3072)
                   │                                                    │
                   │                                                    ↓
                   │                                              adapter ─> compact_knowledge(K,256)
                   │                                                    │
                   ├─> decision_encoder ─> agent_state(1,256) ────────┬┘
                   │                                                    │
                   │                                                    ↓
                   │                                              fusion ─> logits(23), attn_weights(K)
                   │
                   └─> critic ─> value(1)
        
        Args:
            state: (batch, 115) 原始状态向量
            
        Returns:
            logits: (batch, 23) 动作概率分布(unnormalized)
            attn_weights: (batch, 1, K) 注意力权重，用于可解释性分析
            value: (batch, 1) 状态价值，用于PPO训练
        """
        # ========== A. 检索流 ==========
        # 目的: 从知识库中找到与当前状态相关的知识
        
        # Step A1: 把状态编码成查询向量
        retrieval_query = self.retrieval_encoder(state)  # (batch, 3072)
        
        # Step A2: 在超图记忆库中检索Top-K条最相关的知识
        raw_knowledge, _ = self.memory(retrieval_query, k=self.top_k)  # (batch, K, 3072)
        
        # Step A3: 把高维知识压缩到低维决策空间
        compact_knowledge = self.adapter(raw_knowledge)  # (batch, K, 256)
        
        # ========== B. 决策流 ==========
        # 目的: 把状态编码成RL可用的表示
        
        # 注意: unsqueeze(1)把(batch, 256)变成(batch, 1, 256)
        #       这是为了和注意力机制的输入格式匹配
        agent_state = self.decision_encoder(state).unsqueeze(1)  # (batch, 1, 256)
        
        # ========== C. 知识融合 ==========
        # 目的: 让Agent“看着K本参考书思考”，然后做决策
        
        # fusion返回:
        #   - logits: (batch, 23) 动作概率
        #   - attn_weights: (batch, 1, K) 每条知识的重要性权重
        logits, attn_weights = self.fusion(agent_state, compact_knowledge)
        
        # D. 计算价值
        value = self.critic(state)
        
        return logits, attn_weights, value
    
    def get_action_distribution(self, state: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """获取动作分布和价值"""
        logits, _, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value


# V1/V2 的 MultiChannelPolicyNet 保留在 networks_correct.py 中
# V3.2 的 HRAMPolicyNet 在此文件中实现


# ============================================================================
# 文档方案: H-RAM with 4 Actors (保留4个专家 + 检索上下文)
# ============================================================================

class ContextCompressor(nn.Module):
    """
    上下文压缩器 - 将3072维检索上下文压缩到128维
    用于适配原有的Actor输入
    """
    def __init__(self, input_dim: int = 3072, output_dim: int = 128):
        super().__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.compressor(context)


class HRAMActorPre(nn.Module):
    """H-RAM 版 ActorPre - 使用检索上下文替代手工q_pre"""
    def __init__(self, context_dim: int = 128, belief_dim: int = 50, 
                 hidden_dim: int = 128, action_dim: int = 23):
        super().__init__()
        # 输入: retrieved_context(128) + belief(50) = 178
        self.fc1 = nn.Linear(context_dim + belief_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        # 初始化最后一层为小值，防止初始logits过大导致梯度爆炸
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, context: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        x = torch.cat([context, belief], dim=-1)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class HRAMActorScene(nn.Module):
    """H-RAM 版 ActorScene"""
    def __init__(self, context_dim: int = 128, location_dim: int = 20,
                 hidden_dim: int = 128, action_dim: int = 23):
        super().__init__()
        self.fc1 = nn.Linear(context_dim + location_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, context: torch.Tensor, location: torch.Tensor) -> torch.Tensor:
        x = torch.cat([context, location], dim=-1)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class HRAMActorEffect(nn.Module):
    """H-RAM 版 ActorEffect"""
    def __init__(self, context_dim: int = 128, hp_dim: int = 10,
                 hidden_dim: int = 128, action_dim: int = 23):
        super().__init__()
        self.fc1 = nn.Linear(context_dim + hp_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, context: torch.Tensor, hp_context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([context, hp_context], dim=-1)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class HRAMActorRule(nn.Module):
    """H-RAM 版 ActorRule"""
    def __init__(self, context_dim: int = 128, inventory_dim: int = 15,
                 hidden_dim: int = 128, action_dim: int = 23):
        super().__init__()
        self.fc1 = nn.Linear(context_dim + inventory_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, context: torch.Tensor, inventory: torch.Tensor) -> torch.Tensor:
        x = torch.cat([context, inventory], dim=-1)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class GumbelRouter(nn.Module):
    """Gumbel-Softmax 路由器 - 硬路由版本"""
    def __init__(self, state_dim: int = 115, hidden_dim: int = 64, tau: float = 1.0):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)  # 4个专家
        self.relu = nn.ReLU()
        self.tau = tau
    
    def forward(self, state: torch.Tensor, use_gumbel: bool = True) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        
        if use_gumbel:
            # 训练时用硬路由
            alpha = F.gumbel_softmax(logits, tau=self.tau, hard=True)
        else:
            alpha = F.softmax(logits, dim=-1)
        
        return alpha


class HRAMPolicyNetDoc(nn.Module):
    """
    H-RAM 策略网络 - 文档方案
    
    特点：
    - 保留4个Actor专家
    - 使用检索上下文替代手工q_pre/q_scene等
    - Gumbel-Softmax硬路由
    
    架构流程：
    1. State → StateEncoder → Query
    2. Query + HypergraphMemory → Context (神经检索)
    3. Context → ContextCompressor → SharedKnowledge (128维)
    4. SharedKnowledge + State切片 → 4个Actor → Logits
    5. State → GumbelRouter → Weights
    6. Σ(Weights * Logits) → Action
    """
    
    def __init__(self,
                 state_dim: int = 115,
                 embed_dim: int = 3072,
                 context_dim: int = 128,
                 action_dim: int = 23,
                 hypergraph_path: str = "data/hypergraph/hypergraph_complete_real.json",
                 embedding_index_path: str = "data/cache/hypergraph_embedding_index_minsup5.pkl",
                 gumbel_tau: float = 1.0):
        super().__init__()
        
        # 1. 状态编码器
        self.state_encoder = StateEncoder(
            raw_obs_dim=state_dim,
            embed_dim=embed_dim
        )
        
        # 2. 超图记忆
        self.hypergraph_memory = HypergraphMemory(
            hypergraph_path=hypergraph_path,
            embedding_index_path=embedding_index_path,
            embed_dim=embed_dim
        )
        
        # 3. 上下文压缩器
        self.context_compressor = ContextCompressor(
            input_dim=embed_dim,
            output_dim=context_dim
        )
        
        # 4. 四个专家Actor (文档方案核心)
        self.actor_pre = HRAMActorPre(context_dim, 50, 128, action_dim)
        self.actor_scene = HRAMActorScene(context_dim, 20, 128, action_dim)
        self.actor_effect = HRAMActorEffect(context_dim, 10, 128, action_dim)
        self.actor_rule = HRAMActorRule(context_dim, 15, 128, action_dim)
        
        # 5. Gumbel路由器
        self.router = GumbelRouter(state_dim, 64, gumbel_tau)
        
        # 6. Critic网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.action_dim = action_dim
    
    def extract_state_slices(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """从state中提取各Actor需要的切片"""
        belief = state[..., :50]  # (batch, 50)
        location = state[..., 20:40]  # (batch, 20)
        hp_context = state[..., 40:50]  # (batch, 10)
        inventory = state[..., 45:60]  # (batch, 15)
        return {
            'belief': belief,
            'location': location,
            'hp_context': hp_context,
            'inventory': inventory
        }
    
    def forward(self, state: torch.Tensor, use_gumbel: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            logits: (batch, action_dim)
            alpha: (batch, 4) 路由权重
            value: (batch, 1)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # 1. 编码状态为查询
        query = self.state_encoder(state)  # (batch, embed_dim)
        
        # 2. 从超图记忆中检索
        context, _ = self.hypergraph_memory(query)  # (batch, k, embed_dim)
        # 平均池化Top-K结果
        context = context.mean(dim=1)  # (batch, embed_dim)
        
        # 3. 压缩上下文
        shared_knowledge = self.context_compressor(context)  # (batch, context_dim)
        
        # 4. 提取状态切片
        slices = self.extract_state_slices(state)
        
        # 5. 四个专家分别处理
        logits_pre = self.actor_pre(shared_knowledge, slices['belief'])
        logits_scene = self.actor_scene(shared_knowledge, slices['location'])
        logits_effect = self.actor_effect(shared_knowledge, slices['hp_context'])
        logits_rule = self.actor_rule(shared_knowledge, slices['inventory'])
        
        # 6. Gumbel路由
        alpha = self.router(state, use_gumbel=use_gumbel)  # (batch, 4)
        
        # 7. 加权融合
        logits_stack = torch.stack([logits_pre, logits_scene, logits_effect, logits_rule], dim=1)
        alpha_expanded = alpha.unsqueeze(2)
        fused_logits = (alpha_expanded * logits_stack).sum(dim=1)
        
        # 8. 价值估计
        value = self.critic(state)
        
        return fused_logits, alpha, value
    
    def get_action_distribution(self, state: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """获取动作分布和价值"""
        logits, _, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value
