"""正确的多通道RL网络架构 - 基于文档规范"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class ActorPre(nn.Module):
    """前置条件Actor - 输入: q_pre(15) + belief_context(20)"""
    
    def __init__(self, action_dim: int = 33, hidden_dim: int = 128):
        super().__init__()
        # 输入: q_pre(15) + belief_context(20) = 35
        self.fc1 = nn.Linear(35, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
    
    def forward(self, q_pre: torch.Tensor, belief_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_pre: (batch, 15) 前置条件嵌入
            belief_context: (batch, 20) belief向量的前20维
        Returns:
            logits: (batch, action_dim)
        """
        x = torch.cat([q_pre, belief_context], dim=-1)  # (batch, 35)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class ActorScene(nn.Module):
    """场景原子Actor - 输入: q_scene(15) + location_context(20)"""
    
    def __init__(self, action_dim: int = 33, hidden_dim: int = 128):
        super().__init__()
        # 输入: q_scene(15) + location_context(20) = 35
        self.fc1 = nn.Linear(35, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
    
    def forward(self, q_scene: torch.Tensor, location_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_scene: (batch, 15) 场景原子嵌入
            location_context: (batch, 20) belief向量的20-40维
        Returns:
            logits: (batch, action_dim)
        """
        x = torch.cat([q_scene, location_context], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class ActorEffect(nn.Module):
    """效果/风险Actor - 输入: q_effect(8) + hp_context(10)"""
    
    def __init__(self, action_dim: int = 33, hidden_dim: int = 128):
        super().__init__()
        # 输入: q_effect(8) + hp_context(10) = 18
        self.fc1 = nn.Linear(18, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
    
    def forward(self, q_effect: torch.Tensor, hp_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_effect: (batch, 8) 效果/风险嵌入
            hp_context: (batch, 10) belief向量的40-50维
        Returns:
            logits: (batch, action_dim)
        """
        x = torch.cat([q_effect, hp_context], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class ActorRule(nn.Module):
    """规则/条件效果Actor - 输入: q_rule(10) + inventory_context(15)"""
    
    def __init__(self, action_dim: int = 33, hidden_dim: int = 128):
        super().__init__()
        # 输入: q_rule(10) + inventory_context(15) = 25
        self.fc1 = nn.Linear(25, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
    
    def forward(self, q_rule: torch.Tensor, inventory_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_rule: (batch, 10) 规则模式嵌入
            inventory_context: (batch, 15) 从belief和goal构造
        Returns:
            logits: (batch, action_dim)
        """
        x = torch.cat([q_rule, inventory_context], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits


class AttentionWeightNet(nn.Module):
    """注意力权重网络 - 计算α权重用于融合4个Actor的logits"""
    
    def __init__(self, state_dim: int = 115, hidden_dim: int = 64):
        super().__init__()
        # 输入: 完整state(115)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)  # 输出4个通路的权重
        self.relu = nn.ReLU()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, 115) 完整状态向量
        Returns:
            alpha: (batch, 4) softmax后的权重 [α_pre, α_scene, α_effect, α_rule]
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        logits = self.fc3(x)
        alpha = F.softmax(logits, dim=-1)
        return alpha


class CriticNet(nn.Module):
    """共享Critic网络 - 估计状态价值"""
    
    def __init__(self, state_dim: int = 115, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, 115) 完整状态向量
        Returns:
            value: (batch, 1) 状态价值
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class MultiChannelPolicyNet(nn.Module):
    """
    多通道策略网络 - 正确实现
    
    关键点:
    1. 4个Actor各自处理不同的输入子集
    2. AttentionWeightNet计算α权重
    3. 融合: fused_logits = Σ(α_i * logits_i)
    """
    
    def __init__(
        self,
        state_dim: int = 115,
        action_dim: int = 33,
        actor_hidden_dim: int = 128,
        attention_hidden_dim: int = 64,
    ):
        super().__init__()
        
        # 4个独立Actor
        self.actor_pre = ActorPre(action_dim, actor_hidden_dim)
        self.actor_scene = ActorScene(action_dim, actor_hidden_dim)
        self.actor_effect = ActorEffect(action_dim, actor_hidden_dim)
        self.actor_rule = ActorRule(action_dim, actor_hidden_dim)
        
        # 注意力权重网络
        self.attention_net = AttentionWeightNet(state_dim, attention_hidden_dim)
        
        # Critic网络
        self.critic = CriticNet(state_dim, actor_hidden_dim)
        
        self.action_dim = action_dim
    
    def extract_contexts(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        从完整state中提取各个通路需要的context
        
        state结构: [belief(50), q_pre(15), q_scene(15), q_effect(8), q_rule(10), confidence(1), goal(16)]
        """
        belief = state[..., :50]  # (batch, 50)
        q_pre = state[..., 50:65]  # (batch, 15)
        q_scene = state[..., 65:80]  # (batch, 15)
        q_effect = state[..., 80:88]  # (batch, 8)
        q_rule = state[..., 88:98]  # (batch, 10)
        confidence = state[..., 98:99]  # (batch, 1)
        goal = state[..., 99:115]  # (batch, 16)
        
        # 构造各Actor的context
        belief_context = belief[..., :20]  # 前20维
        location_context = belief[..., 20:40]  # 20-40维
        hp_context = belief[..., 40:50]  # 40-50维
        
        # inventory_context从belief末尾和goal构造
        inventory_context = torch.cat([belief[..., 45:50], goal[..., :10]], dim=-1)  # (batch, 15)
        
        return {
            'q_pre': q_pre,
            'q_scene': q_scene,
            'q_effect': q_effect,
            'q_rule': q_rule,
            'belief_context': belief_context,
            'location_context': location_context,
            'hp_context': hp_context,
            'inventory_context': inventory_context,
        }
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: (batch, 115) 或 (115,)
        
        Returns:
            fused_logits: (batch, action_dim) 融合后的动作logits
            alpha: (batch, 4) 注意力权重
            value: (batch, 1) 状态价值
        """
        # 处理单个样本
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # 提取各通路的输入
        contexts = self.extract_contexts(state)
        
        # 4个Actor分别产生logits
        logits_pre = self.actor_pre(contexts['q_pre'], contexts['belief_context'])
        logits_scene = self.actor_scene(contexts['q_scene'], contexts['location_context'])
        logits_effect = self.actor_effect(contexts['q_effect'], contexts['hp_context'])
        logits_rule = self.actor_rule(contexts['q_rule'], contexts['inventory_context'])
        
        # 计算注意力权重
        alpha = self.attention_net(state)  # (batch, 4)
        
        # 单通道模式：只用 pre 通道（通过环境变量控制）
        import os
        if os.getenv("TEDG_SINGLE_CHANNEL", "0") == "1":
            fused_logits = logits_pre
            alpha = alpha * 0 + torch.tensor([1.0, 0.0, 0.0, 0.0], device=alpha.device)
        else:
            # 加权融合logits
            alpha_expanded = alpha.unsqueeze(2)  # (batch, 4, 1)
            logits_stack = torch.stack([logits_pre, logits_scene, logits_effect, logits_rule], dim=1)
            fused_logits = (alpha_expanded * logits_stack).sum(dim=1)  # (batch, action_dim)
        
        # 计算状态价值
        value = self.critic(state)  # (batch, 1)
        
        if squeeze_output:
            fused_logits = fused_logits.squeeze(0)
            alpha = alpha.squeeze(0)
            value = value.squeeze(0)
        
        return fused_logits, alpha, value
    
    def get_action_distribution(self, state: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """获取动作分布和价值"""
        logits, _, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value
    
    def get_alpha_weights(self, state: torch.Tensor) -> torch.Tensor:
        """获取注意力权重 alpha"""
        _, alpha, _ = self.forward(state)
        return alpha
