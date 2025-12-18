"""PPO训练器 - 多通道RL训练循环"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import json
from pathlib import Path


class ReplayBuffer:
    """经验回放缓冲区 - 存储轨迹数据"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, log_prob: float):
        """添加经验"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
        })
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """采样批次"""
        if len(self.buffer) < batch_size:
            indices = range(len(self.buffer))
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor(np.array([b['state'] for b in batch]))
        actions = torch.LongTensor([b['action'] for b in batch])
        rewards = torch.FloatTensor([b['reward'] for b in batch])
        next_states = torch.FloatTensor(np.array([b['next_state'] for b in batch]))
        dones = torch.FloatTensor([b['done'] for b in batch])
        old_log_probs = torch.FloatTensor([b['log_prob'] for b in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'old_log_probs': old_log_probs,
        }
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


class RewardComputer:
    """5分量奖励计算器"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: 奖励权重 {progress, safety, efficiency, feasibility, exploration}
        """
        if weights is None:
            weights = {
                'progress': 0.3,
                'safety': 0.3,
                'efficiency': 0.2,
                'feasibility': 0.1,
                'exploration': 0.1,
            }
        self.weights = weights
    
    def compute(
        self,
        env_reward: float,  # 环境原始奖励
        dlvl_change: float,  # 深度变化
        hp_change: float,  # 血量变化
        steps: int,  # 步数
        is_dead: bool,  # 是否死亡
        violated_precond: bool = False,  # 是否违反前置条件
        discovered_new_effect: bool = False,  # 是否发现新效果
    ) -> float:
        """
        计算多分量奖励
        
        r = w_prog*r_progress + w_safe*r_safety + w_eff*r_efficiency + w_feas*r_feasibility + w_exp*r_exploration
        """
        
        # r_progress: 向下楼探索的进展
        r_progress = dlvl_change * 0.1 + env_reward / 1000.0
        r_progress = np.clip(r_progress, -1, 1)
        
        # r_safety: 避免致命错误
        if is_dead:
            r_safety = -1000.0
        else:
            r_safety = -0.1 * (1 - max(0, hp_change / 100.0))  # hp下降惩罚
            if hp_change < -30:  # 血量大幅下降
                r_safety -= 0.1
        r_safety = np.clip(r_safety, -1000, 0.1)
        
        # r_efficiency: 最少步数到达目标
        r_efficiency = -steps / 1000.0
        r_efficiency = np.clip(r_efficiency, -0.1, 0.1)
        
        # r_feasibility: 遵循超图规则
        r_feasibility = 0.05  # 基础奖励
        if violated_precond:
            r_feasibility -= 0.2
        r_feasibility = np.clip(r_feasibility, -0.3, 0.05)
        
        # r_exploration: 在安全前提下探索新分支
        r_exploration = 0.0
        if discovered_new_effect and not is_dead:
            r_exploration = 0.15
        r_exploration = np.clip(r_exploration, 0, 0.15)
        
        # 加权求和
        total_reward = (
            self.weights['progress'] * r_progress +
            self.weights['safety'] * r_safety +
            self.weights['efficiency'] * r_efficiency +
            self.weights['feasibility'] * r_feasibility +
            self.weights['exploration'] * r_exploration
        )
        
        return float(total_reward)


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(
        self,
        policy_net: nn.Module,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        ppo_epochs: int = 3,
        batch_size: int = 64,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.policy_net = policy_net.to(device)
        self.device = device
        
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # 熵正则化系数 (增加以鼓励探索)
        self.entropy_coef = 0.05  # 从0.01增加到0.05
        self.alpha_entropy_coef = 0.1  # α权重熵正则化系数
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 经验缓冲区
        self.buffer = ReplayBuffer(capacity=10000)
        
        # 奖励计算器
        self.reward_computer = RewardComputer()
        
        # 训练统计
        self.train_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'total_loss': [],
            'avg_reward': [],
            'avg_episode_length': [],
        }
    
    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """
        根据状态选择动作
        
        Args:
            state: (115,) 状态向量
        
        Returns:
            action: 选择的动作索引
            log_prob: 动作的对数概率
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            dist, _ = self.policy_net.get_action_distribution(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, log_prob: float):
        """存储经验"""
        self.buffer.add(state, action, reward, next_state, done, log_prob)
    
    def compute_gae_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: (batch_size,)
            values: (batch_size,)
            dones: (batch_size,)
        
        Returns:
            advantages: (batch_size,)
            returns: (batch_size,)
        """
        batch_size = rewards.shape[0]
        advantages = torch.zeros(batch_size, device=self.device)
        returns = torch.zeros(batch_size, device=self.device)
        
        gae = 0.0
        next_value = 0.0
        
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                next_value = 0.0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[t])
            
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """
        更新策略网络
        
        Returns:
            统计信息字典
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # 采样批次
        batch = self.buffer.sample_batch(self.batch_size)
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        dones = batch['dones'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device)
        
        # 计算旧价值和新价值
        with torch.no_grad():
            _, _, old_values = self.policy_net(states)
            old_values = old_values.squeeze(-1)
        
        # 计算GAE优势
        advantages, returns = self.compute_gae_advantages(rewards, old_values, dones)
        
        # 归一化优势 (MUSA兼容性: 在CPU上计算std)
        adv_mean = advantages.mean()
        adv_std = advantages.cpu().std().to(advantages.device) if advantages.is_cuda or str(advantages.device).startswith('musa') else advantages.std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        # PPO更新循环
        actor_losses = []
        critic_losses = []
        
        for epoch in range(self.ppo_epochs):
            # 前向传播
            logits, alpha, values = self.policy_net(states)
            values = values.squeeze(-1)
            
            # 计算新的对数概率
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 计算α权重的熵 (鼓励均衡分布)
            alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()
            
            # PPO比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Actor损失 (PPO clipped + 熵正则化 + α权重正则化)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy - self.alpha_entropy_coef * alpha_entropy
            
            # Critic损失
            critic_loss = nn.MSELoss()(values, returns)
            
            # 总损失
            total_loss = actor_loss + 0.5 * critic_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 统计信息
        stats = {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'total_loss': np.mean(actor_losses) + 0.5 * np.mean(critic_losses),
            'avg_advantage': advantages.mean().item(),
            'avg_return': returns.mean().item(),
        }
        
        return stats
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_stats': self.train_stats,
        }, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_stats = checkpoint.get('train_stats', self.train_stats)
