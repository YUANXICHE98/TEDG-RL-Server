"""TEDG-RL Agent 实现"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import random


class TEDGRLAgent:
    """TEDG-RL 智能体，使用超图嵌入指导动作选择"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络参数
        self.state_dim = 115  # NLE状态维度
        self.action_dim = 33  # NetHack动作数
        self.hidden_dim = config.get("hidden_dim", 128)
        
        # 初始化网络
        self.actor = ActorNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic = CriticNetwork(self.state_dim, self.hidden_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), 
            lr=config.get("learning_rate", 3e-4)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), 
            lr=config.get("learning_rate", 3e-4)
        )
        
        # 经验缓冲区
        self.buffer = ReplayBuffer(
            capacity=config.get("buffer_size", 10000),
            batch_size=config.get("batch_size", 64)
        )
        
        # 训练参数
        self.gamma = config.get("gamma", 0.99)
        self.clip_ratio = config.get("clip_ratio", 0.2)
        
    def select_action(self, state: np.ndarray, applicable_actions: List[Dict[str, Any]]) -> Tuple[int, float]:
        """根据状态选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        # 如果有可行动作约束，选择最相似的可行动作
        if applicable_actions:
            # 简化：随机选择一个可行动作
            # TODO: 实现基于嵌入相似度的动作选择
            action_idx = random.randint(0, len(applicable_actions) - 1)
            selected_action = applicable_actions[action_idx]
            
            # 映射到 NetHack 动作索引
            action_map = {
                "move": 107, "move_j": 106, "move_h": 104, "move_k": 107,
                "move_l": 108, "move_y": 121, "move_u": 117, "move_b": 98,
                "move_n": 110, "search": 115, "look": 58, "pickup": 44,
                "drop": 100, "wield": 119, "wear": 87, "takeoff": 84,
                "eat": 101, "quaff": 113, "read": 114, "zap": 122,
                "apply": 97, "throw": 116, "wait": 46
            }
            
            operator = selected_action.get("operator", "wait")
            nle_action = action_map.get(operator, 46)  # 默认 wait
            
            return nle_action, log_prob.item()
        
        return action.item(), log_prob.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """存储经验"""
        self.buffer.add(state, action, reward, next_state, done, log_prob)
    
    def update(self):
        """更新网络"""
        if len(self.buffer) < self.buffer.batch_size:
            return
        
        # 采样批次
        batch = self.buffer.sample()
        
        states = torch.FloatTensor(batch['state']).to(self.device)
        actions = torch.LongTensor(batch['action']).to(self.device)
        rewards = torch.FloatTensor(batch['reward']).to(self.device)
        next_states = torch.FloatTensor(batch['next_state']).to(self.device)
        dones = torch.FloatTensor(batch['done']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_prob']).to(self.device)
        
        # 计算优势
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = returns - values
        
        # PPO 更新
        for _ in range(4):  # PPO epochs
            # 计算新的动作概率
            action_probs = self.actor(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 更新 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 更新 Critic
            value_pred = self.critic(states)
            critic_loss = nn.MSELoss()(value_pred, returns)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


class ActorNetwork(nn.Module):
    """Actor 网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.activation = nn.ReLU()
        
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


class CriticNetwork(nn.Module):
    """Critic 网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.activation = nn.ReLU()
        
    def forward(self, state):
        x = self.activation(self.fc1(state))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int, batch_size: int):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        
        return {
            'state': np.array([item[0] for item in batch]),
            'action': np.array([item[1] for item in batch]),
            'reward': np.array([item[2] for item in batch]),
            'next_state': np.array([item[3] for item in batch]),
            'done': np.array([item[4] for item in batch]),
            'log_prob': np.array([item[5] for item in batch])
        }
    
    def __len__(self):
        return len(self.buffer)
