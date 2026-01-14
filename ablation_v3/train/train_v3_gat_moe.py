#!/usr/bin/env python3
"""
TEDG-RL NetHack训练 - V3版本
GAT-Guided Hierarchical MoE with Sparsemax Routing

核心特性:
1. GAT推理层 - 动态激活超图节点
2. Sparsemax路由 - 软中带硬，避免塌缩
3. 语义专家 - Survival/Combat/Exploration/General
4. 三阶段训练 - Warmup → Transition → Fine-tune
5. 多重稳定性措施 - 负载均衡、多样性、NaN检测
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import numpy as np
import gymnasium as gym
import nle.env
import nle.nethack as nh

# 允许直接运行
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 导入核心模块
from src.core.state_constructor import StateConstructor
from src.core.networks_v3_gat_moe import GATGuidedMoEPolicy
from src.core.ppo_trainer import PPOTrainer
from src.core.action_masking import ActionMasker
from src.core.hypergraph_matcher import HypergraphMatcher
from src.core.operator_expert_mapping import OPERATOR_TO_EXPERT, EXPERT_NAMES


# ============================================================================
# 辅助损失函数 (Auxiliary Loss Functions)
# ============================================================================

def load_balance_loss(alpha: torch.Tensor, num_experts: int = 4) -> torch.Tensor:
    """
    负载均衡损失 - 防止专家塌缩
    
    鼓励每个专家被均匀使用，避免某个专家主导
    
    Args:
        alpha: (batch, num_experts) 专家权重
        num_experts: 专家数量
    
    Returns:
        loss: 标量损失
    """
    # 计算每个专家的平均使用率
    expert_usage = alpha.mean(dim=0)  # (num_experts,)
    
    # 理想情况: 每个专家使用率 = 1/num_experts
    target_usage = torch.ones_like(expert_usage) / num_experts
    
    # L2损失
    loss = F.mse_loss(expert_usage, target_usage)
    
    return loss


def expert_diversity_loss(expert_logits: torch.Tensor) -> torch.Tensor:
    """
    专家多样性损失 - 鼓励专家学到不同策略
    
    最小化专家间的余弦相似度，鼓励差异化
    
    Args:
        expert_logits: (batch, num_experts, action_dim) 专家输出
    
    Returns:
        loss: 标量损失
    """
    num_experts = expert_logits.size(1)
    
    if num_experts < 2:
        return torch.tensor(0.0, device=expert_logits.device)
    
    diversity = 0.0
    count = 0
    
    for i in range(num_experts):
        for j in range(i+1, num_experts):
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(
                expert_logits[:, i, :], 
                expert_logits[:, j, :], 
                dim=-1
            ).mean()
            diversity += cos_sim
            count += 1
    
    # 归一化
    if count > 0:
        diversity /= count
    
    return diversity


# ============================================================================
# Manager内层约束 (Manager Inner Constraints)
# ============================================================================

def aggregate_operators_to_experts(
    operator_scores: torch.Tensor,
    operator_names: List[str],
    num_experts: int = 4
) -> torch.Tensor:
    """
    将Operator分数聚合为Expert分数
    
    这是Manager内层约束的核心：将超图GAT输出的Operator激活分数
    映射到对应的Expert，从而为Router提供"应该选哪个专家"的目标
    
    Args:
        operator_scores: (batch, num_operators) GAT输出的Operator激活分数
        operator_names: Operator名称列表（长度=num_operators）
        num_experts: 专家数量
    
    Returns:
        expert_scores: (batch, num_experts) 聚合后的Expert分数
    """
    batch_size = operator_scores.size(0)
    num_operators = operator_scores.size(1)
    device = operator_scores.device
    
    # 创建映射矩阵: (num_operators, num_experts)
    # mapping[i, j] = 1 表示第i个operator属于第j个expert
    mapping = torch.zeros(num_operators, num_experts, device=device)
    
    for i, op_name in enumerate(operator_names):
        # 从operator名称中提取基础名称（去掉variant后缀）
        # 例如: "move_ac717ec4" -> "move"
        base_op_name = op_name.split('_')[0] if '_' in op_name else op_name
        expert_idx = OPERATOR_TO_EXPERT.get(base_op_name, 3)  # 默认General
        mapping[i, expert_idx] = 1.0
    
    # 归一化：每个expert的分数 = 属于它的operators的平均分数
    expert_counts = mapping.sum(dim=0, keepdim=True).clamp(min=1.0)  # (1, num_experts)
    mapping = mapping / expert_counts  # 归一化
    
    # 聚合: (batch, num_operators) @ (num_operators, num_experts) -> (batch, num_experts)
    expert_scores = torch.matmul(operator_scores, mapping)
    
    return expert_scores


def hypergraph_alignment_loss(
    operator_scores: torch.Tensor,
    alpha: torch.Tensor,
    operator_names: List[str],
    temperature: float = 1.0
) -> torch.Tensor:
    """
    超图-路由对齐损失（Manager内层约束的核心）
    
    强制Router的专家选择与GAT的超图推理一致。
    这是密集监督信号，告诉Router"在当前场景下应该选哪个专家"。
    
    物理含义：
    - 当GAT推理出"当前应该战斗"（Combat Operators激活高）时
    - Router应该倾向于选择Combat Expert
    - 这是一种因果引导（Causal Guidance）
    
    Args:
        operator_scores: (batch, num_operators) GAT输出的Operator激活分数
        alpha: (batch, num_experts) Router输出的专家权重
        operator_names: Operator名称列表
        temperature: 温度参数（控制对齐强度，越低越严格）
    
    Returns:
        loss: 标量损失（KL散度）
    """
    # 1. 将Operator分数聚合为Expert分数
    expert_scores = aggregate_operators_to_experts(
        operator_scores, 
        operator_names, 
        num_experts=alpha.size(1)
    )
    
    # 2. 从GAT推理结果创建目标分布
    target_alpha = F.softmax(expert_scores / temperature, dim=-1)
    
    # 3. KL散度: KL(target || current)
    # 含义：让当前的alpha分布接近GAT建议的target_alpha分布
    loss = F.kl_div(
        F.log_softmax(alpha, dim=-1),
        target_alpha,
        reduction='batchmean'
    )
    
    return loss


def enhanced_semantic_orthogonality_loss(expert_logits: torch.Tensor) -> torch.Tensor:
    """
    增强的语义正交损失
    
    强制不同专家有不同的策略（比原有的diversity_loss更强）
    
    Args:
        expert_logits: (batch, num_experts, action_dim) 专家输出
    
    Returns:
        loss: 标量损失
    """
    batch_size, num_experts, action_dim = expert_logits.shape
    
    if num_experts < 2:
        return torch.tensor(0.0, device=expert_logits.device)
    
    # L2归一化
    expert_norm = F.normalize(expert_logits, p=2, dim=2)  # (batch, num_experts, action_dim)
    
    # 计算专家间的余弦相似度矩阵
    # (batch, num_experts, action_dim) @ (batch, action_dim, num_experts)
    similarity = torch.bmm(expert_norm, expert_norm.transpose(1, 2))  # (batch, num_experts, num_experts)
    
    # 创建mask：只惩罚非对角线元素（专家应该不同）
    mask = 1 - torch.eye(num_experts, device=similarity.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # 惩罚相似度
    loss = (similarity * mask).abs().mean()
    
    return loss


def expert_overlap_penalty(
    alpha: torch.Tensor,
    expert_logits: torch.Tensor
) -> torch.Tensor:
    """
    专家重叠惩罚（高级机制3）
    
    惩罚同时激活多个功能相似的专家。
    逼迫Router: 要么只激活一个专家，要么激活输出完全不同的专家。
    
    物理含义：
    - 如果专家i和j同时被激活（α_i, α_j都大）
    - 并且它们的输出很像（CosSim高）
    - 那就重罚
    
    Args:
        alpha: (batch, num_experts) 专家权重
        expert_logits: (batch, num_experts, action_dim) 专家输出
    
    Returns:
        loss: 标量损失
    """
    batch_size, num_experts, action_dim = expert_logits.shape
    
    if num_experts < 2:
        return torch.tensor(0.0, device=alpha.device)
    
    # L2归一化专家输出
    expert_norm = F.normalize(expert_logits, p=2, dim=2)
    
    # 计算余弦相似度矩阵: (batch, num_experts, num_experts)
    similarity = torch.bmm(expert_norm, expert_norm.transpose(1, 2))
    
    # 计算权重乘积矩阵: (batch, num_experts, num_experts)
    alpha_product = torch.bmm(
        alpha.unsqueeze(2),  # (batch, num_experts, 1)
        alpha.unsqueeze(1)   # (batch, 1, num_experts)
    )  # (batch, num_experts, num_experts)
    
    # 只惩罚非对角线元素（不同专家之间）
    mask = 1 - torch.eye(num_experts, device=similarity.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # 重叠惩罚 = 权重乘积 * 相似度
    # 含义：如果两个专家同时被激活且输出相似，则惩罚大
    overlap = (alpha_product * similarity * mask).sum(dim=(1, 2)).mean()
    
    return overlap



# ============================================================================
# 训练配置 (Training Configuration)
# ============================================================================

def get_training_config(episode: int) -> Dict:
    """
    根据训练阶段返回配置
    
    三阶段训练:
    - Warmup (0-1000): Softmax路由，高学习率，强负载均衡
    - Transition (1000-3000): 温度退火，中学习率，中负载均衡
    - Fine-tune (3000+): Sparsemax路由，低学习率，弱负载均衡
    
    Args:
        episode: 当前episode数
    
    Returns:
        config: 配置字典
    """
    if episode < 1000:
        # Warmup阶段: 让专家学到基础策略
        return {
            'phase': 'warmup',
            'use_sparsemax': False,  # 使用Softmax
            'sparsemax_temp': 1.0,
            'learning_rate': 1e-4,
            'entropy_coef': 0.05,
            'alpha_entropy_coef': 0.1,
            'alpha_entropy_sign': -1,   # 最大化熵（防塌缩）
            'load_balance_coef': 0.02,  # 强制均衡
            'diversity_coef': 0.01,
            # Manager内层约束
            'alignment_coef': 0.1,      # 超图-路由对齐（重要！）
            'alignment_temperature': 1.0,  # 对齐温度
            'semantic_coef': 0.05,      # 语义正交
            # 高级机制
            'temporal_coef': 0.0,       # 时间一致性（Warmup不使用）
            'overlap_coef': 0.0,        # 重叠惩罚（Warmup不使用）
        }
    elif episode < 3000:
        # Transition阶段: 平滑过渡到稀疏路由
        progress = (episode - 1000) / 2000  # 0 → 1
        temp = 1.0 - 0.5 * progress  # 1.0 → 0.5
        alpha_entropy_coef = 0.1 * (1 - progress)  # 0.1 → 0（逐渐减小）
        
        return {
            'phase': 'transition',
            'use_sparsemax': True,
            'sparsemax_temp': temp,
            'learning_rate': 5e-5,
            'entropy_coef': 0.02,
            'alpha_entropy_coef': alpha_entropy_coef,
            'alpha_entropy_sign': -1,   # 仍然最大化熵
            'load_balance_coef': 0.01,
            'diversity_coef': 0.01,
            # Manager内层约束
            'alignment_coef': 0.1,
            'alignment_temperature': 1.0,
            'semantic_coef': 0.05,
            # 高级机制
            'temporal_coef': 0.01,      # 开始使用时间一致性
            'overlap_coef': 0.03,       # 开始使用重叠惩罚
        }
    else:
        # Fine-tune阶段: 精细调整专家分工
        return {
            'phase': 'fine-tune',
            'use_sparsemax': True,
            'sparsemax_temp': 0.5,
            'learning_rate': 1e-5,
            'entropy_coef': 0.01,
            'alpha_entropy_coef': 0.05,
            'alpha_entropy_sign': +1,   # 最小化熵（强制专业化）！
            'load_balance_coef': 0.005,
            'diversity_coef': 0.005,
            # Manager内层约束
            'alignment_coef': 0.1,
            'alignment_temperature': 1.0,
            'semantic_coef': 0.05,
            # 高级机制
            'temporal_coef': 0.02,      # 强时间一致性
            'overlap_coef': 0.05,       # 强重叠惩罚
        }


def get_lr_scheduler(optimizer, warmup_steps: int = 1000, max_steps: int = 100000):
    """
    学习率调度器: Warmup + CosineAnnealing
    
    Args:
        optimizer: 优化器
        warmup_steps: Warmup步数
        max_steps: 最大步数
    
    Returns:
        scheduler: 学习率调度器
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # 线性Warmup
            return step / warmup_steps
        else:
            # Cosine退火
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)



# ============================================================================
# 监控和诊断 (Monitoring & Diagnosis)
# ============================================================================

class TrainingMonitor:
    """训练监控器 - 实时监控和异常检测"""
    
    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self.metrics = defaultdict(list)
    
    def log(self, episode: int, metrics: Dict):
        """记录指标"""
        for k, v in metrics.items():
            self.metrics[k].append(v)
        
        if episode % self.log_interval == 0 and episode > 0:
            self.print_summary(episode)
            self.check_anomalies(episode)
    
    def print_summary(self, episode: int):
        """打印摘要"""
        print(f"\n{'='*60}")
        print(f"Episode {episode} Summary")
        print(f"{'='*60}")
        
        for k in ['episode_score', 'alpha_entropy', 'gradient_norm', 'expert_usage_variance']:
            if k in self.metrics and len(self.metrics[k]) > 0:
                recent = self.metrics[k][-self.log_interval:]
                mean_val = np.mean(recent)
                std_val = np.std(recent)
                print(f"  {k}: {mean_val:.4f} ± {std_val:.4f}")
    
    def check_anomalies(self, episode: int):
        """检查异常"""
        warnings = []
        
        # 检查专家塌缩
        if 'alpha_entropy' in self.metrics and len(self.metrics['alpha_entropy']) >= 50:
            recent_entropy = np.mean(self.metrics['alpha_entropy'][-50:])
            if recent_entropy < 0.3:
                warnings.append(f"⚠️ 专家塌缩: α熵={recent_entropy:.4f} (正常>0.5)")
            elif recent_entropy > 1.2:
                warnings.append(f"⚠️ 专家混乱: α熵={recent_entropy:.4f} (正常<1.0)")
        
        # 检查梯度爆炸
        if 'gradient_norm' in self.metrics and len(self.metrics['gradient_norm']) >= 10:
            recent_grad = np.mean(self.metrics['gradient_norm'][-10:])
            if recent_grad > 10.0:
                warnings.append(f"⚠️ 梯度爆炸: 梯度范数={recent_grad:.4f} (正常<5.0)")
        
        # 检查GAT过平滑
        if 'gat_attention_variance' in self.metrics and len(self.metrics['gat_attention_variance']) >= 50:
            recent_var = np.mean(self.metrics['gat_attention_variance'][-50:])
            if recent_var < 0.05:
                warnings.append(f"⚠️ GAT过平滑: 注意力方差={recent_var:.4f} (正常>0.1)")
        
        # 打印警告
        if warnings:
            print(f"\n{'!'*60}")
            print("异常检测:")
            for w in warnings:
                print(f"  {w}")
            print(f"{'!'*60}\n")
    
    def get_stats(self, key: str, window: int = 100) -> Dict:
        """获取统计信息"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return {}
        
        recent = self.metrics[key][-window:]
        return {
            'mean': np.mean(recent),
            'std': np.std(recent),
            'min': np.min(recent),
            'max': np.max(recent),
        }


class NaNDetector:
    """NaN检测器 - 自动检测和回滚"""
    
    def __init__(self, model):
        self.model = model
        self.last_good_state = None
        self.nan_count = 0
    
    def save_checkpoint(self):
        """保存当前状态"""
        self.last_good_state = {
            k: v.clone().cpu() for k, v in self.model.state_dict().items()
        }
    
    def check_and_rollback(self, loss: torch.Tensor) -> bool:
        """
        检查NaN/Inf并回滚
        
        Returns:
            True if NaN detected and rolled back
        """
        if torch.isnan(loss) or torch.isinf(loss):
            self.nan_count += 1
            print(f"\n⚠️ NaN/Inf detected in loss! (count: {self.nan_count})")
            
            if self.last_good_state is not None:
                print("  → Rolling back to last good checkpoint...")
                self.model.load_state_dict(self.last_good_state)
                return True
            else:
                print("  → No checkpoint available, skipping batch...")
                return True
        
        return False


class RewardNormalizer:
    """奖励归一化器 - 滑动平均归一化"""
    
    def __init__(self, clip_range: float = 10.0):
        self.mean = 0.0
        self.std = 1.0
        self.clip_range = clip_range
        self.count = 0
    
    def update(self, reward: float):
        """更新统计量"""
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.std = np.sqrt((self.std**2 * (self.count-1) + delta**2) / self.count)
    
    def normalize(self, reward: float) -> float:
        """归一化奖励"""
        if self.count < 2:
            return reward
        
        normalized = (reward - self.mean) / (self.std + 1e-8)
        return np.clip(normalized, -self.clip_range, self.clip_range)



# ============================================================================
# 工具函数 (Utility Functions)
# ============================================================================

def get_device():
    """自动检测可用设备：MUSA > CUDA > CPU"""
    if os.getenv("TEDG_DEVICE", "").lower() == "cpu" or os.getenv("TEDG_FORCE_CPU", "0") == "1":
        print("✓ 强制使用CPU设备 (TEDG_DEVICE)")
        return torch.device('cpu')
    try:
        import torch_musa
        if torch.musa.is_available():
            device = torch.device('musa:0')
            print(f"✓ 使用MUSA设备: {torch.musa.get_device_name(0)}")
            return device
    except:
        pass
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"✓ 使用CUDA设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ 使用CPU设备")
    
    return device


def extract_atoms_from_obs(obs) -> Dict[str, List[str]]:
    """
    从NetHack观测中提取atoms (增强版 - 包含环境感知)
    
    Args:
        obs: NetHack观测
    
    Returns:
        atoms: {"pre_nodes": [...], "scene_atoms": [...]}
    """
    blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
    glyphs = obs.get("glyphs", None)
    
    # 提取基础信息
    hp = blstats[nh.NLE_BL_HP]
    hp_max = blstats[nh.NLE_BL_HPMAX]
    depth = blstats[nh.NLE_BL_DEPTH]
    gold = blstats[nh.NLE_BL_GOLD]
    hunger = blstats[nh.NLE_BL_HUNGER]
    
    pre_nodes = []
    scene_atoms = []
    
    # 1. 玩家状态
    if hp > 0:
        pre_nodes.append("player_alive")
    
    if hp_max > 0:
        hp_ratio = hp / hp_max
        if hp_ratio >= 0.8:
            pre_nodes.append("hp_full")
        elif hp_ratio >= 0.5:
            pre_nodes.append("hp_medium")
        elif hp_ratio < 0.3:
            pre_nodes.append("hp_low")
    
    # 2. 深度
    if depth > 0:
        scene_atoms.append(f"dlvl_{min(depth, 10)}")
    
    # 3. 资源状态
    if gold > 0:
        pre_nodes.append("has_gold")
    
    if hunger == 0:
        pre_nodes.append("hunger_satiated")
    elif hunger > 500:
        pre_nodes.append("hunger_hungry")
    elif hunger > 1000:
        pre_nodes.append("hunger_starving")
    
    # 4. 环境感知 (从glyphs提取)
    if glyphs is not None:
        player_y = blstats[nh.NLE_BL_Y]
        player_x = blstats[nh.NLE_BL_X]
        
        # 提取周围3x3区域
        y_min = max(0, player_y - 1)
        y_max = min(glyphs.shape[0], player_y + 2)
        x_min = max(0, player_x - 1)
        x_max = min(glyphs.shape[1], player_x + 2)
        
        surrounding = glyphs[y_min:y_max, x_min:x_max]
        
        # 统计周围的实体
        monster_count = 0
        wall_count = 0
        door_count = 0
        item_count = 0
        
        for glyph in surrounding.flatten():
            # NetHack glyph编码 (简化版)
            if 0 < glyph < 400:  # 怪物范围
                monster_count += 1
            elif glyph in [2359, 2360, 2361, 2362]:  # 墙壁
                wall_count += 1
            elif glyph in [2363, 2364, 2365]:  # 门
                door_count += 1
            elif 400 < glyph < 800:  # 物品范围
                item_count += 1
        
        # 添加环境atoms
        if monster_count > 0:
            scene_atoms.append("monsters_nearby")
            if monster_count >= 3:
                scene_atoms.append("many_monsters")
        
        if wall_count >= 6:
            scene_atoms.append("in_corridor")
        elif wall_count <= 2:
            scene_atoms.append("in_room")
        
        if door_count > 0:
            scene_atoms.append("door_nearby")
        
        if item_count > 0:
            scene_atoms.append("items_nearby")
    
    return {
        "pre_nodes": pre_nodes,
        "scene_atoms": scene_atoms
    }


def extract_state_from_obs(obs, state_constructor, matcher, t_now: int = 0) -> Tuple[np.ndarray, Dict]:
    """
    从NetHack观测中提取状态向量和atoms (增强版)
    
    Args:
        obs: NetHack观测
        state_constructor: 状态构造器
        matcher: 超图匹配器
        t_now: 当前时间步
    
    Returns:
        state: (115,) 状态向量
        atoms: atoms字典
    """
    blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
    glyphs = obs.get("glyphs", None)
    
    # 提取atoms
    atoms = extract_atoms_from_obs(obs)
    
    # 构建belief向量 (增强版 - 50维)
    belief = np.zeros(50, dtype=np.float32)
    hp = blstats[nh.NLE_BL_HP]
    hp_max = blstats[nh.NLE_BL_HPMAX]
    
    # 基础状态 (0-9)
    belief[0] = 1.0  # player_alive
    belief[1] = 1.0 if hp > 0 else 0.0  # game_active
    belief[2] = 1.0  # any_hp
    belief[3] = hp / max(hp_max, 1)  # hp_ratio
    belief[4] = 1.0 if blstats[nh.NLE_BL_HUNGER] == 0 else 0.0  # hunger_satiated
    belief[5] = 1.0 if blstats[nh.NLE_BL_GOLD] == 0 else 0.0  # no_gold
    belief[6] = min(blstats[nh.NLE_BL_DEPTH] / 10.0, 1.0)  # depth_normalized
    belief[7] = min(blstats[nh.NLE_BL_GOLD] / 1000.0, 1.0)  # gold_normalized
    belief[8] = min(blstats[nh.NLE_BL_HUNGER] / 1500.0, 1.0)  # hunger_normalized
    belief[9] = min(t_now / 1000.0, 1.0)  # time_normalized
    
    # 环境感知 (10-19) - 从glyphs提取
    if glyphs is not None:
        player_y = blstats[nh.NLE_BL_Y]
        player_x = blstats[nh.NLE_BL_X]
        
        # 提取周围3x3区域
        y_min = max(0, player_y - 1)
        y_max = min(glyphs.shape[0], player_y + 2)
        x_min = max(0, player_x - 1)
        x_max = min(glyphs.shape[1], player_x + 2)
        
        surrounding = glyphs[y_min:y_max, x_min:x_max]
        
        # 统计周围实体
        monster_count = 0
        wall_count = 0
        door_count = 0
        item_count = 0
        floor_count = 0
        
        for glyph in surrounding.flatten():
            if 0 < glyph < 400:  # 怪物
                monster_count += 1
            elif glyph in [2359, 2360, 2361, 2362]:  # 墙壁
                wall_count += 1
            elif glyph in [2363, 2364, 2365]:  # 门
                door_count += 1
            elif 400 < glyph < 800:  # 物品
                item_count += 1
            elif glyph == 2358:  # 地板
                floor_count += 1
        
        belief[10] = monster_count / 9.0  # monster_density
        belief[11] = wall_count / 9.0  # wall_density
        belief[12] = door_count / 9.0  # door_density
        belief[13] = item_count / 9.0  # item_density
        belief[14] = floor_count / 9.0  # floor_density
        belief[15] = 1.0 if monster_count > 0 else 0.0  # has_nearby_monster
        belief[16] = 1.0 if item_count > 0 else 0.0  # has_nearby_item
        belief[17] = 1.0 if door_count > 0 else 0.0  # has_nearby_door
        belief[18] = 1.0 if wall_count >= 6 else 0.0  # in_corridor
        belief[19] = 1.0 if wall_count <= 2 else 0.0  # in_room
    
    # 超图匹配 (使用正确的格式)
    plot_atoms = {
        "pre": atoms["pre_nodes"],
        "scene": atoms["scene_atoms"],
        "effect": [],
        "rule": []
    }
    
    match_results = matcher.match(plot_atoms, t_now=t_now, top_k=8)
    
    # 计算confidence (使用匹配结果的平均分数)
    if match_results:
        confidence = np.mean([r.score for r in match_results])
    else:
        confidence = 0.0
    
    # 构建状态向量
    goal = np.zeros(16, dtype=np.float32)
    goal[0] = 1.0
    
    state = state_constructor.construct_state(
        belief_vector=belief,
        pre_nodes=atoms["pre_nodes"],
        scene_atoms=atoms["scene_atoms"],
        eff_metadata=[],
        conditional_effects=[],
        confidence=confidence,
        goal_embedding=goal,
    )
    
    return state, atoms


def log_gradient_norms(model) -> float:
    """
    记录梯度范数
    
    Returns:
        total_norm: 总梯度范数
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    return total_norm



# ============================================================================
# 主训练函数 (Main Training Function)
# ============================================================================

def main():
    """V3主训练循环"""
    parser = argparse.ArgumentParser(description="TEDG-RL V3 训练脚本 - GAT-Guided MoE")
    parser.add_argument("--exp-name", type=str, default="v3_full", help="实验名称")
    parser.add_argument("--episodes", type=int, default=10000, help="训练episodes数")
    parser.add_argument("--max-steps", type=int, default=2000, help="每episode最大步数")
    parser.add_argument("--no-mask", action="store_true", help="禁用动作掩码")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
    parser.add_argument("--freeze-gat", action="store_true", help="冻结GAT参数")
    parser.add_argument("--num-experts", type=int, default=4, help="专家数量")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"TEDG-RL V3 训练启动 - GAT-Guided Hierarchical MoE")
    print(f"{'='*70}")
    print(f"实验名称: {args.exp_name}")
    print(f"训练配置: {args.episodes} episodes, {args.max_steps} steps/episode")
    print(f"专家数量: {args.num_experts}")
    print(f"冻结GAT: {args.freeze_gat}")
    
    # 设备检测
    device = get_device()
    
    # 创建输出目录
    output_dir = Path(f"ablation_v3/results/{args.exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    
    # 加载超图
    print(f"\n[加载超图数据]")
    state_constructor = StateConstructor("data/hypergraph/hypergraph_complete_real.json")
    matcher = HypergraphMatcher(
        state_constructor.hypergraph,
        weights=(0.35, 0.35, 0.2, 0.1),
        tau=200.0
    )
    print(f"✓ 超图加载完成: {len(matcher.edges)} 条超边")
    
    # 创建环境
    print(f"\n[创建NetHack环境]")
    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")
    print(f"✓ 动作空间: {env.action_space.n}个动作")
    
    # 创建V3网络
    print(f"\n[初始化V3网络]")
    policy_net = GATGuidedMoEPolicy(
        hypergraph_path="data/hypergraph/hypergraph_gat_structure.json",
        state_dim=115,
        hidden_dim=256,
        action_dim=23,
        num_experts=args.num_experts,
        use_sparsemax=True
    ).to(device)
    
    # 加载超图结构以获取operator_names（用于Manager约束）
    print(f"\n[加载超图结构用于Manager约束]")
    with open("data/hypergraph/hypergraph_gat_structure.json", 'r') as f:
        hypergraph_structure = json.load(f)
    operator_names = [node['label'] for node in hypergraph_structure['nodes'] if node['type'] == 'operator']
    print(f"✓ 提取了 {len(operator_names)} 个Operator节点")
    
    # 冻结GAT (如果需要)
    if args.freeze_gat:
        print("  → 冻结GAT参数")
        for param in policy_net.gat.parameters():
            param.requires_grad = False
    
    total_params = sum(p.numel() for p in policy_net.parameters())
    trainable_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)
    print(f"✓ 总参数: {total_params:,}, 可训练: {trainable_params:,}")
    
    # 创建优化器 (初始学习率1e-4)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    
    # 学习率调度器
    lr_scheduler = get_lr_scheduler(optimizer, warmup_steps=1000, max_steps=args.episodes*args.max_steps)
    
    # 创建训练器 (V3推荐参数)
    trainer = PPOTrainer(
        policy_net=policy_net,
        learning_rate=1e-4,
        clip_ratio=0.15,  # V3: 更保守
        gamma=0.995,      # V3: 更长视野
        gae_lambda=0.97,  # V3: 更平滑
        ppo_epochs=4,     # V3: 更充分
        batch_size=256,   # V3: 更大
        device=device,
        alpha_entropy_coef=0.05,  # V3: 专家熵正则
    )
    
    # 初始化监控器
    monitor = TrainingMonitor(log_interval=50)
    nan_detector = NaNDetector(policy_net)
    reward_normalizer = RewardNormalizer(clip_range=10.0)
    
    # 动作掩码
    action_masker = ActionMasker(state_constructor.hypergraph, num_actions=23)
    
    # 训练统计
    episode_rewards = []
    episode_lengths = []
    episode_scores = []
    best_reward = float("-inf")
    best_score = 0
    start_episode = 0
    
    # 恢复checkpoint
    if args.resume and os.path.exists(args.resume):
        print(f"\n[恢复checkpoint: {args.resume}]")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        policy_net.load_state_dict(checkpoint["policy_net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_episode = checkpoint.get("episode", 0) + 1
        best_reward = checkpoint.get("best_reward", float("-inf"))
        best_score = checkpoint.get("best_score", 0)
        print(f"✓ 从Episode {start_episode}继续, Best Score: {best_score}")
    
    print(f"\n{'='*70}")
    print(f"开始训练")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    global_step = 0


    
    # ========================================================================
    # 主训练循环
    # ========================================================================
    
    for episode in range(start_episode, args.episodes):
        # 获取当前阶段配置
        config = get_training_config(episode)
        
        # 更新网络配置
        policy_net.use_sparsemax = config['use_sparsemax']
        
        # [新增] 打印当前路由方式，防止配置没生效
        if episode % 10 == 0:
            print(f"DEBUG: Episode {episode}, Routing: {'Sparsemax' if policy_net.use_sparsemax else 'Softmax'}, Phase: {config['phase']}")
        
        # 更新学习率 (根据阶段)
        for param_group in optimizer.param_groups:
            param_group['lr'] = config['learning_rate']
        
        # 打印阶段信息
        if episode in [0, 1000, 3000]:
            print(f"\n{'='*70}")
            print(f"进入 {config['phase'].upper()} 阶段 (Episode {episode})")
            print(f"  - 路由方式: {'Sparsemax' if config['use_sparsemax'] else 'Softmax'}")
            print(f"  - 学习率: {config['learning_rate']}")
            print(f"  - 负载均衡系数: {config['load_balance_coef']}")
            print(f"{'='*70}\n")
        
        # 重置环境
        obs, info = env.reset()
        state, atoms = extract_state_from_obs(obs, state_constructor, matcher, t_now=0)
        
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        # Episode内的统计
        episode_alphas = []
        episode_expert_logits = []
        episode_gat_attentions = []
        episode_alignment_losses = []
        episode_semantic_losses = []
        episode_temporal_losses = []
        episode_overlap_losses = []
        
        # 时间一致性追踪
        last_alpha = None
        
        # ====================================================================
        # Episode循环
        # ====================================================================
        
        while not (done or truncated) and steps < args.max_steps:
            # 获取动作
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # V3前向传播
                logits, alpha, value, aux_info = policy_net(
                    state_tensor,
                    atoms=atoms
                )
                
                # 记录统计信息
                episode_alphas.append(alpha.cpu())
                if aux_info['expert_logits'] is not None:
                    episode_expert_logits.append(aux_info['expert_logits'].cpu())
                
                # 应用动作掩码
                action_mask = None
                if not args.no_mask:
                    action_mask = action_masker.get_action_mask(
                        atoms["pre_nodes"],
                        atoms["scene_atoms"],
                        0.5  # confidence
                    )
                    mask_t = torch.as_tensor(action_mask, device=logits.device, dtype=torch.bool)
                    masked_logits = logits.masked_fill(~mask_t, float("-inf"))
                    
                    # 兜底: 避免全-inf
                    if not torch.isfinite(masked_logits).any():
                        masked_logits = logits
                        action_mask = np.ones(23, dtype=bool)
                    
                    logits = masked_logits
                
                # 数值稳定性
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                
                # 采样动作
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action.item())
            
            # 奖励归一化
            reward_normalizer.update(reward)
            normalized_reward = reward_normalizer.normalize(reward)
            
            total_reward += reward
            steps += 1
            global_step += 1
            
            # 提取下一状态
            next_state, next_atoms = extract_state_from_obs(obs, state_constructor, matcher, t_now=steps)
            
            # 存储经验
            trainer.buffer.add(
                state=state,
                action=action.item(),
                reward=normalized_reward,
                next_state=next_state,
                done=done or truncated,
                log_prob=log_prob.item(),
                action_mask=action_mask,
            )
            
            state = next_state
            atoms = next_atoms
            
            # 更新网络
            if len(trainer.buffer) >= trainer.batch_size:
                # 保存checkpoint (用于NaN回滚)
                nan_detector.save_checkpoint()
                
                # 采样批次
                batch = trainer.buffer.sample_batch(trainer.batch_size)
                states = batch['states'].to(device)
                actions = batch['actions'].to(device)
                rewards = batch['rewards'].to(device)
                dones = batch['dones'].to(device)
                old_log_probs = batch['old_log_probs'].to(device)
                
                # 计算GAE优势
                with torch.no_grad():
                    _, _, old_values, _ = policy_net(states)
                    old_values = old_values.squeeze(-1)
                
                advantages, returns = trainer.compute_gae_advantages(rewards, old_values, dones)
                
                # 归一化优势
                adv_mean = advantages.mean()
                adv_std = advantages.cpu().std().to(advantages.device) if advantages.is_cuda or str(advantages.device).startswith('musa') else advantages.std()
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                
                # PPO更新循环
                for ppo_epoch in range(trainer.ppo_epochs):
                    # 前向传播
                    logits, alpha, values, aux_info = policy_net(states)
                    values = values.squeeze(-1)
                    
                    # 计算新的对数概率
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()
                    
                    # 计算α熵
                    alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1).mean()
                    
                    # PPO比率
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # Actor损失
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - trainer.clip_ratio, 1 + trainer.clip_ratio) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Critic损失
                    critic_loss = F.mse_loss(values, returns)
                    
                    # 辅助损失
                    lb_loss = load_balance_loss(alpha, num_experts=args.num_experts)
                    
                    div_loss = torch.tensor(0.0, device=device)
                    if aux_info['expert_logits'] is not None:
                        div_loss = expert_diversity_loss(aux_info['expert_logits'])
                    
                    # ===== Manager内层约束 =====
                    alignment_loss = torch.tensor(0.0, device=device)
                    semantic_loss = torch.tensor(0.0, device=device)
                    
                    if aux_info['operator_scores'] is not None and aux_info['expert_logits'] is not None:
                        # 1. 超图-路由对齐损失
                        alignment_loss = hypergraph_alignment_loss(
                            aux_info['operator_scores'],
                            alpha,
                            operator_names,
                            temperature=config.get('alignment_temperature', 1.0)
                        )
                        
                        # 2. 增强的语义正交损失
                        semantic_loss = enhanced_semantic_orthogonality_loss(
                            aux_info['expert_logits']
                        )
                    
                    # ===== 高级机制 =====
                    # 3. 时间一致性损失（高级机制2）
                    temporal_loss = torch.tensor(0.0, device=device)
                    if last_alpha is not None and config.get('temporal_coef', 0.0) > 0:
                        # MSE: 惩罚相邻时间步的alpha剧烈变化
                        temporal_loss = F.mse_loss(alpha, last_alpha)
                    
                    # 4. 专家重叠惩罚（高级机制3）
                    overlap_loss = torch.tensor(0.0, device=device)
                    if aux_info['expert_logits'] is not None and config.get('overlap_coef', 0.0) > 0:
                        overlap_loss = expert_overlap_penalty(alpha, aux_info['expert_logits'])
                    
                    # 总损失（包含所有约束）
                    total_loss = (
                        actor_loss +
                        0.5 * critic_loss -
                        config['entropy_coef'] * entropy +
                        config.get('alpha_entropy_sign', -1) * config['alpha_entropy_coef'] * alpha_entropy +  # 高级机制1：符号可变
                        config['load_balance_coef'] * lb_loss +
                        config['diversity_coef'] * div_loss +
                        config.get('alignment_coef', 0.1) * alignment_loss +      # Manager约束
                        config.get('semantic_coef', 0.05) * semantic_loss +       # Manager约束
                        config.get('temporal_coef', 0.0) * temporal_loss +        # 高级机制2
                        config.get('overlap_coef', 0.0) * overlap_loss            # 高级机制3
                    )
                    
                    # NaN检测
                    if nan_detector.check_and_rollback(total_loss):
                        break
                    
                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()
                    
                    # 梯度裁剪
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    
                    # 更新
                    optimizer.step()
                    lr_scheduler.step()
                
                # 记录Manager约束损失（用于episode级别统计）
                if alignment_loss.item() > 0:
                    episode_alignment_losses.append(alignment_loss.item())
                    episode_semantic_losses.append(semantic_loss.item())
                
                # 记录高级机制损失
                if temporal_loss.item() > 0:
                    episode_temporal_losses.append(temporal_loss.item())
                if overlap_loss.item() > 0:
                    episode_overlap_losses.append(overlap_loss.item())
                
                # 更新last_alpha（用于下一次时间一致性计算）
                last_alpha = alpha.detach()
                
                # 清空缓冲区
                trainer.buffer.clear()
        
        # ====================================================================
        # Episode结束统计
        # ====================================================================
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        final_score = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)[nh.NLE_BL_SCORE]
        episode_scores.append(final_score)
        
        # 计算专家统计
        if episode_alphas:
            alpha_tensor = torch.stack(episode_alphas, dim=0)  # (steps, num_experts)
            mean_alpha = alpha_tensor.mean(dim=0).numpy()
            alpha_entropy_val = -(mean_alpha * np.log(mean_alpha + 1e-8)).sum()
            expert_usage_variance = alpha_tensor.var(dim=0).mean().item()
        else:
            alpha_entropy_val = 0.0
            expert_usage_variance = 0.0
        
        # 记录监控指标
        monitor.log(episode, {
            'episode_score': final_score,
            'episode_reward': total_reward,
            'episode_length': steps,
            'alpha_entropy': alpha_entropy_val,
            'expert_usage_variance': expert_usage_variance,
            'gradient_norm': grad_norm.item() if 'grad_norm' in locals() else 0.0,
        })
        
        # 更新最佳记录
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({
                "policy_net": policy_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode,
                "best_reward": best_reward,
                "best_score": final_score,
                "config": vars(args)
            }, output_dir / "checkpoints" / "best_model.pth")
        
        if final_score > best_score:
            best_score = final_score
        
        # 定期保存checkpoint
        if (episode + 1) % 100 == 0:
            torch.save({
                "policy_net": policy_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "episode": episode,
                "reward": total_reward,
                "score": final_score,
                "config": vars(args)
            }, output_dir / "checkpoints" / f"model_{episode+1:05d}.pth")
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_score = np.mean(episode_scores[-10:])
            print(f"Episode {episode+1}/{args.episodes} | "
                  f"Score: {final_score} | "
                  f"Reward: {total_reward:.2f} | "
                  f"Steps: {steps} | "
                  f"α_entropy: {alpha_entropy_val:.3f} | "
                  f"Phase: {config['phase']}")
            
            # 打印Manager约束的loss（如果有）
            if episode_alignment_losses:
                avg_alignment = np.mean(episode_alignment_losses)
                avg_semantic = np.mean(episode_semantic_losses)
                print(f"  → Manager Constraints: "
                      f"Alignment={avg_alignment:.4f}, "
                      f"Semantic={avg_semantic:.4f}")
            
            # 打印高级机制的loss（如果有）
            if episode_temporal_losses or episode_overlap_losses:
                losses_str = "  → Advanced Mechanisms: "
                if episode_temporal_losses:
                    avg_temporal = np.mean(episode_temporal_losses)
                    losses_str += f"Temporal={avg_temporal:.4f}, "
                if episode_overlap_losses:
                    avg_overlap = np.mean(episode_overlap_losses)
                    losses_str += f"Overlap={avg_overlap:.4f}"
                print(losses_str)


    
    # ========================================================================
    # 训练结束
    # ========================================================================
    
    # 保存最终模型
    torch.save({
        "policy_net": policy_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": args.episodes,
        "best_reward": best_reward,
        "best_score": best_score,
        "config": vars(args)
    }, output_dir / "checkpoints" / "model_final.pth")
    
    # 保存训练日志
    training_log = {
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_lengths": [int(l) for l in episode_lengths],
        "episode_scores": [int(s) for s in episode_scores],
        "best_reward": float(best_reward),
        "best_score": int(best_score),
        "config": vars(args),
        "monitor_metrics": {k: [float(v) for v in vals] for k, vals in monitor.metrics.items()},
    }
    
    with open(output_dir / "logs" / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    
    # 打印最终统计
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"训练完成")
    print(f"{'='*70}")
    print(f"总耗时: {elapsed_time/3600:.2f} 小时")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"最佳分数: {best_score}")
    print(f"平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"平均分数: {np.mean(episode_scores):.1f}")
    print(f"结果保存在: {output_dir}")
    
    # 打印专家使用统计
    # if episode_alphas:
    #     print(f"\n专家使用统计 (最后100 episodes):")
    #     recent_alphas = torch.stack(episode_alphas[-min(100*args.max_steps, len(episode_alphas)):], dim=0)
    #     mean_usage = recent_alphas.mean(dim=0).tolist()
    #     for i, name in enumerate(policy_net.expert_names):
    #         print(f"  {name}: {mean_usage[i]*100:.2f}%")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

