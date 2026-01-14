#!/usr/bin/env python3
"""
V3 Episode时序可视化工具

展示一个完整episode中的动态变化:
1. 每个时间步的obs → atoms → GAT → 专家选择 → 动作
2. 生成时序热图，显示专家权重随时间的变化
3. 生成动作选择热图，显示动作分布的演化
4. 生成GAT注意力热图，显示节点激活的变化
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
import torch
import torch.nn.functional as F

import gymnasium as gym
import nle.env
import nle.nethack as nh

# 添加项目路径
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.networks_v3_gat_moe import GATGuidedMoEPolicy
from src.core.state_constructor import StateConstructor
from src.core.hypergraph_matcher import HypergraphMatcher


def extract_atoms_from_obs(obs):
    """从obs提取atoms (增强版)"""
    blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
    glyphs = obs.get("glyphs", None)
    
    hp = blstats[nh.NLE_BL_HP]
    hp_max = blstats[nh.NLE_BL_HPMAX]
    depth = blstats[nh.NLE_BL_DEPTH]
    gold = blstats[nh.NLE_BL_GOLD]
    hunger = blstats[nh.NLE_BL_HUNGER]
    
    pre_nodes = []
    scene_atoms = []
    
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
    
    if depth > 0:
        scene_atoms.append(f"dlvl_{min(depth, 10)}")
    
    if gold > 0:
        pre_nodes.append("has_gold")
    
    if hunger == 0:
        pre_nodes.append("hunger_satiated")
    elif hunger > 500:
        pre_nodes.append("hunger_hungry")
    elif hunger > 1000:
        pre_nodes.append("hunger_starving")
    
    # 环境感知
    if glyphs is not None:
        player_y = blstats[nh.NLE_BL_Y]
        player_x = blstats[nh.NLE_BL_X]
        
        y_min = max(0, player_y - 1)
        y_max = min(glyphs.shape[0], player_y + 2)
        x_min = max(0, player_x - 1)
        x_max = min(glyphs.shape[1], player_x + 2)
        
        surrounding = glyphs[y_min:y_max, x_min:x_max]
        
        monster_count = 0
        wall_count = 0
        door_count = 0
        item_count = 0
        
        for glyph in surrounding.flatten():
            if 0 < glyph < 400:
                monster_count += 1
            elif glyph in [2359, 2360, 2361, 2362]:
                wall_count += 1
            elif glyph in [2363, 2364, 2365]:
                door_count += 1
            elif 400 < glyph < 800:
                item_count += 1
        
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
    
    return {"pre_nodes": pre_nodes, "scene_atoms": scene_atoms}


def extract_state_from_obs(obs, state_constructor, matcher, t_now=0):
    """从obs提取状态向量 (增强版)"""
    blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
    glyphs = obs.get("glyphs", None)
    atoms = extract_atoms_from_obs(obs)
    
    belief = np.zeros(50, dtype=np.float32)
    hp = blstats[nh.NLE_BL_HP]
    hp_max = blstats[nh.NLE_BL_HPMAX]
    
    # 基础状态
    belief[0] = 1.0
    belief[1] = 1.0 if hp > 0 else 0.0
    belief[2] = 1.0
    belief[3] = hp / max(hp_max, 1)
    belief[4] = 1.0 if blstats[nh.NLE_BL_HUNGER] == 0 else 0.0
    belief[5] = 1.0 if blstats[nh.NLE_BL_GOLD] == 0 else 0.0
    belief[6] = min(blstats[nh.NLE_BL_DEPTH] / 10.0, 1.0)
    belief[7] = min(blstats[nh.NLE_BL_GOLD] / 1000.0, 1.0)
    belief[8] = min(blstats[nh.NLE_BL_HUNGER] / 1500.0, 1.0)
    belief[9] = min(t_now / 1000.0, 1.0)
    
    # 环境感知
    if glyphs is not None:
        player_y = blstats[nh.NLE_BL_Y]
        player_x = blstats[nh.NLE_BL_X]
        
        y_min = max(0, player_y - 1)
        y_max = min(glyphs.shape[0], player_y + 2)
        x_min = max(0, player_x - 1)
        x_max = min(glyphs.shape[1], player_x + 2)
        
        surrounding = glyphs[y_min:y_max, x_min:x_max]
        
        monster_count = 0
        wall_count = 0
        door_count = 0
        item_count = 0
        floor_count = 0
        
        for glyph in surrounding.flatten():
            if 0 < glyph < 400:
                monster_count += 1
            elif glyph in [2359, 2360, 2361, 2362]:
                wall_count += 1
            elif glyph in [2363, 2364, 2365]:
                door_count += 1
            elif 400 < glyph < 800:
                item_count += 1
            elif glyph == 2358:
                floor_count += 1
        
        belief[10] = monster_count / 9.0
        belief[11] = wall_count / 9.0
        belief[12] = door_count / 9.0
        belief[13] = item_count / 9.0
        belief[14] = floor_count / 9.0
        belief[15] = 1.0 if monster_count > 0 else 0.0
        belief[16] = 1.0 if item_count > 0 else 0.0
        belief[17] = 1.0 if door_count > 0 else 0.0
        belief[18] = 1.0 if wall_count >= 6 else 0.0
        belief[19] = 1.0 if wall_count <= 2 else 0.0
    
    plot_atoms = {
        "pre": atoms["pre_nodes"],
        "scene": atoms["scene_atoms"],
        "effect": [],
        "rule": []
    }
    
    match_results = matcher.match(plot_atoms, t_now=t_now, top_k=8)
    confidence = np.mean([r.score for r in match_results]) if match_results else 0.0
    
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


class EpisodeVisualizer:
    """Episode时序可视化器"""
    
    def __init__(self, policy_net, state_constructor, matcher):
        self.policy_net = policy_net
        self.state_constructor = state_constructor
        self.matcher = matcher
        
        self.action_names = [
            'N', 'E', 'S', 'W', 'NE', 'SE', 'SW', 'NW',
            'search', 'kick', 'eat', 'apply', 'read',
            'quaff', 'zap', 'throw', 'fire', 'pray',
            'teleport', 'open', 'close', 'wait', 'look'
        ]
        
        self.expert_names = policy_net.expert_names
    
    def collect_episode_data(self, env, max_steps=50):
        """收集一个episode的数据"""
        print(f"\n[收集Episode数据] 最多{max_steps}步...")
        
        obs, info = env.reset()
        done = False
        truncated = False
        steps = 0
        
        # 存储每一步的数据
        episode_data = {
            'states': [],
            'atoms': [],
            'alpha': [],
            'actions': [],
            'action_probs': [],
            'operator_scores': [],
            'expert_logits': [],
            'rewards': [],
            'hp': [],
            'score': []
        }
        
        while not (done or truncated) and steps < max_steps:
            # 提取状态
            state, atoms = extract_state_from_obs(obs, self.state_constructor, 
                                                 self.matcher, t_now=steps)
            
            # 前向传播
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits, alpha, value, aux_info = self.policy_net(
                    state_tensor, atoms=atoms
                )
                
                probs = F.softmax(logits[0], dim=-1)
                action = torch.argmax(probs).item()
            
            # 记录数据
            episode_data['states'].append(state)
            episode_data['atoms'].append(atoms)
            episode_data['alpha'].append(alpha[0].numpy())
            episode_data['actions'].append(action)
            episode_data['action_probs'].append(probs.numpy())
            
            if aux_info['operator_scores'] is not None:
                episode_data['operator_scores'].append(
                    aux_info['operator_scores'][0].numpy()
                )
            
            if aux_info['expert_logits'] is not None:
                episode_data['expert_logits'].append(
                    aux_info['expert_logits'][0].numpy()
                )
            
            # 执行动作
            obs, reward, done, truncated, info = env.step(action)
            
            blstats = obs.get("blstats", [0] * nh.NLE_BLSTATS_SIZE)
            episode_data['rewards'].append(reward)
            episode_data['hp'].append(blstats[nh.NLE_BL_HP])
            episode_data['score'].append(blstats[nh.NLE_BL_SCORE])
            
            steps += 1
            
            if steps % 10 == 0:
                print(f"  步数: {steps}, HP: {blstats[nh.NLE_BL_HP]}, "
                      f"Score: {blstats[nh.NLE_BL_SCORE]}, "
                      f"动作: {self.action_names[action]}")
        
        print(f"✓ 收集完成: {steps}步")
        return episode_data
    
    def visualize_episode_heatmaps(self, episode_data, save_path=None):
        """生成episode时序热图"""
        steps = len(episode_data['alpha'])
        
        # 转换为numpy数组
        alpha_matrix = np.array(episode_data['alpha'])  # (steps, 4)
        action_matrix = np.array(episode_data['action_probs'])  # (steps, 23)
        
        # 创建大图
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # ================================================================
        # 1. 专家权重时序热图
        # ================================================================
        ax1 = fig.add_subplot(gs[0, :])
        im1 = ax1.imshow(alpha_matrix.T, aspect='auto', cmap='YlOrRd', 
                        interpolation='nearest')
        ax1.set_yticks(range(len(self.expert_names)))
        ax1.set_yticklabels(self.expert_names)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Expert')
        ax1.set_title('专家权重时序变化 (α权重)', fontsize=14, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='Weight')
        
        # 标记主导专家
        for t in range(steps):
            dominant = np.argmax(alpha_matrix[t])
            ax1.plot(t, dominant, 'w*', markersize=8)
        
        # ================================================================
        # 2. 动作概率时序热图
        # ================================================================
        ax2 = fig.add_subplot(gs[1, :])
        im2 = ax2.imshow(action_matrix.T, aspect='auto', cmap='viridis',
                        interpolation='nearest')
        ax2.set_yticks(range(0, 23, 2))
        ax2.set_yticklabels([self.action_names[i] for i in range(0, 23, 2)])
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Action')
        ax2.set_title('动作概率时序变化', fontsize=14, fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='Probability')
        
        # 标记实际选择的动作
        for t, action in enumerate(episode_data['actions']):
            ax2.plot(t, action, 'r*', markersize=6)
        
        # ================================================================
        # 3. 专家使用统计
        # ================================================================
        ax3 = fig.add_subplot(gs[2, 0])
        mean_alpha = alpha_matrix.mean(axis=0)
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']
        bars = ax3.bar(self.expert_names, mean_alpha, color=colors, 
                      alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Average Weight')
        ax3.set_title('平均专家使用率', fontsize=12, fontweight='bold')
        ax3.set_ylim([0, 1])
        
        # 添加数值标签
        for bar, val in zip(bars, mean_alpha):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2%}', ha='center', va='bottom')
        
        # ================================================================
        # 4. 专家切换分析
        # ================================================================
        ax4 = fig.add_subplot(gs[2, 1])
        dominant_experts = np.argmax(alpha_matrix, axis=1)
        
        # 计算切换次数
        switches = np.sum(dominant_experts[1:] != dominant_experts[:-1])
        
        # 绘制主导专家时序
        ax4.plot(dominant_experts, 'o-', linewidth=2, markersize=6)
        ax4.set_yticks(range(len(self.expert_names)))
        ax4.set_yticklabels(self.expert_names)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Dominant Expert')
        ax4.set_title(f'主导专家切换 (切换{switches}次)', 
                     fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # ================================================================
        # 5. 游戏状态变化
        # ================================================================
        ax5 = fig.add_subplot(gs[3, 0])
        ax5_twin = ax5.twinx()
        
        line1 = ax5.plot(episode_data['hp'], 'r-', linewidth=2, label='HP')
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('HP', color='r')
        ax5.tick_params(axis='y', labelcolor='r')
        
        line2 = ax5_twin.plot(episode_data['score'], 'b-', linewidth=2, 
                             label='Score')
        ax5_twin.set_ylabel('Score', color='b')
        ax5_twin.tick_params(axis='y', labelcolor='b')
        
        ax5.set_title('游戏状态变化', fontsize=12, fontweight='bold')
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper left')
        
        # ================================================================
        # 6. 奖励累积
        # ================================================================
        ax6 = fig.add_subplot(gs[3, 1])
        cumulative_reward = np.cumsum(episode_data['rewards'])
        ax6.plot(cumulative_reward, 'g-', linewidth=2)
        ax6.fill_between(range(len(cumulative_reward)), cumulative_reward, 
                        alpha=0.3, color='green')
        ax6.set_xlabel('Time Step')
        ax6.set_ylabel('Cumulative Reward')
        ax6.set_title('累积奖励', fontsize=12, fontweight='bold')
        ax6.grid(alpha=0.3)
        
        # 总标题
        fig.suptitle(f'V3 Episode时序分析 ({steps}步)', 
                    fontsize=16, fontweight='bold')
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 时序热图已保存: {save_path}")
        
        plt.close()
    
    def visualize_key_moments(self, episode_data, save_dir=None):
        """可视化关键时刻 (专家切换、高奖励等)"""
        if save_dir is None:
            return
        
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        alpha_matrix = np.array(episode_data['alpha'])
        dominant_experts = np.argmax(alpha_matrix, axis=1)
        
        # 找到专家切换的时刻
        switch_points = []
        for t in range(1, len(dominant_experts)):
            if dominant_experts[t] != dominant_experts[t-1]:
                switch_points.append(t)
        
        print(f"\n[关键时刻] 发现{len(switch_points)}个专家切换点")
        
        # 为每个切换点生成详细可视化
        for i, t in enumerate(switch_points[:5]):  # 最多5个
            self._visualize_single_moment(
                episode_data, t,
                save_path=f"{save_dir}/switch_{i+1}_step_{t}.png"
            )
    
    def _visualize_single_moment(self, episode_data, t, save_path):
        """可视化单个时刻的详细信息"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Step {t} - 专家切换时刻', fontsize=14, fontweight='bold')
        
        # 1. 当前专家权重
        ax = axes[0, 0]
        alpha = episode_data['alpha'][t]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']
        ax.bar(self.expert_names, alpha, color=colors, alpha=0.7, 
              edgecolor='black')
        ax.set_ylabel('Weight')
        ax.set_title(f'专家权重 (t={t})')
        ax.set_ylim([0, 1])
        
        # 2. 前后专家对比
        ax = axes[0, 1]
        if t > 0:
            alpha_prev = episode_data['alpha'][t-1]
            x = np.arange(len(self.expert_names))
            width = 0.35
            ax.bar(x - width/2, alpha_prev, width, label=f't={t-1}', 
                  alpha=0.7)
            ax.bar(x + width/2, alpha, width, label=f't={t}', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(self.expert_names, rotation=45)
            ax.set_ylabel('Weight')
            ax.set_title('专家权重变化')
            ax.legend()
        
        # 3. 动作概率
        ax = axes[0, 2]
        action_probs = episode_data['action_probs'][t]
        top_5_idx = np.argsort(action_probs)[-5:][::-1]
        top_5_probs = action_probs[top_5_idx]
        top_5_names = [self.action_names[i] for i in top_5_idx]
        
        ax.barh(range(5), top_5_probs, color='steelblue', alpha=0.7)
        ax.set_yticks(range(5))
        ax.set_yticklabels(top_5_names)
        ax.set_xlabel('Probability')
        ax.set_title('Top-5 动作')
        ax.invert_yaxis()
        
        # 4. Atoms信息
        ax = axes[1, 0]
        ax.axis('off')
        atoms = episode_data['atoms'][t]
        text = "Activated Atoms:\n\n"
        text += "Pre-conditions:\n"
        for node in atoms['pre_nodes'][:5]:
            text += f"  • {node}\n"
        text += "\nScene:\n"
        for atom in atoms['scene_atoms'][:5]:
            text += f"  • {atom}\n"
        ax.text(0.1, 0.5, text, fontsize=10, verticalalignment='center',
               family='monospace')
        
        # 5. 游戏状态
        ax = axes[1, 1]
        ax.axis('off')
        text = f"Game State (t={t}):\n\n"
        text += f"HP: {episode_data['hp'][t]}\n"
        text += f"Score: {episode_data['score'][t]}\n"
        text += f"Reward: {episode_data['rewards'][t]:.2f}\n"
        text += f"\nAction: {self.action_names[episode_data['actions'][t]]}\n"
        ax.text(0.1, 0.5, text, fontsize=12, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.3))
        
        # 6. 专家输出对比
        ax = axes[1, 2]
        if len(episode_data['expert_logits']) > t:
            expert_logits = episode_data['expert_logits'][t]
            expert_probs = F.softmax(torch.tensor(expert_logits), dim=-1).numpy()
            
            im = ax.imshow(expert_probs, aspect='auto', cmap='YlOrRd')
            ax.set_yticks(range(len(self.expert_names)))
            ax.set_yticklabels(self.expert_names)
            ax.set_xticks(range(0, 23, 5))
            ax.set_xticklabels([self.action_names[i] for i in range(0, 23, 5)],
                              rotation=45)
            ax.set_title('专家输出分布')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 保存: {save_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V3 Episode时序可视化")
    parser.add_argument("--checkpoint", type=str,
                       default="ablation_v3/results/v3_convergence_cpu/checkpoints/best_model.pth",
                       help="模型checkpoint")
    parser.add_argument("--steps", type=int, default=50,
                       help="最大步数")
    parser.add_argument("--output-dir", type=str,
                       default="ablation_v3/visualizations/episode",
                       help="输出目录")
    args = parser.parse_args()
    
    print("="*60)
    print("V3 Episode时序可视化")
    print("="*60)
    
    # 加载模型
    print("\n[1] 加载模型...")
    device = torch.device('cpu')
    policy_net = GATGuidedMoEPolicy(
        hypergraph_path="data/hypergraph/hypergraph_gat_structure.json",
        state_dim=115,
        hidden_dim=256,
        action_dim=23,
        num_experts=4,
        use_sparsemax=True
    ).to(device)
    
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device, 
                               weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net'])
        print(f"✓ 已加载: {args.checkpoint}")
    else:
        print(f"⚠️ 使用随机初始化")
    
    policy_net.eval()
    
    # 加载超图
    print("\n[2] 加载超图...")
    state_constructor = StateConstructor("data/hypergraph/hypergraph_complete_real.json")
    matcher = HypergraphMatcher(state_constructor.hypergraph,
                                weights=(0.35, 0.35, 0.2, 0.1), tau=200.0)
    print(f"✓ 超图加载完成")
    
    # 创建环境
    print("\n[3] 创建环境...")
    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")
    print(f"✓ 环境创建完成")
    
    # 创建可视化器
    print("\n[4] 创建可视化器...")
    visualizer = EpisodeVisualizer(policy_net, state_constructor, matcher)
    
    # 收集episode数据
    episode_data = visualizer.collect_episode_data(env, max_steps=args.steps)
    
    # 生成时序热图
    print("\n[5] 生成时序热图...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    visualizer.visualize_episode_heatmaps(
        episode_data,
        save_path=f"{args.output_dir}/episode_heatmaps.png"
    )
    
    # 生成关键时刻可视化
    print("\n[6] 生成关键时刻可视化...")
    visualizer.visualize_key_moments(
        episode_data,
        save_dir=f"{args.output_dir}/key_moments"
    )
    
    print(f"\n✓ 可视化完成!")
    print(f"  输出目录: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

