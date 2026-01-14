#!/usr/bin/env python3
"""
测试V3路由的动态性
强制agent移动，观察路由变化
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym
import nle.nethack as nh

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


def main():
    print("="*60)
    print("V3 路由动态性测试 - 强制移动")
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
    
    checkpoint_path = "ablation_v3/results/v3_convergence_cpu/checkpoints/best_model.pth"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net'])
        print(f"✓ 已加载checkpoint")
    
    policy_net.eval()
    
    # 加载超图
    print("\n[2] 加载超图...")
    state_constructor = StateConstructor("data/hypergraph/hypergraph_complete_real.json")
    matcher = HypergraphMatcher(state_constructor.hypergraph,
                                weights=(0.35, 0.35, 0.2, 0.1), tau=200.0)
    
    # 创建环境
    print("\n[3] 创建环境...")
    try:
        env = gym.make("NetHackScore-v0")
    except:
        env = gym.make("NetHack-v0")
    
    # 强制移动测试
    print("\n[4] 强制移动测试 (N, E, S, W各5步)...")
    obs, info = env.reset()
    
    expert_names = ['Survival', 'Combat', 'Exploration', 'General']
    
    # 定义移动序列: N, E, S, W
    move_actions = [0, 1, 2, 3]  # N, E, S, W
    move_names = ['N', 'E', 'S', 'W']
    
    alpha_history = []
    atoms_history = []
    
    for direction_idx, (action, name) in enumerate(zip(move_actions, move_names)):
        print(f"\n{'='*60}")
        print(f"方向: {name} (连续5步)")
        print(f"{'='*60}")
        
        for step in range(5):
            # 提取状态
            state, atoms = extract_state_from_obs(obs, state_constructor, matcher, 
                                                 t_now=direction_idx*5 + step)
            
            # 前向传播
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                logits, alpha, value, aux_info = policy_net(state_tensor, atoms=atoms)
            
            alpha_history.append(alpha[0].numpy())
            atoms_history.append(atoms)
            
            print(f"\nStep {direction_idx*5 + step}:")
            print(f"  Atoms: {atoms['scene_atoms']}")
            print(f"  Alpha: [{alpha[0][0]:.3f}, {alpha[0][1]:.3f}, {alpha[0][2]:.3f}, {alpha[0][3]:.3f}]")
            print(f"  Dominant: {expert_names[alpha[0].argmax()]}")
            
            # 执行强制动作
            obs, reward, done, truncated, info = env.step(action)
            
            if done or truncated:
                print("\n[Episode结束]")
                obs, info = env.reset()
                break
    
    # 分析Alpha变化
    print(f"\n{'='*60}")
    print("Alpha变化分析")
    print(f"{'='*60}")
    
    alpha_matrix = np.array(alpha_history)
    
    print(f"\n各专家的Alpha统计:")
    for i, name in enumerate(expert_names):
        mean_alpha = alpha_matrix[:, i].mean()
        std_alpha = alpha_matrix[:, i].std()
        max_alpha = alpha_matrix[:, i].max()
        min_alpha = alpha_matrix[:, i].min()
        print(f"  {name}:")
        print(f"    Mean: {mean_alpha:.4f}, Std: {std_alpha:.4f}")
        print(f"    Range: [{min_alpha:.4f}, {max_alpha:.4f}]")
    
    # 计算切换次数
    dominant_experts = alpha_matrix.argmax(axis=1)
    switches = np.sum(dominant_experts[1:] != dominant_experts[:-1])
    
    print(f"\n专家切换次数: {switches} / {len(alpha_history)-1}")
    print(f"切换率: {switches/(len(alpha_history)-1)*100:.1f}%")
    
    # 检查Atoms变化
    unique_atoms = set()
    for atoms in atoms_history:
        unique_atoms.update(atoms['scene_atoms'])
    
    print(f"\n观察到的唯一Scene Atoms: {len(unique_atoms)}")
    print(f"  {sorted(unique_atoms)}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
