#!/usr/bin/env python3
"""
消融实验训练脚本 - 可配置版本

支持三种模式:
1. baseline - 无Manager约束，无熵最小化
2. manager - 有Manager约束，无熵最小化  
3. full - 有Manager约束，有熵最小化（完整版）
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# 导入原始训练脚本的main函数
# 但我们需要修改配置

def get_ablation_config(mode: str, episode: int):
    """
    根据消融模式返回配置
    
    Args:
        mode: 'baseline', 'manager', 'full'
        episode: 当前episode数
    """
    # 基础配置（所有模式共享）
    if episode < 1000:
        base_config = {
            'phase': 'warmup',
            'use_sparsemax': False,
            'sparsemax_temp': 1.0,
            'learning_rate': 1e-4,
            'entropy_coef': 0.05,
            'alpha_entropy_coef': 0.1,
            'alpha_entropy_sign': -1,  # 最大化熵
            'load_balance_coef': 0.02,
            'diversity_coef': 0.01,
        }
    elif episode < 3000:
        progress = (episode - 1000) / 2000
        temp = 1.0 - 0.5 * progress
        alpha_entropy_coef = 0.1 * (1 - progress)
        
        base_config = {
            'phase': 'transition',
            'use_sparsemax': True,
            'sparsemax_temp': temp,
            'learning_rate': 5e-5,
            'entropy_coef': 0.02,
            'alpha_entropy_coef': alpha_entropy_coef,
            'alpha_entropy_sign': -1,  # 仍然最大化熵
            'load_balance_coef': 0.01,
            'diversity_coef': 0.01,
        }
    else:
        base_config = {
            'phase': 'fine-tune',
            'use_sparsemax': True,
            'sparsemax_temp': 0.5,
            'learning_rate': 1e-5,
            'entropy_coef': 0.01,
            'alpha_entropy_coef': 0.05,
            'alpha_entropy_sign': -1,  # baseline和manager保持-1
            'load_balance_coef': 0.005,
            'diversity_coef': 0.005,
        }
    
    # 根据模式添加特定配置
    if mode == 'baseline':
        # 无Manager约束，无熵最小化
        base_config.update({
            'alignment_coef': 0.0,      # 关闭
            'alignment_temperature': 1.0,
            'semantic_coef': 0.0,        # 关闭
            'temporal_coef': 0.0,
            'overlap_coef': 0.0,
        })
    
    elif mode == 'manager':
        # 有Manager约束，无熵最小化
        base_config.update({
            'alignment_coef': 0.1,       # 开启
            'alignment_temperature': 1.0,
            'semantic_coef': 0.05,       # 开启
            'temporal_coef': 0.01 if episode >= 1000 else 0.0,
            'overlap_coef': 0.03 if episode >= 1000 else 0.0,
        })
        # 保持alpha_entropy_sign=-1（不改）
    
    elif mode == 'full':
        # 有Manager约束，有熵最小化（完整版）
        base_config.update({
            'alignment_coef': 0.1,
            'alignment_temperature': 1.0,
            'semantic_coef': 0.05,
            'temporal_coef': 0.01 if episode >= 1000 else 0.0,
            'overlap_coef': 0.03 if episode >= 1000 else 0.0,
        })
        # Fine-tune阶段改为熵最小化
        if episode >= 3000:
            base_config['alpha_entropy_sign'] = +1  # 最小化熵
    
    return base_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="消融实验训练脚本")
    parser.add_argument("--mode", type=str, required=True,
                        choices=['baseline', 'manager', 'full'],
                        help="消融模式: baseline/manager/full")
    parser.add_argument("--exp-name", type=str, required=True,
                        help="实验名称")
    parser.add_argument("--episodes", type=int, default=500,
                        help="训练episodes数")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="每episode最大步数")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"消融实验 - 模式: {args.mode.upper()}")
    print(f"{'='*70}")
    print(f"实验名称: {args.exp_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Max Steps: {args.max_steps}")
    
    if args.mode == 'baseline':
        print("配置: 无Manager约束，无熵最小化")
    elif args.mode == 'manager':
        print("配置: 有Manager约束，无熵最小化")
    elif args.mode == 'full':
        print("配置: 有Manager约束，有熵最小化（完整版）")
    
    print(f"{'='*70}\n")
    
    # 导入并修改原始训练脚本
    # 这里我们需要修改get_training_config函数
    import ablation_v3.train.train_v3_gat_moe as train_module
    
    # 保存原始函数
    original_get_config = train_module.get_training_config
    
    # 替换为消融版本
    def ablation_get_config(episode: int):
        return get_ablation_config(args.mode, episode)
    
    train_module.get_training_config = ablation_get_config
    
    # 修改sys.argv以传递参数
    sys.argv = [
        'train_v3_gat_moe.py',
        '--exp-name', args.exp_name,
        '--episodes', str(args.episodes),
        '--max-steps', str(args.max_steps),
    ]
    
    # 运行训练
    train_module.main()
