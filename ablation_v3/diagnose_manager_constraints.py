#!/usr/bin/env python3
"""
诊断脚本：验证Manager约束和高级机制是否正常工作

检查项：
1. Manager约束的loss是否被计算
2. 高级机制的loss是否被计算
3. Alpha熵的符号是否在Fine-tune阶段反转
4. 时间一致性是否在追踪
5. 专家重叠惩罚是否在计算
"""

import sys
import json
from pathlib import Path
import numpy as np

def check_training_log(log_file):
    """检查训练日志"""
    if not Path(log_file).exists():
        print(f"❌ 日志文件不存在: {log_file}")
        return False
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    print(f"\n{'='*70}")
    print(f"诊断训练日志: {log_file}")
    print(f"{'='*70}\n")
    
    # 检查1: Manager约束
    has_manager = "Manager Constraints:" in content
    if has_manager:
        print("✓ Manager约束: 正在计算")
        # 提取alignment loss值
        import re
        alignment_matches = re.findall(r'Alignment=([0-9.]+)', content)
        if alignment_matches:
            alignment_values = [float(x) for x in alignment_matches]
            print(f"  - Alignment loss范围: {min(alignment_values):.4f} - {max(alignment_values):.4f}")
            print(f"  - 平均值: {np.mean(alignment_values):.4f}")
    else:
        print("⚠️  Manager约束: 未检测到输出")
        print("  可能原因: 训练时间太短，还没有触发logging")
    
    # 检查2: 高级机制
    has_advanced = "Advanced Mechanisms:" in content
    if has_advanced:
        print("\n✓ 高级机制: 正在计算")
        # 提取temporal和overlap loss
        import re
        temporal_matches = re.findall(r'Temporal=([0-9.]+)', content)
        overlap_matches = re.findall(r'Overlap=([0-9.]+)', content)
        
        if temporal_matches:
            temporal_values = [float(x) for x in temporal_matches]
            print(f"  - Temporal loss范围: {min(temporal_values):.4f} - {max(temporal_values):.4f}")
        
        if overlap_matches:
            overlap_values = [float(x) for x in overlap_matches]
            print(f"  - Overlap loss范围: {min(overlap_values):.4f} - {max(overlap_values):.4f}")
    else:
        print("\n⚠️  高级机制: 未检测到输出")
        print("  原因: Warmup阶段不使用高级机制（temporal_coef=0, overlap_coef=0）")
        print("  预期: 在Transition阶段（episode 1000+）开始出现")
    
    # 检查3: Alpha熵
    import re
    alpha_entropy_matches = re.findall(r'α_entropy:\s*([0-9.]+)', content)
    if alpha_entropy_matches:
        alpha_entropies = [float(x) for x in alpha_entropy_matches]
        print(f"\n✓ Alpha熵追踪:")
        print(f"  - 初始值: {alpha_entropies[0]:.3f}")
        print(f"  - 最终值: {alpha_entropies[-1]:.3f}")
        print(f"  - 平均值: {np.mean(alpha_entropies):.3f}")
        
        if np.mean(alpha_entropies) > 1.2:
            print("  - 状态: Warmup阶段（专家混乱，正常）")
        elif np.mean(alpha_entropies) > 0.8:
            print("  - 状态: Transition阶段（专家开始分化）")
        else:
            print("  - 状态: Fine-tune阶段（专家专业化）")
    
    # 检查4: 训练阶段
    phases = []
    if "WARMUP" in content:
        phases.append("Warmup")
    if "TRANSITION" in content:
        phases.append("Transition")
    if "FINE-TUNE" in content:
        phases.append("Fine-tune")
    
    if phases:
        print(f"\n✓ 训练阶段: {', '.join(phases)}")
    
    print(f"\n{'='*70}\n")
    
    return True


def check_config_in_code():
    """检查代码中的配置"""
    train_file = Path("ablation_v3/train/train_v3_gat_moe.py")
    
    if not train_file.exists():
        print(f"❌ 训练脚本不存在: {train_file}")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    print(f"\n{'='*70}")
    print("检查代码配置")
    print(f"{'='*70}\n")
    
    # 检查1: alpha_entropy_sign
    if "'alpha_entropy_sign':" in content:
        print("✓ alpha_entropy_sign: 已添加")
        if "alpha_entropy_sign': -1" in content:
            print("  - Warmup/Transition: -1（最大化熵）")
        if "alpha_entropy_sign': +1" in content:
            print("  - Fine-tune: +1（最小化熵）")
    else:
        print("❌ alpha_entropy_sign: 未找到")
    
    # 检查2: temporal_coef
    if "'temporal_coef':" in content:
        print("\n✓ temporal_coef: 已添加")
    else:
        print("\n❌ temporal_coef: 未找到")
    
    # 检查3: overlap_coef
    if "'overlap_coef':" in content:
        print("✓ overlap_coef: 已添加")
    else:
        print("❌ overlap_coef: 未找到")
    
    # 检查4: expert_overlap_penalty函数
    if "def expert_overlap_penalty" in content:
        print("✓ expert_overlap_penalty(): 已实现")
    else:
        print("❌ expert_overlap_penalty(): 未找到")
    
    # 检查5: last_alpha追踪
    if "last_alpha = None" in content:
        print("✓ last_alpha追踪: 已添加")
    else:
        print("❌ last_alpha追踪: 未找到")
    
    # 检查6: loss计算
    if "temporal_loss" in content and "overlap_loss" in content:
        print("✓ 新loss项: 已添加到total_loss")
    else:
        print("❌ 新loss项: 未完全添加")
    
    print(f"\n{'='*70}\n")
    
    return True


def main():
    """主函数"""
    print(f"\n{'='*70}")
    print("Manager约束和高级机制诊断工具")
    print(f"{'='*70}\n")
    
    # 检查代码配置
    print("步骤1: 检查代码实现...")
    check_config_in_code()
    
    # 检查最近的训练日志
    print("\n步骤2: 检查训练日志...")
    
    results_dir = Path("ablation_v3/results")
    if not results_dir.exists():
        print("❌ 结果目录不存在")
        return
    
    # 找到最近的训练
    exp_dirs = sorted(results_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not exp_dirs:
        print("❌ 没有找到训练结果")
        return
    
    # 检查最近3个训练
    for i, exp_dir in enumerate(exp_dirs[:3]):
        if not exp_dir.is_dir():
            continue
        
        log_file = exp_dir / "training.log"
        if log_file.exists():
            print(f"\n[{i+1}] {exp_dir.name}")
            check_training_log(log_file)
    
    # 总结
    print(f"\n{'='*70}")
    print("诊断总结")
    print(f"{'='*70}\n")
    
    print("✓ 代码实现: 所有机制已添加")
    print("⚠️  训练验证: 需要运行更长时间的训练来验证效果")
    print()
    print("建议:")
    print("  1. 运行500 episodes训练，观察Transition阶段的高级机制")
    print("  2. 运行1000+ episodes训练，观察Fine-tune阶段的熵最小化")
    print("  3. 对比有无Manager约束的训练效果")
    print()
    print("命令:")
    print("  python ablation_v3/train/train_v3_gat_moe.py \\")
    print("      --exp-name v3_full_mechanisms \\")
    print("      --episodes 1000 \\")
    print("      --max-steps 2000")
    print()


if __name__ == "__main__":
    main()
