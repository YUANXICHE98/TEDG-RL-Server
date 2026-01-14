#!/usr/bin/env python3
"""
快速对比测试：验证Manager约束是否有效

运行两个短期训练（各100 episodes）：
1. Baseline：alignment_coef=0, semantic_coef=0
2. With Manager：alignment_coef=0.1, semantic_coef=0.05

对比指标：
- Alpha熵的下降速度
- 平均分数的提升
- 训练稳定性
"""

import os
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

def run_training(exp_name, episodes, alignment_coef, semantic_coef):
    """运行一次训练实验"""
    print(f"\n{'='*70}")
    print(f"运行实验: {exp_name}")
    print(f"  - Episodes: {episodes}")
    print(f"  - Alignment coef: {alignment_coef}")
    print(f"  - Semantic coef: {semantic_coef}")
    print(f"{'='*70}\n")
    
    # 临时修改配置
    train_script = Path("ablation_v3/train/train_v3_gat_moe.py")
    backup_script = Path("ablation_v3/train/train_v3_gat_moe_backup.py")
    
    # 备份原始文件
    if not backup_script.exists():
        import shutil
        shutil.copy(train_script, backup_script)
    
    # 读取训练脚本
    with open(train_script, 'r') as f:
        content = f.read()
    
    # 修改Manager约束系数
    content_modified = re.sub(
        r"'alignment_coef':\s*[0-9.]+",
        f"'alignment_coef': {alignment_coef}",
        content
    )
    content_modified = re.sub(
        r"'semantic_coef':\s*[0-9.]+",
        f"'semantic_coef': {semantic_coef}",
        content_modified
    )
    
    # 写入修改后的文件
    with open(train_script, 'w') as f:
        f.write(content_modified)
    
    # 运行训练
    cmd = [
        "python", "ablation_v3/train/train_v3_gat_moe.py",
        "--exp-name", exp_name,
        "--episodes", str(episodes),
        "--max-steps", "2000"
    ]
    
    log_file = Path(f"ablation_v3/results/{exp_name}/training.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                print(line, end='')
                f.write(line)
            
            process.wait()
            
            if process.returncode != 0:
                print(f"\n⚠️  训练出现错误（返回码: {process.returncode}）")
                return None
        
        print(f"\n✓ 训练完成，日志保存到: {log_file}")
        return log_file
        
    finally:
        # 恢复原始文件
        if backup_script.exists():
            import shutil
            shutil.copy(backup_script, train_script)
            backup_script.unlink()


def extract_metrics(log_file):
    """从日志文件中提取关键指标"""
    if not log_file or not log_file.exists():
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 提取所有episode的指标
    alpha_entropies = []
    scores = []
    alignment_losses = []
    
    for line in content.split('\n'):
        # 提取Alpha熵
        match = re.search(r'α_entropy:\s*([0-9.]+)', line)
        if match:
            alpha_entropies.append(float(match.group(1)))
        
        # 提取分数
        match = re.search(r'Score:\s*([0-9]+)', line)
        if match:
            scores.append(int(match.group(1)))
        
        # 提取Alignment loss
        match = re.search(r'Alignment=([0-9.]+)', line)
        if match:
            alignment_losses.append(float(match.group(1)))
    
    if not alpha_entropies or not scores:
        return None
    
    return {
        'alpha_entropy_initial': alpha_entropies[0] if alpha_entropies else None,
        'alpha_entropy_final': alpha_entropies[-1] if alpha_entropies else None,
        'alpha_entropy_mean': sum(alpha_entropies) / len(alpha_entropies) if alpha_entropies else None,
        'score_initial': sum(scores[:10]) / min(10, len(scores)) if scores else None,
        'score_final': sum(scores[-10:]) / min(10, len(scores[-10:])) if scores else None,
        'score_mean': sum(scores) / len(scores) if scores else None,
        'alignment_loss_mean': sum(alignment_losses) / len(alignment_losses) if alignment_losses else None,
        'num_episodes': len(scores),
    }


def print_comparison(baseline_metrics, manager_metrics):
    """打印对比结果"""
    print(f"\n{'='*70}")
    print("对比结果")
    print(f"{'='*70}\n")
    
    if not baseline_metrics or not manager_metrics:
        print("⚠️  无法提取指标，请检查训练日志")
        return
    
    print(f"{'指标':<30} | {'Baseline':<12} | {'Manager':<12} | {'改进':<12}")
    print(f"{'-'*30}+{'-'*14}+{'-'*14}+{'-'*14}")
    
    # Alpha熵（初始）
    b_alpha_init = baseline_metrics['alpha_entropy_initial']
    m_alpha_init = manager_metrics['alpha_entropy_initial']
    print(f"{'Alpha熵（初始）':<30} | {b_alpha_init:<12.3f} | {m_alpha_init:<12.3f} | {'-':<12}")
    
    # Alpha熵（终态）
    b_alpha_final = baseline_metrics['alpha_entropy_final']
    m_alpha_final = manager_metrics['alpha_entropy_final']
    alpha_improve = (b_alpha_final - m_alpha_final) / b_alpha_final * 100 if b_alpha_final > 0 else 0
    print(f"{'Alpha熵（终态）':<30} | {b_alpha_final:<12.3f} | {m_alpha_final:<12.3f} | {alpha_improve:>11.1f}%")
    
    # 平均分数（初始10ep）
    b_score_init = baseline_metrics['score_initial']
    m_score_init = manager_metrics['score_initial']
    print(f"{'平均分数（初始10ep）':<30} | {b_score_init:<12.1f} | {m_score_init:<12.1f} | {'-':<12}")
    
    # 平均分数（最后10ep）
    b_score_final = baseline_metrics['score_final']
    m_score_final = manager_metrics['score_final']
    score_improve = (m_score_final - b_score_final) / b_score_final * 100 if b_score_final > 0 else 0
    print(f"{'平均分数（最后10ep）':<30} | {b_score_final:<12.1f} | {m_score_final:<12.1f} | {score_improve:>11.1f}%")
    
    # Alignment Loss
    m_alignment = manager_metrics['alignment_loss_mean']
    if m_alignment:
        print(f"{'Alignment Loss（平均）':<30} | {'N/A':<12} | {m_alignment:<12.4f} | {'-':<12}")
    
    print()
    
    # 结论
    print(f"{'='*70}")
    print("结论")
    print(f"{'='*70}\n")
    
    if alpha_improve > 10 or score_improve > 10:
        print("✓ Manager约束显著改善了训练效果！")
        print(f"  - Alpha熵降低: {alpha_improve:.1f}%（专家更专业化）")
        print(f"  - 分数提升: {score_improve:.1f}%（性能更好）")
        print("\n建议：继续进行完整训练（1000-5000 episodes）")
    elif alpha_improve > 0 or score_improve > 0:
        print("⚠ Manager约束有轻微改善，但不明显")
        print(f"  - Alpha熵降低: {alpha_improve:.1f}%")
        print(f"  - 分数提升: {score_improve:.1f}%")
        print("\n建议：")
        print("  1. 增加训练episodes（100 → 500）")
        print("  2. 调整系数（alignment_coef: 0.1 → 0.2）")
    else:
        print("✗ Manager约束未显示改善")
        print("\n建议：")
        print("  1. 检查实现是否正确")
        print("  2. 调整超参数")
        print("  3. 增加训练时间")
    
    print()


def main():
    """主函数"""
    print(f"\n{'='*70}")
    print("快速对比测试：Baseline vs Manager约束")
    print(f"{'='*70}\n")
    print("测试配置：")
    print("  - Episodes: 100")
    print("  - Max steps: 2000")
    print("  - 对比指标: Alpha熵、平均分数")
    print()
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(f"ablation_v3/results/quick_comparison_{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"结果将保存到: {result_dir}\n")
    
    # 实验1：Baseline
    baseline_log = run_training(
        exp_name=f"quick_baseline_{timestamp}",
        episodes=100,
        alignment_coef=0.0,  # 禁用Manager约束
        semantic_coef=0.0
    )
    
    # 实验2：With Manager
    manager_log = run_training(
        exp_name=f"quick_manager_{timestamp}",
        episodes=100,
        alignment_coef=0.1,  # 启用Manager约束
        semantic_coef=0.05
    )
    
    # 提取指标
    print(f"\n{'='*70}")
    print("提取指标...")
    print(f"{'='*70}\n")
    
    baseline_metrics = extract_metrics(baseline_log)
    manager_metrics = extract_metrics(manager_log)
    
    if baseline_metrics:
        print("✓ Baseline指标提取成功")
    else:
        print("✗ Baseline指标提取失败")
    
    if manager_metrics:
        print("✓ Manager指标提取成功")
    else:
        print("✗ Manager指标提取失败")
    
    # 打印对比
    print_comparison(baseline_metrics, manager_metrics)
    
    # 保存结果
    summary_file = result_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'baseline': baseline_metrics,
            'manager': manager_metrics,
        }, f, indent=2)
    
    print(f"详细结果已保存到: {summary_file}\n")


if __name__ == "__main__":
    main()
