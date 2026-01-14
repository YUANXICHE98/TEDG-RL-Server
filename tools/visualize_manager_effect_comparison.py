#!/usr/bin/env python3
"""
全面对比有/无Manager约束（内部奖励）的训练效果
展示多个episodes下的效果对比和改进分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_training_log(result_dir):
    """加载训练日志"""
    log_file = Path(result_dir) / "logs" / "training_log.json"
    
    if not log_file.exists():
        print(f"⚠️  日志文件不存在: {log_file}")
        return None
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    return data

def extract_metrics(data):
    """提取关键指标"""
    if data is None:
        return None
    
    # 处理不同的日志格式
    if isinstance(data, dict):
        # 新格式：字典包含列表
        episodes = list(range(len(data.get('episode_rewards', []))))
        scores = data.get('episode_scores', data.get('episode_rewards', []))
        rewards = data.get('episode_rewards', [])
        alpha_entropies = data.get('alpha_entropies', [0] * len(rewards))
        alignment_losses = data.get('alignment_losses', [None] * len(rewards))
        semantic_losses = data.get('semantic_losses', [None] * len(rewards))
    else:
        # 旧格式：列表包含字典
        episodes = [entry['episode'] for entry in data]
        scores = [entry.get('score', entry.get('reward', 0)) for entry in data]
        rewards = [entry.get('reward', 0) for entry in data]
        alpha_entropies = [entry.get('alpha_entropy', 0) for entry in data]
        alignment_losses = [entry.get('alignment_loss', None) for entry in data]
        semantic_losses = [entry.get('semantic_loss', None) for entry in data]
    
    return {
        'episodes': episodes,
        'scores': scores,
        'rewards': rewards,
        'alpha_entropies': alpha_entropies,
        'alignment_losses': alignment_losses,
        'semantic_losses': semantic_losses
    }

def moving_average(values, window=50):
    """计算移动平均"""
    if len(values) < window:
        window = max(1, len(values) // 10)
    
    ma = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        ma.append(np.mean(values[start:i+1]))
    return ma

def compute_improvement(baseline, manager):
    """计算改进百分比"""
    if baseline == 0:
        return 0
    return ((manager - baseline) / abs(baseline)) * 100

def plot_comprehensive_comparison(baseline_metrics, manager_metrics, output_dir):
    """绘制全面对比图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建大图：3行2列
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    fig.suptitle('Manager约束（内部奖励）效果全面对比', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # ========================================
    # 1. 分数对比（左上）
    # ========================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    if baseline_metrics:
        baseline_ma = moving_average(baseline_metrics['scores'])
        ax1.plot(baseline_metrics['episodes'], baseline_ma, 
                label='无Manager约束 (Baseline)', color='#2E86AB', linewidth=2.5, alpha=0.8)
    
    if manager_metrics:
        manager_ma = moving_average(manager_metrics['scores'])
        ax1.plot(manager_metrics['episodes'], manager_ma, 
                label='有Manager约束 (With Manager)', color='#A23B72', linewidth=2.5, alpha=0.8)
    
    ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax1.set_ylabel('平均分数 (Score)', fontsize=12, fontweight='bold')
    ax1.set_title('分数对比 - 多Episode效果', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ========================================
    # 2. Alpha熵对比（右上）
    # ========================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    if baseline_metrics:
        ax2.plot(baseline_metrics['episodes'], baseline_metrics['alpha_entropies'], 
                label='无Manager约束', color='#2E86AB', linewidth=2.5, alpha=0.8)
    
    if manager_metrics:
        ax2.plot(manager_metrics['episodes'], manager_metrics['alpha_entropies'], 
                label='有Manager约束', color='#A23B72', linewidth=2.5, alpha=0.8)
    
    ax2.axhline(y=1.386, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='理论最大值 (ln(4))')
    ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Alpha熵', fontsize=12, fontweight='bold')
    ax2.set_title('专家专业化程度 (熵越低越专业)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # ========================================
    # 3. 奖励对比（左中）
    # ========================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    if baseline_metrics:
        baseline_reward_ma = moving_average(baseline_metrics['rewards'])
        ax3.plot(baseline_metrics['episodes'], baseline_reward_ma, 
                label='无Manager约束', color='#2E86AB', linewidth=2.5, alpha=0.8)
    
    if manager_metrics:
        manager_reward_ma = moving_average(manager_metrics['rewards'])
        ax3.plot(manager_metrics['episodes'], manager_reward_ma, 
                label='有Manager约束', color='#A23B72', linewidth=2.5, alpha=0.8)
    
    ax3.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax3.set_ylabel('平均奖励 (Reward)', fontsize=12, fontweight='bold')
    ax3.set_title('奖励对比', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    # ========================================
    # 4. 改进率曲线（右中）
    # ========================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    if baseline_metrics and manager_metrics:
        # 计算每个episode的改进率
        min_len = min(len(baseline_metrics['scores']), len(manager_metrics['scores']))
        improvements = []
        episodes_imp = []
        
        for i in range(min_len):
            baseline_val = baseline_metrics['scores'][i]
            manager_val = manager_metrics['scores'][i]
            if baseline_val != 0:
                imp = ((manager_val - baseline_val) / abs(baseline_val)) * 100
                improvements.append(imp)
                episodes_imp.append(i)
        
        # 移动平均平滑改进率
        if improvements:
            imp_ma = moving_average(improvements, window=50)
            ax4.plot(episodes_imp, imp_ma, color='#F18F01', linewidth=2.5)
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
            ax4.fill_between(episodes_imp, 0, imp_ma, where=[x > 0 for x in imp_ma], 
                           alpha=0.3, color='green', label='改进')
            ax4.fill_between(episodes_imp, 0, imp_ma, where=[x < 0 for x in imp_ma], 
                           alpha=0.3, color='red', label='退步')
    
    ax4.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax4.set_ylabel('改进率 (%)', fontsize=12, fontweight='bold')
    ax4.set_title('分数改进率变化', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11, loc='best')
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # ========================================
    # 5. Manager约束损失（左下）
    # ========================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    if manager_metrics:
        # 对齐损失
        if any(v is not None for v in manager_metrics['alignment_losses']):
            alignment = [v for v in manager_metrics['alignment_losses'] if v is not None]
            episodes_align = [e for e, v in zip(manager_metrics['episodes'], 
                                                manager_metrics['alignment_losses']) if v is not None]
            if alignment:
                ax5.plot(episodes_align, alignment, label='对齐损失 (Alignment)', 
                        color='#06A77D', linewidth=2.5, alpha=0.8)
        
        # 语义损失
        if any(v is not None for v in manager_metrics['semantic_losses']):
            semantic = [v for v in manager_metrics['semantic_losses'] if v is not None]
            episodes_sem = [e for e, v in zip(manager_metrics['episodes'], 
                                              manager_metrics['semantic_losses']) if v is not None]
            if semantic:
                ax5.plot(episodes_sem, semantic, label='语义正交损失 (Semantic)', 
                        color='#D62246', linewidth=2.5, alpha=0.8)
    
    ax5.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax5.set_ylabel('损失值', fontsize=12, fontweight='bold')
    ax5.set_title('Manager约束损失变化 (越低越好)', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=11, loc='best')
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # ========================================
    # 6. 统计对比表格（右下）
    # ========================================
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # 计算统计数据
    stats_data = []
    
    if baseline_metrics and manager_metrics:
        # 分数统计
        baseline_score_mean = np.mean(baseline_metrics['scores'])
        manager_score_mean = np.mean(manager_metrics['scores'])
        score_imp = compute_improvement(baseline_score_mean, manager_score_mean)
        
        baseline_score_final = baseline_metrics['scores'][-1] if baseline_metrics['scores'] else 0
        manager_score_final = manager_metrics['scores'][-1] if manager_metrics['scores'] else 0
        score_final_imp = compute_improvement(baseline_score_final, manager_score_final)
        
        # Alpha熵统计
        baseline_entropy_mean = np.mean(baseline_metrics['alpha_entropies'])
        manager_entropy_mean = np.mean(manager_metrics['alpha_entropies'])
        entropy_change = compute_improvement(baseline_entropy_mean, manager_entropy_mean)
        
        baseline_entropy_final = baseline_metrics['alpha_entropies'][-1] if baseline_metrics['alpha_entropies'] else 0
        manager_entropy_final = manager_metrics['alpha_entropies'][-1] if manager_metrics['alpha_entropies'] else 0
        entropy_final_change = compute_improvement(baseline_entropy_final, manager_entropy_final)
        
        # 奖励统计
        baseline_reward_mean = np.mean(baseline_metrics['rewards'])
        manager_reward_mean = np.mean(manager_metrics['rewards'])
        reward_imp = compute_improvement(baseline_reward_mean, manager_reward_mean)
        
        stats_data = [
            ['指标', '无Manager约束', '有Manager约束', '改进'],
            ['', '', '', ''],
            ['平均分数', f'{baseline_score_mean:.2f}', f'{manager_score_mean:.2f}', f'{score_imp:+.1f}%'],
            ['最终分数', f'{baseline_score_final:.2f}', f'{manager_score_final:.2f}', f'{score_final_imp:+.1f}%'],
            ['', '', '', ''],
            ['平均Alpha熵', f'{baseline_entropy_mean:.4f}', f'{manager_entropy_mean:.4f}', f'{entropy_change:+.1f}%'],
            ['最终Alpha熵', f'{baseline_entropy_final:.4f}', f'{manager_entropy_final:.4f}', f'{entropy_final_change:+.1f}%'],
            ['', '', '', ''],
            ['平均奖励', f'{baseline_reward_mean:.2f}', f'{manager_reward_mean:.2f}', f'{reward_imp:+.1f}%'],
        ]
        
        # Manager约束统计
        if any(v is not None for v in manager_metrics['alignment_losses']):
            alignment = [v for v in manager_metrics['alignment_losses'] if v is not None]
            if alignment:
                stats_data.append(['', '', '', ''])
                stats_data.append(['对齐损失 (初始)', '', f'{alignment[0]:.4f}', ''])
                stats_data.append(['对齐损失 (最终)', '', f'{alignment[-1]:.4f}', f'{compute_improvement(alignment[0], alignment[-1]):+.1f}%'])
    
    # 绘制表格
    if stats_data:
        table = ax6.table(cellText=stats_data, cellLoc='center', loc='center',
                         colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # 设置表头样式
        for i in range(4):
            table[(0, i)].set_facecolor('#4A90E2')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置改进列颜色
        for i in range(2, len(stats_data)):
            if i < len(stats_data) and len(stats_data[i]) > 3:
                cell_text = stats_data[i][3]
                if '+' in cell_text:
                    table[(i, 3)].set_facecolor('#D4EDDA')
                    table[(i, 3)].set_text_props(weight='bold', color='#155724')
                elif '-' in cell_text and '%' in cell_text:
                    table[(i, 3)].set_facecolor('#F8D7DA')
                    table[(i, 3)].set_text_props(weight='bold', color='#721C24')
    
    ax6.set_title('关键指标对比统计', fontsize=14, fontweight='bold', pad=20)
    
    # 保存图片
    plt.savefig(output_dir / 'manager_effect_comprehensive_comparison.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 保存全面对比图: {output_dir / 'manager_effect_comprehensive_comparison.png'}")
    plt.close()

def print_detailed_analysis(baseline_metrics, manager_metrics):
    """打印详细分析报告"""
    print(f"\n{'='*100}")
    print(f"{'Manager约束（内部奖励）效果详细分析':^100}")
    print(f"{'='*100}\n")
    
    if not baseline_metrics or not manager_metrics:
        print("⚠️  缺少对比数据")
        return
    
    # 1. 分数分析
    print(f"📊 分数分析")
    print(f"{'-'*100}")
    
    baseline_scores = baseline_metrics['scores']
    manager_scores = manager_metrics['scores']
    
    baseline_mean = np.mean(baseline_scores)
    manager_mean = np.mean(manager_scores)
    score_imp = compute_improvement(baseline_mean, manager_mean)
    
    baseline_final = baseline_scores[-1]
    manager_final = manager_scores[-1]
    final_imp = compute_improvement(baseline_final, manager_final)
    
    print(f"  平均分数:")
    print(f"    无Manager约束: {baseline_mean:.2f}")
    print(f"    有Manager约束: {manager_mean:.2f}")
    print(f"    改进: {score_imp:+.1f}%")
    print()
    print(f"  最终分数:")
    print(f"    无Manager约束: {baseline_final:.2f}")
    print(f"    有Manager约束: {manager_final:.2f}")
    print(f"    改进: {final_imp:+.1f}%")
    print()
    
    # 2. 专业化分析
    print(f"🎯 专家专业化分析 (Alpha熵)")
    print(f"{'-'*100}")
    
    baseline_entropy = baseline_metrics['alpha_entropies']
    manager_entropy = manager_metrics['alpha_entropies']
    
    baseline_entropy_mean = np.mean(baseline_entropy)
    manager_entropy_mean = np.mean(manager_entropy)
    entropy_change = compute_improvement(baseline_entropy_mean, manager_entropy_mean)
    
    baseline_entropy_final = baseline_entropy[-1]
    manager_entropy_final = manager_entropy[-1]
    entropy_final_change = compute_improvement(baseline_entropy_final, manager_entropy_final)
    
    print(f"  平均Alpha熵:")
    print(f"    无Manager约束: {baseline_entropy_mean:.4f}")
    print(f"    有Manager约束: {manager_entropy_mean:.4f}")
    print(f"    变化: {entropy_change:+.1f}% (负值表示更专业)")
    print()
    print(f"  最终Alpha熵:")
    print(f"    无Manager约束: {baseline_entropy_final:.4f}")
    print(f"    有Manager约束: {manager_entropy_final:.4f}")
    print(f"    变化: {entropy_final_change:+.1f}%")
    print()
    print(f"  专业化程度:")
    baseline_spec = (1.386 - baseline_entropy_final) / 1.386 * 100
    manager_spec = (1.386 - manager_entropy_final) / 1.386 * 100
    print(f"    无Manager约束: {baseline_spec:.1f}%")
    print(f"    有Manager约束: {manager_spec:.1f}%")
    print(f"    提升: {manager_spec - baseline_spec:+.1f}个百分点")
    print()
    
    # 3. Manager约束效果
    if any(v is not None for v in manager_metrics['alignment_losses']):
        print(f"🔗 Manager约束效果")
        print(f"{'-'*100}")
        
        alignment = [v for v in manager_metrics['alignment_losses'] if v is not None]
        if alignment:
            print(f"  对齐损失 (Alignment Loss):")
            print(f"    初始值: {alignment[0]:.4f}")
            print(f"    最终值: {alignment[-1]:.4f}")
            print(f"    下降: {compute_improvement(alignment[0], alignment[-1]):+.1f}%")
            print(f"    解读: 对齐损失下降表示Router越来越听从GAT的建议")
        print()
        
        semantic = [v for v in manager_metrics['semantic_losses'] if v is not None]
        if semantic:
            print(f"  语义正交损失 (Semantic Loss):")
            print(f"    初始值: {semantic[0]:.4f}")
            print(f"    最终值: {semantic[-1]:.4f}")
            print(f"    下降: {compute_improvement(semantic[0], semantic[-1]):+.1f}%")
            print(f"    解读: 语义损失下降表示专家策略越来越不同")
        print()
    
    # 4. 总结
    print(f"📝 总结")
    print(f"{'-'*100}")
    print(f"  ✅ 分数提升: {score_imp:+.1f}%")
    print(f"  ✅ 专业化提升: {manager_spec - baseline_spec:+.1f}个百分点")
    print(f"  ✅ Alpha熵下降: {entropy_final_change:+.1f}%")
    
    if score_imp > 20:
        print(f"\n  🎉 效果显著！Manager约束带来了明显的性能提升！")
    elif score_imp > 10:
        print(f"\n  👍 效果良好！Manager约束有明显帮助！")
    elif score_imp > 0:
        print(f"\n  ✓ 有改进，但可能需要更长时间训练才能看到显著效果")
    else:
        print(f"\n  ⚠️  当前阶段效果不明显，建议继续训练更多episodes")
    
    print(f"\n{'='*100}\n")

def main():
    parser = argparse.ArgumentParser(description='全面对比有/无Manager约束的训练效果')
    parser.add_argument('--baseline', required=True, help='Baseline结果目录（无Manager约束）')
    parser.add_argument('--manager', required=True, help='新版本结果目录（有Manager约束）')
    parser.add_argument('--output', default='ablation_v3/visualizations/manager_comparison/', 
                       help='输出目录')
    
    args = parser.parse_args()
    
    print(f"\n{'='*100}")
    print(f"{'加载训练数据':^100}")
    print(f"{'='*100}\n")
    print(f"  Baseline (无Manager约束): {args.baseline}")
    print(f"  Manager (有Manager约束): {args.manager}")
    
    # 加载数据
    baseline_data = load_training_log(args.baseline)
    manager_data = load_training_log(args.manager)
    
    # 提取指标
    baseline_metrics = extract_metrics(baseline_data)
    manager_metrics = extract_metrics(manager_data)
    
    if baseline_metrics:
        print(f"\n  ✅ Baseline数据: {len(baseline_metrics['episodes'])} episodes")
    if manager_metrics:
        print(f"  ✅ Manager数据: {len(manager_metrics['episodes'])} episodes")
    
    # 打印详细分析
    print_detailed_analysis(baseline_metrics, manager_metrics)
    
    # 绘制全面对比图
    print(f"生成全面对比可视化...")
    plot_comprehensive_comparison(baseline_metrics, manager_metrics, args.output)
    
    print(f"\n✅ 分析完成！")
    print(f"结果保存在: {args.output}")
    print(f"\n查看图片: open {args.output}/manager_effect_comprehensive_comparison.png\n")

if __name__ == '__main__':
    main()
