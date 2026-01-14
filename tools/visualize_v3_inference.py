#!/usr/bin/env python3
"""
V3推理过程可视化工具

展示完整的推理链路:
1. 输入状态 → atoms激活
2. GAT注意力传播 → Operator激活
3. Intent Vector提取
4. 专家路由 (α权重)
5. 专家输出融合
6. 最终动作选择

生成热图:
- GAT注意力热图 (节点激活)
- 专家选择热图 (α权重)
- 动作分布热图
- 完整推理流程图
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import torch
import torch.nn.functional as F

# 添加项目路径
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.core.networks_v3_gat_moe import GATGuidedMoEPolicy
from src.core.hypergraph_gat_loader import HypergraphGATLoader


class V3InferenceVisualizer:
    """V3推理过程可视化器"""
    
    def __init__(self, policy_net: GATGuidedMoEPolicy):
        self.policy_net = policy_net
        self.loader = policy_net.gat.loader
        
        # 动作名称 (NetHack 23个动作)
        self.action_names = [
            'move_N', 'move_E', 'move_S', 'move_W',
            'move_NE', 'move_SE', 'move_SW', 'move_NW',
            'search', 'kick', 'eat', 'apply', 'read',
            'quaff', 'zap', 'throw', 'fire', 'pray',
            'teleport', 'open', 'close', 'wait', 'look'
        ]
    
    def visualize_single_step(self, state: torch.Tensor, atoms: dict, 
                             save_path: str = None, show: bool = True):
        """
        可视化单步推理过程
        
        Args:
            state: (115,) 状态向量
            atoms: {"pre_nodes": [...], "scene_atoms": [...]}
            save_path: 保存路径
            show: 是否显示
        """
        # 前向传播，获取所有中间结果
        with torch.no_grad():
            logits, alpha, value, aux_info = self.policy_net(
                state.unsqueeze(0),
                atoms=atoms
            )
        
        # 提取数据
        h_vis = aux_info['h_vis'][0]  # (256,)
        h_logic = aux_info['h_logic'][0]  # (256,)
        operator_scores = aux_info['operator_scores'][0] if aux_info['operator_scores'] is not None else None  # (279,)
        expert_logits = aux_info['expert_logits'][0]  # (4, 23)
        alpha = alpha[0]  # (4,)
        logits = logits[0]  # (23,)
        
        # 创建大图
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # ================================================================
        # 第1行: 输入和GAT推理
        # ================================================================
        
        # 1.1 激活的atoms
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_activated_atoms(ax1, atoms)
        
        # 1.2 GAT节点激活热图
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_gat_node_activation(ax2, operator_scores)
        
        # 1.3 Intent Vector
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_intent_vector(ax3, h_logic)
        
        # 1.4 双流特征对比
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_dual_stream(ax4, h_vis, h_logic)
        
        # ================================================================
        # 第2行: 专家系统
        # ================================================================
        
        # 2.1 专家路由 (α权重)
        ax5 = fig.add_subplot(gs[1, 0])
        self._plot_expert_routing(ax5, alpha)
        
        # 2.2 专家输出热图
        ax6 = fig.add_subplot(gs[1, 1:3])
        self._plot_expert_outputs(ax6, expert_logits, alpha)
        
        # 2.3 专家贡献分解
        ax7 = fig.add_subplot(gs[1, 3])
        self._plot_expert_contribution(ax7, expert_logits, alpha)
        
        # ================================================================
        # 第3行: 最终决策
        # ================================================================
        
        # 3.1 融合后的动作分布
        ax8 = fig.add_subplot(gs[2, 0:2])
        self._plot_action_distribution(ax8, logits)
        
        # 3.2 Top-5动作
        ax9 = fig.add_subplot(gs[2, 2])
        self._plot_top_actions(ax9, logits)
        
        # 3.3 推理流程图
        ax10 = fig.add_subplot(gs[2, 3])
        self._plot_inference_flow(ax10, atoms, alpha, logits)
        
        # 总标题
        fig.suptitle('V3 推理过程可视化', fontsize=16, fontweight='bold')
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 可视化已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def _plot_activated_atoms(self, ax, atoms):
        """绘制激活的atoms"""
        ax.axis('off')
        ax.set_title('1. 激活的Atoms', fontweight='bold')
        
        pre_nodes = atoms.get('pre_nodes', [])
        scene_atoms = atoms.get('scene_atoms', [])
        
        text = "Pre-conditions:\n"
        for node in pre_nodes[:5]:  # 最多显示5个
            text += f"  • {node}\n"
        
        text += "\nScene Atoms:\n"
        for atom in scene_atoms[:5]:
            text += f"  • {atom}\n"
        
        ax.text(0.1, 0.5, text, fontsize=10, verticalalignment='center',
                family='monospace')
    
    def _plot_gat_node_activation(self, ax, operator_scores):
        """绘制GAT节点激活热图"""
        if operator_scores is None:
            ax.text(0.5, 0.5, 'No GAT data', ha='center', va='center')
            ax.set_title('2. GAT Operator激活')
            return
        
        # 取Top-20 Operator
        scores = operator_scores.numpy()
        top_indices = np.argsort(scores)[-20:][::-1]
        top_scores = scores[top_indices]
        
        # 获取Operator名称
        operator_nodes = self.loader.get_operator_nodes()
        top_names = [operator_nodes[i]['label'][:20] for i in top_indices]
        
        # 绘制热图
        colors = plt.cm.Reds(top_scores / (top_scores.max() + 1e-8))
        ax.barh(range(len(top_scores)), top_scores, color=colors)
        ax.set_yticks(range(len(top_scores)))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel('Activation Score')
        ax.set_title('2. GAT Operator激活 (Top-20)', fontweight='bold')
        ax.invert_yaxis()
    
    def _plot_intent_vector(self, ax, h_logic):
        """绘制Intent Vector"""
        intent = h_logic.numpy()
        
        # 显示分布
        ax.hist(intent, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(intent.mean(), color='red', linestyle='--', 
                   label=f'Mean: {intent.mean():.3f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('3. Intent Vector分布', fontweight='bold')
        ax.legend()
    
    def _plot_dual_stream(self, ax, h_vis, h_logic):
        """绘制双流特征对比"""
        vis_norm = torch.norm(h_vis).item()
        logic_norm = torch.norm(h_logic).item()
        
        streams = ['Visual\nStream', 'Logic\nStream']
        norms = [vis_norm, logic_norm]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(streams, norms, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('L2 Norm')
        ax.set_title('4. 双流特征强度', fontweight='bold')
        
        # 添加数值标签
        for bar, norm in zip(bars, norms):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{norm:.2f}', ha='center', va='bottom')
    
    def _plot_expert_routing(self, ax, alpha):
        """绘制专家路由"""
        expert_names = self.policy_net.expert_names
        alpha_np = alpha.numpy()
        
        # 饼图
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']
        explode = [0.1 if a == alpha_np.max() else 0 for a in alpha_np]
        
        wedges, texts, autotexts = ax.pie(
            alpha_np, labels=expert_names, autopct='%1.1f%%',
            colors=colors, explode=explode, startangle=90
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('5. 专家路由 (α权重)', fontweight='bold')
    
    def _plot_expert_outputs(self, ax, expert_logits, alpha):
        """绘制专家输出热图"""
        expert_names = self.policy_net.expert_names
        
        # 转换为概率
        expert_probs = F.softmax(expert_logits, dim=-1).numpy()
        
        # 绘制热图
        im = ax.imshow(expert_probs, aspect='auto', cmap='YlOrRd')
        
        ax.set_yticks(range(len(expert_names)))
        ax.set_yticklabels(expert_names)
        ax.set_xticks(range(0, 23, 2))
        ax.set_xticklabels([self.action_names[i][:6] for i in range(0, 23, 2)], 
                          rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Actions')
        ax.set_title('6. 专家输出热图 (概率)', fontweight='bold')
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 标记每个专家的α权重
        for i, (name, a) in enumerate(zip(expert_names, alpha)):
            ax.text(-1.5, i, f'α={a:.2f}', va='center', fontweight='bold')
    
    def _plot_expert_contribution(self, ax, expert_logits, alpha):
        """绘制专家贡献分解"""
        expert_names = self.policy_net.expert_names
        
        # 计算每个专家对最终决策的贡献
        expert_probs = F.softmax(expert_logits, dim=-1)
        weighted_probs = expert_probs * alpha.unsqueeze(1)
        contributions = weighted_probs.sum(dim=1).numpy()
        
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181']
        bars = ax.barh(expert_names, contributions, color=colors, alpha=0.7, 
                      edgecolor='black')
        ax.set_xlabel('Contribution')
        ax.set_title('7. 专家贡献度', fontweight='bold')
        
        # 添加数值标签
        for bar, contrib in zip(bars, contributions):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{contrib:.3f}', ha='left', va='center', fontsize=9)
    
    def _plot_action_distribution(self, ax, logits):
        """绘制动作分布"""
        probs = F.softmax(logits, dim=-1).numpy()
        
        colors = plt.cm.viridis(probs / probs.max())
        bars = ax.bar(range(len(probs)), probs, color=colors, alpha=0.7, 
                     edgecolor='black')
        
        ax.set_xticks(range(len(self.action_names)))
        ax.set_xticklabels(self.action_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Probability')
        ax.set_title('8. 最终动作分布', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_top_actions(self, ax, logits):
        """绘制Top-5动作"""
        probs = F.softmax(logits, dim=-1).numpy()
        top_indices = np.argsort(probs)[-5:][::-1]
        top_probs = probs[top_indices]
        top_names = [self.action_names[i] for i in top_indices]
        
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, 5))
        bars = ax.barh(range(5), top_probs, color=colors, alpha=0.7, 
                      edgecolor='black')
        ax.set_yticks(range(5))
        ax.set_yticklabels([f'{i+1}. {name}' for i, name in enumerate(top_names)])
        ax.set_xlabel('Probability')
        ax.set_title('9. Top-5 动作', fontweight='bold')
        ax.invert_yaxis()
        
        # 添加概率标签
        for bar, prob in zip(bars, top_probs):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{prob:.1%}', ha='left', va='center', fontsize=9)
    
    def _plot_inference_flow(self, ax, atoms, alpha, logits):
        """绘制推理流程图"""
        ax.axis('off')
        ax.set_title('10. 推理流程', fontweight='bold')
        
        # 选择的动作
        action_idx = torch.argmax(logits).item()
        action_name = self.action_names[action_idx]
        action_prob = F.softmax(logits, dim=-1)[action_idx].item()
        
        # 主导专家
        dominant_expert_idx = torch.argmax(alpha).item()
        dominant_expert = self.policy_net.expert_names[dominant_expert_idx]
        dominant_alpha = alpha[dominant_expert_idx].item()
        
        # 流程文本
        flow_text = f"""
推理链路:

1. 输入状态
   ↓
2. Atoms激活
   • {len(atoms.get('pre_nodes', []))} pre-conditions
   • {len(atoms.get('scene_atoms', []))} scene atoms
   ↓
3. GAT推理
   • 节点消息传递
   • Intent Vector提取
   ↓
4. 专家路由
   • 主导专家: {dominant_expert}
   • α权重: {dominant_alpha:.2%}
   ↓
5. 动作决策
   • 选择: {action_name}
   • 概率: {action_prob:.2%}
        """
        
        ax.text(0.1, 0.5, flow_text, fontsize=9, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))


def main():
    """主函数 - 演示可视化"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V3推理过程可视化")
    parser.add_argument("--checkpoint", type=str, 
                       default="ablation_v3/results/v3_convergence_cpu/checkpoints/best_model.pth",
                       help="模型checkpoint路径")
    parser.add_argument("--output", type=str, 
                       default="ablation_v3/visualizations/v3_inference.png",
                       help="输出图片路径")
    args = parser.parse_args()
    
    print("="*60)
    print("V3 推理过程可视化")
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
    
    # 加载checkpoint (如果存在)
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        policy_net.load_state_dict(checkpoint['policy_net'])
        print(f"✓ 已加载checkpoint: {args.checkpoint}")
    else:
        print(f"⚠️ Checkpoint不存在，使用随机初始化")
    
    policy_net.eval()
    
    # 创建可视化器
    print("\n[2] 创建可视化器...")
    visualizer = V3InferenceVisualizer(policy_net)
    
    # 创建测试输入
    print("\n[3] 生成测试输入...")
    state = torch.randn(115)  # 随机状态
    atoms = {
        "pre_nodes": ["player_alive", "hp_low", "has_gold"],
        "scene_atoms": ["dlvl_3", "monsters_present"]
    }
    
    print(f"  状态维度: {state.shape}")
    print(f"  Atoms: {atoms}")
    
    # 可视化
    print("\n[4] 生成可视化...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    visualizer.visualize_single_step(
        state, atoms,
        save_path=args.output,
        show=False
    )
    
    print(f"\n✓ 可视化完成!")
    print(f"  输出: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()

