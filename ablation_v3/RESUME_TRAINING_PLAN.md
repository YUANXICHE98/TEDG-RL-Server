# Resume训练计划 - 从100到500 Episodes

## 背景

### 旧版本训练（1月7-9日）
- **不包含新机制**：训练时还没有实现Manager约束等4个机制
- 训练结果：
  - Warmup (1000ep): 分数7.53, Alpha熵1.3853
  - Transition (3000ep): 分数10.35, Alpha熵0.6929
  - Fine-tune (5000ep): 分数15.46, Alpha熵0.6928

### 新版本快速测试（1月11-12日）
- **包含所有4个新机制**：
  1. Manager内层约束（超图-路由对齐）
  2. 熵最小化（Fine-tune阶段）
  3. 时间一致性
  4. 专家重叠惩罚
- 100 episodes快速对比测试结果：
  - Baseline: 平均分数9.08, Alpha熵1.3834
  - With Manager: 平均分数11.12, Alpha熵1.3832
  - **改进: +22.5%分数提升**

## 当前计划

### 目标
从100 episodes checkpoint继续训练到500 episodes，验证新机制的中期效果

### 为什么选择500 episodes？
1. **快速验证**：2-3小时即可完成（vs 5000 episodes需要20-30小时）
2. **仍在Warmup阶段**：可以观察新机制在Warmup阶段的完整效果
3. **对比基准**：可以和旧版本的Warmup 1000结果对比

### 训练配置
- **起点**: Episode 100 (checkpoint: `quick_manager_20260111_230845/checkpoints/model_00100.pth`)
- **终点**: Episode 500
- **实际训练**: 400 episodes
- **预计时间**: 2-3小时
- **阶段**: Warmup (0-1000)
- **路由方式**: Softmax
- **学习率**: 1e-4

### Warmup阶段配置（Episode 0-1000）
```python
{
    'phase': 'warmup',
    'use_sparsemax': False,  # Softmax路由
    'learning_rate': 1e-4,
    'entropy_coef': 0.05,
    'alpha_entropy_coef': 0.1,
    'alpha_entropy_sign': -1,   # 最大化熵（防塌缩）
    'load_balance_coef': 0.02,
    'diversity_coef': 0.01,
    # Manager内层约束
    'alignment_coef': 0.1,      # 超图-路由对齐
    'alignment_temperature': 1.0,
    'semantic_coef': 0.05,      # 语义正交
    # 高级机制（Warmup不使用）
    'temporal_coef': 0.0,
    'overlap_coef': 0.0,
}
```

## 启动命令

```bash
# 方式1: 使用脚本（推荐）
bash ablation_v3/scripts/resume_100_to_500.sh

# 方式2: 直接命令
conda activate tedg-rl-demo
python -u ablation_v3/train/train_v3_gat_moe.py \
    --exp-name resume_500_from_100 \
    --episodes 500 \
    --max-steps 2000 \
    --resume ablation_v3/results/quick_manager_20260111_230845/checkpoints/model_00100.pth
```

## 预期观察指标

### 1. 分数提升
- **100 episodes**: 11.12
- **500 episodes**: 预期 12-14（持续提升）

### 2. Alpha熵变化
- **100 episodes**: 1.3832（接近初始值）
- **500 episodes**: 预期 1.2-1.3（Warmup阶段应保持较高熵）
- **注意**: Warmup阶段`alpha_entropy_sign=-1`，目标是**最大化熵**，防止专家塌缩

### 3. Manager约束效果
- **超图-路由对齐损失**: 应该逐渐下降（Router学会听从GAT建议）
- **专家使用率**: 应该相对均衡（4个专家都被使用）

### 4. 训练稳定性
- **梯度范数**: 应该稳定在<5.0
- **NaN检测**: 不应该出现NaN/Inf
- **GAT注意力方差**: 应该>0.1（避免过平滑）

## 对比分析

训练完成后，可以对比：

### 新版本 vs 旧版本（Warmup阶段）
| 指标 | 旧版本@1000ep | 新版本@500ep | 改进 |
|------|--------------|-------------|------|
| 平均分数 | 7.53 | ? | ? |
| Alpha熵 | 1.3853 | ? | ? |
| 专家使用率方差 | ? | ? | ? |

### 新版本内部对比
| 指标 | @100ep | @500ep | 变化 |
|------|--------|--------|------|
| 平均分数 | 11.12 | ? | ? |
| Alpha熵 | 1.3832 | ? | ? |
| 对齐损失 | ? | ? | ? |

## 后续计划

### 如果500 episodes效果好
继续训练到1000 episodes（完成Warmup阶段），然后进入Transition阶段

### 如果效果不理想
分析问题：
1. 检查Manager约束是否生效（对齐损失是否下降）
2. 检查专家是否塌缩（Alpha熵是否过低）
3. 检查GAT是否过平滑（注意力方差是否过低）

## 文件位置

- **启动脚本**: `ablation_v3/scripts/resume_100_to_500.sh`
- **Checkpoint**: `ablation_v3/results/quick_manager_20260111_230845/checkpoints/model_00100.pth`
- **输出目录**: `ablation_v3/results/resume_500_from_100/`
- **训练日志**: `ablation_v3/results/resume_500_from_100/training.log`
- **训练数据**: `ablation_v3/results/resume_500_from_100/logs/training_log.json`

## 监控命令

```bash
# 实时查看训练日志
tail -f ablation_v3/results/resume_500_from_100/training.log

# 查