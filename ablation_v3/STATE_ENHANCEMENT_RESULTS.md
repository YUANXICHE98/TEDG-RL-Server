# V3 状态增强与初始化修正结果

## 修改时间
2026-01-06

## 问题诊断

### 之前的问题
1. **Alpha变化太小** - 每步只变化0.001，几乎固定在初始值
2. **Scene atoms不变** - Agent没有移动，一直在同一位置
3. **动作概率接近均匀** - Logits太小（~0.02），无法产生有效梯度

### 根本原因
1. **Expert初始化增益0.01太小** - 导致初始logits接近0，经过Softmax后动作概率极其平坦
2. **Sparsemax过早稀疏化** - Warmup阶段应该用Softmax，但被覆盖了
3. **Router输入特征可能被淹没** - belief向量后30维全是0，导致特征稀疏
4. **过度的clamp限制** - logits被限制在[-20, 20]，如果本来就很小，clamp没意义

## 修改内容

### 1. Expert初始化增益修正 ✅
**文件**: `src/core/networks_v3_gat_moe.py`

```python
# 修改前
nn.init.orthogonal_(self.network[-1].weight, gain=0.01)

# 修改后
nn.init.orthogonal_(self.network[-1].weight, gain=0.5)
```

**原因**: 0.01太小导致初始梯度极小，专家学不动。改为0.5让专家有更强的信号传递。

### 2. Router初始化增强 ✅
**文件**: `src/core/networks_v3_gat_moe.py`

```python
# 新增
nn.init.orthogonal_(self.router[-1].weight, gain=0.1)
```

**原因**: Router最后一层初始化稍微大一点，让Router大胆选择专家。

### 3. 移除过度clamp限制 ✅
**文件**: `src/core/networks_v3_gat_moe.py`

```python
# 修改前
logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

# 修改后
logits = torch.nan_to_num(logits, nan=0.0)
```

**原因**: 之前的clamp(-20, 20)限制太死，如果logits本来就很小，clamp没意义。只保留数值稳定性处理。

### 4. 添加调试输出 ✅
**文件**: `ablation_v3/train/train_v3_gat_moe.py`

```python
# 新增
if episode % 10 == 0:
    print(f"DEBUG: Episode {episode}, Routing: {'Sparsemax' if policy_net.use_sparsemax else 'Softmax'}, Phase: {config['phase']}")
```

**原因**: 确保Warmup阶段真的在使用Softmax，防止配置没生效。

## 预期效果

### Warmup阶段 (0-1000 episodes, Softmax)
- **Alpha分布**: 比较均匀（例如 [0.25, 0.25, 0.25, 0.25]）
- **Alpha变化**: 随着训练进行，某个专家的权重会缓慢上升
- **Reward**: 应该能更快看到Reward的增长
- **动作logits**: 不再是微弱的噪音（~0.02），而是有力的动作尝试（>0.1）

### Transition阶段 (1000-3000 episodes, Sparsemax)
- **Alpha分布**: 开始稀疏化，Top-2专家权重上升
- **专家分工**: 不同场景下使用不同专家
- **Reward**: 持续增长

### Fine-tune阶段 (3000+ episodes, Sparsemax)
- **Alpha分布**: 高度稀疏，主导专家权重>0.5
- **专家分工**: 明确的语义分工
- **Reward**: 稳定在较高水平

## 下一步训练计划

### 训练配置
```bash
cd ablation_v3/scripts
bash test_convergence_cpu.sh
```

### 训练参数
- **Episodes**: 50-100（先快速验证）
- **Max steps**: 500
- **Device**: CPU（快速测试）

### 监控指标
1. **Alpha变化** - 每10 episodes打印一次，观察是否有明显变化
2. **Reward曲线** - 是否比之前更快增长
3. **动作logits** - 是否从~0.02增长到>0.1
4. **专家使用率** - Warmup阶段是否均匀，Transition阶段是否开始分化

## 理论依据

### 为什么gain=0.5合适？
- **Actor网络通常用0.01** - 为了初始动作平滑，避免过于激进
- **MoE架构需要更强信号** - 专家内部层需要更强的信号传递，否则Router学不到"哪个专家更好"
- **0.5是折中值** - 既不会太大导致梯度爆炸，也不会太小导致梯度消失

### 为什么Warmup阶段用Softmax？
- **所有专家都是小白** - 训练初期，所有专家都不知道怎么做
- **Sparsemax强制稀疏** - 把权重分配给Top-2或Top-1，其他专家拿不到梯度（梯度为0）
- **Softmax雨露均沾** - 让所有专家都能分到一点梯度，避免"饿死"

### 为什么移除clamp？
- **Clamp限制梯度流** - 如果logits在[-20, 20]内，clamp不起作用；如果超出，clamp会截断梯度
- **初始logits很小** - 由于gain=0.01，初始logits接近0，clamp完全没用
- **只保留NaN处理** - 数值稳定性处理足够，不需要过度限制

## 参考文献
- Switch Transformer (Google, 2021) - MoE初始化策略
- Sparse Mixture of Experts (Shazeer et al., 2017) - 负载均衡和路由策略
- NetHack Challenge (NeurIPS 2021) - 强化学习训练技巧
