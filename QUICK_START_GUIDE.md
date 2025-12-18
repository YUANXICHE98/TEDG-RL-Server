# TEDG-RL 快速启动指南

## 当前状态总结

### ✅ 已修复
- **blstats 索引错误**：所有训练脚本已使用 `nle.nethack.NLE_BL_*` 常量
- **三个脚本**：train_confmatch.py, train_verbose.py, train_nethack.py

### ⚠️ 旧训练数据
- 如果之前的训练使用了错误索引，**不能作为有效实验结果**
- 建议重新训练

---

## 推荐使用脚本

**主实验脚本：`train_confmatch.py`**

原因：
- ✅ 真实超图匹配逻辑（4通道覆盖度）
- ✅ 置信度基于匹配分数（不是HP伪置信度）
- ✅ 已修复 blstats 索引
- ✅ 输出到独立目录 `results_confmatch/`

---

## 快速测试（验证修复）

### 1. 语法检查
```bash
cd /root/autodl-tmp/TEDG-RL-Server
python -m py_compile train_confmatch.py
```

### 2. 最小运行测试（1个episode）
```bash
# 设置少量episode快速验证
TEDG_NUM_EPISODES=1 TEDG_MAX_STEPS=100 TEDG_VERBOSE_INTERVAL=1 \
python train_confmatch.py
```

**预期输出：**
```
[观测解析]
  HP: 14/14 (100%)        # ✅ 正确：开局满血
  深度: 1层                # ✅ 正确：从第1层开始
  金币: 0                  # ✅ 正确：开局无金币
  分数: 0                  # ✅ 正确：开局0分
```

**如果看到异常值（如 HP: 0/14, 深度: 14），说明修复未生效**

---

## 正式训练

### 基础训练（完整10000 episodes）
```bash
python train_confmatch.py
```

### 自定义参数训练
```bash
# 短期测试（500 episodes）
TEDG_NUM_EPISODES=500 TEDG_MAX_STEPS=500 \
TEDG_OUTPUT_DIR=results_test \
python train_confmatch.py

# 长期训练（20000 episodes）
TEDG_NUM_EPISODES=20000 \
TEDG_OUTPUT_DIR=results_long \
python train_confmatch.py
```

---

## 并行消融实验（待实现）

### 实验设计
| 实验名 | 时间衰减 | 置信度分支 | Query动作 | LLM | 输出目录 |
|--------|---------|-----------|----------|-----|---------|
| full | ✅ | ✅ | ✅ | ✅ | results_full |
| no_decay | ❌ | ✅ | ✅ | ✅ | results_no_decay |
| no_branch | ✅ | ❌ | ✅ | ✅ | results_no_branch |
| no_query | ✅ | ✅ | ❌ | ✅ | results_no_query |
| no_llm | ✅ | ✅ | ✅ | ❌ | results_no_llm |
| baseline | ❌ | ❌ | ❌ | ❌ | results_baseline |

### 启动脚本（需先实现特性开关）
```bash
# 完整版
TEDG_USE_TIME_DECAY=1 TEDG_USE_CONF_BRANCH=1 TEDG_USE_QUERY=1 TEDG_USE_LLM=1 \
TEDG_OUTPUT_DIR=results_full python train_confmatch.py &

# 无时间衰减
TEDG_USE_TIME_DECAY=0 TEDG_USE_CONF_BRANCH=1 TEDG_USE_QUERY=1 TEDG_USE_LLM=1 \
TEDG_OUTPUT_DIR=results_no_decay python train_confmatch.py &

# 基线（仅多通道匹配）
TEDG_USE_TIME_DECAY=0 TEDG_USE_CONF_BRANCH=0 TEDG_USE_QUERY=0 TEDG_USE_LLM=0 \
TEDG_OUTPUT_DIR=results_baseline python train_confmatch.py &
```

---

## 监控训练

### 查看实时日志
```bash
tail -f results_confmatch/logs/training_log.json
```

### 检查检查点
```bash
ls -lh results_confmatch/checkpoints/
```

### 查看α权重分布
训练完成后，检查日志中的α权重统计：
```
α权重分布:
  α_pre:    0.XXX ± 0.XXX
  α_scene:  0.XXX ± 0.XXX
  α_effect: 0.XXX ± 0.XXX
  α_rule:   0.XXX ± 0.XXX
```

---

## 故障排查

### 问题1：ImportError: No module named 'nle'
```bash
pip install nle
```

### 问题2：CUDA out of memory
减小 batch_size：
```bash
# 在 train_confmatch.py 中修改 batch_size=64 或 32
```

### 问题3：训练速度慢
- 确认使用 GPU（输出应显示 "✓ CUDA GPU" 或 "✓ MUSA GPU"）
- 减小 max_steps 或 num_episodes

### 问题4：置信度始终为0
- 检查超图文件是否存在：`data/hypergraph/hypergraph_complete_real.json`
- 检查超图是否为空

---

## 下一步工作

### 立即可做
1. ✅ 运行最小测试验证修复
2. ✅ 启动基础训练（train_confmatch.py）

### 需要实现
1. ⏳ 时间衰减真正启用（维护 t_i 历史）
2. ⏳ 置信度阈值分支逻辑
3. ⏳ Query 动作集定义
4. ⏳ LLM 调用封装
5. ⏳ 特性开关（环境变量控制）

### 实验验证
1. ⏳ 短期训练（100 episodes）验证收敛趋势
2. ⏳ 对比实验：train_confmatch.py vs train_verbose.py
3. ⏳ 消融实验：逐个关闭特性验证贡献

---

## 联系与支持

如有问题，检查：
1. `TRAIN_SCRIPTS_COMPARISON.md` - 脚本详细对比
2. `train_confmatch.py` 顶部注释 - 实现细节
3. `src/core/` - 核心模块代码
