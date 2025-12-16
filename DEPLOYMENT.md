# TEDG-RL 服务器部署指南

## 快速部署

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo> TEDG-RL-Server
cd TEDG-RL-Server

# 创建虚拟环境
conda create -n tedg-rl python=3.9
conda activate tedg-rl

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
# 嵌入 API 配置
export EMBEDDING_API_KEY="your_embedding_api_key"
export EMBEDDING_BASE_URL="https://api.openai-hk.com/v1"
export EMBEDDING_MODEL_NAME="text-embedding-3-large"

# OpenAI API（可选）
export OPENAI_API_KEY="your_openai_api_key"
```

### 3. 数据准备

确保以下数据文件存在：

```
data/
├── hypergraph/
│   └── hypergraph_complete_real.json
└── cache/
    ├── hypergraph_embedding_index_minsup5.pkl
    └── hypergraph_mixed_embedding_minsup10_w0.30_0.40_0.20_0.10.pkl
```

如果没有缓存文件，首次运行会自动生成。

### 4. 运行训练

```bash
# 基础训练
python train.py

# 自定义配置
python train.py --config config.yaml
```

## 项目结构

```
TEDG-RL-Server/
├── src/
│   └── core/
│       ├── hypergraph_loader.py    # 超图加载器
│       ├── mixed_embedding.py      # 混合信道嵌入
│       └── rl_agent.py            # TEDG-RL Agent
├── data/
│   ├── hypergraph/                # 超图数据
│   └── cache/                     # 嵌入缓存
├── results/
│   ├── checkpoints/               # 模型检查点
│   ├── logs/                      # 训练日志
│   └── visualizations/            # 结果图表
├── config.yaml                    # 配置文件
├── train.py                      # 训练脚本
└── requirements.txt              # 依赖列表
```

## 核心特性

1. **混合信道嵌入**
   - 4通道独立嵌入：pre_nodes, scene_atoms, eff_nodes, operator
   - 可配置权重：默认 [0.3, 0.4, 0.2, 0.1]
   - 自动缓存机制

2. **超图约束的动作选择**
   - 基于当前状态过滤可行动作
   - 支持动作匹配和相似度计算

3. **PPO 强化学习**
   - Actor-Critic 架构
   - 经验回放缓冲区
   - 自动保存最佳模型

## 配置说明

### config.yaml

```yaml
hypergraph:
  file: "data/hypergraph/hypergraph_complete_real.json"
  embedding_cache: "data/cache/hypergraph_embedding_index_minsup5.pkl"
  mixed_embedding_cache: "data/cache/hypergraph_mixed_embedding.pkl"

embedding:
  api_key: "${EMBEDDING_API_KEY}"
  base_url: "https://api.openai-hk.com/v1"
  model: "text-embedding-3-large"
  dim: 1536
  channel_weights: [0.3, 0.4, 0.2, 0.1]

rl:
  learning_rate: 3e-4
  batch_size: 64
  num_episodes: 1000
  max_steps: 1000
  hidden_dim: 128
  gamma: 0.99
  clip_ratio: 0.2
  buffer_size: 10000

nle:
  character: "mon-hum-neu-mal"
  max_steps: 1000
  savedir: "results/nle_episodes"
```

## 常见问题

### Q: 嵌入 API 调用失败
A: 检查环境变量是否正确设置，确保 API key 有效。

### Q: 内存不足
A: 减少 batch_size 或 buffer_size，或使用 CPU 模式。

### Q: 训练速度慢
A: 可以：
- 使用预计算的嵌入缓存
- 减少 num_episodes 进行测试
- 使用 GPU 加速

## 监控和日志

训练过程会自动：
- 每 10 个 episode 打印进度
- 保存最佳模型到 `results/checkpoints/best_model.pth`
- 每 100 个 episode 保存检查点
- 生成训练曲线图

## 扩展开发

### 添加新的嵌入通道

1. 在 `mixed_embedding.py` 中添加新通道的文本提取
2. 更新 `channel_weights` 配置
3. 修改融合逻辑

### 自定义奖励函数

在 `rl_agent.py` 中修改 `store_transition` 方法，加入自定义奖励计算。

### 部署到多 GPU

修改 `rl_agent.py` 中的设备选择：
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

使用 `DataParallel` 进行多卡训练。
