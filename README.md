# TEDG-RL Server Deployment

精简版 TEDG-RL 实现，用于服务器训练。

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 设置环境变量：
```bash
export EMBEDDING_API_KEY="your_embedding_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

3. 运行训练：
```bash
python train.py
```

## 项目结构

```
TEDG-RL-Server/
├── src/core/          # 核心模块
├── data/              # 数据文件
├── results/           # 结果输出
├── config.yaml        # 配置文件
├── train.py          # 训练脚本
└── requirements.txt  # 依赖列表
```

## 核心特性

- 混合信道嵌入（4通道加权融合）
- 超图约束的动作空间
- TEDG-RL 算法实现
- NetHack 环境支持
