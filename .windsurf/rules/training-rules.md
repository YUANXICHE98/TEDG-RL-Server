---
trigger: manual
---


1. 凡是修改了网络结构（尤其是 H-RAM/Attention），先跑一个独立的 Python 测试脚本。
查设备：确保 Tensor 都在同一设备（CPU/GPU）。
查维度：输入输出 Shape 必须对齐。
查爆炸：模拟跑 100 次前向传播，打印 Logits 的 Min/Max。如果出现 `NaN`、`Inf` 或数值超过 [-10, 10] 区间，视为测试失败，禁止开始训练。


2. 网络设计的“数值防御”规范：
写代码时必须默认植入防炸机制，Attention：必须加 Scaling（除以 \sqrt{d_k}），防止 Softmax 饱和。中间层：Actor/Critic 深层网络必须加 `LayerNorm`。输出层必须对 Logits 用 `torch.clamp` 截断，或使用 `gain=0.01` 的正交初始化，防止初始策略过于自信导致梯度爆炸。

3. 启动前的“预飞行”自检 (Pre-flight Check)：在正式训练循环开始前，脚本必须自动执行一轮检查：
数据检查：超图文件路径、缓存文件是否存在。
语义检查*：关键索引（如 blstats 的 HP/Score 位置）是否与 NLE 官方常量一致。
资源检查：显存是否足够。
原则：报错要趁早，严禁训练跑了 1 小时后才因为基础配置错误而崩溃。

