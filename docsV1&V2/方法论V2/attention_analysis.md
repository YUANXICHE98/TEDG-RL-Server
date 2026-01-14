## MA vs eval 是什么区别？

- **MA（Moving Average）**：对逐 episode（或逐 step）得到的原始序列做滑动平均平滑（如 MA100/MA200）。作用是压噪声、看趋势；它不是额外的一条曲线，而是“同一条序列的平滑版”。
- **eval（评估点/评估块）**：训练过程中按固定频率（比如每 50 episode）跑一次评估/统计输出的平均分（avgS/avgR）。它更像“离散采样点”，通常更少、更稳定，但时间分辨率低。

经验上：
- 想看“整体趋势”→ 看 **MA**
- 想看“对比/检查点”→ 看 **eval**

---

## 我想看 Attention（模型是否学到：不同场景选不同动作）

你现在有 **三类** “attention/权重”可以看（注意：不同 ablation 看到的 attention 不是同一种东西）：

1) **4 通道融合权重 α（V1/V2 MultiChannelPolicyNet）**
- 位置：`src/core/networks_correct.py` 的 `AttentionWeightNet` / `MultiChannelPolicyNet.forward()`
- 形状：`alpha = [α_pre, α_scene, α_effect, α_rule]`，和为 1
- 含义：在当前 state 下，模型更“听”哪个知识通道
  - V1：软融合（softmax）
  - V2：可能是 Gumbel/稀疏路由（hard/soft 取决于实现与设置）

2) **HRAM V3.2 的知识注意力 attn_weights（CrossAttentionFusion）**
- 位置：`src/core/networks_hram.py` 的 `CrossAttentionFusion`
- 形状：`attn_weights: (1, K)`（对 Top-K 条知识的注意力分布）
- 含义：在当前 state 下，模型更“用”哪条检索到的知识

3) **HRAM-Doc 的 4 通道路由 α（HRAMPolicyNetDoc）**
- 位置：`src/core/networks_hram.py` 的 `HRAMPolicyNetDoc.router`
- 形状：同样是 `alpha = (4,)`，但它基于“检索上下文+切片 state”的专家网络（不是 V1/V2 的手工 q_pre/q_scene）
- 含义：在当前 state 下，四个专家谁更该主导决策（与 V1/V2 的 α 不完全可比）

---

## 快速生成热力图（不影响正在训练的进程）

下面的脚本会：
- **加载某个实验的 checkpoint**（默认 `best_model.pth`）
- 用 NetHack 环境跑少量 eval episode（CPU，低优先级）
- 统计“场景 → α(4通道)” + “场景 → 动作分布/采样频率”
- 输出热力图（含转置版：横纵轴切换）

脚本：`tools/analyze_attention_heatmaps.py`

示例（baseline）：
```bash
python tools/analyze_attention_heatmaps.py --exp baseline --episodes 5 --max-steps 500 --device cpu --top-scenarios 20
```

示例（HRAM e2e / HRAM doc）：
```bash
python tools/analyze_attention_heatmaps.py --exp hram_e2e --episodes 5 --max-steps 500 --device cpu --top-scenarios 20
python tools/analyze_attention_heatmaps.py --exp hram_doc_fixed --episodes 5 --max-steps 500 --device cpu --top-scenarios 20
```

输出目录：
- `docs/visualizations/attention/<exp>/alpha_heatmap.png`（场景→4通道权重）
- `docs/visualizations/attention/<exp>/action_freq_heatmap.png`（场景→采样动作频率）
- `docs/visualizations/attention/<exp>/action_prob_heatmap.png`（场景→平均动作概率）
- `docs/visualizations/attention/<exp>/action_*_heatmap_T.png`（转置版：动作→场景）
- `docs/visualizations/attention/<exp>/summary.json`（机器可读汇总）

备注：
- `scenario_mode=simple`：用 `dlvl bucket + low_hp + gold` 做场景键（更聚合，热力图更清晰）
- `scenario_mode=atoms`：把部分 `pre_nodes/scene_atoms` 拼到场景键里（更细粒度，但更稀疏）

### 为什么有的图“看起来没说什么”？

常见原因（你现在看到的“几乎一整块颜色”就是这类情况）：
- **场景太粗**：`scenario_mode=simple` 只分 `dlvl/low_hp/gold`，很多 step 被归到同一格里；想看差异用 `scenario_mode=atoms` 或增加场景特征。
- **样本太少**：只跑 1-2 个 episode，很多格子统计不稳定；至少 `episodes=10`、`max-steps=1000` 才像样。
- **策略还没学会分化**：early stage 往往是 α 接近均匀，动作分布也比较“随机+偏向某几个安全动作”。
- **hard routing 的影响**：若用 Gumbel hard（每步几乎 one-hot），平均后可能仍呈现某个专家长期占优；可以用 `--hard-routing`/关闭来对比。
