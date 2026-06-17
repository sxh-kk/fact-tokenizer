# FACT Tokenizer v0.1 训练流程与实验总结

日期：2026-06-16
项目目录：`/data_all/sxh/FACT_tokenizer`
本次主训练 run：

```text
/data_all/sxh/FACT_tokenizer/outputs/fact_tokenizer/egoexo_diverse_500takes_k64_20k_ddp4_resumable_20260615_181053
```

## 1. 本次实验目标

本次实验的目标不是直接证明 token 已经具有很强的语义纯度，而是先验证 FACT tokenizer 第一阶段 MVP 的核心机制是否跑通：

1. 使用 EgoExo4D 的 ego/exo paired transition 数据训练 tokenizer。
2. 将原始 latent action 拆成 shared action branch 和 private residual branch。
3. ego/exo 两个视角共用同一个 shared action codebook。
4. 训练时 private residual 参与重建，导出时只导出 ego 分支的 shared action token。
5. 验证 self reconstruction 和 swapped reconstruction 是否都能收敛。
6. 导出 `q_act_ego`，检查 token shape、code usage、confidence 和 private residual 是否被正确排除。

最终希望得到的第一版产物是：

```text
ego-accessible shared action token: q_act_ego
```

这个 token 后续可以作为 WAM 或下游 action-conditioned world model 的离散动作标签。

## 2. 数据流程

### 2.1 数据来源

本次训练使用 EgoExo4D 的 paired ego/exo 视频数据。为了先验证机制，没有直接使用全量数据，而是抽取了一个 diverse 规模的训练 shard。

训练 NPZ：

```text
/data_all/sxh/FACT_tokenizer/data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz
```

数据内容：

```text
ego: (7999, 2, 224, 224, 3)
exo: (7999, 2, 224, 224, 3)
```

其中：

- `7999` 表示 sampled transitions 数量。
- `2` 表示每个样本取两帧，即 `current -> future`。
- `224 x 224` 是 resize 后的图像大小。
- `3` 是 RGB channel。

### 2.2 Batch schema

训练时 dataset 会统一输出：

```text
batch["ego"]["videos"]: (B, 2, C, H, W)
batch["exo"]["videos"]: (B, 2, C, H, W)
```

本阶段只训练两帧 transition：

```text
current frame -> future frame
```

虽然接口保留了 `T >= 2` 的扩展空间，但当前 v0.1 的主要目标是把 token 机制跑通，所以训练时先使用两帧 transition。

## 3. 模型思路

### 3.1 总体结构

本次实现的核心模型是：

```text
FACTTokenizer
```

它包含以下主要模块：

1. frozen DINOv2 backbone
2. ego view encoder
3. exo view encoder
4. shared VectorQuantizer codebook
5. private residual branch
6. DINO feature decoder

设计思想是：

```text
视频 transition
  -> DINO patch features
  -> view-specific encoder
  -> shared action latent z_act
  -> shared codebook quantization
  -> q_act
```

同时每个 view 还有一个 private residual：

```text
r_priv_ego
r_priv_exo
```

private residual 只用于训练期辅助重建，不导出给下游模型。

### 3.2 Shared action branch

ego 和 exo 分别编码出：

```text
z_act_ego
z_act_exo
```

二者进入同一个 shared action codebook：

```text
q_act_ego = VQ(z_act_ego)
q_act_exo = VQ(z_act_exo)
```

这一步是 FACT tokenizer 的关键：不同视角观察到同一个动作时，模型被鼓励把它们映射到共享的离散 action space。

本次训练的 codebook 配置：

```text
K = 64 codes
num_action_slots = 4
latent_dim = 32
```

因此每个 transition 最终导出的 hard token shape 是：

```text
(transitions, num_action_slots)
```

本次导出结果：

```text
indices: (7999, 1, 4)
```

### 3.3 Private residual branch

每个 view encoder 还会输出：

```text
r_priv_ego
r_priv_exo
```

配置：

```text
num_private_slots = 1
private_dim = 8
```

它的作用是吸收视角相关、外观相关、背景相关的信息，比如：

- ego/exo 视角差异
- 相机位置差异
- 局部遮挡
- 外观细节

但 private residual 不应该成为下游 action label，因此导出 token 时不会保存它。

本次验证结果中：

```text
private_residual_exported: false
```

符合设计目标。

## 4. 训练路径

本次训练有 4 条 reconstruction path：

```text
ego self:
ego current + q_act_ego + r_priv_ego -> ego future

exo self:
exo current + q_act_exo + r_priv_exo -> exo future

ego swap:
ego current + q_act_exo + r_priv_ego -> ego future

exo swap:
exo current + q_act_ego + r_priv_exo -> exo future
```

其中 self path 让模型先学会正常重建当前视角的 future feature。

swap path 是更关键的约束：它要求模型使用另一个视角的 shared action token，也能重建本视角的 future feature。

如果 swap reconstruction 可以做到接近 self reconstruction，说明 shared token 至少在机制上具有跨视角可替换性。

## 5. Loss 设计

本次训练目标是 DINO feature reconstruction，不做 RGB pixel reconstruction。

主要 loss 包括：

### 5.1 Self reconstruction loss

```text
L_self = ego_self_mse + exo_self_mse
```

约束同视角 token 可以重建 future DINO feature。

### 5.2 Swap reconstruction loss

```text
L_swap = ego_swap_mse + exo_swap_mse
```

约束跨视角 shared token 可以互换。

这是 FACT tokenizer 中最重要的结构性约束。

### 5.3 VQ loss

```text
L_vq = codebook_loss + beta * commitment_loss
```

本次使用：

```text
vq_beta = 0.25
```

VQ loss 保证 continuous latent 和 codebook embedding 能稳定对齐。

### 5.4 Confidence-gated KL

```text
KL(stopgrad(p_exo) || p_ego) * confidence_exo
```

它鼓励 ego/exo 在 soft assignment distribution 上接近。

注意：本次日志里最终实际 KL 权重是：

```text
weight_kl = 0.01
```

原因是 schedule 的 `0.1` 又乘上了 config 里的 `kl_weight=0.1`。这不是训练错误，但说明这次跨视角概率对齐约束比较弱。

### 5.5 Code usage balance loss

防止 codebook collapse，让 batch-level assignment 分布更接近均匀分布。

本次配置：

```text
balance_weight = 0.05
```

### 5.6 Private residual regularization

对 private residual 做 L2 正则，避免它无限制吸收所有信息。

本次配置：

```text
private_reg_weight = 0.01
```

## 6. Loss schedule

训练 schedule 分成两个阶段。

前 20% steps：

```text
self = 1.0
swap: 0 -> 0.5
kl: 10% steps 后逐渐开启
```

20% steps 之后：

```text
self = 0.2
swap = 1.0
kl schedule weight = 0.1
```

因为最终 KL 还要乘 `kl_weight=0.1`，所以日志中的最终实际 KL 系数是：

```text
0.1 * 0.1 = 0.01
```

这个 schedule 的直觉是：

1. 先让模型学会基本 self reconstruction。
2. 再逐渐提高 swap reconstruction 的重要性。
3. 后期主要依赖 swap 来逼迫 shared action token 对齐跨视角动作信息。

## 7. 训练配置

本次 checkpoint 中记录的主要模型配置：

```json
{
  "image_channels": 3,
  "model_dim": 128,
  "dino_dim": 768,
  "latent_dim": 32,
  "private_dim": 8,
  "num_latents": 64,
  "num_action_slots": 4,
  "num_private_slots": 1,
  "patch_size": 14,
  "enc_blocks": 1,
  "dec_blocks": 1,
  "num_heads": 4,
  "dropout": 0.0,
  "vq_temperature": 0.1,
  "backbone": "dino",
  "view_names": ["ego", "exo"]
}
```

训练步数：

```text
20000 steps
```

训练完成 checkpoint：

```text
/data_all/sxh/FACT_tokenizer/outputs/fact_tokenizer/egoexo_diverse_500takes_k64_20k_ddp4_resumable_20260615_181053/fact_tokenizer.ckpt
```

## 8. 训练结果

### 8.1 总体 loss

训练稳定完成，没有出现 NaN、VQ 爆炸或进程异常。

核心 loss 变化：

```text
total loss:
前 200 step 平均: 1.530
最后 500 step 平均: 0.695

self loss:
前 200 step 平均: 1.479
最后 500 step 平均: 0.578

swap loss:
前 200 step 平均: 1.479
最后 500 step 平均: 0.578

vq loss:
前 200 step 平均: 0.034
最后 500 step 平均: 0.00127
```

这说明训练是正常收敛的。

### 8.2 Reconstruction MSE

DINO feature reconstruction MSE：

```text
ego self:
前 200 step 平均: 0.746
最后 500 step 平均: 0.375

ego swap:
前 200 step 平均: 0.746
最后 500 step 平均: 0.375

exo self:
前 200 step 平均: 0.732
最后 500 step 平均: 0.203

exo swap:
前 200 step 平均: 0.732
最后 500 step 平均: 0.203
```

最重要的观察：

```text
self reconstruction 和 swap reconstruction 几乎完全重合。
```

最后 500 step：

```text
ego swap - self 平均差: -0.00034
exo swap - self 平均差: -0.000024
```

这说明从重建角度看，另一视角的 shared action token 几乎可以无损替换本视角 shared action token。

这是 FACT tokenizer v0.1 中最积极的信号。

## 9. Token 导出与验证

导出文件：

```text
/data_all/sxh/FACT_tokenizer/outputs/fact_tokenizer/egoexo_diverse_500takes_k64_20k_ddp4_resumable_20260615_181053/extracted/ego_tokens.npz
```

导出内容：

```text
indices:     (7999, 1, 4)
soft_probs:  (7999, 1, 4, 64)
confidence:  (7999, 1, 4)
```

验证结果：

```text
token_min: 0
token_max: 63
tokens_in_range: true
private_residual_exported: false
```

说明：

1. token index 范围正确。
2. 导出的是 ego branch token。
3. private residual 没有泄露到导出文件中。

## 10. Codebook 使用情况

整体使用：

```text
num_latents: 64
total_tokens: 31996
used_codes: 49
usage_fraction: 0.765625
usage_perplexity: 17.66
normalized usage entropy: 0.690
```

结论：

```text
整体没有严重 codebook collapse。
```

但分布并不均匀：

```text
top1 code 占比: 16.2%
top5 codes 合计占比约: 57.8%
top10 codes 合计占比约: 78.1%
zero-count codes: 15
```

这说明 codebook 已经被使用起来，但部分 code 明显更常用。

### 10.1 Action slot 分工

不同 action slot 的 code 使用情况差异很大：

```text
slot 0:
used_codes = 42
perplexity = 14.25

slot 1:
used_codes = 6
perplexity = 2.15

slot 2:
used_codes = 9
perplexity = 2.28

slot 3:
used_codes = 45
perplexity = 20.53
```

解释：

1. slot 0 和 slot 3 比较分散，承担了更多离散变化。
2. slot 1 和 slot 2 比较集中，可能出现了局部 collapse，或者它们只学到了很粗粒度的二分/少数状态。
3. 目前 4 个 action slots 的分工还不均匀。

下一版可以考虑：

1. 加强 per-slot balance。
2. 增大 data diversity。
3. 降低 private residual 泄露能力。
4. 调整 temperature 或 commitment/balance 权重。

## 11. Confidence 分析

导出的 assignment confidence：

```text
confidence_mean: 0.028
confidence_min: 0.0043
confidence_max: 0.0836
confidence_median: 0.0144
```

soft max probability：

```text
median: 0.0389
max: 0.148
```

这个 confidence 明显偏低。

因为 confidence 的定义是：

```text
1 - entropy(p_act) / log(num_codes)
```

所以低 confidence 表示 soft assignment distribution 的 entropy 仍然比较高，接近均匀分布。

这意味着：

1. hard token 虽然已经能导出。
2. codebook 也没有整体塌缩。
3. 但模型对每个样本到底应该选择哪个 code 还不够确定。

这也是当前 v0.1 最主要的不足之一。

## 12. 本次实验的核心结论

本次实验可以认为成功完成了 FACT tokenizer 第一阶段 MVP。

已经成立的部分：

1. EgoExo paired data 可以进入统一 batch schema。
2. DINO feature backbone、view encoder、shared VQ codebook、private residual、decoder 可以端到端训练。
3. self reconstruction 和 swapped reconstruction 都能稳定下降。
4. swapped reconstruction 几乎和 self reconstruction 一样好。
5. ego shared action token 可以独立导出。
6. private residual 没有被导出。
7. codebook 整体使用了 49/64 个 code，没有明显整体 collapse。

还不能过早下结论的部分：

1. 现在还不能说 token 已经具有明确 action 语义。
2. private residual 可能携带了部分动作信息。
3. decoder 可能存在 shortcut，使 swap/self 接近。
4. confidence 偏低，说明 assignment 还不够 sharp。
5. slot 1/2 有局部 collapse 倾向。

因此当前最准确的判断是：

```text
FACT tokenizer v0.1 的工程机制已经跑通；
q_act_ego 已经是一个可导出、可验证的 shared action token 雏形；
但它是否真正是高质量 action token，还需要 ablation 和语义 probe 验证。
```

## 13. 下一步最重要的验证

下一步不建议只盲目扩大训练，而应该先做几个关键 ablation。

### 13.1 Token causality ablation

比较以下几种情况的 reconstruction MSE：

```text
correct q_act
shuffled q_act
zero q_act
random-take q_act
```

如果 correct q_act 明显优于 shuffled/zero/random，说明 future reconstruction 真正依赖 shared action token。

### 13.2 Private leakage ablation

比较：

```text
normal r_priv
zero r_priv
shuffled r_priv
reduced private_dim
stronger private dropout
```

如果 private residual 被破坏后，模型仍然必须依赖 q_act 才能重建 future，说明 q_act 更可能承担 action 信息。

### 13.3 Ego/exo token alignment

检查同一 transition 下：

```text
q_act_ego
q_act_exo
```

是否在 hard index 或 soft distribution 上接近。

可以看：

1. hard token match rate
2. ego/exo assignment KL
3. same-take temporal consistency
4. cross-view retrieval accuracy

### 13.4 Semantic probe

利用 EgoExo4D annotation 做初步语义验证：

```text
object label
contact label
verb/action label
narration-derived action text
```

检查：

1. 同一 action 是否更容易落到相似 code。
2. 同一 object/contact 是否有 code 聚类。
3. token 对 action label 的 linear probe / kNN probe 是否高于随机 baseline。

## 14. 后续训练方向

如果 ablation 证明 q_act 确实有因果作用，可以继续扩大训练。

建议方向：

1. 将训练步数从 20k 扩到 50k。
2. 保持 K=64 做稳定性验证，再尝试 K=128。
3. 加强 KL 或 cross-view alignment 权重。
4. 加入 per-slot usage balance，缓解 slot 1/2 局部 collapse。
5. 降低 private residual 能力，例如更小 private_dim、更强 dropout、stop-gradient 策略或 adversarial separation。
6. 从两帧 transition 扩展到多帧 temporal window，让 token 更像 action segment，而不是单步 visual delta。

## 15. 本次实验的可视化结果

训练曲线：

```text
docs/assets/fact_tokenizer_visualizations/egoexo_diverse_500takes_k64_20k_ddp4_resumable_20260615_181053_training_curves.png
```

Token 诊断图：

```text
docs/assets/fact_tokenizer_visualizations/egoexo_diverse_500takes_k64_20k_ddp4_resumable_20260615_181053_token_diagnostics.png
```

可视化摘要：

```text
outputs/fact_tokenizer/egoexo_diverse_500takes_k64_20k_ddp4_resumable_20260615_181053/visualizations/visual_summary.json
```

## 16. 一句话总结

这次实验完成了 FACT tokenizer 第一阶段从数据、训练、导出到验证的完整闭环；结果显示 shared action token 机制已经可训练、可导出、可验证，并且 swap reconstruction 表现非常接近 self reconstruction。下一阶段的关键不是继续证明它能重建，而是证明 `q_act_ego` 对 future prediction 有不可替代的因果作用，并进一步验证它是否与真实动作语义对齐。
