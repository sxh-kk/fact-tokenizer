# FACT Tokenizer v6e-Control Comparison Report

更新时间：2026-06-24

本文记录 `v6e-control` temporal same-take donor 对照实验，以及它与 `v6e-a` offline DINO-mined same-take hard negative 实验的 heldout 对比。结论只覆盖当前 Stage-1 FACT tokenizer，不扩展到 WAM、Action Head、SONIC 或机器人部署。

## 1. 对照目的

`v6e-a` 在 `92000` first gate 中出现了轻微 random-take 提升，但 same-take action discrimination 没有改善。为了判断这点提升来自哪里，需要跑 `v6e-control`。

核心问题是：

```text
v6e-a 的小幅 random-take 提升，是否来自 DINO offline hard mining？
还是仅来自 pair-batch / same-take donor-in-batch 训练结构本身？
```

因此 `v6e-control` 保持和 `v6e-a` 尽量一致的训练结构，只替换 donor 选择方式。

## 2. v6e-a 与 v6e-control 的区别

| 项目 | v6e-a | v6e-control |
|---|---|---|
| 起点 | `v6b@90000` | `v6b@90000` |
| 目标 step | `92000` | `92000` |
| sampler | anchor + donor 同 batch | anchor + donor 同 batch |
| mined contrast 权重 | 相同 | 相同 |
| no-private mined contrast 权重 | 相同 | 相同 |
| donor 选择 | DINO transition/context score 离线 hard mining | 普通 same-take temporal negative |
| pair map mode | `mined` | `temporal` |
| 目的 | 测 DINO hard mining 是否有效 | 排除 batch 结构本身导致的假提升 |

解释：

- 如果 `v6e-a` 明显优于 `v6e-control`，说明 DINO hard mining 有贡献。
- 如果两者接近，说明提升主要来自 pair-batch / same-take donor structure。
- 如果 `v6e-control` 更好，说明当前 DINO miner 可能噪声较大。
- 如果两者都不能提升 same-take，则当前路线没有解决核心问题。

## 3. 运行资产

服务器仓库：

```text
/data_all/intern02/fact-tokenizer
```

v6e-a run：

```text
outputs/fact_tokenizer/v6e_mined_same_take_from_v6b_8gpu_gate92000_20260623_queued
```

v6e-a heldout probe：

```text
outputs/fact_tokenizer/v6e_mined_same_take_from_v6b_8gpu_gate92000_20260623_queued/heldout_probe_20260624_120930
```

v6e-control run：

```text
outputs/fact_tokenizer/v6e_control_temporal_from_v6b_8gpu_gate92000_20260624_1229
```

v6e-control heldout probe：

```text
outputs/fact_tokenizer/v6e_control_temporal_from_v6b_8gpu_gate92000_20260624_1229/heldout_probe_20260624_140429
```

v6e-control script：

```text
scripts/run_fact_transition48_v6e_control_temporal_from_v6b_8gpu.sh
```

v6e-control pair map：

```text
outputs/fact_tokenizer/v6e_pair_maps/v6e_control_temporal_same_take_pairs_train.npz
```

训练数据：

```text
data/fact_egoexo_sxh_handoff/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz
```

heldout 数据：

```text
data/fact_egoexo_sxh_handoff/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz
data/fact_egoexo_sxh_handoff/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_labels.jsonl
```

## 4. v6e-control 配方

v6e-control 从 `v6b@90000` 继续训练：

```text
v6b@90000
    -> v6e-control temporal same-take donor
    -> step 91999
```

关键参数：

```text
PAIR_MODE=temporal
PAIR_LABEL=control_temporal
STEPS=92000
--mined-pair-batches
--mined-same-take-contrast-weight 0.080
--no-private-mined-same-take-contrast-weight 0.100
```

虽然参数名仍然叫 `mined-*`，但 control 的 pair map 来自 temporal donor：

```text
score = -abs(abs(delta_t) - temporal_offset)
```

也就是说，control 保留同 batch donor 结构和 loss 名称，但 donor 不经过 DINO hard mining。

## 5. v6e-control 训练结果

v6e-control 完成情况：

- final step：`91999`
- `exit_code = 0`
- final checkpoint 成功保存
- 中间 checkpoint：`fact_tokenizer_step_091000.ckpt`、`fact_tokenizer_step_092000.ckpt`
- heldout probe 成功完成

最终 checkpoint：

```text
outputs/fact_tokenizer/v6e_control_temporal_from_v6b_8gpu_gate92000_20260624_1229/fact_tokenizer.ckpt
```

训练末尾关键记录：

```text
step = 91999
loss = 2.45340
mined_same_take_contrast_loss = 0.02315
```

资源备注：

- v6e-control 运行时服务器上仍有 openpi 任务占用 8 张 GPU。
- v6e-control 与该任务短时共享 GPU 显存和算力。
- 训练没有 OOM，final checkpoint 正常保存。
- wall-clock 不应作为算法结论依据。

## 6. Heldout Probe 结果

v6e-control heldout probe：

```text
outputs/fact_tokenizer/v6e_control_temporal_from_v6b_8gpu_gate92000_20260624_1229/heldout_probe_20260624_140429
```

token usage：

| view | used codes | note |
|---|---:|---|
| ego | 27 | 与 v6b@90000 接近 |
| exo | 33 | 与 v6e-a 相同 |

causality / leakage：

| metric | value |
|---|---:|
| ego same-take delta | 0.00355 |
| ego random-take delta | 0.00731 |
| ego random-code delta | 0.03251 |
| exo random-take delta | 0.00164 |
| exo random-code delta | 0.02367 |
| exo zero delta | 0.01924 |
| ego take NMI | 0.34147 |
| exo take NMI | 0.39194 |
| tuple view NMI | 0.05221 |

## 7. v6e-a vs v6e-control

| metric | v6e-a mined | v6e-control temporal | control - mined |
|---|---:|---:|---:|
| ego same-take delta | 0.00349 | 0.00355 | +0.00006 |
| ego random-take delta | 0.00730 | 0.00731 | +0.00001 |
| ego random-code delta | 0.03051 | 0.03251 | +0.00199 |
| exo random-take delta | 0.00167 | 0.00164 | -0.00002 |
| exo random-code delta | 0.02303 | 0.02367 | +0.00064 |
| exo zero delta | 0.01953 | 0.01924 | -0.00029 |
| ego take NMI | 0.33991 | 0.34147 | +0.00156 |
| exo take NMI | 0.39160 | 0.39194 | +0.00034 |
| ego used codes | 26 | 27 | +1 |
| exo used codes | 33 | 33 | 0 |

关键观察：

- `ego random-take` 基本完全相同：`0.00730` vs `0.00731`。
- `ego same-take` 也基本完全相同：`0.00349` vs `0.00355`。
- `v6e-control` 的 `ego random-code` 反而更高：`0.03251` vs `0.03051`。
- `v6e-control` 的 ego used codes 为 `27`，略高于 v6e-a 的 `26`。
- take leakage 基本相同，control 略高但差距很小。
- exo random-take 差距可忽略。

## 8. 结论

v6e-control 的结果说明：

```text
v6e-a 的小幅 random-take 提升，不能归因于 DINO hard mining。
普通 temporal same-take donor 在相同 pair-batch / loss 结构下取得了几乎一样的结果。
```

因此：

- 当前 DINO transition/context miner 没有提供清晰额外收益。
- 当前 pair-batch + same-take donor contrast 结构可能有轻微 random-take 帮助。
- 但核心 same-take action discrimination 仍然没有改善。
- v6e-a 和 v6e-control 都没有突破 `ego same-take delta ~= 0.0035`。
- 这条 recipe 不应继续直接延长到 `95000` 作为主路线。

## 9. 路线判断

不建议：

- 继续延长当前 v6e-a 到 `95000`。
- 继续延长当前 v6e-control 到 `95000`。
- 把当前 DINO hard mining 当作有效突破点。
- 只围绕当前 mixed diverse data 继续堆 loss。

建议：

- 暂停 v6e-a / v6e-control 延长训练。
- 将 v6e 记录为“工程路径打通，但 hard mining 未被证明有效”的混合负结果。
- 下一步优先推进 task-relevant data filtering。
- 如果继续做 miner，应重做 donor quality criterion，而不是复用当前 DINO delta/context scoring。
- 后续所有 same-take / phase negative 实验应先在 high-interaction subset 上复测。

## 10. 一句话结论

```text
v6e-control 证明：v6e-a 的轻微 random-take 改善主要不是来自 DINO hard negative mining；
当前 v6e recipe 没有解决 same-take action discrimination，
下一步应转向数据筛选或重做 miner，而不是继续延长训练。
```
