# FACT Tokenizer v6e Same-Take Hard Negative Report

更新时间：2026-06-24

本文记录本次在 `intern02@10.249.42.141` 上实现和补跑的 v6e same-take hard negative sprint，包括离线 hard negative pair mining、pair-batch 训练路径、200-step smoke、`92000` first gate，以及 heldout probe 结果。结论只覆盖当前 Stage-1 FACT tokenizer，不扩展到 WAM、Action Head、SONIC 或机器人部署。

## 1. 实验目的

v6d action-aware continuation 已经证明：在 v6b anchor 上直接加 batch-local action-aware contrast，不能有效解决同一个 take 内 action phase discrimination 弱的问题。

v6e 的目标是验证一个更显式的假设：

```text
如果给每个 anchor 放入离线挖出的 same-take / different-phase donor，
并在 batch 内显式构造 mined same-take action negative，
FACT action token 是否能更好地区分同 take 内不同 transition / action phase。
```

本轮重点检查：

- `ego_swap_same_take_delta` 是否从约 `0.0034` 提升到 `>= 0.0055`
- `ego_swap_random_take_delta` 是否从约 `0.006` 提升，最好接近或超过 `0.008`
- `ego_swap_random_code_delta` 是否保持 `>= 0.025`
- `exo` 侧是否不明显崩坏
- take leakage 是否保持在可接受范围，尤其 `take_nmi <= 0.40`
- heldout ego used codes 是否不低于约 `27`

## 2. 新增实现

新增离线 pair mining：

```text
scripts/build_fact_same_take_hard_negative_pairs.py
```

新增 pair visualization：

```text
scripts/visualize_fact_mined_pairs.py
```

新增 v6e 运行脚本：

```text
scripts/run_fact_transition48_v6e_mined_from_v6b_8gpu.sh
scripts/run_fact_transition48_v6e_control_temporal_from_v6b_8gpu.sh
scripts/run_fact_transition48_v6e_mined_from_v6b086_8gpu.sh
```

训练代码新增参数：

```text
--mined-negative-map
--mined-negative-top-k
--mined-pair-batches
--mined-same-take-contrast-weight
--no-private-mined-same-take-contrast-weight
```

核心改动：

- 每个 anchor 从离线 mined pair map 中找 same-take donor。
- `MinedPairBatchSampler` 保证 anchor 和 donor 同 batch。
- 训练 loop 建立 global sample id 到 local batch index 的映射。
- model 内新增 `_mined_same_take_action` reconstruction path。
- loss 内新增 `mined_same_take_contrast_loss` 和 `no_private_mined_same_take_contrast_loss`。

## 3. Pair Mining 设置

训练 pair map：

```text
outputs/fact_tokenizer/v6e_pair_maps/v6e_mined_same_take_pairs_train.npz
```

heldout pair map：

```text
outputs/fact_tokenizer/v6e_pair_maps/v6e_mined_same_take_pairs_heldout.npz
```

pair visualization：

```text
outputs/fact_tokenizer/v6e_pair_maps/visual_mined_train
outputs/fact_tokenizer/v6e_pair_maps/visual_mined_heldout
```

离线 mining 约束：

- donor 必须和 anchor 同 take
- donor 不能等于 anchor
- 时间间隔满足 `3 <= |dt| <= 24`
- 每个样本保留 top-4 donor

scoring 使用 DINO transition feature：

- 奖励 ego / exo transition delta 差异
- 惩罚 current/context 差异，降低纯场景变化 shortcut
- 轻微奖励合理时间间隔

pair map validation 结果：

- donor same-take 约束通过
- donor non-self 约束通过
- 时间间隔约束通过
- valid fraction 为 `1.0`

人工查看 `visual_mined_train` 后的附加观察：

- mined pair 中有一部分确实像 same-take different phase。
- 但当前 diverse Ego-Exo 数据中也有不少 ego 片段缺少手、物体或明确交互区域。
- 因此 v6e 本轮只能证明算法路径在混合数据上的早期信号，不能替代后续 task-relevant subset filtering。

## 4. v6e 配方

v6e-a 不是从头训练，而是从 v6b anchor 继续：

```text
v6b@90000
    -> v6e 200-step smoke, step 90199
    -> v6e first gate, step 91999
```

相对 v6b，v6e 保留：

- full current context
- mild delta 相关约束
- random-code contrast
- no-private 相关约束
- view-invariance 相关约束
- 不启用 v6c hard usage entropy / strong usage repair
- 不进入 WAM / Action Head / SONIC

v6e 新增关键权重：

```text
--mined-same-take-contrast-weight 0.080
--no-private-mined-same-take-contrast-weight 0.100
```

v6e 不使用 v6d 的 action-aware contrast：

```text
weight_action_aware_contrast = 0.0
weight_no_private_action_aware_contrast = 0.0
```

## 5. Smoke Test

先跑了 200-step smoke：

```text
outputs/fact_tokenizer/_smoke_v6e_mined_from_v6b_8gpu_200step_20260623_181343
```

结果：

- 从 v6b@90000 恢复。
- 跑到 step `90199`。
- 8GPU DDP 正常。
- DINOv2 加载正常。
- `mined_same_take_contrast_loss` 和 `no_private_mined_same_take_contrast_loss` 正常记录且非零。
- checkpoint 成功保存。
- heldout probe 可正常运行。

最后一条 smoke 训练记录：

```text
step = 90199
loss = 2.43935
mined_same_take_contrast_loss = 0.02343
no_private_mined_same_take_contrast_loss = 0.02473
ego_swap_mined_same_take_action/feature_mse = 0.39969
ego_swap_random_code_action/feature_mse = 0.42089
```

smoke heldout probe 关键指标：

| metric | value |
|---|---:|
| ego same-take delta | 0.00345 |
| ego random-take delta | 0.00638 |
| ego random-code delta | 0.03201 |
| exo random-take delta | 0.00129 |
| exo random-code delta | 0.02582 |
| ego used codes | 27 |
| exo used codes | 34 |
| ego take NMI | 0.33657 |
| exo take NMI | 0.38190 |

结论：smoke 通过，说明 v6e 的 pair map、sampler、model reconstruction path、loss path、checkpoint 和 heldout probe 都可运行。

## 6. v6e First Gate Run

first gate run：

```text
outputs/fact_tokenizer/v6e_mined_same_take_from_v6b_8gpu_gate92000_20260623_queued
```

运行说明：

- 起点为 v6e 200-step smoke checkpoint。
- 目标为 step `92000`。
- 实际完成到 step `91999`。
- `exit_code = 0`。
- final checkpoint 成功保存。

最终 checkpoint：

```text
outputs/fact_tokenizer/v6e_mined_same_take_from_v6b_8gpu_gate92000_20260623_queued/fact_tokenizer.ckpt
```

训练末尾记录：

```text
step = 91999
loss = 2.45078
mined_same_take_contrast_loss = 0.02498
no_private_mined_same_take_contrast_loss = 0.02407
ego_swap_mined_same_take_action/feature_mse = 0.39675
ego_swap_random_code_action/feature_mse = 0.41270
usage_capacity_loss = 0.91566
take_uniformity_loss = 0.21319
```

资源备注：

- 该 run 启动时服务器上另有 openpi 任务占用 8 张 GPU。
- v6e 与该任务短时共享 GPU 显存和算力。
- v6e 最终没有 OOM，checkpoint 正常保存。
- 这会影响 wall-clock，不应作为算法结论的一部分。

## 7. Heldout Probe

heldout probe 输出：

```text
outputs/fact_tokenizer/v6e_mined_same_take_from_v6b_8gpu_gate92000_20260623_queued/heldout_probe_20260624_120930
```

probe checkpoint step：

```text
91999
```

token usage：

| view | used codes | usage fraction | confidence mean |
|---|---:|---:|---:|
| ego | 26 | 0.40625 | 0.12711 |
| exo | 33 | 0.51563 | 0.12896 |

causality ablation：

| path | action mode | mean | delta vs correct |
|---|---|---:|---:|
| ego_swap | same_take_shuffle | 0.39618 | 0.00349 |
| ego_swap | random_take | 0.39999 | 0.00730 |
| ego_swap | random_code | 0.42320 | 0.03051 |
| exo_swap | random_take | 0.19526 | 0.00167 |
| exo_swap | random_code | 0.21663 | 0.02303 |
| exo_swap | zero | 0.21313 | 0.01953 |

semantic controls：

| metric | value |
|---|---:|
| ego take NMI | 0.33991 |
| exo take NMI | 0.39160 |
| tuple view NMI | 0.05419 |

## 8. 对比基线

| checkpoint | ego random-take | ego random-code | ego same-take | exo random-take | exo random-code | take NMI | ego used |
|---|---:|---:|---:|---:|---:|---:|---:|
| v6b@090000 | 0.00617 | 0.03230 | 0.00346 | 0.00120 | n/a | 0.38239 | 27 |
| v6d@098000 | 0.00607 | 0.03147 | 0.00345 | 0.00113 | n/a | 0.38522 | 30 |
| v6e@90199 smoke | 0.00638 | 0.03201 | 0.00345 | 0.00129 | 0.02582 | 0.33657 | 27 |
| v6e@91999 gate | 0.00730 | 0.03051 | 0.00349 | 0.00167 | 0.02303 | 0.33991 | 26 |

关键观察：

- v6e@91999 的 `ego random-take` 从 v6b/v6d 的约 `0.006` 提升到 `0.00730`。
- v6e@91999 的 `exo random-take` 从约 `0.0012` 提升到 `0.00167`。
- v6e@91999 的 `ego random-code` 仍保持通过，`0.03051 >= 0.025`。
- v6e@91999 的 `ego same-take` 仍只有 `0.00349`，和 v6b/v6d 基本相同。
- v6e@91999 的 `ego used codes` 为 `26`，没有改善。
- v6e@91999 的 `exo random-code` 为 `0.02303`，低于 smoke 的 `0.02582`，需要警惕。
- take leakage 没有恶化，ego take NMI 约 `0.340`，exo take NMI 约 `0.392`。

## 9. Gate 判断

v6e@91999 没有通过 first gate。

通过或接近通过的部分：

- `ego_swap_random_code_delta >= 0.025`：通过。
- `ego_swap_random_take_delta` 比 v6b/v6d 有抬升，但仍未达到理想 `>= 0.008`。
- `exo_swap_random_take_delta` 有抬升，但仍未达到更强标准。
- take NMI 没有明显恶化。
- view NMI 仍低。

失败部分：

- `ego_swap_same_take_delta` 没有从 `0.003-0.004` 区间脱离。
- `ego used codes` 下降到 `26`，没有达到“不低于约 27”的最低期望。
- `exo_swap_random_code_delta` 降到 `0.02303`，略低于理想线。

因此，v6e@92000 是一个混合结果：

```text
工程路径成立，random-take 有早期正向信号；
但核心 same-take action discrimination 没有改善，
不能把 v6e-a 直接判为正结果。
```

## 10. 路线判断

不建议：

- 直接把 v6e-a 继续跑到 95000 当作默认下一步。
- 仅凭训练日志中的 mined loss 或 batch feature MSE 判断成功。
- 在当前 mixed diverse data 上继续无限延长 v6e-a。

建议：

- 先跑 `v6e-control` 到相同 gate。
- `v6e-control` 使用相同 pair-batch / donor-in-batch 结构，但 donor 由普通 temporal same-take negative 产生。
- 目的：判断 v6e 的 random-take 小幅改善来自 hard mining，还是来自 pair-batch 结构本身。
- 若 `v6e-a > v6e-control`，说明 offline hard mining 有贡献，可考虑修 miner 和数据过滤后继续。
- 若 `v6e-control` 接近或优于 v6e-a，说明当前 hard mining 噪声较大，不能作为主线。
- 若二者都不改善 same-take，则应把重心转到 task-relevant subset filtering，而不是继续调 loss。

## 11. 一句话结论

```text
v6e same-take hard negative 路径已经工程打通，并带来轻微 random-take 改善；
但 first gate 没有解决 same-take action discrimination，
所以当前不能继续盲目延长 v6e-a，应先跑 v6e-control 对照并并行推进数据筛选。
```
