# FACT Tokenizer Filtering v0 / Filtered-v6b / Filtered-v6e Report

更新时间：2026-06-24

本文记录在 `filtering_v0` task-relevant Ego-Exo subset 上补跑的 FACT tokenizer 实验：`v6b@90000` filtered-heldout baseline、`filtered-v6b` 200-step smoke、`filtered-v6b` 92000 first gate，以及 `filtered-v6e` mined same-take 200-step smoke。结论只覆盖当前 Stage-1 FACT tokenizer，不扩展到 WAM、Action Head、SONIC 或机器人部署。

## 1. 实验目的

此前 `v6e` same-take hard negative 在原始 diverse 500-take split 上没有解决 same-take action discrimination 弱的问题。人工查看 mined pair contact sheets 后，一个新的问题变得更明确：

```text
原始 diverse split 中有不少 ego 片段缺少手、物体或明确交互区域，
容易让 tokenizer 学到 scene / take / camera-motion shortcut，
而不是 coarse loco-manipulation action token。
```

因此本轮先不急着继续加 loss，而是先做一个数据侧验证：

```text
如果只使用 filtering_v0 选出的 interaction-rich / task-relevant subset，
v6b anchor 是否会在 filtered heldout 上产生更强的 action discrimination 信号？
```

本轮 gate 顺序：

1. 先评估 `v6b@90000` on filtered heldout。
2. 跑 `filtered-v6b` 200-step smoke。
3. 用同一个 filtered heldout probe 比较。
4. 若 `filtered-v6b` 有正向信号，再继续到 `92000`。
5. 若 `filtered-v6b` 92000 正向，再跑 `filtered-v6e` 200-step smoke。
6. `filtered-v6e` smoke 必须打过同长度 `filtered-v6b` smoke，才继续到 `92000`。

## 2. Filtering v0 数据资产

原始 source split：

```text
data/fact_egoexo_sxh_handoff/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20
```

filtering output：

```text
/data_all/intern02/egoexo-task-filter/outputs/filtering_v0_500takes_20260624
```

生成的 filtered NPZ：

```text
/data_all/intern02/egoexo-task-filter/outputs/filtering_v0_500takes_20260624/filtered_npz/train_by_take.npz
/data_all/intern02/egoexo-task-filter/outputs/filtering_v0_500takes_20260624/filtered_npz/heldout_by_take.npz
```

take-level bucket 结果：

| split | fact_main | loco_aux | discard |
|---|---:|---:|---:|
| train | 121 | 1 | 220 |
| heldout | 34 | 1 | 50 |

transition-level filtered samples：

| split | takes | samples | samples / take |
|---|---:|---:|---:|
| train | 121 | 5808 | 48 |
| heldout | 34 | 1632 | 48 |

task 分布观察：

- `fact_main` 主要保留 `Cooking`、`Bike Repair`、`Health`，少量 `Music`。
- `discard` 主要包含 `Dance`、`Soccer`、`Basketball`、`Rock Climbing`、大量 `Music`。
- 这说明 `filtering_v0` 更像一个 interaction-rich / manipulation-heavy subset，不是 balanced coarse loco-manipulation subset。
- 当前筛选仍是 metadata / proxy-driven，还没有人工黄金集校准。

## 3. 运行资产

服务器仓库：

```text
/data_all/intern02/fact-tokenizer
```

baseline probe，`v6b@90000` on filtered heldout：

```text
outputs/fact_tokenizer/v6b90000_on_filtering_v0_interaction_heldout_probe_20260624_164425
```

`filtered-v6b` 200-step smoke：

```text
outputs/fact_tokenizer/_smoke_filtered_v6b_from_v6b90000_filtering_v0_interaction_8gpu_20260624_171149
```

`filtered-v6b` 92000 gate：

```text
outputs/fact_tokenizer/filtered_v6b_from_v6b90000_filtering_v0_interaction_8gpu_gate92000_20260624_172321
```

`filtered-v6e` mined same-take 200-step smoke：

```text
outputs/fact_tokenizer/_smoke_filtered_v6e_mined_from_v6b90000_filtering_v0_interaction_8gpu_20260624_180921
```

`filtered-v6e` pair maps：

```text
outputs/fact_tokenizer/v6e_pair_maps_filtering_v0_interaction/v6e_mined_filtering_v0_interaction_same_take_pairs_train.npz
outputs/fact_tokenizer/v6e_pair_maps_filtering_v0_interaction/v6e_mined_filtering_v0_interaction_same_take_pairs_heldout.npz
```

pair visualization：

```text
outputs/fact_tokenizer/v6e_pair_maps_filtering_v0_interaction/visual_mined_filtering_v0_interaction_train
outputs/fact_tokenizer/v6e_pair_maps_filtering_v0_interaction/visual_mined_filtering_v0_interaction_heldout
```

## 4. 训练完成情况

`filtered-v6b` 200-step smoke：

- 起点：`v6b@90000`
- final step：`90199`
- `exit_code = 0`
- checkpoint 成功保存
- heldout probe 成功完成
- 无 NaN / OOM

`filtered-v6b` 92000 gate：

- 起点：`filtered-v6b` smoke checkpoint
- final step：`91999`
- `exit_code = 0`
- `fact_tokenizer.ckpt` 和 `fact_tokenizer_step_092000.ckpt` 成功保存
- heldout probe 成功完成
- 无 NaN / OOM

`filtered-v6e` 200-step smoke：

- 起点：`v6b@90000`
- final step：`90199`
- `exit_code = 0`
- mined train / heldout pair map 均成功生成
- pair map valid anchor fraction = `1.0`
- pair map valid pair fraction = `1.0`
- `mined_same_take_contrast_loss` 和 `no_private_mined_same_take_contrast_loss` 正常记录且非零
- checkpoint 成功保存
- heldout probe 成功完成
- 无 NaN / OOM

资源备注：

- 本轮实验运行时服务器上存在另一个 8-GPU workload。
- FACT tokenizer 训练没有触发 CUDA OOM。
- wall-clock 时间不作为算法结论依据。

## 5. Filtered Heldout Probe 对比

所有结果均使用同一个 filtered heldout：

```text
/data_all/intern02/egoexo-task-filter/outputs/filtering_v0_500takes_20260624/filtered_npz/heldout_by_take.npz
```

| run | step | ego used | exo used | ego same | ego random-take | ego random-code | ego temporal-4 | exo same | exo random-take | exo random-code | exo temporal-4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v6b@90000` | 89999 | 24 | 28 | 0.004939 | 0.005485 | 0.031744 | 0.002877 | 0.001506 | 0.001565 | 0.026624 | 0.001021 |
| `filtered-v6b` smoke | 90199 | 25 | 29 | 0.005077 | 0.005601 | 0.035093 | 0.002971 | 0.001498 | 0.001555 | 0.026958 | 0.001021 |
| `filtered-v6b` 92000 | 91999 | 27 | 29 | 0.005725 | 0.006105 | 0.035348 | 0.003179 | 0.001576 | 0.001503 | 0.024935 | 0.001122 |
| `filtered-v6e` smoke | 90199 | 25 | 28 | 0.005064 | 0.005726 | 0.033411 | 0.002866 | 0.001497 | 0.001503 | 0.026326 | 0.001010 |

`filtered-v6b` 92000 vs `v6b@90000`：

| metric | delta |
|---|---:|
| ego used codes | +3 |
| exo used codes | +1 |
| ego same | +0.000786 |
| ego random-take | +0.000621 |
| ego random-code | +0.003604 |
| ego temporal-4 | +0.000303 |
| exo same | +0.000071 |
| exo random-take | -0.000062 |
| exo random-code | -0.001689 |
| exo temporal-4 | +0.000101 |

`filtered-v6e` smoke vs same-length `filtered-v6b` smoke：

| metric | delta |
|---|---:|
| ego used codes | 0 |
| exo used codes | -1 |
| ego same | -0.000013 |
| ego random-take | +0.000124 |
| ego random-code | -0.001682 |
| ego temporal-4 | -0.000105 |
| exo same | -0.000001 |
| exo random-take | -0.000052 |
| exo random-code | -0.000632 |
| exo temporal-4 | -0.000011 |

## 6. Take Leakage / Semantic Probe

对 `filtered-v6b` 92000 额外跑了 semantic/take probe：

```text
outputs/fact_tokenizer/filtered_v6b_from_v6b90000_filtering_v0_interaction_8gpu_gate92000_20260624_172321/semantic_take_probe_20260624_180412
```

label coverage：

- `1611 / 1632 = 0.9871`
- `take_uid` values：34
- `parent_task_name` values：4
- `task_name` values：12

tuple-level take NMI：

| view | take_uid nmi_sqrt | purity_label_given_token |
|---|---:|---:|
| ego | 0.32699 | 0.25760 |
| exo | 0.40461 | 0.32464 |

slot-level take NMI：

| view | slot 0 | slot 1 | slot 2 | slot 3 |
|---|---:|---:|---:|---:|
| ego | 0.07844 | 0.09907 | 0.09895 | 0.09756 |
| exo | 0.16451 | 0.19254 | 0.18762 | 0.16755 |

观察：

- slot-level take leakage 不高。
- tuple-level exo take NMI 约 `0.405`，接近此前原始 heldout 上的量级，不是新的明显红线。
- 当前 filtered split 只有 34 heldout takes，tuple-level NMI 对 take 数和 subset composition 更敏感，后续应继续用同一口径追踪。

## 7. 结论

### 7.1 数据筛选有正向信号

`filtered-v6b` 92000 是本轮最重要的结果：

- `ego same` 从 `0.004939` 提升到 `0.005725`。
- `ego same` 已经越过此前 sprint 里设定的 `0.0055` 最低推进线。
- `ego random-take` 从 `0.005485` 提升到 `0.006105`，但仍低于理想的 `0.008`。
- `ego random-code` 从 `0.031744` 提升到 `0.035348`，保持强信号。
- ego used codes 从 `24` 增加到 `27`。

这支持一个判断：

```text
此前 same-take action discrimination 弱，不完全是 loss 设计问题；
原始 diverse split 的数据混杂和 scene/take shortcut 也在压制 action-token 学习。
```

### 7.2 exo 侧仍然没有同步改善

`filtered-v6b` 92000 的 exo 指标比较混合：

- `exo same` 微升。
- `exo temporal-4` 微升。
- `exo random-take` 微降。
- `exo random-code` 明显低于 baseline。

因此不能说 filtering_v0 已经全面改善 ego-exo action token；目前正向信号主要来自 ego 侧。

### 7.3 当前 filtered-v6e 不值得继续

`filtered-v6e` 200-step smoke 虽然训练路径可运行，但没有打过同长度 `filtered-v6b` smoke：

- `ego same` 基本持平略低。
- `ego random-take` 只小幅提高。
- `ego random-code` 下降。
- exo 指标整体更差或持平。

因此本轮不应继续 `filtered-v6e` 到 92000。当前 DINO mined same-take hard negative 仍可能噪声较大，或者它的收益被 filtered-v6b 自身的数据改善覆盖。

## 8. 下一步建议

短期建议：

1. 不继续当前 `filtered-v6e` smoke。
2. 继续 `filtered-v6b` 主线，从 `92000` 往 `95000` 跑，每 `1000` step 保存并 probe。
3. 继续用同一个 filtered heldout 追踪 `ego same`、`ego random-take`、`ego random-code`、used codes 和 take NMI。
4. 如果 `ego random-take` 能继续靠近 `0.008`，再考虑在 filtered data 上重新设计 v6e miner。

数据侧建议：

1. 给 `filtering_v0` 加小规模人工审查，尤其检查 `fact_main` 是否真的包含 hand/object/body interaction。
2. 不要把 `filtering_v0` 当成最终数据集；它目前偏 interaction-rich，明显缺少 balanced loco-body subset。
3. 构建三个 split：
   - `interaction_main`
   - `loco_aux`
   - `diagnostic_heldout`
4. 后续对 `filtered-v6b` 和 `filtered-v6e` 都使用同一个人工校准后的 split，避免数据变化和方法变化混在一起。

方法侧建议：

1. v6e miner 需要重新评估 donor quality，不建议只靠 DINO transition delta。
2. 如果继续 v6e，应增加人工抽检通过率门槛，或者引入 hand/object/body-motion relevance score。
3. 暂时不要进入 WAM / Action Head / SONIC；Stage-1 tokenizer 还需要先稳定 action discrimination 和数据筛选。

## 9. 当前决策

本轮决策为：

```text
保留 filtered-v6b 作为下一阶段主线。
暂停当前 filtered-v6e mined hard negative。
优先推进数据筛选校准和 filtered-v6b longer gate。
```
