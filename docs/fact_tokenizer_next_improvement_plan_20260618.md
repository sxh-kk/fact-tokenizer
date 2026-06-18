# FACT Tokenizer 下一轮改进计划

更新时间：2026-06-18

本文基于当前综合实验结论和 v5l transition48 heldout gate 结果，定义下一轮一阶段 FACT tokenizer 优化方案。当前阶段仍然不加入完整 multi-exo teacher 聚合，继续聚焦 paired Ego/Exo 两视角下的 shared action tokenizer。

## 1. 改进目标

下一轮目标不是单纯降低 reconstruction loss，而是推动 tokenizer 通过 Stage-1 gate：

- `tuple_take_nmi <= 0.35`
- `tuple_view_nmi <= 0.10`
- `ego_used_codes` 进入 45-55 区间
- `ego_swap_random_take_delta >= 0.015`
- `ego_swap_random_code_delta >= 0.025`
- `exo_swap_random_take_delta >= 0.002`
- `exo_swap_zero_delta >= 0.006`
- `ego_action_without_private_saving >= 0.018`
- `ego_private_only_gap <= 0.012`

v5l 已经把 `tuple_take_nmi` 和 `tuple_view_nmi` 压到阈值内，但 action causality 和 code usage 明显失败。因此 v5l 之后的优先级排序调整为：

1. 提高 ego/exo action causality。
2. 恢复并稳定 codebook usage。
3. 保持 take leakage 不反弹。
4. 保持 view invariance 不退化。
5. 保持 private residual 不进入后续 action signal。

## 2. 已验证策略：transition48 数据分布

旧 16 transitions/take 的数据分布已被证明是 take leakage 的重要瓶颈。v5l 已采用更 dense 的 transition sampling：

- `transition_sec = 0.5`
- `stride_sec = 1.0`
- 每个可用 take 严格采样 48 个 transition
- train / heldout 继续按 take 划分
- 首轮使用 max 48，不直接上 64。

v5l 结果：

- `tuple_take_nmi = 0.339 <= 0.35`
- `tuple_view_nmi = 0.055 <= 0.10`
- `ego_used_codes = 21 < 45`
- `ego_swap_random_take_delta = 0.0039 < 0.015`
- `exo_swap_zero_delta = 0.0013 < 0.006`

结论：

- transition48 方向有效，应继续保留。
- take leakage 已经不是当前唯一最大问题。
- 下一轮不应继续主要堆 anti-take loss，而应在 transition48 数据上修 action causality 和 codebook health。

风险：

- 0.5s 可能对慢动作过短，motion delta 不够强。
- 48 transitions/take 会增加数据量和训练时间。
- 如果 DINO feature 对短时变化不敏感，可能仍需增强 delta/motion loss 或改 future window。

控制：

- 保留 48 作为主数据设置。
- 如果后续 action causality 仍弱，做 `transition_sec=1.0, stride_sec=1.0, 48 transitions/take` 对照，判断 0.5s 是否过短。
- 保留 64 作为后备 coverage 实验，不与 loss 大改同时混在一起。

## 3. 新数据版本

已创建并使用新 shard：

- `data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000.npz`

已创建并使用新 split：

- `data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz`
- `data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz`
- `data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_labels.jsonl`
- `data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_labels.jsonl`

生成命令模板：

```bash
/home/sxh/.conda/envs/fact_tokenizer/bin/python scripts/prepare_fact_transition48_split.py \
  --egoexo-root data/egoexo4d \
  --selected-jsonl data/egoexo4d/fact_debug/selected_takes_500_diverse.jsonl \
  --output-npz data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000.npz \
  --failed-jsonl data/fact_egoexo/failed_transition48_samples.jsonl \
  --prepare-report data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000_report.json \
  --split-dir data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20 \
  --samples-per-take 48 \
  --stride-sec 1.0 \
  --transition-sec 0.5 \
  --resize 224 \
  --heldout-fraction 0.2 \
  --seed 123
```

默认不加 `--allow-short-takes`，因此短 take 会被跳过，每个保留 take 必须严格写出 48 个 transition。

当前 v5l split 使用新 transition48 可用 take 集合，短 take 已跳过。后续应固定这套 transition48 heldout，方便和 v5l 横向比较。

## 4. v5l 后训练 recipe

建议不要直接从 v5k 或 v5l 作为唯一主线继续。v5k/v5l 都出现明显 code usage 收缩；v5l 更适合作为 transition48 有效但 usage/causality 失败的负结果参考。下一轮更合理的起点有三类：

- v4d：整体最均衡，code usage 和 confidence 最好，但 view/take leakage 在 heldout gate 下仍偏高。
- v5d：view leakage 最低之一，slot dropout/private separation 有效，但 usage 和 causality 仍不足。
- v5f：4.0 非 multi-exo weak teacher 主线，view leakage 很低，effective codes 较好，但 usage 数量和 causality 不足。

推荐主线：

- 首选起点 checkpoint：`outputs/fact_tokenizer/v5f_4p0_teacher_usage_8gpu_20260617_192843/fact_tokenizer.ckpt`
- 对照起点 checkpoint：`outputs/fact_tokenizer/v5d_v4d_finetune_slot_dropout_8gpu_20260617_185920/fact_tokenizer.ckpt`
- 高 usage 对照起点：`outputs/fact_tokenizer/egoexo_diverse_500takes_k64_35k_v4d_gentle_usage_from_v2_20260616_215152/fact_tokenizer.ckpt`
- 不建议主线起点：`outputs/fact_tokenizer/v5l_transition48_dense_from_v5f_20260618_121935/fact_tokenizer.ckpt`
- 训练名建议：`v5m_transition48_usage_causality_repair_8gpu`
- GPU：8 卡
- global batch：128 或 256
- per-GPU batch：16 或 32
- steps：先 12k-14k finetune

核心训练逻辑：

- 保留 transition48 split。
- 保留 confidence-gated exo weak teacher，但不加 full multi-exo aggregation。
- 保留 private dropout / no-private contrast，继续限制 private residual 承担 action。
- 保留 action contrast、same-take contrast。
- usage balancing 不再继续盲目加大，避免进一步 code collapse。
- take uniform 系列从 v5l 权重下调，防止继续把 token 压成低泄漏但低因果的少数 prototype。

推荐相对 v5l 的变化：

- `take_uniform_weight` / `take_slot_uniform_weight` / `take_pair_uniform_weight` 下调 30%-50%，因为 v5l 的 take leakage 已过阈值。
- `action_consistency_weight` 和 `kl_weight` 下调，避免 ego/exo 被拉成过平滑的任务/take prototype。
- `assignment_entropy_weight` 继续保持很低，必要时把 target 从 0.88 降到 0.80-0.84，减少过度软分配。
- `hard_usage_balance_weight` 不要继续上调；优先通过高 usage 起点、较低 consistency、较强 token causality 让 code 自然打开。
- `no_private_contrast_weight` 与 `action_only_weight` 保持或小幅提高，重点观察 `ego_action_without_private_saving`。
- same-take contrast 保留，但要检查 negative 是否真来自不同时间动作阶段，而不是相似静态片段。
- `exo_aux_multiplier` 不再单独大幅提高；优先增强 exo motion/delta 敏感性，避免 exo teacher 继续产生平滑弱 token。

建议做两个小对照，而不是单一路线：

| run | 起点 | 目的 |
|---|---|---|
| `v5m_transition48_from_v5f_usage_causality` | v5f | 保留 4.0 weak teacher 主线，减弱 anti-take/consistency，修 causality |
| `v5n_transition48_from_v4d_usage_check` | v4d | 检查高 usage 起点在 transition48 上是否能保住 40+ codes |

## 5. 自动评估与停止条件

每轮训练完成后必须自动运行：

```bash
/home/sxh/.conda/envs/fact_tokenizer/bin/python scripts/probe_fact_action_tokens.py \
  --checkpoint <run_dir>/fact_tokenizer.ckpt \
  --input-npz data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz \
  --output-dir <run_dir>/action_token_probe_heldout \
  --labels data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_labels.jsonl \
  --label-columns parent_task_name task_name university_name \
  --source-view-keys ego exo \
  --resize 224 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda
```

```bash
/home/sxh/.conda/envs/fact_tokenizer/bin/python scripts/evaluate_fact_stage1_gate.py \
  --probe-dir <run_dir>/action_token_probe_heldout \
  --output-json <run_dir>/action_token_probe_heldout/stage1_gate.json
```

停止条件：

- 如果 Stage-1 gate 全部通过，停止 tokenizer 一阶段训练，冻结 shared action tokenizer。
- 如果未通过，但 `ego_used_codes` 回升且 action delta 变大，继续小步调 recipe。
- 如果 `tuple_take_nmi` 反弹但 action/usage 变好，说明 anti-take 权重下调过多，应小幅加回。
- 如果 code usage 继续下降，停止 loss 堆叠，回到高 usage 起点或检查 assignment/codebook 机制。

## 6. 下一轮结果判读规则

### 6.1 理想进展

v5l 已满足 transition48 方向有效的 leakage 标准：

- `tuple_take_nmi` 从 v5f 的 0.433 降到 v5l 的 0.339。
- `tuple_view_nmi` 从 v5f 的 0.062 降到 v5l 的 0.055。

下一轮的理想进展标准改为：

- `ego_used_codes` 从 21 回到 35+，下一阶段再逼近 45。
- `ego_swap_random_take_delta` 从 0.0039 提高到 0.010+。
- `ego_swap_random_code_delta` 从 0.0034 提高到 0.015+。
- `exo_swap_zero_delta` 从 0.0013 提高到 0.003+。
- `tuple_take_nmi` 保持在 0.35 附近或以下。
- `tuple_view_nmi` 保持低于 0.10。

### 6.2 失败模式 A：take_nmi 降了，但 code usage 仍低

v5l 正属于这个失败模式：dense transitions 有效，但 usage balancing/assignment 仍不健康。下一步：

- 降低 `action_consistency_weight`
- 降低 `kl_weight`
- 不再直接提高 `hard_usage_balance_weight`
- 从 v4d 高 usage checkpoint 做 transition48 对照
- 检查 code histogram 是否少数 code 统治

### 6.3 失败模式 B：code usage 回升，但 take_nmi 仍高

说明 codebook 打开了，但打开的是 take/context code，不是 action code。下一步：

- 提高 same-take contrast
- 增强 same-take temporal offset negative
- 加强 take-balanced batch
- 考虑每 batch 覆盖更多 take，避免 batch distribution 本身泄漏

### 6.4 失败模式 C：ego 好转，exo 仍弱

说明 exo branch 没有足够 action-causal。下一步：

- 提高 exo path 的 action-only / motion / delta weight
- 检查 exo frame pair 是否真的有明显动态
- 对 exo 使用更强 motion-focused loss，但不要同步加大 ego-exo consistency

### 6.5 失败模式 D：所有指标均无改善

说明当前 DINO first-last feature reconstruction 和 0.5s transition 对 action token 不够敏感。下一步应做诊断：

- 可视化同一 take 内 48 transitions 的 frame pair。
- 统计 DINO feature delta 分布。
- 对比 transition_sec 0.5 vs 1.0。
- 对比 stride_sec 1.0 vs 0.5。
- 考虑引入更直接的 motion/contact proxy，而不是继续调 VQ loss。

## 7. 不建议现在做的事情

当前不建议：

- 进入 WAM / Action Head 训练。
- 导出 private residual 作为下游 label。
- 加完整 multi-exo teacher 聚合。
- 继续在旧 16 transitions/take split 上大幅堆 anti-take loss。
- 在 v5l 配方上继续无差别加 anti-take / consistency loss。
- 只看 train reconstruction loss 判断模型好坏。

原因是当前失败项集中在 code usage 和 action causality。未过 gate 前，后续模块会继承错误 token，越训越难定位问题。

## 8. 当前推荐执行顺序

1. 固定当前 transition48 train/heldout split。
2. 开一轮 `v5m_transition48_from_v5f_usage_causality`，下调 anti-take/consistency，保留 action/private 压力。
3. 并行或随后开 `v5n_transition48_from_v4d_usage_check`，验证高 usage 起点是否能在 transition48 上保住 codebook。
4. 每轮结束立即跑 heldout probe/gate。
5. 生成可视化：training curves、action diagnostics、code usage、probe/gate summary。
6. 如果 action delta 与 used codes 回升且 take/view 不反弹，继续小步推进。
7. 如果仍然低因果低 usage，转向 transition_sec 1.0 对照和 DINO delta 诊断。

## 9. 一句话结论

v5l 已证明 dense 48/take 能削弱 take identity 捷径；下一轮核心不是更复杂的 teacher，而是在 transition48 数据上把 shared action token 重新变成 decoder 真正依赖的因果变量，同时把 ego code usage 从 21 个恢复到至少 35+。
