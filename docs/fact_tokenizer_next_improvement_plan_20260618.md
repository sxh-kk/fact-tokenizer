# FACT Tokenizer 下一轮改进计划

更新时间：2026-06-18

本文基于当前综合实验结论，定义下一轮一阶段 FACT tokenizer 优化方案。当前阶段仍然不加入完整 multi-exo teacher 聚合，继续聚焦 paired Ego/Exo 两视角下的 shared action tokenizer。

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

优先级排序：

1. 降低 take leakage。
2. 恢复并稳定 codebook usage。
3. 提高 ego/exo action causality。
4. 保持 view invariance 不退化。
5. 保持 private residual 不进入后续 action signal。

## 2. 关键策略：先改 transition 数据分布

当前 16 transitions/take 的数据分布已成为瓶颈。下一轮优先采用更 dense 的 transition sampling：

- `transition_sec = 0.5`
- `stride_sec = 1.0`
- 每个 take 最多采样 48 个 transition
- train / heldout 继续按 take 划分
- 首轮使用 max 48，不直接上 64；如果 48 对 take leakage 有改善但 coverage 仍不足，再做 64 对照。

理由：

- 0.5s transition 保持短时动作变化，减少 long-horizon scene/task progress 对 token 的污染。
- 1.0s stride 提高每个 take 内动作阶段覆盖，提供更多 same-take variation。
- 48 transitions/take 让模型在同一 take 内看到更多不同动作阶段，降低把 take identity 当作 token 捷径的收益。
- 按 take 划分 heldout 继续保证泛化评估严格。

风险：

- 0.5s 可能对慢动作过短，motion delta 不够强。
- 48 transitions/take 会增加数据量和训练时间。
- 如果 DINO feature 对短时变化不敏感，可能仍需增强 delta/motion loss 或改 future window。

控制：

- 第一轮只做 48，观察 take_nmi、used_codes、ego/exo causality。
- 保留 64 作为后备数据实验，不与 loss 大改同时混在一起。

## 3. 新数据版本

建议创建新 shard：

- `data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000.npz`

建议创建新 split：

- `data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz`
- `data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz`
- `data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_labels.jsonl`
- `data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_labels.jsonl`

建议命令：

```bash
/home/sxh/.conda/envs/fact_tokenizer/bin/python scripts/prepare_fact_egoexo_npz.py \
  --egoexo-root data/egoexo4d \
  --selected-jsonl data/egoexo4d/fact_debug/selected_takes_500_diverse.jsonl \
  --output-npz data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000.npz \
  --failed-jsonl data/fact_egoexo/failed_samples_500takes_t0p5_s1_48t.jsonl \
  --samples-per-take 48 \
  --stride-sec 1.0 \
  --transition-sec 0.5 \
  --resize 224
```

```bash
/home/sxh/.conda/envs/fact_tokenizer/bin/python scripts/split_fact_npz_by_take.py \
  --input-npz data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000.npz \
  --output-dir data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20 \
  --heldout-fraction 0.2 \
  --seed 123 \
  --labels-jsonl data/egoexo4d/fact_debug/selected_takes_500_diverse.jsonl
```

如果需要与旧 split 使用同一批 heldout take，可加：

```bash
--heldout-uids data/fact_egoexo/splits/diverse_500takes_seed123_80_20/heldout_uids.txt
```

推荐使用同一批 heldout take，方便和 v4d/v5 系列横向比较。

## 4. 下一轮训练 recipe

建议不要直接从 v5k 作为唯一主线继续。v5k 已出现明显 code collapse，更适合作为负结果参考。下一轮更合理的起点有三类：

- v4d：整体最均衡，code usage 和 confidence 最好，但 view/take leakage 在 heldout gate 下仍偏高。
- v5d：view leakage 最低之一，slot dropout/private separation 有效，但 usage 和 causality 仍不足。
- v5f：4.0 非 multi-exo weak teacher 主线，view leakage 很低，effective codes 较好，但 usage 数量和 causality 不足。

推荐主线：

- 首选起点 checkpoint：`outputs/fact_tokenizer/v5f_4p0_teacher_usage_8gpu_20260617_192843/fact_tokenizer.ckpt`
- 对照起点 checkpoint：`outputs/fact_tokenizer/v5d_v4d_finetune_slot_dropout_8gpu_20260617_185920/fact_tokenizer.ckpt`
- 保底回退 checkpoint：`outputs/fact_tokenizer/egoexo_diverse_500takes_k64_35k_v4d_gentle_usage_from_v2_20260616_215152/fact_tokenizer.ckpt`
- 训练名：`v5l_4p0_transition48_dense_take_repair_8gpu`
- GPU：8 卡
- global batch：128 或 256
- per-GPU batch：16 或 32
- steps：先 12k-14k finetune

核心训练逻辑：

- 保留 confidence-gated exo weak teacher，但不加 full multi-exo aggregation。
- 保留 private dropout / no-private contrast，继续限制 private residual 承担 action。
- 保留 action contrast、same-take contrast。
- usage balancing 不再继续盲目加大，避免进一步 code collapse。
- take uniform 系列保留中等权重，观察 dense data 是否自然降低 take leakage。

推荐相对 v5k 的变化：

- `samples_per_take` in grouped batch 从 4 提到 8，让 batch 内同 take variation 更丰富。
- `vq_temperature` 适度提高到约 0.08-0.085，帮助探索 codebook。
- `balance_weight` 和 `hard_usage_balance_weight` 适度提高，但不再极端堆叠。
- `assignment_entropy_target` 保持高一点，但 entropy weight 降低，避免把 assignment 推成过度平滑。
- `action_consistency_weight` 不要过强，避免 ego/exo 被拉成任务/take 语义平均。
- `exo_aux_multiplier` 可以略高，专门修 exo causality，但需观察 exo zero/random-take delta。

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
- 如果未通过，但 `tuple_take_nmi` 明显下降且 used codes 回升，继续小步调 recipe。
- 如果 `tuple_take_nmi` 未下降且 code usage 继续下降，停止 loss 堆叠，进入数据/label/feature 诊断。

## 6. 下一轮结果判读规则

### 6.1 理想进展

认为 transition48 方向有效的标准：

- `tuple_take_nmi` 从 0.43-0.52 降到 0.35-0.40 附近。
- `ego_used_codes` 从 30-38 回到 40+，最好接近 45。
- `tuple_view_nmi` 仍低于 0.10。
- `ego_swap_random_take_delta` 和 `ego_swap_random_code_delta` 不下降。
- `exo_swap_random_take_delta` 或 `exo_swap_zero_delta` 至少一个改善。

### 6.2 失败模式 A：take_nmi 降了，但 code usage 仍低

说明 dense transitions 有效，但 usage balancing/assignment 仍不健康。下一步：

- 降低 `action_consistency_weight`
- 降低 `kl_weight`
- 提高 `hard_usage_balance_weight` 小幅，不超过 v5k 太多
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
- 只看 train reconstruction loss 判断模型好坏。

原因是当前失败项集中在 take leakage、code usage 和 action causality。未过 gate 前，后续模块会继承错误 token，越训越难定位问题。

## 8. 当前推荐执行顺序

1. 生成 transition48 新 shard。
2. 使用旧 heldout_uids 保持可比性，生成新 train/heldout split。
3. 从 v5f 或 v5d checkpoint 开始 v5l 8-GPU finetune。
4. 训练后立即跑 heldout probe/gate。
5. 生成可视化：training curves、action diagnostics、code usage、probe/gate summary。
6. 如果通过 gate，停止一阶段训练。
7. 如果未通过，按失败模式小步迭代。

## 9. 一句话结论

下一轮核心不是更复杂的 teacher，而是先把 transition sampling 从 16/take 改成 dense 48/take，让模型在同一 take 内看到足够动作阶段变化，从数据分布上削弱 take identity 捷径，再用现有 4.0 非 multi-exo loss 做温和约束。
