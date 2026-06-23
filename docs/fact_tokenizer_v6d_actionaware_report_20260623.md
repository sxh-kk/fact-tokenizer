# FACT Tokenizer v6d Action-Aware Experiment Report

更新时间：2026-06-23

本文记录本次在 `intern02@10.249.42.141` 上补跑的 v6d action-aware continuation 实验、heldout checkpoint sweep，以及 v6b 中间 checkpoint 对照。结论只覆盖当前一阶段 FACT tokenizer，不扩展到 WAM、Action Head、SONIC 或机器人部署。

## 1. 实验目的

前序结论中，v6b 是当前较健康的参考点：通过了 random-code、exo zero、action-without-private、private gap 和 view-invariance，但仍失败于同 take 内动作区分、random-take causality、take leakage 和 heldout code usage。

本次 v6d 的目标是验证：在 v6b 基础上继续加入 action-aware contrast，是否能让 FACT token 学会“同一个 take 内不同 transition / action phase 应该不同”。

重点检查：

- `ego_swap_same_take_delta >= 0.010`
- `ego_swap_random_take_delta >= 0.015`
- `exo_swap_random_take_delta >= 0.002`
- `tuple_take_nmi <= 0.35`
- `ego_used_codes >= 45`

## 2. 运行资产

服务器仓库：

```text
/data_all/intern02/fact-tokenizer
```

训练脚本：

```text
/data_all/intern02/fact-tokenizer/scripts/run_fact_transition48_v6d_actionaware_from_v6b_8gpu.sh
```

训练数据：

```text
data/fact_egoexo_sxh_handoff/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz
```

Heldout gate 数据：

```text
data/fact_egoexo_sxh_handoff/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz
data/fact_egoexo_sxh_handoff/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_labels.jsonl
```

v6d 起点：

```text
outputs/fact_tokenizer/v6b_transition48_delta_full_from_v5p_8gpu_20260620_2200/fact_tokenizer.ckpt
```

v6d 正式输出：

```text
outputs/fact_tokenizer/v6d_transition48_actionaware_from_v6b_8gpu_20260623_145201
```

v6d checkpoint sweep：

```text
outputs/fact_tokenizer/v6d_transition48_actionaware_from_v6b_8gpu_20260623_145201/heldout_checkpoint_sweep_20260623_154907/summary.json
```

v6b checkpoint sweep：

```text
outputs/fact_tokenizer/_sweep_v6b_checkpoints_heldout_20260623_161957/summary.json
```

## 3. v6d 配方

v6d 不是从头训练，而是从 v6b final checkpoint 继续训练：

```text
v6b final around step 90000
        -> v6d action-aware continuation
        -> target step 98000
```

相对 v6b，v6d 保留 full current context 和温和 delta 设置，并新增：

```text
--action-aware-contrast-weight 0.045
--no-private-action-aware-contrast-weight 0.065
--action-aware-context-weight 0.35
--lr 1.5e-6
```

其它关键点：

- 8 GPU DDP，`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- per-GPU batch size 16
- `save-every 1000`
- 从 v6b step 90000 继续到 step 98000
- 不启用 v6c 的 hard usage entropy / usage capacity 方向

## 4. Smoke Test

先跑了 200 step smoke：

```text
outputs/fact_tokenizer/_smoke_v6d_actionaware_from_v6b_8gpu_20260623_143956
```

结果：

- 从 v6b step 90000 恢复。
- 跑到 step 90199。
- 8GPU DDP 正常。
- DINOv2 加载正常。
- `action_aware_contrast_loss` 和 `no_private_action_aware_contrast_loss` 正常记录。
- checkpoint 成功保存。

最后一条 smoke 训练记录中：

```text
loss = 2.4964
action_aware_contrast_loss = 0.02876
no_private_action_aware_contrast_loss = 0.02804
same_take_contrast_loss = 0.02689
random_code_contrast_loss = 0.00430
action_top1_agreement = 0.34375
```

结论：smoke 通过，说明脚本、数据、DINO、resume、DDP 和新 loss 路径都可运行。

## 5. 正式 v6d 训练

正式 run：

```text
outputs/fact_tokenizer/v6d_transition48_actionaware_from_v6b_8gpu_20260623_145201
```

训练完成情况：

- 最终 checkpoint：`fact_tokenizer.ckpt`
- 最终 step：97999
- 中间 checkpoint：`091000` 到 `098000`
- final code usage：30/64 左右
- 训练主过程没有 Traceback / RuntimeError / NaN
- 日志末尾出现 NCCL OOM warning，但发生在收尾清理阶段，final checkpoint 已保存并可加载

最终训练日志末尾：

```text
step = 97999
loss = 2.5859
action_aware_contrast_loss = 0.02628
no_private_action_aware_contrast_loss = 0.02615
same_take_contrast_loss = 0.02762
random_code_contrast_loss = 0.00591
action_top1_agreement = 0.390625
```

## 6. v6d Heldout Checkpoint Sweep

为避免只赌 final checkpoint，已对 v6d 的 8 个中间 checkpoint 跑 heldout probe + Stage-1 gate：

| checkpoint | pass | ego random-take | ego random-code | ego same-take | exo random-take | exo zero | action w/o private | view NMI | take NMI | ego used |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6d@091000 | no | 0.00629 | 0.03315 | 0.00354 | 0.00118 | 0.01813 | 0.02039 | 0.05724 | 0.38042 | 29 |
| v6d@092000 | no | 0.00612 | 0.03219 | 0.00345 | 0.00119 | 0.01873 | 0.02071 | 0.05764 | 0.38353 | 31 |
| v6d@093000 | no | 0.00614 | 0.03251 | 0.00341 | 0.00119 | 0.01863 | 0.02028 | 0.05536 | 0.38009 | 27 |
| v6d@094000 | no | 0.00599 | 0.03107 | 0.00340 | 0.00119 | 0.01804 | 0.02074 | 0.05574 | 0.38460 | 27 |
| v6d@095000 | no | 0.00613 | 0.03136 | 0.00353 | 0.00113 | 0.01775 | 0.02028 | 0.05671 | 0.38527 | 29 |
| v6d@096000 | no | 0.00601 | 0.03226 | 0.00347 | 0.00112 | 0.01852 | 0.01948 | 0.05955 | 0.38349 | 29 |
| v6d@097000 | no | 0.00603 | 0.03129 | 0.00343 | 0.00112 | 0.01767 | 0.02046 | 0.06096 | 0.38682 | 30 |
| v6d@098000 | no | 0.00607 | 0.03147 | 0.00345 | 0.00113 | 0.01809 | 0.01993 | 0.06035 | 0.38522 | 30 |

Gate 解释：

- 继续通过：
  - `ego_swap_random_code_delta >= 0.025`
  - `exo_swap_zero_delta >= 0.006`
  - `ego_action_without_private_saving >= 0.018`
  - `ego_private_only_gap <= 0.012`
  - `tuple_view_nmi <= 0.10`
- 持续失败：
  - `ego_swap_random_take_delta < 0.015`
  - `ego_swap_same_take_delta < 0.010`
  - `exo_swap_random_take_delta < 0.002`
  - `tuple_take_nmi > 0.35`
  - `ego_used_codes < 45`

关键观察：

- v6d 没有出现中途 checkpoint 明显变好的情况。
- `ego_swap_same_take_delta` 全程约 0.0034-0.0035，距离 0.010 阈值很远。
- `ego_swap_random_take_delta` 全程约 0.0060-0.0063，距离 0.015 阈值很远。
- `tuple_take_nmi` 稳定在 0.380-0.387，仍高于 0.35。
- `ego_used_codes` 只在 27-31 之间，仍远低于 45。

## 7. v6b Checkpoint Sweep

为验证 v6b 的 90000 是否只是历史终点，而不是更早 checkpoint 可能更好，补扫了 v6b 的 86k / 88k / 90k。

临时 patch 说明：

- 原 sxh checkpoint 的 `model_config.torch_home` 指向 `/data_all/sxh/FACT_tokenizer/checkpoints/torch_hub`。
- 为在 intern02 环境可加载，只在我们自己的 sweep 目录中复制 checkpoint 并把 `torch_home` 改为 `checkpoints/torch_hub`。
- 未修改 sxh 原目录和原 checkpoint。

结果：

| checkpoint | pass | ego random-take | ego random-code | ego same-take | exo random-take | exo zero | action w/o private | view NMI | take NMI | ego used |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v6b@086000 | no | 0.00682 | 0.03105 | 0.00342 | 0.00138 | 0.01839 | 0.01972 | 0.05798 | 0.38250 | 25 |
| v6b@088000 | no | 0.00658 | 0.03237 | 0.00341 | 0.00125 | 0.01879 | 0.02026 | 0.05588 | 0.38045 | 27 |
| v6b@090000 | no | 0.00617 | 0.03230 | 0.00346 | 0.00120 | 0.01929 | 0.02050 | 0.05614 | 0.38239 | 27 |

解释：

- v6b@086000 在 random-take 和 exo random-take 上略高，但 used codes 更低。
- v6b@090000 在 exo zero、action-without-private、same-take 上略好。
- 三个 checkpoint 都没有通过 gate。
- 没有证据说明 v6b@086000 或 v6b@088000 明显优于 v6b@090000。

因此，v6b@90000 仍可作为“历史健康参考点”，但不应被表述为严格最优 checkpoint。

## 8. 综合结论

本次 v6d 是负结果。

v6d 保住了 v6b 的一部分优点：

- random-code causality 保持通过。
- exo zero delta 保持通过。
- action-without-private 保持通过。
- view-invariance 保持通过。

但 v6d 没有解决核心问题：

- 同 take 内 action phase discrimination 没有改善。
- random-take causality 没有改善。
- exo random-take 仍弱。
- take leakage 没有下降。
- heldout ego code usage 没有打开。

这说明当前 v6d 的 action-aware continuation 并没有把 shared action token 推向“同一个 take 内不同动作阶段可区分”的方向。

## 9. 路线判断

不建议：

- 继续把当前 v6d 硬延长到 100k / 105k。
- 把 v6d 当前 recipe 作为新主线。
- 直接从头训练 v6d。
- 在当前 tokenizer 未过 Stage-1 gate 前进入 WAM / Action Head / SONIC。

建议：

- 保留 v6b 作为当前 anchor / reference。
- 将 v6d action-aware continuation 记录为负结果。
- 下一步需要改 recipe，而不是继续加 step。
- 下一轮应更直接地针对 same-take / temporal action discrimination 设计训练信号。

## 10. 下一步建议

建议开 v6e targeted sprint，而不是延长 v6d：

- 从 v6b@90000 或 v6b@086000 做小规模对照。
- 保留 v6b 的 full context + mild delta。
- 不引入 v6c 的 hard usage entropy。
- 更强地构造同 take 内 temporal/action hard negative，但避免只把 take/context 信息摊进 codebook。
- 每 1000 step 保存 checkpoint，并强制跑 heldout checkpoint sweep。
- 以 `ego_swap_same_take_delta`、`ego_swap_random_take_delta`、`tuple_take_nmi`、`ego_used_codes` 为主指标，不只看训练 loss。

一句话结论：

```text
v6d 证明“在 v6b 上直接加当前 action-aware contrast”不足以解决同 take 内动作区分问题；当前路线应回到 v6b anchor，重新设计 same-take / temporal action discrimination，而不是继续堆训练步数。
```
