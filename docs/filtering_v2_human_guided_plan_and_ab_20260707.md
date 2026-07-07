# filtering_v2 人工引导筛选方案与 500-step A/B 结论

日期：2026-07-07

本文记录 `filtering_v2` 的下一步完善思路，以及 `filtering_v2 fact_main` 与 no-filter baseline 的最小验证实验结论。本文只覆盖 FACT tokenizer 第一阶段数据筛选，不扩展到 WAM、Action Head 或机器人部署阶段。

## 1. 核心结论

当前最合理的方向是：

```text
人工定义筛选方向和边界
  -> 自动模型提取多信号并推广到全量数据
  -> 小规模人工复核校准模型
  -> 生成 filtered split / NPZ
  -> 用 FACT tokenizer A/B probe 判断筛选是否真的有效
```

人工不需要全量标注，也不需要标注 FACT token。人工的作用是给出少量高价值样本的筛选判断、边界案例和失败原因，让 relevance ranker / VLM judge / detector features 学会“什么样的数据适合训练 shared action token”。

## 2. 为什么需要 human-guided filtering

这次 500-step A/B 显示，当前 minimal filter 已经有正向信号，但并不完整：

- filtered 数据提高了训练端跨视角 action assignment 一致性。
- filtered 数据没有在 heldout correct reconstruction 上赢过 no-filter。
- ego-side random-take / global-shuffle causality gap 没有提升，说明当前筛选可能偏向“exo 清楚、交互强”，但没有充分保证 ego 端可预测性和分布覆盖。

因此，下一版 filter 不应该继续只调 motion proxy 或固定阈值，而应该把人工判断作为校准信号，引导自动模型推广。

## 3. 人工需要标注什么

### 3.1 take-level 校准标签

第一轮人工主要标 take-level，不做密集 token 标注。

推荐字段：

```csv
take_uid
ego_hand_visibility
exo_body_visibility
object_interaction
phase_diversity
scene_only_risk
ego_exo_sync_quality
take_relevance
usable_for
confidence
reason
```

字段定义：

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `ego_hand_visibility` | `0/1/2` | ego 中手或交互区域不可见 / 部分可见 / 清楚可见 |
| `exo_body_visibility` | `0/1/2` | exo 中身体或空间关系不可见 / 部分可见 / 清楚可见 |
| `object_interaction` | `0/1/2` | 无交互 / 可疑或弱交互 / 明确手物或人-物交互 |
| `phase_diversity` | `0/1/2` | 单一状态 / 轻微阶段变化 / 明显多阶段变化 |
| `scene_only_risk` | `0/1/2` | scene shortcut 风险低 / 中 / 高 |
| `ego_exo_sync_quality` | `ok/minor_issue/bad` | ego-exo 同步质量 |
| `take_relevance` | `A/B/C/D/E/F` | take 类型，见下表 |
| `usable_for` | `tokenizer_main/loco_aux/discard/diagnostic_candidate` | 最终用途 |
| `confidence` | `0-1` | 人工判断置信度 |
| `reason` | free text | 一句话说明原因 |

`take_relevance` 建议定义：

| 类别 | 名称 | 用途 |
| --- | --- | --- |
| `A_interaction_rich` | ego/exo 都有较清楚交互和阶段变化 | `tokenizer_main` 首选 |
| `B_loco_body` | exo 身体/空间运动清楚，但手物交互较弱 | 少量进入 `loco_aux` |
| `C_active_view_only` | ego 有运动但交互语义弱或 exo 帮助有限 | 多数作为 diagnostic |
| `D_scene_only` | 背景、头动、场景变化主导 | discard |
| `E_fine_dexterous` | 细粒度手部动作，但 coarse phase 不明显 | diagnostic 或少量 main |
| `F_bad_or_unclear` | 质量差、同步差或不可判断 | discard |

### 3.2 transition-level 少量标签

第二层只对一小批 take 做 transition-level 标签，用于改进 transition selection 和 same-take hard negatives。

推荐字段：

```csv
take_uid
transition_id
timestamp
contact_state
phase_label
ego_action_visible
exo_body_motion_visible
keep_for_fact_main
hard_negative_candidate
note
```

推荐枚举：

- `contact_state`: `no_contact / approach / contact_manipulate / release / unclear`
- `phase_label`: `approach / reach / grasp_contact / manipulate_carry / place_release / idle_scene`
- `ego_action_visible`: `0/1/2`
- `exo_body_motion_visible`: `0/1/2`
- `keep_for_fact_main`: `yes/no/diagnostic`
- `hard_negative_candidate`: `yes/no`

## 4. 建议标注规模

当前可用 take 规模约 427 个，第一轮不需要全量人工标注。

建议第一轮：

| 标注对象 | 规模 | 采样方式 |
| --- | ---: | --- |
| take-level | `120-160` takes | 高分、边界、冲突、false drop、随机抽检、parent task 覆盖 |
| transition-level | `500-900` transitions | 从 `60-80` 个 takes 中每个抽 `8-12` 个 transition |

take-level 主动学习采样建议：

- 自动高分 `fact_main`：约 40 个，用于检查 false positive。
- 阈值附近边界样本：约 40 个，用于校准 policy。
- 模型/规则冲突样本：约 30 个，例如 VLM 高分但 contact proxy 低。
- 自动 discard 样本：约 20 个，用于检查 false drop。
- 每个 parent task 至少抽 2-3 个，避免任务分布偏置。

## 5. 自动信号设计

下一版 `filtering_v2` 应把当前 cheap proxy 升级为多信号特征。

### 5.1 Ego hand / object / contact

新增或强化字段：

```text
ego_hand_score
ego_hand_visibility_prob
object_presence_score
object_motion_score
hand_object_contact_score
interacting_object_score
contact_state_change_score
```

目标：

- 保证进入 `fact_main` 的样本在 ego 端有可预测的交互线索。
- 减少“exo 很清楚但 ego 看不到关键动作”的样本。
- 降低纯头动、纯背景运动对 shared action token 的污染。

可选模型：

- EgoHOS：egocentric hand-object segmentation。
- ORMNet / CaRe-Ego：contact-aware egocentric interaction。
- GroundingDINO + SAM2：开放词表物体检测和视频分割。

### 5.2 Exo body / loco / phase

新增或强化字段：

```text
exo_body_visibility_score
exo_pose_confidence
exo_body_motion_score_v2
body_phase_diversity_score
loco_motion_score
pose_state_change_score
```

目标：

- 利用 exo 提供 whole-body interaction 和空间关系。
- 防止 `loco_aux` 变成纯背景或纯相机运动数据。
- 支持少量 capped loco 数据辅助 exo branch，但不能替代 `fact_main`。

优先使用 Ego-Exo4D 原生 pose / body annotations；不足时再补 OpenPose、MediaPipe Holistic 等通用姿态模型。

### 5.3 Temporal phase diversity

新增或强化字段：

```text
phase_diversity_score_v2
motion_state_change_score
contact_state_change_score
pose_state_change_score
feature_cluster_change_score
```

目标：

- 过滤同一状态重复片段。
- 让每个保留 take 内覆盖 approach、reach、contact、manipulate、place、release 等阶段。
- 为 same-take hard negative 构造提供可靠候选。

### 5.4 VLM relevance judge

VLM 不单独决定 split，而是作为 teacher / judge。

建议输出字段：

```text
vlm_prob_tokenizer_main
vlm_prob_loco_aux
vlm_prob_discard
vlm_prob_diagnostic_candidate
vlm_take_relevance
vlm_usable_for
vlm_confidence
vlm_reason
```

输入：

```text
contact sheet
take_uid
parent_task_name
task_name
optional short clip
```

输出必须是严格 JSON，便于自动解析和 ranker 训练。

## 6. 模型推广与融合策略

自动模型不直接输出“保留/丢弃”二值判断，而是输出 calibrated probabilities：

```text
prob_tokenizer_main
prob_loco_aux
prob_discard
prob_diagnostic_candidate
auto_confidence
auto_disagreement_score
auto_review_priority
```

建议生成多个 split，而不是单一 filtered set：

| split | 目标 |
| --- | --- |
| `fact_main_strict` | 高 precision，优先 hand-object/contact + phase diversity |
| `fact_main_balanced` | 保留更多 ego 可预测样本，避免筛后分布过窄 |
| `fact_main_plus_loco_capped` | `fact_main` 加少量 capped `loco_aux`，测试 exo/body 辅助是否有益 |
| `diagnostic_candidate` | 边界、冲突、疑似 false drop 样本，只用于分析 |
| `discard` | 明显 scene-only、质量差、同步差或无阶段变化 |

## 7. 针对当前 A/B 结果的修正重点

本次实验显示 filtered 的训练端 cross-view agreement 明显更高，但 heldout ego 端没有赢。因此下一版 policy 应加入：

- `ego_visibility_floor`：ego hand/contact 不够清楚的 take 不进入 `fact_main_strict`。
- `ego_predictability_score`：优先保留从 ego 当前和短历史能预测未来 shared action 的 transition。
- `distribution_quota`：按 parent task、scene、take type 设置限额，防止筛后数据过窄。
- `phase_balanced_transition_selection`：每个 take 内覆盖不同 phase，而不是只选高 motion transition。
- `false_drop_review`：抽查自动 discard 中可能有用的样本。
- `conflict_review`：VLM、detector、pose、proxy 互相冲突时进入人工复核。
- `loco_aux_cap`：如果加入 loco 辅助，每个 take transition 数量 capped，避免 exo/body 信号淹没 hand-object 主目标。

## 8. 验证实验记录：500-step A/B

### 8.1 实验设置

共同设置：

- 起点 checkpoint：`outputs/fact_tokenizer/v6b_transition48_delta_full_from_v5p_8gpu_20260620_2200/fact_tokenizer.ckpt`
- 训练步数：`90000 -> 90499`，共 500 steps。
- 设备：单卡训练，filtered / no-filter 分别在 GPU0 / GPU1 并行。
- batch size：8。
- heldout probe：`scripts/probe_fact_action_tokens.py`
- probe eval paths：`ego_self ego_swap exo_self exo_swap`

数据：

| 条件 | train data | heldout data |
| --- | --- | --- |
| filtered | `outputs/filtering_v2_minimal_real/filtered_npy/fact_main/train_by_take` | `outputs/filtering_v2_minimal_real/filtered_npy/fact_main/heldout_by_take` |
| no-filter | `outputs/filtering_v2_minimal_real/nofilter_npy/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take` | `outputs/filtering_v2_minimal_real/nofilter_npy/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take` |

输出：

| 条件 | run dir |
| --- | --- |
| filtered | `outputs/fact_tokenizer/filtering_v2_fact_main_from_v6b_to_90500_1gpu_20260707_164437` |
| no-filter | `outputs/fact_tokenizer/nofilter_from_v6b_to_90500_1gpu_20260707_164437` |
| summary | `outputs/fact_tokenizer/ab_logs/ab_500step_20260707_164437_summary.json` |

### 8.2 训练端结果

| metric | filtered | no-filter | 观察 |
| --- | ---: | ---: | --- |
| `loss` | 2.1330 | 2.0283 | no-filter 更低 |
| `ego_self/feature_mse` | 0.4541 | 0.3499 | no-filter 更好 |
| `ego_swap/feature_mse` | 0.4609 | 0.3495 | no-filter 更好 |
| `exo_self/feature_mse` | 0.1693 | 0.2269 | filtered 更好 |
| `exo_swap/feature_mse` | 0.1696 | 0.2267 | filtered 更好 |
| `action_top1_agreement` | 0.5625 | 0.2500 | filtered 明显更好 |
| used codes | 35 / 64 | 37 / 64 | 接近 |
| usage fraction | 0.5469 | 0.5781 | no-filter 略高 |

训练端结论：

- filtered 明显提高 ego/exo action assignment 一致性。
- filtered 让 exo reconstruction 更好，但 ego reconstruction 变差。
- 这说明当前 filter 可能强化了跨视角一致动作原型，但数据分布或 ego 可预测性仍有问题。

### 8.3 Heldout probe correct MSE

| path | filtered correct | no-filter correct | filtered - no-filter |
| --- | ---: | ---: | ---: |
| `ego_self` | 0.4236 | 0.3896 | +0.0340 |
| `ego_swap` | 0.4262 | 0.3924 | +0.0338 |
| `exo_self` | 0.2069 | 0.1937 | +0.0132 |
| `exo_swap` | 0.2066 | 0.1935 | +0.0131 |

heldout correct MSE 结论：

- filtered 在 heldout reconstruction 上没有赢。
- 这不是直接否定 filter，但说明当前版本还不能作为最终数据筛选器。

### 8.4 Heldout causality gaps

gap 定义：`ablation MSE - correct MSE`。gap 越大，说明替换或破坏 action token 后性能下降越明显，action token 对 decoder 越重要。

| path | mode | filtered gap | no-filter gap | filtered - no-filter |
| --- | --- | ---: | ---: | ---: |
| `ego_self` | `random_code` | 0.03595 | 0.03595 | +0.00000 |
| `ego_self` | `random_take` | 0.00666 | 0.00906 | -0.00240 |
| `ego_self` | `global_shuffle` | 0.00641 | 0.00897 | -0.00256 |
| `ego_self` | `zero` | 0.00433 | 0.00734 | -0.00302 |
| `ego_swap` | `random_code` | 0.03335 | 0.03319 | +0.00016 |
| `ego_swap` | `random_take` | 0.00414 | 0.00592 | -0.00178 |
| `ego_swap` | `global_shuffle` | 0.00406 | 0.00586 | -0.00180 |
| `ego_swap` | `zero` | 0.00173 | 0.00458 | -0.00286 |
| `exo_self` | `random_code` | 0.02776 | 0.02659 | +0.00117 |
| `exo_self` | `random_take` | 0.00132 | 0.00095 | +0.00037 |
| `exo_self` | `global_shuffle` | 0.00128 | 0.00087 | +0.00041 |
| `exo_self` | `zero` | 0.02232 | 0.01618 | +0.00615 |
| `exo_swap` | `random_code` | 0.02806 | 0.02678 | +0.00127 |
| `exo_swap` | `random_take` | 0.00183 | 0.00106 | +0.00077 |
| `exo_swap` | `global_shuffle` | 0.00161 | 0.00106 | +0.00055 |
| `exo_swap` | `zero` | 0.02262 | 0.01638 | +0.00625 |

causality 结论：

- `random_code` gap：filtered 在 ego 上基本持平，在 exo 上略好。
- `random_take/global_shuffle/zero`：filtered 在 exo 上更好，但在 ego 上更差。
- 当前 filter 更偏向增强 exo-side action dependence，还没有改善 ego-side action discrimination。

## 9. 总体判断

当前 minimal `filtering_v2 fact_main` 有价值，但不是最终版：

- 正向：跨视角 action code 一致性明显增强。
- 正向：exo-side action dependence 和 zero-action sensitivity 更强。
- 风险：ego heldout reconstruction 明显变差。
- 风险：ego-side random-take / global-shuffle causality gap 没有提升。
- 风险：filtered split 可能过窄，或包含 exo 清楚但 ego 不够可预测的样本。

因此，下一步应该继续完善 filter 流程，而不是直接长训当前 filtered split。

## 10. 下一步执行计划

1. 固定人工标注 schema，导出 active-review CSV。
2. 对 120-160 个 takes 做 take-level 人工校准。
3. 对 500-900 个 transitions 做少量 phase/contact 标注。
4. 接入 Ego hand/object/contact、Exo pose/body、temporal phase diversity、VLM JSON judge。
5. 训练 relevance ranker，输出 calibrated bucket probabilities。
6. 生成 `fact_main_strict`、`fact_main_balanced`、`fact_main_plus_loco_capped` 三个候选 split。
7. 对每个 split materialize NPY/NPZ。
8. 从同一 v6b checkpoint 跑 500-step A/B quick gate。
9. 对最好的 split 再跑 2k-5k steps 验证趋势是否稳定。
10. 只有当 heldout correct MSE 不明显恶化，且 ego/exo causality gap 同时改善时，才把该 filter 版本作为后续主线。

## 11. 判定标准

一个改进后的 filter 版本应满足：

- `action_top1_agreement` 高于 no-filter。
- heldout `ego_self/ego_swap/exo_self/exo_swap` correct MSE 至少不明显恶化。
- ego 和 exo 的 `random_code_gap`、`random_take_gap`、`global_shuffle_gap` 有同步改善。
- heldout code usage 不塌缩，ego/exo used codes 接近或高于当前水平。
- take/scene shortcut 风险下降。
- filtered split 的 parent task / scene / phase 分布不过窄。

