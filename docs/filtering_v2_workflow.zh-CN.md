# FACT filtering_v2 工作流

`filtering_v2` 的目标是把已有 FACT 风格 NPZ split 筛成更适合第一阶段 FACT tokenizer 的训练资产。它不替代 tokenizer 训练；它只负责数据侧的多信号预标注、人工主动复核、ranker 融合、split 生成和审计。

补充记录：

- [filtering_v2 人工引导筛选方案与 500-step A/B 结论](filtering_v2_human_guided_plan_and_ab_20260707.md)
- [filtering_v2 human-guided v1 人工标注方法](filtering_v2_human_annotation_guide.zh-CN.md)
- [transition-level filter v1 人工标注流程](transition_filter_human_annotation_guide.zh-CN.md)

## 输入

已有 split：

```text
data/fact_egoexo/splits/.../train_by_take.npz
data/fact_egoexo/splits/.../heldout_by_take.npz
data/fact_egoexo/splits/.../train_labels.jsonl
data/fact_egoexo/splits/.../heldout_labels.jsonl
```

## 推荐流程

```bash
BASE=data/fact_egoexo/splits/diverse_500takes_t1p0_s1_48t_seed123_80_20
OUT=outputs/filtering_v2

conda run -n fact_tokenizer python tools/make_take_contact_sheets.py \
  --split $BASE/train_by_take.npz \
  --labels-jsonl $BASE/train_labels.jsonl \
  --out $OUT/contact_sheets_train

conda run -n fact_tokenizer python tools/extract_relevance_features.py \
  --split $BASE/train_by_take.npz \
  --split-name train \
  --labels-jsonl $BASE/train_labels.jsonl \
  --contact-sheet-manifest $OUT/contact_sheets_train/contact_sheet_manifest.csv \
  --out $OUT/relevance_v0_train.csv

conda run -n fact_tokenizer python tools/extract_ego_hand_object_features.py \
  --split $BASE/train_by_take.npz \
  --split-name train \
  --out $OUT/ego_hand_object_train.csv

conda run -n fact_tokenizer python tools/extract_exo_pose_phase_features.py \
  --split $BASE/train_by_take.npz \
  --split-name train \
  --out $OUT/exo_pose_phase_train.csv

conda run -n fact_tokenizer python tools/extract_vlm_relevance_features.py \
  --contact-sheet-manifest $OUT/contact_sheets_train/contact_sheet_manifest.csv \
  --labels-jsonl $BASE/train_labels.jsonl \
  --out-request-jsonl $OUT/vlm_requests_train.jsonl \
  --out $OUT/vlm_relevance_train.csv
```

把 `vlm_requests_train.jsonl` 交给 Qwen2.5-VL / LLaVA-NeXT-Video 后，将严格 JSON 响应保存为 JSONL，再重新运行：

```bash
conda run -n fact_tokenizer python tools/extract_vlm_relevance_features.py \
  --responses-jsonl $OUT/vlm_responses_train.jsonl \
  --out $OUT/vlm_relevance_train.csv
```

合并多信号并导出主动复核 CSV：

```bash
conda run -n fact_tokenizer python tools/merge_relevance_features.py \
  --base $OUT/relevance_v0_train.csv \
  --ego-hand-object $OUT/ego_hand_object_train.csv \
  --exo-pose-phase $OUT/exo_pose_phase_train.csv \
  --vlm $OUT/vlm_relevance_train.csv \
  --policy configs/filter_policy_v2.yaml \
  --out $OUT/take_relevance_scores_v2_train.csv

conda run -n fact_tokenizer python tools/export_active_review_csv.py \
  --scores $OUT/take_relevance_scores_v2_train.csv \
  --out $OUT/annotation_batch_v2_review.csv
```

人工只填写 `needs_human_review=1` 的样本。校准后训练/应用 ranker：

```bash
conda run -n fact_tokenizer python tools/train_relevance_ranker.py \
  --features $OUT/take_relevance_scores_v2_train.csv \
  --labels $OUT/annotation_batch_v2_labeled.csv \
  --label-column usable_for \
  --model auto \
  --out $OUT/relevance_ranker_v2.joblib

conda run -n fact_tokenizer python tools/apply_relevance_ranker.py \
  --features $OUT/take_relevance_scores_v2_train.csv \
  --ranker $OUT/relevance_ranker_v2.joblib \
  --out $OUT/take_relevance_ranked_v2_train.csv
```

对 train 和 heldout 分别生成 ranked CSV 后合并，再生成 policy/strict/balanced 三个 split：

```bash
conda run -n fact_tokenizer python tools/build_filtered_split.py \
  --ranked $OUT/take_relevance_ranked_v2_all.csv \
  --policy configs/filter_policy_v2.yaml \
  --out $OUT/filtered_split_v2.json

conda run -n fact_tokenizer python tools/build_filtered_split.py \
  --ranked $OUT/take_relevance_ranked_v2_all.csv \
  --policy configs/filter_policy_v2.yaml \
  --fact-main-mode strict \
  --out $OUT/filtered_split_v2_fact_main_strict.json

conda run -n fact_tokenizer python tools/build_filtered_split.py \
  --ranked $OUT/take_relevance_ranked_v2_all.csv \
  --policy configs/filter_policy_v2.yaml \
  --fact-main-mode balanced \
  --out $OUT/filtered_split_v2_fact_main_balanced.json
```

## 接入当前 FACT NPZ 构建

如果要复用当前仓库的 `scripts/prepare_fact_transition48_split.py`，先从 split 导出 selected JSONL：

```bash
conda run -n fact_tokenizer python tools/export_filtered_selected_takes.py \
  --filtered-split $OUT/filtered_split_v2.json \
  --labels-jsonl $BASE/train_labels.jsonl \
  --include-splits train \
  --include-buckets fact_main \
  --out data/egoexo4d/fact_debug/selected_takes_filtering_v2_fact_main_train.jsonl
```

然后把这个 JSONL 作为 `--selected-jsonl` 传给现有 preparation 脚本。

## 原则

- `fact_main` 优先 hand-object/contact 和 phase diversity。
- `loco_aux` 只作为 capped ablation，默认每 take 12 transitions。
- `diagnostic_candidate` 不直接混入主训练。
- VLM 只是 teacher/judge，不单独决定最终 split。
- ranker 只学习人工标注和数据侧特征，不使用 tokenizer loss/probe/MSE 作为训练标签。
- tokenizer A/B 只作为固定参数下的下游验证，不自动反向校正 filter。

## Soft filter / weighted sampling 接入

当前实验证明，hard-filter-only 会明显损失动作和任务多样性。因此推荐把 human-guided filter 的输出先转成 take-level 采样权重，而不是直接删除大部分 takes。

### weighted soft filter v2

`weighted soft filter v2` 是当前推荐版本。它相对 v1 的变化：

- 不只按 `ranker_bucket` 给固定权重，而是使用 `ranker_prob_*`、`interaction_score`、`object_motion_proxy`、`phase_diversity_score_v2`、`motion_state_change_score` 等连续特征。
- 对 `scene_only_score`、`prob_discard`、`diagnostic` 风险做更明确的 penalty。
- 对 parent task / task 做温和的 diversity re-balance，避免权重质量过度集中到少数任务。
- `discard` 默认仍为 0；`loco_aux` 和 `diagnostic_candidate` 保留低权重，避免 hard filter 带来的 codebook 多样性损失。

生成 v2 take 权重 CSV：

```bash
OUT=outputs/filtering_v2_annotation_review_no_vlm

conda run -n fact_tokenizer python tools/build_take_weight_csv.py \
  --ranked $OUT/take_relevance_ranked_v2_all.csv \
  --out $OUT/take_quality_weights_v2_train.csv \
  --split train \
  --preset v2
```

当前 `take_quality_weights_v2_train.csv` 审计结果：

| bucket | takes | positive takes | mean positive weight |
| --- | ---: | ---: | ---: |
| `tokenizer_main` | 94 | 94 | 0.5962 |
| `loco_aux` | 105 | 105 | 0.1857 |
| `diagnostic_candidate` | 93 | 93 | 0.1408 |
| `discard` | 50 | 0 | 0.0000 |

整体：342 个 train takes 全部匹配训练数据，其中 292 个正权重；positive mean weight 为 0.3036，effective takes 约 194.6。

### weighted soft filter v1 复现

```bash
OUT=outputs/filtering_v2_annotation_review_no_vlm

conda run -n fact_tokenizer python tools/build_take_weight_csv.py \
  --ranked $OUT/take_relevance_ranked_v2_all.csv \
  --out $OUT/take_quality_weights_v1_train.csv \
  --split train \
  --preset v1
```

默认权重含义：

| bucket | 默认作用 |
| --- | --- |
| `tokenizer_main` | 主训练样本，高权重 |
| `loco_aux` | 少量身体/空间阶段辅助，低权重保留 |
| `diagnostic_candidate` | 边界样本，极低权重保留多样性 |
| `discard` | 默认 0 权重，不参与采样 |

单卡稳定训练脚本：

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_NAME=weighted_filter_v1_from_v6b_single_fg_$(date +%Y%m%d_%H%M%S) \
TRAIN_NPZ=outputs/fact_tokenizer/nofilter_t1p0_train_by_take_npy_mmap \
RESUME_CHECKPOINT=outputs/fact_tokenizer/v6b_transition48_delta_full_from_v5p_8gpu_20260620_2200/fact_tokenizer.ckpt \
STEPS=91000 \
PER_GPU_BATCH=16 \
NUM_WORKERS=0 \
PREFETCH_FACTOR=2 \
TAKE_WEIGHT_CSV=$OUT/take_quality_weights_v1_train.csv \
scripts/run_fact_transition48_v6b_delta_full_single_gpu_train.sh
```

使用 v2 时，把 `TAKE_WEIGHT_CSV` 改为：

```bash
TAKE_WEIGHT_CSV=$OUT/take_quality_weights_v2_train.csv
```

同 heldout probe：

```bash
RUN_DIR=outputs/fact_tokenizer/weighted_filter_v1_from_v6b_single_fg_YYYYMMDD_HHMMSS

conda run -n fact_tokenizer python scripts/probe_fact_action_tokens.py \
  --checkpoint $RUN_DIR/fact_tokenizer.ckpt \
  --input-npz outputs/fact_tokenizer/nofilter_t1p0_heldout_by_take_npy_mmap \
  --output-dir $RUN_DIR/probe_heldout_weighted_filter_v1 \
  --source-view-keys ego exo \
  --batch-size 8 \
  --num-workers 0 \
  --resize 224 \
  --device cuda
```

## Light augmentation 接入

训练脚本支持 `LIGHT_AUGMENT=1`，对应 `scripts/train_fact_npz_debug.py --light-augment`。本轮 2026-07-08 对照显示，light augmentation 虽然没有提高 train loss，但降低了 action top1，并显著恶化 view-invariance。因此当前不建议作为主线，只建议作为后续小权重辅助实验。

```bash
CUDA_VISIBLE_DEVICES=1 \
RUN_NAME=nofilter_t1p0_lightaug_from_v6b_single_fg_$(date +%Y%m%d_%H%M%S) \
TRAIN_NPZ=outputs/fact_tokenizer/nofilter_t1p0_train_by_take_npy_mmap \
RESUME_CHECKPOINT=outputs/fact_tokenizer/v6b_transition48_delta_full_from_v5p_8gpu_20260620_2200/fact_tokenizer.ckpt \
STEPS=91000 \
PER_GPU_BATCH=16 \
NUM_WORKERS=0 \
PREFETCH_FACTOR=2 \
LIGHT_AUGMENT=1 \
scripts/run_fact_transition48_v6b_delta_full_single_gpu_train.sh
```

## 2026-07-08 对比实验结论

完整结果见：

- `outputs/fact_tokenizer/ab_logs/filter_weight_aug_ab_20260708_summary.json`
- `outputs/fact_tokenizer/ab_logs/filter_weight_aug_ab_20260708_metrics.csv`
- `docs/filtering_v2_human_guided_plan_and_ab_20260707.md`

简要结论：

- hard filter 有轻微信号收益，但删掉太多多样性，不适合直接作为主训练方案。
- weighted soft filter 保住了 code usage 和 view-invariance，是下一版 filter 的主线。
- light augmentation 当前会恶化 view-invariance，不作为优先方向。
- 后续完善 filter 应继续基于人工标注和数据侧特征，不用 tokenizer probe 反向改 filter 标签。

## 2026-07-09 weighted soft filter v2 验证结论

v2 验证已完成：

- 训练 run：`outputs/fact_tokenizer/weighted_filter_v2_from_v6b_single_fg_20260709_132629`
- probe run：`outputs/fact_tokenizer/weighted_filter_v2_from_v6b_single_fg_20260709_132629/probe_heldout_weighted_filter_v2_20260709_133340`
- 权重文件：`outputs/filtering_v2_annotation_review_no_vlm/take_quality_weights_v2_train.csv`
- 汇总文件：`outputs/fact_tokenizer/ab_logs/filter_weight_aug_ab_20260708_metrics.csv`

简要结果：

| metric | no-filter | weighted v1 | weighted v2 |
| --- | ---: | ---: | ---: |
| action top1 last100 | 0.3814 | 0.3991 | 0.3983 |
| ego/exo used codes | 28 / 31 | 28 / 32 | 26 / 31 |
| ego self random-code gap | 0.03145 | 0.03222 | 0.03227 |
| ego swap random-code gap | 0.02906 | 0.02982 | 0.03011 |
| view-invariance NMI sqrt | 0.06835 | 0.06595 | 0.07470 |

结论：v2 相对 no-filter 有一点 action/causality 信号，但没有超过 weighted v1；同时 reconstruction loss 更差、ego code usage 下降、view-invariance 变差。因此当前不应把 v2 作为“filter 已有效”的证据，只能说明 soft weighting 方向比 hard deletion 更值得继续。下一步应优先补充数据侧 hand/object/contact、phase、sync 特征，再做 v3，而不是继续只调 bucket 权重。

## Transition-Level Filter v1

take-level filter 的几轮验证没有形成稳定收益，因此下一版改为筛 transition/window。核心变化：

- 不再问“这个 take 是否适合训练”，而是问“这个 take 内哪些 48-frame transition 有 shared action dynamics”。
- 保留好 take 内的高质量片段，也允许普通 take 中的好片段进入训练。
- 训练时使用 `--transition-weight-csv`，在 take-grouped batch 内按 transition 权重采样。

已新增工具：

| 工具 | 作用 |
| --- | --- |
| `tools/extract_transition_filter_features.py` | 不使用 VLM，提取 transition-level motion/sync/scene-only proxy |
| `tools/export_transition_review_csv.py` | 分层抽样导出人工 review CSV |
| `tools/make_transition_contact_sheets.py` | 为 review rows 生成 ego/exo 首末帧图片 |
| `tools/build_transition_weight_csv.py` | 合并自动/人工标签，生成训练用 `transition_weight` CSV |

当前自动版本产物：

```text
outputs/filtering_v2_transition_filter/transition_features_v1_train.csv
outputs/filtering_v2_transition_filter/transition_weights_v1_train.csv
outputs/filtering_v2_transition_filter/transition_review_batch_v1_train_with_sheets.csv
outputs/filtering_v2_transition_filter/transition_review_sheets_v1_train/
```

训练脚本已支持：

```bash
TRANSITION_WEIGHT_CSV=outputs/filtering_v2_transition_filter/transition_weights_v1_train.csv \
scripts/run_fact_transition48_v6b_delta_full_single_gpu_train.sh
```

详细人工标注方法见 `docs/transition_filter_human_annotation_guide.zh-CN.md`。
