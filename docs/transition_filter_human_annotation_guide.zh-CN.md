# Transition-Level Filter v1 人工标注流程

本文档记录 FACT tokenizer 的片段级筛选流程。目标不是再判断一个 take 是否整体可用，而是判断每个 48-frame transition/window 是否包含对 shared action token 有价值的交互动态。

## 1. 标签定义

只标 4 类：

| label | 含义 | 默认训练权重 |
| --- | --- | ---: |
| `main_interaction` | 明确 hand/object/contact 或身体-环境交互，动作会改变接触关系、物体状态或任务阶段 | 1.00 |
| `phase_context` | 接近目标、准备、释放后整理、阶段切换等上下文；有动作阶段信息但交互不如主片段明确 | 0.35 |
| `loco_only` | 主要是走动、身体移动、视角移动，无明确对象操作；只少量辅助保留 | 0.10 |
| `discard` | idle、纯背景、严重遮挡、不同步、无可见动作变化、scene-only | 0.00 |

标注时优先看：ego 是否能看到操作、exo 是否能对应同一动作阶段、是否有接触/物体/身体姿态阶段变化。不要因为任务名称相关就标 `main_interaction`，必须看当前 transition 图像本身。

## 2. 自动生成 transition proxy

第一版不使用 VLM，只用首末帧差分和 Ego/Exo 同步 proxy：

```bash
TRAIN=outputs/fact_tokenizer/nofilter_t1p0_train_by_take_npy_mmap
OUT=outputs/filtering_v2_transition_filter
LABELS=data/fact_egoexo/splits/diverse_500takes_t1p0_s1_48t_seed123_80_20/train_labels.jsonl

/home/sxh/.conda/envs/fact_tokenizer/bin/python -u tools/extract_transition_filter_features.py \
  --input-npz "$TRAIN" \
  --labels-jsonl "$LABELS" \
  --out "$OUT/transition_features_v1_train.csv" \
  --split-name train \
  --chunk-size 4096 \
  --spatial-stride 8
```

输出文件包含：

- `ego_motion_score` / `exo_motion_score`：首末帧整体运动强度。
- `ego_local_motion_score` / `exo_local_motion_score`：局部区域运动强度。
- `object_motion_score`：ego 中心/局部运动 proxy。
- `phase_change_score`：同一 take 内运动阶段变化 proxy。
- `ego_exo_sync_score`：ego/exo 同时有动作变化的程度。
- `scene_only_score`：更像全局相机/背景变化、缺少局部交互的风险。
- `interaction_score`：综合交互分数。
- `auto_label`：自动初筛标签。
- `transition_weight`：自动权重。

当前一次生成结果：

```text
16416 rows
main_interaction=1838
phase_context=7733
loco_only=3569
discard=3276
```

## 3. 导出人工 review CSV

默认抽 500 个 transition，按标签分层，并限制每个 take 最多 4 个样本：

```bash
/home/sxh/.conda/envs/fact_tokenizer/bin/python -u tools/export_transition_review_csv.py \
  --features "$OUT/transition_features_v1_train.csv" \
  --out "$OUT/transition_review_batch_v1_train.csv" \
  --max-review 500 \
  --max-per-take 4
```

再生成图片表：

```bash
/home/sxh/.conda/envs/fact_tokenizer/bin/python -u tools/make_transition_contact_sheets.py \
  --input-npz "$TRAIN" \
  --review-csv "$OUT/transition_review_batch_v1_train.csv" \
  --out-dir "$OUT/transition_review_sheets_v1_train" \
  --out-csv "$OUT/transition_review_batch_v1_train_with_sheets.csv" \
  --thumb-size 144 \
  --progress-every 100
```

人工标注主要填写：

| column | 是否必填 | 填写方式 |
| --- | --- | --- |
| `transition_label` | 必填 | `main_interaction` / `phase_context` / `loco_only` / `discard` |
| `confidence` | 建议填 | `1.0` 明确，`0.75` 大致确定，`0.5` 边界样本 |
| `reason` | 可选 | 简短说明，可写中文，不参与训练 |
| `notes` | 可选 | 任何备注，可空 |

查看 `contact_sheet_path` 对应图片。图片每行是一个 transition，包含 ego/exo 的首帧和末帧。标注时以图片为主，proxy 分数只作为参考。

## 4. 标注判断细则

标 `main_interaction`：

- ego 中能看到手/工具/物体/身体部位正在与目标交互。
- 有明显接触开始、稳定接触、释放、放置、抓取、按压、切、倒、推拉、击球、攀爬等。
- exo 也能看到同一动作阶段，或者至少没有明显不同步。

标 `phase_context`：

- 有走向目标、准备拿取、动作完成后的整理、从一个阶段切到另一个阶段。
- 交互不强，但对理解 action phase 有帮助。
- ego/exo 大致同步。

标 `loco_only`：

- 主要是身体移动、走路、跑动、相机跟随、站位变化。
- 没有清晰 hand-object/contact，但保留少量可帮助身体/空间阶段建模。

标 `discard`：

- 基本 idle 或无动作变化。
- 纯背景、纯相机晃动、场景切换。
- ego 看不到可学习操作，exo 也无法补充。
- ego/exo 明显不同步或不是同一事件。
- 严重遮挡、黑屏、画面损坏。

## 5. 从人工标注生成训练权重

如果还没有人工标注，可以直接用自动标签生成权重：

```bash
/home/sxh/.conda/envs/fact_tokenizer/bin/python -u tools/build_transition_weight_csv.py \
  --features "$OUT/transition_features_v1_train.csv" \
  --out "$OUT/transition_weights_v1_train.csv"
```

如果已经在 `transition_review_batch_v1_train_with_sheets.csv` 填好人工标签：

```bash
/home/sxh/.conda/envs/fact_tokenizer/bin/python -u tools/build_transition_weight_csv.py \
  --features "$OUT/transition_features_v1_train.csv" \
  --annotations "$OUT/transition_review_batch_v1_train_with_sheets.csv" \
  --out "$OUT/transition_weights_v1_train_human.csv" \
  --min-confidence 0.5
```

脚本会优先使用人工 `transition_label`，没有标注的 transition 回退到 `auto_label`。

## 6. 接入 tokenizer 训练

单卡验证命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_NAME=transition_filter_v1_from_v6b_single_fg_$(date +%Y%m%d_%H%M%S) \
TRAIN_NPZ=outputs/fact_tokenizer/nofilter_t1p0_train_by_take_npy_mmap \
RESUME_CHECKPOINT=outputs/fact_tokenizer/v6b_transition48_delta_full_from_v5p_8gpu_20260620_2200/fact_tokenizer.ckpt \
STEPS=91000 \
PER_GPU_BATCH=16 \
NUM_WORKERS=0 \
PREFETCH_FACTOR=2 \
TRANSITION_WEIGHT_CSV=outputs/filtering_v2_transition_filter/transition_weights_v1_train.csv \
scripts/run_fact_transition48_v6b_delta_full_single_gpu_train.sh
```

如果使用人工校准版本，把 `TRANSITION_WEIGHT_CSV` 改成：

```bash
TRANSITION_WEIGHT_CSV=outputs/filtering_v2_transition_filter/transition_weights_v1_train_human.csv
```

训练脚本会：

1. 用 `transition_weight` 匹配每个 `sample_id` / `row_index`。
2. 自动从 positive transition mass 推导 take 采样概率。
3. 在 take-grouped batch 内按 transition 权重采样具体片段。
4. 权重为 0 的 transition 不会被采到。

## 7. 验证标准

下一轮对比建议：

| variant | 含义 |
| --- | --- |
| `nofilter_baseline` | 原始 full transition |
| `weighted_soft_filter_v1` | take-level soft filter |
| `transition_filter_v1_auto` | 自动 transition-level filter |
| `transition_filter_v1_human` | 人工校准后的 transition-level filter |

只有同时满足以下趋势，才说明片段级 filter 真有效：

- `action_top1_agreement` 上升。
- heldout random-code gap 上升。
- temporal-offset gap 上升，说明 token 更关心动作阶段。
- view-invariance NMI 不恶化。
- code usage 不下降。
- correct reconstruction MSE 不明显恶化。
