# FACT filtering_v2 工作流

`filtering_v2` 的目标是把已有 FACT 风格 NPZ split 筛成更适合第一阶段 FACT tokenizer 的训练资产。它不替代 tokenizer 训练；它只负责数据侧的多信号预标注、人工主动复核、ranker 融合、split 生成和审计。

补充记录：

- [filtering_v2 人工引导筛选方案与 500-step A/B 结论](filtering_v2_human_guided_plan_and_ab_20260707.md)

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
  --out $OUT/relevance_ranker_v2.json

conda run -n fact_tokenizer python tools/apply_relevance_ranker.py \
  --features $OUT/take_relevance_scores_v2_train.csv \
  --ranker $OUT/relevance_ranker_v2.json \
  --out $OUT/take_relevance_ranked_v2_train.csv
```

对 train 和 heldout 分别生成 ranked CSV 后合并，再生成 split：

```bash
conda run -n fact_tokenizer python tools/build_filtered_split.py \
  --ranked $OUT/take_relevance_ranked_v2_all.csv \
  --policy configs/filter_policy_v2.yaml \
  --out $OUT/filtered_split_v2.json
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
- 最终是否推进，看 FACT tokenizer probe，而不是只看标注准确率。
