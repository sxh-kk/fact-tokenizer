# FACT v7 gold300 标注交付与封存手册

本文面向数据管理员和项目负责人，说明如何把已冻结的 FACT v7 gold300 样本安全地交给两位标注者、接收结果并保持 locked test 隔离。标注者只应收到公开 delivery，不得接触 canonical manifest、split、take UID 或 sealed mapping。

## 1. 角色与工作量

| 角色 | 工作量 | 权限 |
| --- | ---: | --- |
| 标注者 A | 300 条 | 只能看到自己的盲化图片和公开 `task.csv` |
| 标注者 B | 60 条 dual 子集 | 只能看到自己的盲化图片和公开 `task.csv` |
| 数据管理员 | 生成、分发、回收、校验、映射和封存 | 独占 admin/sealed 与 locked canonical 标签 |
| 模型开发人员 | 不参与 gold 标注 | 最终 campaign 前不得查看 locked canonical 标签 |

每条包含两个相互独立的标签：一个 8 类 `effect_label` 和一个 5 类 `contact_label`。gold 不回传 effect encoder，只用于固定线性 probe、弱标签校准和评价。

## 2. 冻结后的 canonical 文件

先从已经审计并冻结的三个候选 manifest 生成 canonical gold 根目录：

```bash
python scripts/prepare_fact_gold300.py \
  --train-candidates <TRAIN_CANDIDATES.jsonl> \
  --heldout-candidates <HELDOUT_CANDIDATES.jsonl> \
  --short73-candidates <SHORT73_CANDIDATES.jsonl> \
  --diagnostic500 <DIAGNOSTIC500.csv> \
  --required-locked-takes <PARTICIPANT_FRESH_TAKES.json> \
  --seed 20260711 \
  --output-dir <GOLD_ROOT>
```

`scripts/prepare_fact_gold300.py` 会生成：

```text
gold300_root/
  gold300_frozen.jsonl
  gold300_freeze.json
  gold140_probe_train_annotations.csv
  gold60_calibration_dev_annotations.csv
  gold100_locked_test_annotations.csv
  representation_train_excluded_takes.txt
  ANNOTATION_GUIDE.zh-CN.md
```

精确契约：

- `probe_train`：140 条，47 个旧 train takes。
- `calibration_dev`：60 条，20 个旧 heldout takes。
- `locked_test`：100 条，50 个 short73 takes。
- 每个 split 确定性选 20 条 dual，共 60 条。
- 所有 gold 行均为 `representation_training_valid=false`。
- 500 条 transition diagnostic sample IDs 已被排除。

这些 canonical 文件不是标注者的工作表，禁止直接发送给标注者。

指南内容属于 freeze 契约。指南更新后必须用相同候选和 seed 重新运行本命令生成新的 `gold300_freeze.json`；禁止只在旧 gold 根目录或旧 delivery 中手工替换 Markdown 文件。

## 3. 公开 delivery 的目录

一次 materialize 会原子生成三份互相分离的 artifact：

```text
admin_output/                 # 管理员私有，包含 sealed mapping
annotator_a_output/           # A 的公开任务
annotator_b_output/           # B 的 dual-only 公开任务
```

公开任务内部固定包含：

```text
ANNOTATION_GUIDE.zh-CN.md
task.csv
task.template.csv
validate_submission.py
delivery_manifest.json
image_inventory.jsonl
images/
```

`task.csv` 只有 opaque `review_id`、图片路径和五个可编辑字段，不包含 `sample_id`、`take_uid`、split、任务名、弱标签或模型预测。

### 3.1 正式交付前的 non-locked pilot

在冻结正式指南前，先从未进入 gold 的 heldout takes 生成 24 条 A/B 重叠 pilot：

```bash
python scripts/prepare_fact_gold300_pilot.py \
  --candidates <HELDOUT_BASE_MANIFEST.jsonl> \
  --gold-manifest <GOLD_ROOT>/gold300_frozen.jsonl \
  --annotation-guide docs/fact_gold300_annotator_guide.zh-CN.md \
  --count 24 \
  --seed 20260711 \
  --output-dir <PILOT_ROOT>/selection
```

该脚本会排除正式 gold 的全部 sample ID 和 take UID，从 24 个不同的 heldout takes 各选一个确定性中位 transition。pilot freeze 明确禁止用于 encoder、linear probe、模型选择、正式评价或 locked test。

然后用现有 materializer 生成两个公开 pilot 包和一个管理员包：

```bash
python scripts/materialize_fact_gold300_review.py \
  --gold-manifest <PILOT_ROOT>/selection/pilot_frozen.jsonl \
  --gold-freeze <PILOT_ROOT>/selection/pilot_freeze.json \
  --include-split calibration_dev \
  --source heldout=<HELDOUT_NPY_DIR> \
  --annotation-template <PILOT_ROOT>/selection/pilot_annotations.csv \
  --annotation-guide docs/fact_gold300_annotator_guide.zh-CN.md \
  --admin-output-dir <PILOT_ROOT>/admin \
  --annotator-a-output-dir <PILOT_ROOT>/annotator_a \
  --annotator-b-output-dir <PILOT_ROOT>/annotator_b \
  --allow-noncanonical-count
```

pilot 不使用 formal source contract，因为它选择的是未进入正式 gold/raw-frame audit 的 heldout 样本；materializer 仍会绑定完整 NPY 文件、所选 RGB 内容、manifest、模板和指南 hash。A/B 都独立完成 24 条后才讨论类别边界。若修改指南，必须重新冻结 gold 并重新生成正式 v2 包；pilot 标签不得复制进正式提交。

## 4. train/dev 与 locked 必须分开生成

train/dev 可在同一次命令中生成；locked 必须单独生成到不同的 admin、A、B 目录。下面使用占位路径，实际运行时替换尖括号内容。

### 4.1 生成 probe-train + calibration-dev delivery

```bash
python scripts/materialize_fact_gold300_review.py \
  --gold-manifest <GOLD_ROOT>/gold300_frozen.jsonl \
  --gold-freeze <GOLD_ROOT>/gold300_freeze.json \
  --include-split probe_train \
  --include-split calibration_dev \
  --source train=<TRAIN_NPY_DIR> \
  --source heldout=<HELDOUT_NPY_DIR> \
  --source-contract train=<TRAIN_SOURCE_CONTRACT.json> \
  --source-contract heldout=<HELDOUT_SOURCE_CONTRACT.json> \
  --annotation-template <GOLD_ROOT>/gold140_probe_train_annotations.csv \
  --annotation-template <GOLD_ROOT>/gold60_calibration_dev_annotations.csv \
  --annotation-guide <GOLD_ROOT>/ANNOTATION_GUIDE.zh-CN.md \
  --admin-output-dir <PRIVATE_ADMIN_ROOT>/gold200_admin \
  --annotator-a-output-dir <DELIVERY_ROOT>/gold200_annotator_a \
  --annotator-b-output-dir <DELIVERY_ROOT>/gold40_annotator_b
```

预期公开任务数：A 为 200，B 为 40。

### 4.2 单独生成 locked delivery

```bash
python scripts/materialize_fact_gold300_review.py \
  --gold-manifest <GOLD_ROOT>/gold300_frozen.jsonl \
  --gold-freeze <GOLD_ROOT>/gold300_freeze.json \
  --include-split locked_test \
  --source short73=<SHORT73_NPY_DIR> \
  --source-contract short73=<SHORT73_SOURCE_CONTRACT.json> \
  --annotation-template <GOLD_ROOT>/gold100_locked_test_annotations.csv \
  --annotation-guide <GOLD_ROOT>/ANNOTATION_GUIDE.zh-CN.md \
  --admin-output-dir <LOCKED_PRIVATE_ROOT>/gold100_admin \
  --annotator-a-output-dir <LOCKED_DELIVERY_ROOT>/gold100_annotator_a \
  --annotator-b-output-dir <LOCKED_DELIVERY_ROOT>/gold20_annotator_b
```

预期公开任务数：A 为 100，B 为 20。

materializer 会拒绝把 `locked_test` 与其他 split 放进同一个 pack，也会拒绝重叠或嵌套的输出目录。

## 5. 分发前检查

管理员在发送每个公开包前检查：

- `delivery_manifest.json` 的 `weak_or_model_labels_in_pack=false`。
- `contains_frozen_sample_mapping=false`。
- `contains_other_annotator_task=false`。
- 图片数量与 manifest 的 `tasks` 一致。
- `task.template.csv` 与 blank task 的 SHA256 一致。
- A 和 B 收到不同目录；B 只收到 dual 子集。
- admin 目录权限为私有；在 Linux 上应为 `0700`，且不位于公开 delivery 内。

不要通过公共 GitHub、群文件或所有项目成员可读的共享目录分发 locked 包。

## 6. 标注者操作约定

给 A/B 的统一说明：

1. 只编辑 `task.csv`。
2. `task.template.csv` 和其他证据文件保持不变。
3. A/B 使用不同、稳定的匿名 `annotator_id`。
4. 提交前运行 `validate_submission.py`。
5. 只提交 `task.csv` 与 `submission_validation_report.json`。
6. 两人提交前不得交换或讨论逐样本标签。

建议先用少量非 locked 样本做规则培训，但正式 dual 样本必须在指南版本冻结后重新独立完成。

## 7. 回收和公开提交校验

对每位标注者的每个包，优先使用包内自检脚本：

```bash
python validate_submission.py \
  --submission task.csv \
  --template task.template.csv \
  --delivery-manifest delivery_manifest.json \
  --output-report submission_validation_report.json
```

管理员还应确认：

- A/B 的 `annotator_id` 不同。
- `review_id` 精确覆盖各自模板，无缺失、重复或新增。
- `image` 与模板逐条一致。
- effect/contact 枚举合法，`ambiguous` 均有原因。
- result、template、delivery manifest 和 admin mapping 的 SHA256 被记录。

公开提交不能直接交给 `validate_fact_gold300.py`：该脚本需要 canonical `sample_id`。必须由管理员使用 sealed `review_mapping.csv` 进行安全导入，标注者不得接触 mapping。

### 7.1 导入 non-locked 的 200/40 条提交

```bash
python scripts/import_fact_gold300_submissions.py \
  --review-mapping <PRIVATE_ADMIN_ROOT>/gold200_admin/sealed/review_mapping.csv \
  --template-a <DELIVERY_ROOT>/gold200_annotator_a/task.template.csv \
  --submission-a <RETURN_ROOT>/gold200_annotator_a/task.csv \
  --template-b <DELIVERY_ROOT>/gold40_annotator_b/task.template.csv \
  --submission-b <RETURN_ROOT>/gold40_annotator_b/task.csv \
  --delivery-manifest-a <DELIVERY_ROOT>/gold200_annotator_a/delivery_manifest.json \
  --delivery-manifest-b <DELIVERY_ROOT>/gold40_annotator_b/delivery_manifest.json \
  --output-dir <PRIVATE_IMPORT_ROOT>/gold200_imported
```

### 7.2 在 locked 私有根目录导入 100/20 条提交

```bash
python scripts/import_fact_gold300_submissions.py \
  --review-mapping <LOCKED_PRIVATE_ROOT>/gold100_admin/sealed/review_mapping.csv \
  --template-a <LOCKED_DELIVERY_ROOT>/gold100_annotator_a/task.template.csv \
  --submission-a <LOCKED_RETURN_ROOT>/gold100_annotator_a/task.csv \
  --template-b <LOCKED_DELIVERY_ROOT>/gold20_annotator_b/task.template.csv \
  --submission-b <LOCKED_RETURN_ROOT>/gold20_annotator_b/task.csv \
  --delivery-manifest-a <LOCKED_DELIVERY_ROOT>/gold100_annotator_a/delivery_manifest.json \
  --delivery-manifest-b <LOCKED_DELIVERY_ROOT>/gold20_annotator_b/delivery_manifest.json \
  --output-dir <LOCKED_PRIVATE_ROOT>/gold100_imported
```

importer 会完成以下检查：

- 公开 7 列 schema、review ID 和 image 路径精确不变。
- A 精确覆盖本包全部任务，B 精确覆盖 dual-only 子集。
- A/B 的匿名 ID 稳定且不同。
- 标签枚举与 ambiguous 原因合法。
- delivery manifest 强绑定正确的 sealed mapping SHA256。
- locked 与 non-locked 不能混合导入。
- 输出目录必须不存在，成功后原子发布；Linux locked 输出权限固定为 `0700`。

每次导入会输出 combined canonical CSV、按 split 拆分的 canonical CSV、`result_manifest.json` 及其 SHA256。例如 non-locked A 会同时得到：

```text
annotations_a_canonical.csv
annotations_a_probe_train.csv
annotations_a_calibration_dev.csv
```

probe 使用按 split 文件，不要人工拆分 combined CSV。

## 8. κ 门与指南修订

完成安全导入后，对 60 条 dual 样本分别计算：

- effect Cohen's κ。
- contact Cohen's κ。

管理员可以在不生成混合 CSV 的情况下，同时读取物理隔离的 40 条 non-locked 和 20 条 locked canonical 文件：

```bash
python scripts/validate_fact_gold300.py \
  --annotations-a <PRIVATE_IMPORT_ROOT>/gold200_imported/annotations_a_canonical.csv \
  --annotations-a <LOCKED_PRIVATE_ROOT>/gold100_imported/annotations_a_canonical.csv \
  --annotations-b <PRIVATE_IMPORT_ROOT>/gold200_imported/annotations_b_dual_canonical.csv \
  --annotations-b <LOCKED_PRIVATE_ROOT>/gold100_imported/annotations_b_dual_canonical.csv \
  --expected-a-count 300 \
  --expected-dual-count 60 \
  --minimum-kappa 0.70 \
  --output-report <LOCKED_PRIVATE_ROOT>/gold300_kappa_report.json
```

该命令只能由数据管理员在可读 locked 标签的私有环境运行。输出报告记录输入 hash、双标数量和两项 κ，不输出逐样本标签。

两项都必须达到 `0.70`。任一低于门槛时：

1. 不查看模型结果，不调整类别以追求下游分数。
2. 汇总分歧类型，优先修订类别边界和示例。
3. 记录新指南版本和 hash。
4. 由 A/B 在互不可见条件下重新完成指定 calibration/dual 批次。
5. 重新计算 κ；不得把讨论后的共识标签伪装成独立标注。

若需要最终单一 gold 标签，分歧裁决必须发生在独立提交和 κ 计算之后，并记录裁决者、原始 A/B 标签、裁决标签、原因和指南版本。

## 9. locked 封存

locked A/B 结果回收后：

- 立即移入 root-owned 或等价的私有目录，建议 Linux 权限 `0700`。
- 不放入普通实验输出、共享 home、Git history 或模型开发目录。
- 模型开发人员只能得到“标注已完成、格式/κ 是否通过”的状态，不得得到逐样本标签。
- 禁止用 locked 标签选择模型、阈值、loss 权重或 checkpoint。
- 只有 P0/P2/P3/P4 × seeds 42/43/44 的配置、checkpoint、probe 和阈值全部冻结后，才能由 `probe_fact_effect_locked_campaign.py` 一次性消费。
- 保留全局 consumed marker；失败的读取尝试也不得通过换输出目录重新打开 locked。

## 10. 最终归档清单

管理员至少归档：

```text
gold300_frozen.jsonl + hash
gold300_freeze.json + hash
ANNOTATION_GUIDE.zh-CN.md + hash
每个 delivery_manifest.json + hash
每个 task.template.csv + hash
A/B 原始提交 + hash
A/B submission_validation_report.json
sealed/review_mapping.csv + hash
安全导入 result manifest
κ 报告
指南修订记录（若有）
裁决记录（若有）
locked consumed marker（最终评估后）
```

任何文件重建或修订都应产生新版本和新 hash，不覆盖已有证据。
