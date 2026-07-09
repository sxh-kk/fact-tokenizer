# filtering_v2 human-guided v1 人工标注方法

本文是第一版纯数据侧 `human-guided filter v1` 的人工标注手册。目标是让标注人员能够独立判断一个 Ego/Exo take 是否适合 FACT tokenizer 第一阶段训练，并把少量人工判断推广成全量数据筛选器。

核心原则：

- 人工标签只表达数据质量和训练用途。
- ranker 只学习人工标签和数据侧特征。
- 不使用 tokenizer loss、probe、MSE、causality gap、A/B 训练结果作为 filter 的训练标签。
- tokenizer A/B 只用于固定参数下验证 filter 是否改善训练数据，不用于反向校正人工标签。

## 1. 标注目标

FACT tokenizer 第一阶段希望学习的是 ego-accessible shared action token：它应该主要编码跨 Ego/Exo 视角一致、并且从 ego 视角可预测的交互动态。

因此，标注时要问的不是“这个视频好不好看”，而是：

```text
这个 take 的数据本身，是否能帮助 tokenizer 学到可从 ego 预测、且跨视角共享的动作/交互 token？
```

优先保留的数据：

- Ego 能看到手、物体、接触区域、被操作物体或主动交互线索。
- Exo 能看到身体姿态、空间关系、移动方向或动作阶段。
- 同一 take 内有阶段变化，例如 approach、reach、contact、manipulate、carry、place、release。
- Ego/Exo 同步可靠，两个视角确实描述同一段动作。
- 动作信号强于场景、背景、任务名称等 shortcut。

优先丢弃的数据：

- 纯头动、纯背景变化、纯场景切换、纯走路但没有明确任务阶段。
- Ego 看不到关键动作，即使 Exo 很清楚。
- 单一状态长时间重复，没有明显 transition。
- Ego/Exo 不同步，或两个视角内容对应关系不可靠。
- 画面质量差、遮挡严重、任务语义不清。
- scene-only shortcut 风险高，模型可能只靠厨房/桌面/房间背景学习，而不是学动作。

## 2. 快速开始

### 2.1 生成待标注 CSV

通常先由自动流程生成：

```text
outputs/.../annotation_batch_v2_review.csv
```

这个 CSV 已经包含自动 proxy、VLM、hand/object/contact、exo pose 等数据侧特征，并会标记哪些样本更需要人工复核。

第一版不接 VLM 时，可以直接用一键脚本生成待标注表：

```bash
BASE=data/fact_egoexo/splits/diverse_500takes_t1p0_s1_48t_seed123_80_20
OUT=outputs/filtering_v2_annotation_review_no_vlm

conda run -n fact_tokenizer python tools/run_filtering_v2_annotation_review_no_vlm.py \
  --base-split-dir $BASE \
  --out-dir $OUT \
  --max-review 160
```

默认流程使用 metadata-fast 模式，不读取 6GB 级视频 NPZ 数组，先快速生成待人工标注 CSV。它适合第一轮决定“哪些 take 要人工看”和“人工字段怎么填”。

输出包括：

```text
$OUT/take_relevance_scores_v2_all_no_vlm.csv
$OUT/annotation_batch_v2_review.csv
```

如果需要同时生成 contact sheets，可以显式加 `--with-contact-sheets`。注意：压缩 `.npz` 生成图片会重新读取视频数组，可能很慢。

```bash
conda run -n fact_tokenizer python tools/run_filtering_v2_annotation_review_no_vlm.py \
  --base-split-dir $BASE \
  --out-dir $OUT \
  --max-review 160 \
  --with-contact-sheets
```

如果需要使用视频 motion proxy，而不是 metadata proxy，可以显式加 `--feature-mode video`。这会读取大视频数组，适合后续更精细版本，不建议第一轮人工标注入口默认使用。

如果只想先标 train，可以加：

```bash
conda run -n fact_tokenizer python tools/run_filtering_v2_annotation_review_no_vlm.py \
  --base-split-dir $BASE \
  --out-dir $OUT \
  --splits train \
  --max-review 160
```

### 2.2 生成图像 review pack

如果已有 contact sheet，可以生成 HTML review pack：

```bash
conda run -n fact_tokenizer python tools/build_annotation_review_pack.py \
  --annotations outputs/.../annotation_batch_v2_review.csv \
  --out-dir outputs/.../annotation_review_pack \
  --copy-images
```

标注时主要看：

- `contact_sheet_path` 或 HTML 图集。
- `parent_task_name` / `task_name`。
- 自动分数和 `review_reasons`。

自动分数只作为提示，不能替代人工判断。

### 2.3 填写人工字段

人工在 CSV 中填写：

```csv
take_relevance
ego_hand_visibility
exo_body_visibility
object_interaction
phase_diversity
scene_only_risk
ego_exo_sync_quality
usable_for
confidence
reason
notes
```

建议保存为：

```text
outputs/.../annotation_batch_v2_labeled.csv
```

### 2.4 校验标注格式

```bash
conda run -n fact_tokenizer python tools/validate_annotations.py \
  --annotations outputs/.../annotation_batch_v2_labeled.csv
```

如果只标了部分 rows，可以允许空行：

```bash
conda run -n fact_tokenizer python tools/validate_annotations.py \
  --annotations outputs/.../annotation_batch_v2_labeled.csv \
  --allow-empty
```

## 3. 标注时只看数据侧证据

本流程的目的，是先构建一个独立于 tokenizer 参数的数据筛选器。因此标注时只允许参考数据侧信息：

- Ego/Exo 图像帧或 contact sheet。
- 任务名、take id、parent task。
- hand/object/contact proxy。
- exo pose/phase proxy。
- VLM 对视频内容的描述或评分。
- 自动规则给出的 review reason。

不要参考：

- 当前 tokenizer loss。
- 当前 tokenizer probe 分数。
- filtered/no-filter A/B 的训练曲线。
- 某次 tokenizer 的 MSE、agreement、causality gap。
- 某个 take 在当前 tokenizer 下“看起来是否容易训练”。

原因是：当前 tokenizer 参数仍在优化中。如果用不稳定 tokenizer 指标反向修正 filter，就会把模型自身偏差写入数据筛选器。第一版 filter 应该先回答“数据本身是否适合”，再用固定 tokenizer 参数做下游验证。

## 4. 三步判断法

每个 take 推荐按三步判断。

### 4.1 第一步：Ego 是否有可预测动作线索

先看 Ego，因为最终 shared action token 必须能从 Ego history 预测。

重点看：

- 是否能看到手、工具、物体、接触区域或被操作对象。
- Ego 视角是否能判断动作正在接近、接触、移动、放置或释放。
- 画面变化是否主要由人的主动交互引起，而不是头部晃动或相机运动。

如果 Ego 完全看不到关键动作，即使 Exo 很清楚，通常不能直接进 `tokenizer_main`。

### 4.2 第二步：Exo 是否补足身体/空间/阶段

再看 Exo 是否提供跨视角共享的信息：

- 是否看到身体姿态、移动方向、人与物体的空间关系。
- 是否能补足 Ego 看不清的动作阶段。
- 是否能判断当前是 approach、contact、manipulate、place、release 等阶段。

Exo 很清楚但 Ego 不可见时，更适合 `loco_aux` 或 `diagnostic_candidate`，不要轻易标主训练正例。

### 4.3 第三步：是否有阶段变化且 shortcut 风险低

最后判断是否真的存在可学习的 transition：

- 是否从一个动作阶段进入另一个动作阶段。
- 是否有物体状态变化或接触状态变化。
- 是否有足够 temporal diversity，而不是一直重复同一状态。
- 是否可能只靠场景、任务背景、固定物体外观就能猜出标签。

如果主要变化来自背景、房间、桌面、头动、光照或镜头位移，应提高 `scene_only_risk`，并谨慎进入主训练。

## 5. 字段定义

### 5.1 `ego_hand_visibility`

| 值 | 含义 | 例子 |
| --- | --- | --- |
| `0` | Ego 中基本看不到手、物体或关键交互区域 | 镜头对着地面/墙面，手和物体都不在画面内 |
| `1` | Ego 中部分可见，但遮挡、裁切或动作线索不稳定 | 偶尔看到手或物体边缘，但关键接触不连续 |
| `2` | Ego 中手、物体或交互区域清楚可见 | 能看到手接近、接触、移动或释放物体 |

判断重点：不是必须一直看到完整双手，而是 Ego 是否提供足够动作预测线索。

### 5.2 `exo_body_visibility`

| 值 | 含义 | 例子 |
| --- | --- | --- |
| `0` | Exo 中看不到身体或空间关系 | 人被遮挡，或只看到局部背景 |
| `1` | Exo 中身体/空间关系部分可见 | 能看到上半身或移动方向，但阶段不稳定 |
| `2` | Exo 中身体、姿态、空间关系清楚 | 能看到人、目标物体、移动方向和动作阶段 |

Exo 的价值在于提供 whole-body、空间关系和阶段 teacher，但它不能替代 Ego 可预测性。

### 5.3 `object_interaction`

| 值 | 含义 | 例子 |
| --- | --- | --- |
| `0` | 无明确手物或人-物交互 | 站立、走路、看场景、相机移动 |
| `1` | 有可疑或较弱交互 | 可能在拿东西，但接触不清楚 |
| `2` | 明确手物/人-物交互，且动作和物体状态有关 | 抓取、放置、打开、倒入、推动、整理、切换工具 |

如果只有身体移动但没有物体或接触状态变化，通常不要给 `2`。

### 5.4 `phase_diversity`

| 值 | 含义 | 例子 |
| --- | --- | --- |
| `0` | 基本单一状态 | 一直站着、一直看、一直走、一直背景变化 |
| `1` | 有轻微阶段变化，但不够清晰 | 看到接近或移动，但接触/释放不明显 |
| `2` | 明显多阶段变化 | approach -> contact -> manipulate -> release |

FACT tokenizer 需要 transition。静态好画质但没有阶段变化，也不是好主训练样本。

### 5.5 `scene_only_risk`

| 值 | 含义 | 例子 |
| --- | --- | --- |
| `0` | 主要变化来自动作/交互，shortcut 风险低 | 手物接触和物体状态变化清楚 |
| `1` | 有一定场景/背景干扰 | 背景明显，但动作仍可见 |
| `2` | 场景、背景、任务名或相机运动主导 | 只靠厨房/房间/桌面背景可能猜出任务 |

`scene_only_risk=2` 的样本不能标 `tokenizer_main`。

### 5.6 `ego_exo_sync_quality`

| 值 | 含义 | 处理建议 |
| --- | --- | --- |
| `ok` | Ego/Exo 同步和内容对应可靠 | 可进入正常判断 |
| `minor_issue` | 有轻微错位，但还能判断同一动作阶段 | 可以保留，但 `confidence` 不宜过高 |
| `bad` | 明显不同步或对应关系不可靠 | 通常 `discard` |

如果两个视角明显不是同一个动作阶段，不要因为单视角质量高而标主训练正例。

### 5.7 `take_relevance`

| 值 | 含义 | 常见用途 |
| --- | --- | --- |
| `A_interaction_rich` | Ego/Exo 都有较清楚交互和阶段变化 | `tokenizer_main` |
| `B_loco_body` | Exo 身体/空间运动清楚，但手物交互弱 | `loco_aux` |
| `C_active_view_only` | 有运动但交互语义弱，或只有单视角有效 | `diagnostic_candidate` |
| `D_scene_only` | 纯场景、纯头动、背景主导 | `discard` |
| `E_fine_dexterous` | 细粒度手部动作明显，但 coarse phase 不清楚 | 少量 `tokenizer_main` 或 `diagnostic_candidate` |
| `F_bad_or_unclear` | 质量差、同步差、看不清或无法判断 | `discard` |

`take_relevance` 描述数据类型，`usable_for` 描述训练用途。两者相关，但不是完全等价。

### 5.8 `usable_for`

| 值 | 含义 | 进入训练方式 |
| --- | --- | --- |
| `tokenizer_main` | 可以进入主 FACT tokenizer 数据 | 主训练数据，严格控制质量 |
| `loco_aux` | 可作为少量身体/空间阶段辅助数据 | capped auxiliary，不替代 hand-object 主数据 |
| `diagnostic_candidate` | 边界、冲突或不确定样本 | 暂不进入主训练，用于分析 |
| `discard` | 不适合训练 | 不进入训练 |

`usable_for` 是 ranker 默认学习的标签列。

### 5.9 `confidence`

填 `0-1` 的人工置信度：

| 范围 | 含义 | 建议 |
| --- | --- | --- |
| `0.9-1.0` | 非常确定 | 强正例或强负例 |
| `0.7-0.9` | 基本确定 | 可用于训练 ranker |
| `0.5-0.7` | 边界或不确定 | 建议更多写明 `reason` |
| `<0.5` | 很不确定 | 优先 `diagnostic_candidate`，不作为强标签 |

不要为了凑数量把不确定样本标成高置信度。

### 5.10 `reason`

`reason` 写一句核心判断，越具体越好。推荐包含至少一个正向或负向证据。

好例子：

```text
clear ego hand-object contact, exo body phase visible, low scene-only risk
exo clear but ego cannot see the manipulation, keep as diagnostic
mostly head motion and background change, no visible object interaction
minor ego/exo sync offset but same contact phase is still identifiable
```

不推荐：

```text
good
bad
maybe useful
score high
```

### 5.11 `notes`

`notes` 记录非核心但有用的信息，例如：

- 任务名可能不准。
- 手被遮挡但物体状态变化明显。
- Exo 有遮挡。
- Contact sheet 帧数不足。
- 该 parent task 样本整体偏难。

## 6. `take_relevance` 与 `usable_for` 对照

| `take_relevance` | 默认 `usable_for` | 可调整情况 |
| --- | --- | --- |
| `A_interaction_rich` | `tokenizer_main` | 如果同步差或 scene-only 风险高，改 `discard` 或 `diagnostic_candidate` |
| `B_loco_body` | `loco_aux` | 如果有清楚 hand-object，可升到 `tokenizer_main`；如果只是背景移动，改 `discard` |
| `C_active_view_only` | `diagnostic_candidate` | 如果 Ego 线索足够且 phase 清楚，可少量升到 `tokenizer_main` |
| `D_scene_only` | `discard` | 通常不升；除非人工确认 contact sheet 缺失导致误判 |
| `E_fine_dexterous` | `diagnostic_candidate` | 如果 Ego 手物接触和阶段非常清楚，可少量进入 `tokenizer_main` |
| `F_bad_or_unclear` | `discard` | 通常不升 |

## 7. 典型判定规则

### 7.1 进入 `tokenizer_main`

通常需要满足：

```text
ego_hand_visibility >= 1
object_interaction >= 1
phase_diversity >= 1
scene_only_risk <= 1
ego_exo_sync_quality != bad
```

强正例通常是：

```text
take_relevance = A_interaction_rich
ego_hand_visibility = 2
exo_body_visibility >= 1
object_interaction = 2
phase_diversity = 2
scene_only_risk = 0
ego_exo_sync_quality = ok
usable_for = tokenizer_main
confidence >= 0.8
```

适合主训练的例子：

- Ego 能看到手接近杯子并拿起，Exo 能看到身体靠近桌子和拿起阶段。
- Ego 能看到手把物体放到目标位置，Exo 能看到空间关系和 release。
- Ego 能看到工具与物体接触，Exo 能看到全身姿态和操作阶段。

### 7.2 进入 `loco_aux`

适合：

```text
exo_body_visibility >= 1
phase_diversity >= 1
scene_only_risk <= 1
object_interaction 可以较弱
```

典型情况：

- 身体移动方向、转身、靠近目标区域很清楚，但 Ego 手物接触弱。
- Exo 能看清 whole-body phase，Ego 只提供部分运动线索。

注意：`loco_aux` 只能作为 capped ablation 或辅助数据，不应大量混入主 hand-object 训练。

### 7.3 进入 `diagnostic_candidate`

适合：

- 人工不确定。
- 自动规则和人工直觉冲突。
- VLM 高分但 hand/contact proxy 低。
- Ego 清楚但 Exo 很差，或 Exo 清楚但 Ego 很差。
- 细粒度 dexterous 很强，但 coarse phase 不明显。
- 看起来可能有用，但不适合直接放入第一版主训练。

`diagnostic_candidate` 是保护带：它防止边界样本污染主训练，同时保留后续分析机会。

### 7.4 进入 `discard`

适合：

- scene-only 或纯头动。
- Ego/Exo 不同步。
- 看不清交互。
- 没有阶段变化。
- 任务背景明显强于动作信号。
- 只有任务名称相关，但画面证据弱。
- 画面质量、遮挡、裁切严重影响判断。

## 8. 常见边界案例

### 8.1 Exo 很清楚，Ego 看不到关键动作

不要直接标 `tokenizer_main`。

推荐：

- 如果身体/空间阶段有价值：`loco_aux`。
- 如果只是用于观察边界：`diagnostic_candidate`。
- 如果 Ego 完全无动作线索：`discard`。

原因：最终 token 必须 ego-accessible。Exo-only 信息会让 tokenizer 学到部署时 Ego 预测不了的动作因素。

### 8.2 Ego 手很清楚，但 Exo 看不到人或不同步

推荐：

- Exo 轻微缺失但同步可判断：可 `diagnostic_candidate` 或低置信 `tokenizer_main`。
- Exo 明显不同步：`discard`。

原因：第一阶段训练依赖 paired Ego/Exo transition。单视角好不等于跨视角训练好。

### 8.3 画面运动很大，但主要是头动或相机晃动

推荐：

- `scene_only_risk=2`
- `take_relevance=D_scene_only` 或 `F_bad_or_unclear`
- `usable_for=discard`

原因：这类样本容易让 token 学相机运动或背景流，而不是交互动作。

### 8.4 细粒度手部操作明显，但阶段变化不清楚

推荐：

- 如果 Ego 手物接触稳定、scene risk 低：`E_fine_dexterous` + `diagnostic_candidate`，少量可进 `tokenizer_main`。
- 如果只是小幅重复动作：`diagnostic_candidate` 或 `discard`。

原因：第一版主训练优先 coarse phase 和共享交互动态，细粒度 dexterous 可以后续专门处理。

### 8.5 任务名很相关，但画面证据弱

不要因任务名标正例。

推荐：

- 画面证据不足：`diagnostic_candidate`。
- 只有背景或任务场景：`discard`。

原因：filter 要避免 task-name/scene shortcut。

### 8.6 VLM 分数高，但 proxy 分数低

人工优先看 contact sheet。

推荐：

- 如果 VLM 确实识别到真实交互：按人工判断标。
- 如果 VLM 被任务名或场景误导：降到 `diagnostic_candidate` 或 `discard`。

`reason` 中写明冲突，例如：

```text
vlm high but ego contact not visible; likely scene/task-name shortcut
```

### 8.7 自动规则给 discard，但人工看到清楚交互

可以人工纠正。

推荐：

- 标正确字段。
- `reason` 写明为什么规则误判，例如 contact sheet 清楚、proxy 漏检。
- 这类样本对 ranker 很有价值。

## 9. 第一轮标注采样建议

第一轮建议标注 `120-160` 个 takes，目标不是覆盖全量，而是给 ranker 足够清楚的方向。

推荐组成：

| 样本类型 | 数量 | 目的 |
| --- | ---: | --- |
| 自动高分 `tokenizer_main` | 35-45 | 检查 false positive |
| 阈值附近边界样本 | 35-45 | 校准 strict/balanced policy |
| 自动规则/VLM/proxy 冲突样本 | 25-35 | 教 ranker 处理复杂情况 |
| 自动 discard | 15-25 | 检查 false drop |
| 每个高频 parent task 的代表样本 | 视情况补齐 | 降低任务偏置 |

建议比例：

- `tokenizer_main`：约 35%-45%。
- `loco_aux`：约 10%-20%。
- `diagnostic_candidate`：约 20%-30%。
- `discard`：约 20%-30%。

不要刻意追求完全均匀，但要避免 120 个样本里几乎全是正例或全是负例。

## 10. 标注流程建议

### 10.1 第一遍：快速粗分

只做四类用途判断：

```text
tokenizer_main / loco_aux / diagnostic_candidate / discard
```

同时给出 `take_relevance` 和一句 `reason`。

### 10.2 第二遍：补齐评分字段

回头补：

```text
ego_hand_visibility
exo_body_visibility
object_interaction
phase_diversity
scene_only_risk
ego_exo_sync_quality
confidence
```

如果第二遍发现第一遍判断不稳，优先降到 `diagnostic_candidate`。

### 10.3 第三遍：质检边界样本

重点复查：

- `confidence < 0.7` 的样本。
- `tokenizer_main` 但 `scene_only_risk=1` 的样本。
- `tokenizer_main` 但 `ego_hand_visibility=1` 的样本。
- VLM 高、proxy 低的冲突样本。
- 自动 discard 但人工升为正例的样本。

## 11. 质量控制

### 11.1 一致性规则

以下组合通常不允许：

| 情况 | 建议修正 |
| --- | --- |
| `usable_for=tokenizer_main` 且 `scene_only_risk=2` | 改 `discard` 或 `diagnostic_candidate` |
| `usable_for=tokenizer_main` 且 `ego_exo_sync_quality=bad` | 改 `discard` |
| `usable_for=tokenizer_main` 且 `ego_hand_visibility=0` | 通常改 `loco_aux` / `diagnostic_candidate` / `discard` |
| `usable_for=discard` 但所有正向字段都是 2 | 复查是否误标 |
| `confidence>=0.9` 但 `reason` 很模糊 | 补具体证据 |

### 11.2 复标建议

如果人力允许：

- 随机抽 20 个样本做二次复标。
- 优先复标所有边界样本和冲突样本。
- 如果两次标注不一致，不要强行平均，写入 `notes` 并降到 `diagnostic_candidate`。

### 11.3 false positive / false drop 检查

训练 ranker 前，人工快速检查两类错误：

- false positive：自动高分但人工认为不适合主训练。
- false drop：自动低分但人工认为是清楚交互。

这两类样本最能改进 ranker。

## 12. 训练和应用 ranker

标注校验通过后训练 ranker：

```bash
conda run -n fact_tokenizer python tools/train_relevance_ranker.py \
  --features outputs/.../take_relevance_scores_v2_train.csv \
  --labels outputs/.../annotation_batch_v2_labeled.csv \
  --label-column usable_for \
  --model auto \
  --out outputs/.../relevance_ranker_v1.joblib
```

应用到 train 或 heldout：

```bash
conda run -n fact_tokenizer python tools/apply_relevance_ranker.py \
  --features outputs/.../take_relevance_scores_v2_train.csv \
  --ranker outputs/.../relevance_ranker_v1.joblib \
  --out outputs/.../take_relevance_ranked_v2_train.csv
```

如果 train 和 heldout 分别生成 ranked CSV，合并后再生成 split。

## 13. 生成 strict / balanced split

生成默认 policy split：

```bash
conda run -n fact_tokenizer python tools/build_filtered_split.py \
  --ranked outputs/.../take_relevance_ranked_v2_all.csv \
  --policy configs/filter_policy_v2.yaml \
  --out outputs/.../filtered_split_v2.json
```

生成更严格的主训练 split：

```bash
conda run -n fact_tokenizer python tools/build_filtered_split.py \
  --ranked outputs/.../take_relevance_ranked_v2_all.csv \
  --policy configs/filter_policy_v2.yaml \
  --fact-main-mode strict \
  --out outputs/.../filtered_split_v2_fact_main_strict.json
```

生成更平衡的主训练 split：

```bash
conda run -n fact_tokenizer python tools/build_filtered_split.py \
  --ranked outputs/.../take_relevance_ranked_v2_all.csv \
  --policy configs/filter_policy_v2.yaml \
  --fact-main-mode balanced \
  --out outputs/.../filtered_split_v2_fact_main_balanced.json
```

建议第一版至少保留：

- `fact_main_strict`：高精度，优先避免脏数据。
- `fact_main_balanced`：稍高召回，观察是否增加有用 diversity。
- `discard`：明确不用。
- `diagnostic_candidate`：后续人工复查和失败分析。

## 14. 与 tokenizer A/B 的关系

完成数据侧 filter 后，可以用固定 tokenizer 参数做 A/B：

```text
no-filter vs fact_main_strict
no-filter vs fact_main_balanced
```

A/B 的作用是验证：

- filtered 数据是否提高 shared action token 的稳定性。
- filtered 数据是否改善 Ego/Exo action agreement。
- filtered 数据是否降低明显由脏数据造成的训练噪声。
- strict 和 balanced 哪个更适合当前阶段。

但 A/B 不直接反向改人工标签。若 A/B 不好，先区分：

- 数据 filter 是否真的选错了样本。
- tokenizer 参数、loss 权重、warm-up、alignment schedule 是否还不稳定。
- split 太小或 diversity 不足。
- heldout 分布是否和 train 不一致。

只有当人工复查确认某类数据确实误选或误丢，才更新标注规则。

## 15. 标注完成标准

第一轮标注完成时，建议满足：

- 至少 `120-160` 个 take 有完整人工字段。
- 每个 `usable_for` 类别都有一定样本，尤其是 `tokenizer_main` 和 `discard`。
- 所有 `tokenizer_main` 样本都没有明显 sync bad 或 scene-only high risk。
- 每个高频 parent task 至少有少量人工样本。
- `reason` 能解释主要判断，而不是只有泛泛的 good/bad。
- `validate_annotations.py` 通过。
- ranker 训练后能输出 `prob_tokenizer_main`、`prob_loco_aux`、`prob_discard`、`prob_diagnostic_candidate`。

## 16. 最小可执行闭环

完整第一版闭环如下：

```text
人工标注 120-160 个 takes
        ↓
校验 annotation_batch_v2_labeled.csv
        ↓
用数据侧特征训练 relevance ranker
        ↓
推广到全部 train/heldout takes
        ↓
生成 fact_main_strict / fact_main_balanced / discard
        ↓
固定 tokenizer 参数做 no-filter vs filtered A/B
        ↓
人工复查错误样本，决定是否进入 filter v2
```

这条链路中，人工标注是 filter 的监督信号；tokenizer A/B 是 filter 的外部验证信号。两者要分开，避免在 tokenizer 尚未稳定时把模型偏差反馈进数据筛选。
