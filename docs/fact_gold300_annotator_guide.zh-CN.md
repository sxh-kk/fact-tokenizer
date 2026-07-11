# FACT v7 effect/contact gold 标注者上手指南

本指南面向标注者 A 和标注者 B。你不需要了解模型结构，也不需要判断动作意图。你的任务只有一个：比较同一条 Ego/Exo transition 在 `t` 和 `t+0.5s` 的可观察变化，分别填写一个 `effect_label` 和一个 `contact_label`。

## 1. 你会收到什么

每个公开标注包的目录结构如下：

```text
annotation_delivery/
  ANNOTATION_GUIDE.zh-CN.md   # 本指南
  task.csv                    # 唯一需要填写并提交的文件
  task.template.csv           # 原始空白模板，只读，不要修改
  validate_submission.py      # 提交前自检脚本
  delivery_manifest.json      # 交付记录，只读，不要修改
  image_inventory.jsonl       # 图片哈希清单，只读，不要修改
  images/
    <review_id>.png           # 每条任务对应的四宫格图片
```

只编辑 `task.csv`。不要修改或删除其他文件，也不要重命名 `images/` 中的图片。

## 2. 五分钟快速开始

1. 打开 `task.csv`，每行通过 `image` 列找到对应图片。
2. 先比较左上与右上的 Ego 两个端点，再比较左下与右下的 Exo 两个端点。
3. 先在 `contact_label` 中填写一个 contact 类别。
4. 再独立地在 `effect_label` 中填写一个 effect 类别；不要从 contact 机械推导 effect。
5. 每一行填写同一个稳定的 `annotator_id`，例如 `ann_a01`；不要填写真实姓名、邮箱或工号。
6. `effect_label=ambiguous` 时必须填写 `ambiguous_reason`。
7. 完成后保存为 UTF-8 CSV，并运行提交前校验。

允许填写的标签必须完全照抄本指南中的英文值，不能翻译、缩写、改变大小写或添加空格。

## 3. 图片怎么看

每张图片固定包含四个面板：

```text
┌─────────────────────┬─────────────────────┐
│ Ego t               │ Ego t+0.5s          │
├─────────────────────┼─────────────────────┤
│ Exo t               │ Exo t+0.5s          │
└─────────────────────┴─────────────────────┘
```

- Ego 和 Exo 是同一段动作的同步第一视角与第三视角。
- 优先使用证据更清楚的视角，另一个视角用于确认。
- 只依据两个端点中可见的变化，不推断中间一定发生了什么，不推断人的意图、物理因果或画面外状态。
- 如果两个视角明显冲突、疑似不同步，或关键证据被遮挡，使用 `ambiguous`/`unknown`，不要猜测。
- 图片标题只包含随机化的 `review_id`。不要尝试恢复 take、任务、split 或原始样本身份。

## 4. `task.csv` 字段

| 字段 | 是否填写 | 说明 |
| --- | --- | --- |
| `review_id` | 不填写 | 样本的盲化编号，不得修改 |
| `image` | 不填写 | 四宫格图片相对路径，不得修改 |
| `effect_label` | 必填 | 从 8 个 effect 类别中选 1 个 |
| `contact_label` | 必填 | 从 5 个 contact 类别中选 1 个 |
| `ambiguous_reason` | 条件必填 | effect 为 `ambiguous` 时必填，否则留空 |
| `annotator_id` | 必填 | 全部行使用同一个匿名、稳定编号 |
| `notes` | 可选 | 只记录遮挡、同步或画质等必要情况 |

## 5. effect 标签：这 0.5 秒主要发生了什么

每行只能选择下面一个值。

| `effect_label` | 什么时候使用 | 不要误标为该类的情况 |
| --- | --- | --- |
| `no_effect` | 证据充分，两个端点之间没有可确认的任务相关变化 | 因遮挡或画质差而看不清；这种情况用 `ambiguous` |
| `approach_align` | 手、工具或目标明显接近、瞄准、对齐或准备接触，但尚未取得稳定控制 | 已经抓住并开始搬运；应选 `acquire_control` 或 `transport_reposition` |
| `acquire_control` | 从未控制变为抓取、夹持、支撑、勾住，或用工具稳定控制目标 | 两个端点都已经稳定持有；通常是 `transport_reposition`、`state_change_or_manipulate` 或 `no_effect` |
| `state_change_or_manipulate` | 目标出现可见状态、构型或操作变化，如开、关、切、旋、搅、按压、折叠、拆装 | 目标基本保持形状，仅被整体移动；应选 `transport_reposition` |
| `transport_reposition` | 已受控目标发生明显整体搬运、平移、抬升、重定位，或处于放置过程 | 刚刚取得控制但还未形成主要位移；应选 `acquire_control` |
| `release_complete` | 控制或任务相关接触明确结束，目标被放下、释放，或一个操作明确完成 | 仍保持接触或仍在移动目标；应选 `stable` 对应的 effect 类别 |
| `recover_abort` | 明显中止、失败、撤回、掉落、滑脱、重试或恢复 | 正常的释放完成；应选 `release_complete` |
| `ambiguous` | 至少两个 effect 类别同样合理，或视觉证据不足以可靠选择 | 明确没有变化；应选 `no_effect` |

### 5.1 推荐判断顺序

1. 证据是否足够？不足则选 `ambiguous`。
2. 是否出现明确失败、掉落、撤回或重试？是则选 `recover_abort`。
3. 是否从控制转为释放/完成？是则选 `release_complete`。
4. 是否从未控制转为控制？是则选 `acquire_control`。
5. 目标是否发生内部状态或构型变化？是则选 `state_change_or_manipulate`。
6. 已受控目标是否主要发生整体位移？是则选 `transport_reposition`。
7. 是否主要在接近、瞄准或对齐？是则选 `approach_align`。
8. 证据充分但以上都不是，则选 `no_effect`。

如果 0.5 秒内似乎跨过多个阶段，选择端点之间最明确、最主要的阶段边界，并在 `notes` 中简短说明。仍无法确定主类别时使用 `ambiguous`。

## 6. contact 标签：任务相关接触如何变化

contact 只看主动执行部位（手、身体或所持工具）与当前任务目标之间的任务相关接触/控制。不要把脚与地面、身体与衣物、普通座椅等持续日常支撑接触当作目标接触。多手、多工具或全身交互时，只要仍存在一个有效的任务相关控制关系，就视为仍在接触；所有相关控制都结束后才是 `release`。

| `t` 时刻 | `t+0.5s` 时刻 | `contact_label` |
| --- | --- | --- |
| 明确未接触 | 明确未接触 | `none` |
| 明确未接触 | 明确接触或取得控制 | `onset` |
| 明确接触或控制 | 明确接触或控制 | `stable` |
| 明确接触或控制 | 明确分离 | `release` |
| 任一端点无法可靠判断 | 任一端点无法可靠判断 | `unknown` |

五个允许值的定义：

- `none`：两个端点都没有可见的任务相关接触。
- `onset`：区间内从未接触变为接触或控制。
- `stable`：区间两个端点都保持接触或控制；即使物体没有移动，也可以是 `stable`。
- `release`：区间内从接触或控制变为分离。
- `unknown`：遮挡、视角、画质或同步问题使接触阶段无法可靠判断。

## 7. effect 与 contact 必须独立判断

接触发生不等于已经产生可见 effect。下面是常见但不是强制绑定的组合：

| 可见情况 | 常见 `effect_label` | 常见 `contact_label` |
| --- | --- | --- |
| 手在靠近物体，尚未接触 | `approach_align` | `none` |
| 指尖刚触碰目标，但看不出已取得控制 | `approach_align` | `onset` |
| 从未接触到抓住物体 | `acquire_control` | `onset` |
| 一直拿着物体但没有可见变化 | `no_effect` | `stable` |
| 持续接触并旋转、打开或按压目标 | `state_change_or_manipulate` | `stable` |
| 持续持有并搬动物体 | `transport_reposition` | `stable` |
| 放下或松开目标 | `release_complete` | `release` |
| 目标已接近放置面，但末帧仍被手控制 | `transport_reposition` | `stable` |
| 目标掉落或抓取失败 | `recover_abort` | `release`、`unknown` 或其他可见阶段 |

不要因为选择了某个 effect 就自动填写 contact；必须重新检查两个端点的接触状态。

## 8. `no_effect`、`ambiguous` 和 `unknown` 的区别

- `no_effect`：你看得清楚，并且能确认没有任务相关 effect。
- `ambiguous`：effect 看不清，或两个 effect 类别同样合理。
- `unknown`：只用于 contact；接触阶段看不清。

因此一条样本可以是：

```text
effect_label = no_effect
contact_label = unknown
```

也可以是：

```text
effect_label = ambiguous
contact_label = stable
```

## 9. `ambiguous_reason` 和 `notes`

当 `effect_label=ambiguous` 时，`ambiguous_reason` 必须简短说明原因。推荐使用以下形式：

```text
occlusion
motion_too_small
view_conflict_or_sync
multiple_effects
endpoint_evidence_insufficient
other: <简短说明>
```

`notes` 可用于记录明显的数据问题，例如：

```text
exo heavily occluded
ego/exo appear misaligned
target leaves frame
contact unknown because target is hidden
```

不要在任何字段中填写真实姓名、个人联系方式、模型预测、任务猜测或从其他数据源获得的信息。

## 10. CSV 填写示例

```csv
review_id,image,effect_label,contact_label,ambiguous_reason,annotator_id,notes
4b7f2b6a9d8e10c1,images/4b7f2b6a9d8e10c1.png,acquire_control,onset,,ann_a01,
73f2d9c1a460be52,images/73f2d9c1a460be52.png,ambiguous,unknown,occlusion,ann_a01,ego hand hidden
```

注意：示例 ID 是虚构的。实际填写时不要新增示例行。

## 11. 保存和提交前校验

保存要求：

- 文件保持 CSV 格式，不要改成 `.xlsx`。
- 编码使用 UTF-8 或 UTF-8 with BOM。
- 不删除、插入或复制行。
- 不修改列名、列顺序、`review_id` 或 `image`。
- `task.template.csv`、manifest、inventory 和图片必须保持不变。

在标注包目录运行：

```bash
python validate_submission.py \
  --submission task.csv \
  --template task.template.csv \
  --delivery-manifest delivery_manifest.json \
  --output-report submission_validation_report.json
```

Windows PowerShell 也可以写成一行：

```powershell
python validate_submission.py --submission task.csv --template task.template.csv --delivery-manifest delivery_manifest.json --output-report submission_validation_report.json
```

只有看到报告中的：

```json
"passed": true
```

才可以提交。提交时发送：

```text
task.csv
submission_validation_report.json
```

不要把图片或其他标注者的文件复制进自己的交付包。

## 12. A/B 独立标注规则

- 标注者 A 完成自己包中的全部任务。
- 标注者 B 只完成自己收到的 dual 子集，不需要补齐 A 的其他样本。
- A 和 B 必须使用不同的 `annotator_id`。
- 在两人都提交前，不得查看、交换或讨论逐样本标签。
- 可以向数据管理员询问“规则如何解释”，但不要询问另一位标注者给某一条样本标了什么。
- 如果指南修订，管理员会发布新版本并明确哪些行需要重新独立标注；不要自行修改旧提交。

## 13. locked 数据保密规则

标注者不需要知道某条样本属于 train、dev 还是 locked。无论包名如何，全部按相同标准盲标。

- 完成的 CSV 只交给指定数据管理员。
- 不发给模型开发人员，不上传到群聊、网盘公共目录或 GitHub。
- 不截图传播，不保留包含另一位标注者结果的副本。
- 数据管理员确认接收后，按项目要求删除个人工作副本。

## 14. 最终提交清单

提交前逐项确认：

- [ ] 每行 `effect_label` 都是 8 个允许值之一。
- [ ] 每行 `contact_label` 都是 5 个允许值之一。
- [ ] 每行都填写同一个 `annotator_id`。
- [ ] 所有 `ambiguous` 行都填写了 `ambiguous_reason`。
- [ ] 没有修改 `review_id`、`image`、列名、模板、manifest 或图片。
- [ ] 没有参考 atomic text、旧 codelabel、weak label、模型预测或另一位标注者结果。
- [ ] 校验报告显示 `"passed": true`。
- [ ] 只把结果交给指定数据管理员。
