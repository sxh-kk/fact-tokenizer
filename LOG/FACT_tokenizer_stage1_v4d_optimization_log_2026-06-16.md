# FACT Tokenizer Stage-1 训练增强与迭代总结

日期：2026-06-16
仓库：`/data_all/sxh/FACT_tokenizer`
当前推荐模型：`v4d_gentle_usage_from_v2`

最终 checkpoint：

```text
/data_all/sxh/FACT_tokenizer/outputs/fact_tokenizer/egoexo_diverse_500takes_k64_35k_v4d_gentle_usage_from_v2_20260616_215152/fact_tokenizer.ckpt
```

## 1. 本轮目标

本轮工作的核心问题不是单纯把 DINO feature reconstruction loss 继续压低，而是验证并增强：

1. `q_act` 是否真的控制 transition/future reconstruction。
2. `q_act` 是否在 ego/exo 两个视角间表达共享动作因素。
3. private residual 是否只是 tokenizer 训练辅助，而不是偷偷承担主要动作信息。
4. codebook 是否既不 collapse，也不过度依赖模糊 soft assignment。
5. 当前模型是否足够作为 FACT tokenizer stage-1 MVP 的候选模型。

根据 `docs/fact_tokenizer_plan_3_0.md` 和 `AGENTS.md`，本阶段应保持目标集中在：

- ego-accessible shared action tokens。
- paired ego/exo transition 学习。
- swapped future reconstruction。
- shared action codebook。
- private residual 与 shared action token 的分离。
- tokenizer quality validation，而不是提前扩展到 WAM、Action Head、SONIC 或机器人部署模块。

## 2. 起点实验

原始训练配置：

- 数据规模：约 500 takes 中抽取出的 7999 条 paired transitions。
- 输入形式：ego/exo 两视角，两帧 transition。
- codebook：`K=64`。
- action slots：4。
- private residual slots：1。
- backbone：frozen DINOv2。
- 训练目标：DINO feature reconstruction，不做 RGB reconstruction。
- 训练路径：self reconstruction 与 ego/exo swapped reconstruction。
- 训练步数：20k steps。

原始结果：

- 训练完整跑完，无 NaN、崩溃或 VQ 爆炸。
- total loss 从约 1.53 降到最后 500 step 平均约 0.695。
- ego/exo self reconstruction 与 swapped reconstruction 几乎一样好。
- ego token 成功导出，shape 为 `(7999, 1, 4)`。
- private residual 没有被导出，符合设计。
- codebook 使用 49/64 个 code。

起点模型的正面信号是 shared-action 机制初步成立：exo shared action token 可以替换 ego token 用于 ego future reconstruction。
但关键问题是：仅靠 reconstruction 与 swap success 还不能证明 token 有明确动作语义，因为 private residual、当前帧 DINO feature、take identity 或场景信息都有可能解释一部分重建能力。

## 3. 本轮新增验证

为了直接回答“它是不是 action token”，补充了三类 probe。

### 3.1 Token Causality Ablation

比较不同 action token 条件下的 reconstruction MSE：

- `correct token`
- `global shuffled token`
- `same-take shuffled token`
- `random-take token`
- `temporal offset token`
- `zero token`
- `random code token`

核心判断：

- 如果 correct token 明显优于 shuffled/zero/random，则说明 `q_act` 对 future/action reconstruction 有因果控制作用。
- random-take 与 global-shuffle 可以检验动作 token 是否携带跨样本 transition 信息。
- same-take shuffle 用来区分“动作信息”与“take/scene/static context”。
- random-code 可以检验 decoder 是否真的读 code，而不是忽略 token。

### 3.2 Private Leakage Ablation

比较 action 与 private residual 的贡献：

- 正确 `q_act` + 正确 `r_priv`。
- 正确 `q_act` + zero private。
- 正确 `q_act` + shuffled private。
- zero/shuffled/random `q_act` + 正确 private。
- private dropout sweep。

核心判断：

- 如果去掉 private 后 correct action 仍然明显优于 zero/shuffled action，说明 action token 是有效瓶颈。
- 如果正确 private 可以在错误 action 下大量恢复重建，说明 private leakage 严重。
- 理想情况是：private 帮助补充视角细节，但不能替代 shared action。

### 3.3 Semantic Probe

使用 EgoExo 标签做 token 的外部语义检查：

- object/contact/action/narration/task label 的 code purity。
- NMI。
- conditional histogram。
- view/take/task leakage probe。

核心判断：

- 同类动作应该在 code 或 code pattern 上更集中。
- token 不应该强烈编码 view identity。
- token 不应该主要编码 take identity 或静态场景。
- 语义 probe 不能单独证明 action token 成立，只能作为 reconstruction causality 之外的外部支持。

实现脚本：

```text
/data_all/sxh/FACT_tokenizer/scripts/probe_fact_action_tokens.py
```

## 4. 训练增强思路

本轮训练增强不是简单加 loss，而是围绕 action bottleneck 做约束。

### 4.1 Action-only / No-private Reconstruction

加入 action-only 或 no-private 路径，让 decoder 在缺少 private residual 时仍必须从 `q_act` 中恢复 transition-relevant future feature。

目的：

- 防止 private residual 成为主通道。
- 强迫 shared action token 承担可交换的动作因素。

### 4.2 Shuffled Action Contrast

加入 shuffled token contrast loss，让 correct action 的 reconstruction 必须优于 shuffled/random action。

目的：

- 避免 decoder 忽略 action token。
- 将“token 可替换”变成“token 替错会变差”的因果约束。

### 4.3 Private Dropout 与 Private Reg

提高 private dropout，并保留 private regularization。

目的：

- 降低 private residual 传递动作信息的能力。
- 让 private 更偏向视角细节、外观残差、不可共享因素。

### 4.4 Assignment Confidence 与 Codebook Balance

调整：

- VQ temperature。
- VQ beta。
- assignment entropy penalty。
- code usage balance。
- slot diversity。

目的：

- 避免 soft assignment 过于模糊。
- 避免 hard code collapse。
- 保持 4 个 action slot 不完全同质化。

### 4.5 Motion / Delta Focus

加入 motion-focused 与 delta-focused reconstruction/contrast 辅助项。

目的：

- 让 token 更关注 transition 变化，而不是静态外观。
- 减少 take/scene leakage。

相关代码修改：

```text
/data_all/sxh/FACT_tokenizer/fact_tokenizer/model.py
/data_all/sxh/FACT_tokenizer/fact_tokenizer/losses.py
/data_all/sxh/FACT_tokenizer/scripts/train_fact_npz_debug.py
```

监控脚本：

```text
/data_all/sxh/FACT_tokenizer/scripts/monitor_fact_run.py
```

## 5. 迭代记录

### 5.1 v1: Enhanced Action Bottleneck

路径：

```text
/data_all/sxh/FACT_tokenizer/outputs/fact_tokenizer/egoexo_diverse_500takes_k64_20k_enhanced_action_bottleneck_20260616_163726
```

主要改动：

- private dropout。
- action-only/no-private reconstruction。
- shuffled action contrast。
- private leakage probe。

结果摘要：

| 指标 | 结果 |
| --- | --- |
| confidence mean | 0.0148 |
| ego used codes | 27 |
| exo used codes | 28 |
| ego effective codes | 18.25 |
| exo effective codes | 15.52 |
| ego random-take causality delta | 0.00875 / 0.00964 |
| exo random-take causality delta | 0.00040 / 0.00047 |
| view NMI | 0.1589 |
| take NMI | 0.6176 / 0.6256 |
| task NMI | 0.4388 / 0.4342 |

判断：

- ego action causality 有提升。
- private leakage 有所下降。
- 但 assignment confidence 仍然极低。
- take/task NMI 偏高，存在 take identity 或静态上下文泄漏风险。

### 5.2 v2: Exo Balance Finetune

路径：

```text
/data_all/sxh/FACT_tokenizer/outputs/fact_tokenizer/egoexo_diverse_500takes_k64_30k_v2_exo_balance_finetune_20260616_174802
```

主要改动：

- 从 v1 checkpoint finetune 到 30k。
- 降低学习率。
- 提高 VQ confidence。
- 加强 exo auxiliary loss。
- 增强 action consistency、assignment entropy、motion focus。
- private dropout 提高到 0.45。

结果摘要：

| 指标 | 结果 |
| --- | --- |
| confidence mean | 0.1953 |
| ego used codes | 23 |
| exo used codes | 26 |
| ego effective codes | 15.10 |
| exo effective codes | 15.37 |
| ego random-take causality delta | 0.01170 / 0.01150 |
| exo random-take causality delta | 0.00053 / 0.00065 |
| exo zero sensitivity | 约 0.0039 / 0.0040 |
| ego save vs zero no-private | 约 0.0171 / 0.0168 |
| exo save vs zero no-private | 约 0.0040 / 0.0042 |
| remove-private delta | ego 约 0.0006-0.0012，exo 近 0 |
| view NMI | 0.0481 |
| take NMI | 0.3732 / 0.4089 |
| task NMI | 0.2138 / 0.2383 |

判断：

- 这是第一个比较稳的增强版本。
- confidence 从 v1 的 0.0148 提升到 0.1953。
- ego/exo causality 都变强。
- private 去掉后 correct action 仍有效，说明 private 不再是主要动作通道。
- 但 code usage 下降，effective codes 只有约 15，表达容量偏窄。

### 5.3 v3b: Diversity / Delta From v0

路径：

```text
/data_all/sxh/FACT_tokenizer/outputs/fact_tokenizer/egoexo_diverse_500takes_k64_30k_v3b_diversity_delta_from_v0_20260616_193103
```

主要改动：

- 从原始高 usage v0 出发，而不是从 v2 出发。
- 使用 private_dim=8 匹配 v0 checkpoint。
- 加入更强 assignment entropy target、slot balance、delta focus。
- exo auxiliary multiplier 提高。

结果摘要：

| 指标 | 结果 |
| --- | --- |
| confidence mean | 0.0104 |
| global used codes | 13 |
| ego used codes | 12 |
| exo used codes | 13 |
| ego effective codes | 6.97 |
| exo effective codes | 6.71 |
| ego random-take causality delta | 0.00754 / 0.00615 |
| exo random-take causality delta | 0.00034 / 0.00042 |
| ego save vs zero no-private | 0.0205 / 0.0176 |
| remove-private delta | 0.0056 / 0.0084 |
| view NMI | 0.1071 |
| take NMI | 0.3970 / 0.4265 |
| task NMI | 0.2474 / 0.2724 |

判断：

- 这是一个有价值的负结果。
- 从高 usage v0 出发并没有保住 code diversity。
- 强 entropy/slot balance 导致 soft assignment 更模糊，hard usage 反而 collapse。
- private 依赖重新变强。
- 该方向不适合作为最终模型。

### 5.4 v4c: Hard Usage From v2

路径：

```text
/data_all/sxh/FACT_tokenizer/outputs/fact_tokenizer/egoexo_diverse_500takes_k64_40k_v4c_hard_usage_from_v2_20260616_214445
```

主要改动：

- 从 v2 出发。
- 加入 ST-hard usage balance。
- 加入 slot diversity loss。
- 试图在保持 v2 confidence 的同时恢复 code usage。

结果：

- 训练早期 entropy 升到约 0.98。
- soft assignment 重新变得极度模糊。
- 提前停止在约 31.6k。

判断：

- hard usage balance 权重过强时会把 soft distribution 推向未使用 code。
- 这会制造“看起来想用更多 code，但实际 assignment 不确定”的失败模式。
- hard usage loss 只能作为很弱的正则，不能作为主导目标。

### 5.5 v4d: Gentle Usage From v2

路径：

```text
/data_all/sxh/FACT_tokenizer/outputs/fact_tokenizer/egoexo_diverse_500takes_k64_35k_v4d_gentle_usage_from_v2_20260616_215152
```

主要改动：

- 从 v2 checkpoint 出发，训练 30k 到 35k。
- 降低学习率到 `1.5e-5`。
- 使用更低 VQ temperature：`0.06`。
- 提高 VQ beta：`0.35`。
- entropy penalty 加强到 `0.06`。
- hard usage balance 降到极弱：`0.002`。
- slot diversity 降到 `0.02`。
- 移除强 slot balance。
- delta/motion 辅助项保持温和。

结果摘要：

| 指标 | v2 | v4d |
| --- | ---: | ---: |
| confidence mean | 0.1953 | 0.3458 |
| global used codes | 约 23-26 | 46 |
| ego used codes | 23 | 47 |
| exo used codes | 26 | 46 |
| ego effective codes | 15.10 | 21.43 |
| exo effective codes | 15.37 | 20.81 |
| ego random-take causality delta | 约 0.0115 | 约 0.0131 |
| exo random-take causality delta | 约 0.00065 | 约 0.00089 |
| exo zero sensitivity | 约 0.0040 | 约 0.0055 |
| view NMI | 0.0481 | 0.1112 |
| ego take NMI | 0.3732 | 0.4259 |
| exo take NMI | 0.4089 | 0.4093 |
| ego task NMI | 0.2138 | 0.2600 |
| exo task NMI | 0.2383 | 0.2440 |

v4d causality probe：

| Path | correct MSE | random-take delta | same-take delta | zero delta | random-code delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| ego_self | 0.382238 | 0.012885 | 0.010717 | 0.009569 | 0.021675 |
| ego_swap | 0.381826 | 0.013103 | 0.010668 | 0.009981 | 0.022087 |
| exo_self | 0.201269 | 0.000742 | 0.000648 | 0.005322 | 0.002006 |
| exo_swap | 0.201113 | 0.000886 | 0.000764 | 0.005478 | 0.002161 |

v4d private leakage probe：

| Path | action saves vs zero no-private | action saves vs shuffled no-private | remove-private delta |
| --- | ---: | ---: | ---: |
| ego_self | 0.021481 | 0.018166 | 0.001122 |
| ego_swap | 0.021259 | 0.018247 | 0.001756 |
| exo_self | 0.005629 | 0.000826 | 0.000066 |
| exo_swap | 0.005846 | 0.000966 | 0.000005 |

判断：

- v4d 是当前最均衡的 stage-1 tokenizer。
- 相比 v2，confidence、code usage、effective code、ego/exo causality 都提升。
- private 去掉后损失变化很小，但 action 被置零或替错时损失明显变差，说明 action token 是主要可交换控制变量。
- exo 的 causality delta 绝对值仍小于 ego，但 zero sensitivity 和 random-code sensitivity 已经可见。
- semantic probe 没有 collapse，task NMI 略有提升。
- view/take NMI 比 v2 略高，需要后续用 held-out take 和 leakage classifier 进一步确认。

因此，本轮选择 v4d 作为当前推荐模型。

## 6. 当前模型仍然存在的问题

### 6.1 Exo causality 仍偏弱

ego token 替错后 MSE 增量约 0.013，而 exo random-take 增量只有约 0.0007-0.0009。
这说明 exo decoder 或 exo feature reconstruction 可能更容易依赖当前帧、视角残差或低频上下文完成重建。

后续需要：

- 更强的 exo transition/delta objective。
- 更明确的 cross-view action consistency。
- 检查 exo DINO feature 的 temporal variance 是否本身小于 ego。

### 6.2 Take leakage 还没有完全排除

v4d 的 take NMI：

- ego：0.4259。
- exo：0.4093。

这不是灾难性数值，但说明 token 可能仍编码了部分 take/scene/task context。
这需要在 held-out takes 上验证，否则 semantic NMI 可能被数据集结构放大。

### 6.3 Slot 分工仍需更清晰

v4d code usage 已经恢复到 46/64，但 4 个 action slot 是否分别承担不同动作因素，还需要更细粒度检查：

- per-slot code usage。
- per-slot entropy。
- per-slot semantic histogram。
- slot-drop ablation。
- slot permutation sensitivity。

### 6.4 Reconstruction 指标不等于动作语义

当前最强证据来自 causality ablation 和 private leakage ablation。
但真正要证明 action semantics，还需要外部标签、下游预测或跨视角检索支持。

## 7. 这个阶段怎样算做好

Stage-1 MVP 不需要达到最终机器人控制 token 的标准，但应该满足以下条件。

### 7.1 必须达标

| 类别 | 建议目标 |
| --- | --- |
| 训练稳定性 | 20k-40k steps 无 NaN、VQ 爆炸、loss 崩溃 |
| token export | ego token 稳定导出，shape 为 `(N, 1, 4)` |
| private export | private residual 不作为下游 action label 导出 |
| code usage | K=64 时 hard used codes 约 35-55 |
| effective codes | ego/exo effective codes 至少 15-20 |
| confidence | mean confidence 最好大于 0.25，当前 v4d 为 0.346 |
| causality | correct token 明显优于 shuffled/zero/random |
| private leakage | zero/shuffle private 不应摧毁 correct action 的优势 |
| swap reconstruction | self 与 swap 不应出现明显断裂 |

### 7.2 进入下一阶段前应补齐

| 类别 | 建议目标 |
| --- | --- |
| held-out take probe | 在 unseen takes 上仍保持 causality delta |
| view leakage classifier | 用 `q_act` 预测 ego/exo 视角接近 chance，至少不要明显高于 60% |
| take leakage classifier | take prediction 不应过强，take NMI 建议小于约 0.45 |
| semantic purity | action/contact label 对 code 有稳定富集 |
| ego predictability | 只用 ego current/history 能预测 `q_act`，并优于 unigram baseline |
| downstream utility | current ego + predicted/true `q_act` 对 future feature 预测优于 no-token/shuffled-token |

v4d 已经满足大部分 stage-1 内部指标，但还没有完成 held-out 与 downstream proof。

## 8. 后续实验建议

### 8.1 Held-out Takes Evaluation

将 take 划分为 train/val/test，而不是只在同一批 7999 transitions 上 probe。
重点看：

- causality delta 是否保持。
- code usage 是否保持。
- task/contact NMI 是否仍存在。
- take leakage 是否下降。

这是下一步最关键实验。

### 8.2 Leakage Classifier

训练轻量 classifier：

- 输入：`q_act` 或 code histogram。
- 预测：view、take、task、object、contact。

理想结果：

- view classifier 接近 chance。
- take classifier 不强。
- contact/action classifier 有明显信号。

这可以把 NMI 观察变成更可解释的诊断。

### 8.3 Ego-only Action Token Predictability

因为最终 token 要能从 ego stream 获得，所以需要验证：

- 用 ego current/history DINO feature 预测 tokenizer 导出的 `q_act`。
- 比较 top-1、top-5、NLL、per-slot accuracy。
- 将预测 token 放回 decoder，比较 future reconstruction 是否优于 shuffled/random token。

如果 ego-only predictor 无法预测 `q_act`，即使 tokenizer 本身很好，也还不能成为后续 WAM/action label。

### 8.4 Slot-level Semantic Analysis

对 4 个 action slots 分别做：

- code usage。
- entropy。
- top label histogram。
- slot dropout。
- pairwise mutual information。

目标是判断 slot 是否形成了互补因素，而不是 4 个 slot 重复编码同一件事。

### 8.5 Cross-view Retrieval

用 ego token 检索同一 transition 的 exo token，或用 exo token 检索 ego token。
比较：

- correct pair rank。
- same-take negative。
- random-take negative。
- same-task negative。

这可以直接验证 shared action code 是否跨视角对齐。

## 9. 本轮结论

本轮从原始 reconstruction tokenizer 出发，补齐了 causality、private leakage、semantic 三类验证，并围绕这些验证进行了多轮训练增强。

最终 v4d 的主要改进是：

- confidence 从 v2 的 0.195 提升到 0.346。
- used codes 恢复到 46/64。
- ego/exo effective codes 提升到约 21。
- ego causality delta 达到约 0.013。
- exo causality 和 zero sensitivity 有稳定正信号。
- action-without-private 仍能保留明显重建优势。
- private residual 没有成为主要动作通道。

当前判断：

```text
v4d 可以作为 FACT tokenizer stage-1 MVP 的当前候选模型。
```

但它还不是最终意义上的“动作语义 token”。
下一步应优先做 held-out take、leakage classifier、ego-only token predictor 和 cross-view retrieval。只有这些实验也站住，才能更有把握地把导出的 ego token 当作后续 WAM 或 action model 的训练信号。
