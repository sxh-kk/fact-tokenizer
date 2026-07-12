# FACT v7 连续效应表征实现

[English version](fact_effect_v7_implementation.md)

本分支以独立模块实现连续、关闭 VQ 的 FACT v7 effect 管线，其准确定位是：

> 基于位姿条件、经过相机运动补偿的二维可观测交互转变表征。

本实现不宣称具备深度、稠密三维光流、物理因果效应或跨 embodiment 有效性。

## 硬性保护

- 当前每条 cache 记录的 `depth_valid` 和 `flow_3d_valid` 都为 `false`。
- 训练开始前即拒绝任何 depth/3D-flow 请求。
- 旧 transition codelabel、gold probe 样本、locked 样本和 Assembly101 probe 均无法进入主训练器。
- 最终 cross decoder 只接收目标视角当前上下文、源视角连续语义和目标视角相机上下文；接口中不存在 private 或源视角图像平面几何参数。
- 最终模型的 state dict 不包含任何 VQ 或 codebook 状态。

## 迁移 NPY 的 transition 修正

不得直接把旧 filtering-v2 快照作为 v7 视频源。原视频重建审计表明：完整 no-filter 数组已经是 `t+0.5s`，而独立生成的 filtering-v2 数组实际使用 `t+1.0s`，尽管其历史目录名包含 `t0p5`。

首先运行 `python scripts/rebuild_fact_npy_transition.py --source-transition-seconds 0.5`，将 no-filter 每个首尾端点锚定并 hash 绑定到原视频的精确帧号；随后使用 `python scripts/materialize_fact_npy_subset.py`，按照冻结的源行索引派生 filtered 数据。重建命令要求显式提供已经实证核验的源间隔，防止混淆两类历史数据。两个工具均以原子方式发布，并记录内容 hash 与原视频 provenance。

基于旧 filtered 数组生成的任何 target cache 都只能作为 diagnostic，必须重新构建后才能用于正式实验。

## 构建顺序

1. 使用 `scripts/build_fact_effect_manifest.py` 构建基础 manifest。
2. 使用 `scripts/index_fact_take_quality.py` 合并人工质量分层和独立派生的采样权重。质量信息永远不能作为 effect target。
3. 使用 `scripts/index_fact_effect_annotations.py` 索引相机、文本和 phase 标注。
4. 使用 `scripts/index_fact_relations_masks.py` 流式索引 Relations mask。如果 annotation anchor 存在非零时间偏移，必须提供真实解码的 anchor-frame 图像；target builder 随后从该真实 anchor 向两个端点分别运行 RAFT。
5. 使用 `scripts/index_fact_sparse_pose_anchors.py` 索引稀疏 body/hand anchor。
6. 使用 `scripts/select_fact_target_audit_samples.py` 冻结一个按质量、位姿和 mask 分层抽取的 50 条审计集；随后仅针对这些行运行 `scripts/build_fact_effect_targets.py --sample-index-npy ... --sample-index-contract .../audit_selection.json`。
7. 使用 `scripts/audit_fact_effect_targets.py` 生成并人工复核可视化包。验证通过后会生成与 cache identity 绑定的 gate。后续任何正式全量 cache 或 shard 都必须提供该 gate；gate 释放前只允许构建 hash 绑定的 50 条审计集。至少 45/50 条通过全部对齐检查后才能释放 gate。
8. 使用 `--visual-audit-gate ...` 构建完整正式 cache。缺少该 gate 时，全量正式构建必须失败；独立 shard 只能通过 `scripts/merge_fact_effect_target_shards.py` 合并。
9. 使用 `scripts/freeze_fact_p3_eligibility.py` 冻结 P0/P1/P2/P3/P4 共用的 eligibility 清单和 P3 donor；该产物必须绑定 gold 的 47-take 排除文件。
10. 使用 `scripts/train_fact_effect_v7.py` 运行 C0-C2/T1-T2 和 P0-P4。脚本会强制检查 5k/20k 阶段、随机种子、optimizer update 数、global batch、GPU 数、take 排除、共同 eligibility 和最终实验契约。P4 每个 micro-batch 执行一次主数据 forward 和一次额外 Ego forward。

冻结的数值配置位于 `configs/fact_effect_v7.yaml`。

## Locked 与 gold 数据

- `scripts/prepare_fact_short73_locked.py` 明确区分 `provisional` 和 `final` 阶段。provisional manifest 只能为感知审计进行解码；final freeze 要求提供 train、heldout、diagnostic 和历史引用，并完成 Ego 与 Exo 的感知最近邻证据。候选数组、sample ID、provisional freeze 和所有引用 NPY 均被 hash 绑定。
- `scripts/materialize_fact_short73_locked.py` 将 73 × 8 条 transition 写入 staging，只有全部 584 条均成功解码后才原子发布。同时写入 `role=locked_test`、`training_valid=false` 和被训练器隔离的 effect manifest。
- `scripts/prepare_fact_gold300.py` 冻结 140/60/100 条样本，排除 500 条 diagnostic，并输出必须从 representation training 排除的 47 个 take。
- `scripts/prepare_fact_gold300_pilot.py` 从 24 个不属于正式 gold 的 heldout take 中各确定性选择一条样本，供 A/B 校准标注指南。其 freeze 明确禁止将 pilot 用于 encoder、probe、模型选择、正式评价或 locked test。
- `scripts/materialize_fact_gold300_review.py` 按精确 sample、take 和时间连接三套冻结源数组，并生成无损、使用不透明 ID 的 Ego/Exo 端点图。locked 样本不能与 train/dev 放入同一个交付包。每次运行都会原子发布三个物理隔离的产物：管理员 mapping/provenance、标注者 A 任务和仅含 dual 子集的标注者 B 任务。所有图像、源数据、source contract 和 freeze hash 都会被记录；系统拒绝 weak label、codelabel、模型预测或预填标签列。公开 delivery v2 还包含角色专属快速上手说明、冻结指南、不可变空白模板和独立的公开提交校验器。
- `scripts/import_fact_gold300_submissions.py` 是管理员专用的桥接工具，用于把不透明 review ID 映射回 canonical sample ID。它将两份 delivery 绑定到 sealed mapping hash，要求 A/B 完整覆盖且 B 严格限制为 dual 子集，拒绝混合导入 locked 与 non-locked，并以原子方式发布按 split 拆分的 canonical CSV；在 POSIX 系统上，locked 输出保持私有权限。
- `scripts/freeze_fact_npy_source_contract.py` 绑定 RGB 通道顺序、`t/t+0.5s` 端点语义、shape、dtype 和五个核心源 NPY 的 hash。root contract 必须引用通过的 `audit_fact_npy_source_semantics.py` 报告：每个被选中的 gold 端点都要从 hash 绑定的原始 Ego/Exo 视频重新解码，并与 NPY 精确一致。filtered 数组只有在逐行验证为父数据的精确子集后，才能继承 parent contract。
- `scripts/validate_fact_gold300.py` 可以读取物理隔离的 non-locked 与 locked canonical 文件而不生成混合 CSV，并要求正好 60 条双标数据的 effect/contact Cohen's kappa 都达到 0.70。
- `scripts/calibrate_fact_weak_semantics.py` 是启用 weak phase/contact target 的唯一入口。它将确定性动词映射绑定到全部 60 条 dev 数据，并要求两项实测 precision 均达到 0.80。
- `scripts/probe_fact_effect_gold.py` 只读取冻结特征，绝不更新 effect encoder。dev 阶段会冻结实际拟合得到的 probe 产物。单模型 locked 模式被明确禁用；`probe_fact_effect_locked_campaign.py` 只允许一次性占用该 locked set，并评估最终全部 12 个 control/seed probe，且严格使用已冻结的 artifact hash。一次性标记保存在 canonical locked asset root 中，因此不能通过更换输出目录绕过。

## 评价

- 冻结语义特征：`scripts/extract_fact_effect_features.py` 和 `scripts/merge_fact_effect_features.py`。
- paired-value GO gate：`scripts/evaluate_fact_paired_value.py`。leakage NMI 必须由冻结的 feature NPZ 计算，不能通过自由数值参数传入。使用 `scripts/extract_fact_leakage_features.py` 生成这些 NPZ；评价器输入绑定 checkpoint、run fingerprint、prediction report 和 manifest hash。
- campaign gate：`scripts/aggregate_fact_paired_campaign.py`。filtered、unfiltered 和 fresh-short73 三类报告缺少任何一类时都拒绝给出 GO。
- Assembly101：使用 `scripts/prepare_fact_assembly101_probe.py` 冻结固定 8/2 split；源 NPZ 可通过 `scripts/unpack_fact_assembly101.py` 解包。模型、配置和阈值全部冻结后，再一次性运行 `scripts/probe_fact_assembly101.py`。其 split 和全局 consumed marker 都绑定到 canonical Assembly asset root。

filtered、完整 unfiltered 和 fresh short73 的结果必须分开报告。Assembly101 不能用于模型或阈值选择。

## 正式运行前置条件

正式 screen/final 训练还必须提供：

- `--gold-freeze` 及其精确对应的 47-take 排除文件；
- P0-P4 共用的 `--paired-eligibility` 产物；
- 正式 target cache，其 manifest、RGB、DINO 和代码 hash 必须与当前输入一致，且配置中包含已经通过的 50 条可视化 release；
- paired run 之间完全相同的 optimizer、sampling、objective、模型维度和 shared control-contract hash；
- 与主数据不相交、规模至少相等的 P4 额外 Ego 数据池；
- 与 cache 绑定的 P1 Exo target/quality provenance。
