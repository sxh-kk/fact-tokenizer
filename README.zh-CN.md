# FACT Tokenizer MVP

[English](README.md)

FACT tokenizer 第一阶段原型，用于从 ego/exo 配对视频 transition 中学习 ego-accessible shared action token。

当前 MVP 支持：

- 使用 canonical `ego` / `exo` 视角的 paired NPZ 数据加载
- 冻结 DINOv2 backbone，使用 DINO feature 作为重建目标
- ego/exo 双 view encoder
- 共享 VQ action codebook
- private residual branch，只参与 tokenizer 训练，不导出给下游 WAM
- self reconstruction 和 swapped reconstruction 两条训练路径
- DDP 训练
- 动态 GPU launcher：有空闲卡就启动，更多卡空出来后可基于 checkpoint 重启扩容
- ego shared action token 导出与机制验证脚本

## 代码结构

```text
fact_tokenizer/
  data.py                  # paired NPZ dataset
  model.py                 # FACT tokenizer model, VQ, DINO feature path
  losses.py                # reconstruction, swap, VQ, KL, balance, private losses
  utils.py                 # shared helpers
scripts/
  select_egoexo_fact_uids.py
  prepare_fact_egoexo_npz.py
  train_fact_npz_debug.py
  launch_fact_dynamic_gpus.py
  extract_fact_tokens.py
  validate_fact_tokens.py
  probe_fact_action_tokens.py
  visualize_fact_run.py
```

本仓库只追踪 FACT tokenizer 相关源码。以下本地大文件目录不会进入 git：

- `data/`
- `outputs/`
- `checkpoints/`
- `logs/`
- vendored `mvp_lam/`
- 无关复现辅助目录

## 环境

本地实验使用的 conda 环境名为 `fact_tokenizer`。

```bash
pip install -r requirements-fact-tokenizer.txt
```

## 数据

第一版大规模 smoke run 使用 EgoExo4D paired NPZ shard：

```text
data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz
```

期望数组：

- `ego`: `(B, 2, 224, 224, 3)` uint8
- `exo`: `(B, 2, 224, 224, 3)` uint8

数据不进入 git。需要先用 EgoExo4D CLI 和本地授权下载必要数据，再转换成统一 NPZ schema：

```bash
python scripts/prepare_fact_egoexo_npz.py
```

当前第一阶段主要需要：

- EgoExo4D metadata
- `downscaled_takes/448`
- annotations，用于后续 token 质量分析和语义验证

## 训练

单脚本训练示例：

```bash
python scripts/train_fact_npz_debug.py \
  --ddp \
  --input-npz data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz \
  --output-dir outputs/fact_tokenizer/run_name \
  --source-view-keys ego exo \
  --steps 20000 \
  --batch-size 8 \
  --resize 224 \
  --backbone dino \
  --device cuda \
  --model-dim 128 \
  --dino-dim 768 \
  --latent-dim 32 \
  --private-dim 8 \
  --num-latents 64 \
  --num-private-slots 1 \
  --num-heads 4 \
  --patch-size 14 \
  --enc-blocks 1 \
  --dec-blocks 1 \
  --lr 5e-5 \
  --vq-beta 0.25 \
  --balance-weight 0.05 \
  --private-reg-weight 0.01 \
  --save-every 1000
```

动态 GPU launcher：

```bash
python scripts/launch_fact_dynamic_gpus.py \
  --gpu-ids 0 1 2 3 4 5 6 7 \
  --min-gpus 1 \
  --max-gpus 8
```

说明：PyTorch DDP 的 `world_size` 在进程启动时固定，因此不能真正热加入 GPU。这里采用更稳的“checkpoint-based restart”：先用空闲 GPU 启动；如果后续发现更多 GPU 空闲，则等到新 checkpoint 出现后重启训练，并从 checkpoint resume 到更多 GPU。

## 模型思路

输入是两帧 transition：

```text
current -> future
```

每个 view encoder 输出两类 latent：

- `z_act`: shared action latent，ego/exo 共用同一个 VQ codebook
- `r_priv`: private residual latent，连续变量，不量化，只用于训练重建

训练路径包括：

- ego self reconstruction：`ego current + q_act_ego + r_priv_ego -> ego future`
- exo self reconstruction：`exo current + q_act_exo + r_priv_exo -> exo future`
- ego swapped reconstruction：`ego current + q_act_exo + r_priv_ego -> ego future`
- exo swapped reconstruction：`exo current + q_act_ego + r_priv_exo -> exo future`

目标是让 shared action codebook 尽量捕获跨视角一致的动作信息，而让 view-specific 细节进入 private residual。

## Loss

当前实现包括：

- `L_self`: view 内 future DINO feature reconstruction
- `L_swap`: swapped shared action reconstruction
- `L_vq`: codebook + commitment loss
- `L_kl`: confidence-gated ego/exo assignment alignment
- `L_balance`: batch-level code usage balancing
- `L_private_reg`: private residual L2 bottleneck 正则

默认训练调度：

- 前 20% steps：`self=1.0`，`swap` 从 `0 -> 0.5`
- 20% 后：`self=0.2`，`swap=1.0`
- `kl` 从 10% steps 后开启，最终权重 `0.1`

## 导出 Token

```bash
python scripts/extract_fact_tokens.py \
  --checkpoint outputs/fact_tokenizer/run_name/fact_tokenizer.ckpt \
  --input-npz data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz \
  --output-npz outputs/fact_tokenizer/run_name/ego_tokens.npz
```

导出的 token 只包含 ego shared action branch：

- `indices`
- `soft_probs`
- `confidence`

private residual 不导出。

## 验证

```bash
python scripts/validate_fact_tokens.py \
  --checkpoint outputs/fact_tokenizer/run_name/fact_tokenizer.ckpt \
  --input-npz data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz
```

建议关注的指标：

- self reconstruction DINO feature MSE
- swapped reconstruction DINO feature MSE
- VQ/codebook loss
- code usage fraction
- ego/exo assignment KL
- token confidence histogram

第一阶段目标是让 token 机制跑通：能训练、能保存、能恢复、能导出、能做基本验证。第一版不要求 token 已经有很强语义纯度。

三类 action-token 机制验证：

```bash
python scripts/probe_fact_action_tokens.py \
  --checkpoint outputs/fact_tokenizer/run_name/fact_tokenizer.ckpt \
  --input-npz data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz \
  --output-dir outputs/fact_tokenizer/run_name/action_token_probe \
  --source-view-keys ego exo \
  --resize 224 \
  --batch-size 8 \
  --device cuda
```

输出文件：

- `causality_ablation.csv/json`：比较 correct token、global shuffle、same-take shuffle、random-take、temporal-offset、zero、random-code 控制组。
- `private_leakage_ablation.csv/json`：action token / private residual 的 3x3 ablation，以及 private dropout sweep。
- `semantic_probe.json`：可选标签的 purity、NMI、conditional histogram，同时自动报告 view-invariance 和 take-leakage 控制项。
- `probe_summary.json`：最重要差值的浓缩摘要，适合直接读实验结论。

如果有 take 级或 interval 级标签，可以直接接入：

```bash
python scripts/probe_fact_action_tokens.py \
  --checkpoint outputs/fact_tokenizer/run_name/fact_tokenizer.ckpt \
  --input-npz data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz \
  --output-dir outputs/fact_tokenizer/run_name/action_token_probe_with_labels \
  --labels data/egoexo4d/fact_debug/selected_takes_500_diverse.jsonl \
  --label-columns parent_task_name task_name university_name \
  --source-view-keys ego exo \
  --resize 224 \
  --batch-size 8 \
  --device cuda
```
