# FACT Tokenizer 交付文档

更新时间：2026-06-21

本文用于把当前 FACT tokenizer MVP 交接给后续开发者。交付范围只覆盖一阶段 FACT tokenizer：从 paired Ego/Exo transition 中学习 ego-accessible shared action token。当前不交付 WAM、Action Head、SONIC、robot deployment 等后续阶段实现。

## 1. GitHub 仓库内交付内容

本次应进入 GitHub 的内容包括：

- `AGENTS.md`：项目级指导，要求优先参考 `docs/fact_tokenizer_plan_3_0.md`。
- `README.md` / `README.zh-CN.md`：项目入口、环境、数据、训练、导出与验证命令。
- `requirements-fact-tokenizer.txt`：Python 依赖。
- `fact_tokenizer/`：核心 tokenizer 代码。
- `scripts/`：数据准备、训练、动态 GPU launcher、token 导出、probe、gate、visualization 和 v5/v6 训练脚本。
- `docs/fact_tokenizer_plan_3_0.md`：研究计划和设计原则。
- `docs/fact_tokenizer_experiment_conclusions_20260618.md`：截至 v6c 的综合实验结论。
- `docs/fact_tokenizer_next_improvement_plan_20260618.md`：下一轮改进计划。
- `docs/assets/fact_tokenizer_visualizations/`：已同步到仓库的训练曲线和诊断图，目前主要到 v5q。
- `LOG/`：早期阶段训练总结和优化日志。
- `backups/`：v6/v6c 关键重写前备份，便于交接时追溯。

## 2. 不进入 GitHub 的大文件资产

以下目录被 `.gitignore` 排除，不能直接作为普通 GitHub 文件上传：

| path | local size | reason |
|---|---:|---|
| `data/fact_egoexo/` | about 37G | NPZ 数据和 split，单文件最大约 8G |
| `outputs/fact_tokenizer/` | about 86G | checkpoint、训练日志、probe 结果、可视化 |
| `checkpoints/` | about 1.3G | DINO/torch hub cache 和其他模型权重 |
| `mvp_lam/` | about 639M | vendored upstream/reproduction checkout，非当前 FACT 主线必需 |

这些资产需要通过 NAS、对象存储、网盘、Hugging Face Dataset/Model、GitHub Release 外链或其他大文件渠道交接。普通 GitHub 仓库只保留代码、文档、轻量结果图和清单。

## 3. 必须单独交接的数据资产

主线继续实验至少需要以下文件：

```text
data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000.npz
data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000_report.json
data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz
data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz
data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_labels.jsonl
data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_labels.jsonl
data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_uids.txt
data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_uids.txt
data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/split_report.json
data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/transition48_data_report.json
data/fact_egoexo/failed_transition48_samples.jsonl
```

可选但建议保留：

```text
data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz
data/fact_egoexo/splits/diverse_500takes_seed123_80_20/
data/fact_egoexo/shards/train_diverse_500takes_t1p0_s1_48t_000000.npz
data/fact_egoexo/splits/diverse_500takes_t1p0_s1_48t_seed123_80_20/
```

## 4. 从零下载和准备 EgoExo4D 数据

如果没有直接交接 `data/fact_egoexo/`，可以用下面命令重新生成主线数据。需要先获得 EgoExo4D / Ego4D 数据访问权限，并在 shell 中配置 AWS 凭据。

### 4.1 环境变量

```bash
cd /data_all/sxh/FACT_tokenizer
conda activate fact_tokenizer

export AWS_ACCESS_KEY_ID=<your_egoexo_aws_access_key>
export AWS_SECRET_ACCESS_KEY=<your_egoexo_aws_secret_key>
export PYTHON_ENV_BIN=/home/sxh/.conda/envs/fact_tokenizer/bin
```

换机器后，把 `PYTHON_ENV_BIN` 改成新环境的 bin 目录。

### 4.2 下载 metadata

推荐先用官方 EgoExo CLI 下载 metadata：

```bash
mkdir -p data/egoexo4d

${PYTHON_ENV_BIN}/egoexo \
  -o data/egoexo4d \
  --release v2 \
  --parts metadata \
  --views ego exo \
  -y
```

完成后应至少有：

```text
data/egoexo4d/takes.json
```

### 4.3 选择 FACT 用的 take UID

当前主线使用 diverse 500 takes：

```bash
${PYTHON_ENV_BIN}/python scripts/select_egoexo_fact_uids.py \
  --egoexo-root data/egoexo4d \
  --output-uids data/egoexo4d/fact_debug/uids_500_diverse.txt \
  --output-jsonl data/egoexo4d/fact_debug/selected_takes_500_diverse.jsonl \
  --max-takes 500 \
  --min-duration-sec 48.5 \
  --diverse
```

如果只想做 smoke test，可先选少量 takes：

```bash
${PYTHON_ENV_BIN}/python scripts/select_egoexo_fact_uids.py \
  --egoexo-root data/egoexo4d \
  --output-uids data/egoexo4d/fact_debug/uids.txt \
  --output-jsonl data/egoexo4d/fact_debug/selected_takes.jsonl \
  --max-takes 3 \
  --min-duration-sec 8 \
  --diverse
```

### 4.4 下载 downscaled Ego/Exo 视频

方式 A：使用官方 EgoExo CLI，适合网络稳定时：

```bash
${PYTHON_ENV_BIN}/egoexo \
  -o data/egoexo4d \
  --release v2 \
  --parts downscaled_takes/448 \
  --views ego exo \
  --uids $(cat data/egoexo4d/fact_debug/uids_500_diverse.txt) \
  -y
```

仓库里也有封装脚本：

```bash
UID_FILE=data/egoexo4d/fact_debug/uids_500_diverse.txt \
OUT_DIR=data/egoexo4d \
RELEASE=v2 \
PYTHON_ENV_BIN=${PYTHON_ENV_BIN} \
bash scripts/download_egoexo_minimal.sh
```

方式 B：使用 manifest + boto3 按 UID 下载，适合官方 CLI 不稳定或需要更细控制并发时：

```bash
${PYTHON_ENV_BIN}/python scripts/download_egoexo_manifest_subset.py \
  --uids data/egoexo4d/fact_debug/uids_500_diverse.txt \
  --out-dir data/egoexo4d \
  --manifest-cache data/egoexo4d/fact_debug/downscaled_448_manifest.json \
  --failed-jsonl data/egoexo4d/fact_debug/download_failed.jsonl \
  --num-workers 32 \
  --transfer-concurrency 4
```

如果网络中途断流，可以改用 ranged 下载：

```bash
${PYTHON_ENV_BIN}/python scripts/download_egoexo_manifest_subset.py \
  --uids data/egoexo4d/fact_debug/uids_500_diverse.txt \
  --out-dir data/egoexo4d \
  --manifest-cache data/egoexo4d/fact_debug/downscaled_448_manifest.json \
  --failed-jsonl data/egoexo4d/fact_debug/download_failed_ranged.jsonl \
  --num-workers 16 \
  --download-mode ranged \
  --chunk-size-mib 1 \
  --chunk-workers 4 \
  --chunk-retries 8
```

如果 `download_failed*.jsonl` 非空，重新运行同一条命令即可跳过已完整下载的文件，继续补缺。

### 4.5 生成 transition48 NPZ 和 train/heldout split

主线数据配置为 `transition_sec=0.5`、`stride_sec=1.0`、每 take 48 个 transition，并按 take 划分 80/20：

```bash
${PYTHON_ENV_BIN}/python scripts/prepare_fact_transition48_split.py \
  --egoexo-root data/egoexo4d \
  --selected-jsonl data/egoexo4d/fact_debug/selected_takes_500_diverse.jsonl \
  --output-npz data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000.npz \
  --failed-jsonl data/fact_egoexo/failed_transition48_samples.jsonl \
  --prepare-report data/fact_egoexo/shards/train_diverse_500takes_t0p5_s1_48t_000000_report.json \
  --split-dir data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20 \
  --samples-per-take 48 \
  --stride-sec 1.0 \
  --transition-sec 0.5 \
  --resize 224 \
  --heldout-fraction 0.2 \
  --seed 123
```

默认不加 `--allow-short-takes`，因此短 take 会被跳过；保留下来的每个 take 必须严格写出 48 个 transition。

### 4.6 可选：生成 transition_sec=1.0 对照

v5p 做过 `transition_sec=1.0` 对照，结论是不推荐作为主线，但如果需要复现：

```bash
${PYTHON_ENV_BIN}/python scripts/prepare_fact_transition48_split.py \
  --egoexo-root data/egoexo4d \
  --selected-jsonl data/egoexo4d/fact_debug/selected_takes_500_diverse.jsonl \
  --output-npz data/fact_egoexo/shards/train_diverse_500takes_t1p0_s1_48t_000000.npz \
  --failed-jsonl data/fact_egoexo/failed_transition1p0_48_samples.jsonl \
  --prepare-report data/fact_egoexo/shards/train_diverse_500takes_t1p0_s1_48t_000000_report.json \
  --split-dir data/fact_egoexo/splits/diverse_500takes_t1p0_s1_48t_seed123_80_20 \
  --samples-per-take 48 \
  --stride-sec 1.0 \
  --transition-sec 1.0 \
  --resize 224 \
  --heldout-fraction 0.2 \
  --seed 123
```

### 4.7 快速检查数据

```bash
${PYTHON_ENV_BIN}/python - <<'PY'
import numpy as np
for path in [
    "data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz",
    "data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz",
]:
    with np.load(path, allow_pickle=False) as data:
        print(path)
        print({k: data[k].shape for k in data.files})
PY
```

期望主线 split 约为：

```text
train:   342 takes / 16416 transitions
heldout: 85 takes / 4080 transitions
```

## 5. 必须单独交接的实验产物

最小 checkpoint / result 资产包建议包含：

```text
outputs/fact_tokenizer/v5f_4p0_teacher_usage_8gpu_20260617_192843/
outputs/fact_tokenizer/egoexo_diverse_500takes_k64_35k_v4d_gentle_usage_from_v2_20260616_215152/
outputs/fact_tokenizer/v5p_transition48_from_v5m_randomcode_take_repair_4gpu_tmux/
outputs/fact_tokenizer/v5q_transition48_from_v5p_temporal_usage_repair_8gpu_20260618_234428/
outputs/fact_tokenizer/v5r_transition48_from_v5p_actionaware_capacity_8gpu_20260620_1145/
outputs/fact_tokenizer/v6a_transition48_delta_bottleneck_from_v5p_8gpu_20260620_2021/
outputs/fact_tokenizer/v6b_transition48_delta_full_from_v5p_8gpu_20260620_2200/
outputs/fact_tokenizer/v6c_transition48_usage_entropy_from_v6b_8gpu_20260621_1449/
```

每个 run 至少保留：

```text
fact_tokenizer.ckpt
metadata.json
code_usage.json
train_history.json
train_stdout.log
stage1_gate.log or action_token_probe_heldout/stage1_gate.json, if available
action_token_probe_heldout/, if available
visualizations/, if available
```

可以删除或低优先级保留的内容：

- 大量 `fact_tokenizer_step_*.ckpt` 中间 checkpoint；一般只需要 final `fact_tokenizer.ckpt` 和少数关键中间点。
- `outputs/lam_tokenizer/`，除非后续开发者要回看 LAM baseline。
- 早期 smoke run 的完整 checkpoint；文档中已经总结工程闭环结果。

## 6. 当前实验状态

当前一阶段 tokenizer 仍未通过 Stage-1 gate，不能冻结作为后续 WAM label tokenizer。

关键结论：

- v5l/v5m 证明 transition48 数据可以降低 take/view leakage。
- v5p 首次让 `ego_swap_random_code_delta` 通过阈值，说明 explicit random-code negative 有效。
- v5q 保住 random-code，但 same-take / temporal hard negative 和 usage pressure 没有解决核心问题。
- v5r 打开了更多 code，但 take/view leakage 明显反弹，说明只修 usage 不够。
- v6a 是 bottleneck + strong delta 的负结果，破坏已建立的 action causality。
- v6b 是当前较健康的参考点：通过 random-code、exo zero、action-without-private 和 view-invariance，但仍失败于 same-take、random-take、take leakage 和 heldout code usage。
- v6c 只有训练侧结果，尚缺 heldout gate；训练侧 code usage 从 v6b 的 37/64 收缩到 23/64，不应直接作为新主线。

详细数值见：

```text
docs/fact_tokenizer_experiment_conclusions_20260618.md
```

## 7. 推荐接手后的第一步

先补跑 v6c 的 heldout probe 和 gate：

```bash
/home/sxh/.conda/envs/fact_tokenizer/bin/python scripts/probe_fact_action_tokens.py \
  --checkpoint outputs/fact_tokenizer/v6c_transition48_usage_entropy_from_v6b_8gpu_20260621_1449/fact_tokenizer.ckpt \
  --input-npz data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz \
  --output-dir outputs/fact_tokenizer/v6c_transition48_usage_entropy_from_v6b_8gpu_20260621_1449/action_token_probe_heldout \
  --labels data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_labels.jsonl \
  --label-columns parent_task_name task_name university_name \
  --source-view-keys ego exo \
  --resize 224 \
  --batch-size 32 \
  --num-workers 4 \
  --device cuda

/home/sxh/.conda/envs/fact_tokenizer/bin/python scripts/evaluate_fact_stage1_gate.py \
  --probe-dir outputs/fact_tokenizer/v6c_transition48_usage_entropy_from_v6b_8gpu_20260621_1449/action_token_probe_heldout \
  --output-json outputs/fact_tokenizer/v6c_transition48_usage_entropy_from_v6b_8gpu_20260621_1449/action_token_probe_heldout/stage1_gate.json
```

如果 v6c gate 未改善，建议回到 v6b/v5p：

- 以 v6b 作为 full-context + mild-delta 参考点。
- 以 v5p 作为 random-code negative 的干净参考点。
- 重点重新设计 same-take / temporal action discrimination。
- 不要只加大 usage regularization；v5r/v6c 已显示这会带来 leakage 或 hard usage 收缩。

## 8. 环境和运行

本地实验使用 conda 环境：

```bash
conda activate fact_tokenizer
pip install -r requirements-fact-tokenizer.txt
```

主数据路径：

```text
data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/train_by_take.npz
data/fact_egoexo/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_by_take.npz
```

推荐从现有脚本继续：

```bash
bash scripts/run_fact_transition48_v6b_delta_full_train.sh
bash scripts/run_fact_transition48_v6c_usage_entropy_train.sh
```

注意：这些脚本中的默认 `TORCHRUN` 路径是本机路径 `/home/sxh/.conda/envs/fact_tokenizer/bin/torchrun`。换机器后需要通过环境变量覆盖：

```bash
TORCHRUN=/path/to/torchrun bash scripts/run_fact_transition48_v6b_delta_full_train.sh
```

## 9. 交付风险

- GitHub 上不包含大文件数据和 checkpoint。没有这些外部资产，后续只能阅读代码和文档，不能直接复现实验。
- v6c 缺正式 heldout gate，是当前最需要补的实验验证。
- 当前 tokenizer 不能冻结为 WAM label tokenizer。
- private residual 仍然只应作为 tokenizer 训练辅助，不能导出给下游 WAM 或 Action Head。
- 不建议在当前状态扩展到 WAM、Action Head、SONIC 或机器人部署模块。
