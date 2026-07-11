# FACT Tokenizer MVP

The isolated continuous FACT v7 effect pipeline is documented in
[`docs/fact_effect_v7_implementation.md`](docs/fact_effect_v7_implementation.md).

[中文版](README.zh-CN.md)

First-stage FACT tokenizer prototype for learning ego-accessible shared action tokens from paired ego/exo video transitions.

The current MVP supports:

- paired NPZ data loading with canonical `ego` and `exo` views
- frozen DINOv2 feature reconstruction
- ego/exo view encoders
- shared VQ action codebook
- private residual branch used only during tokenizer training
- self and swapped reconstruction paths
- DDP training with dynamic GPU launcher
- ego shared action token extraction and validation scripts

## Repository Layout

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

Large local folders such as `data/`, `outputs/`, `checkpoints/`, `logs/`, the vendored `mvp_lam/` checkout, and unrelated reproduction helpers are intentionally ignored by git.

## Environment

The local experiments used a conda environment named `fact_tokenizer`.

```bash
pip install -r requirements-fact-tokenizer.txt
```

## Data

The first large local run uses a paired EgoExo4D NPZ shard:

```text
data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz
```

Expected arrays:

- `ego`: `(B, 2, 224, 224, 3)` uint8
- `exo`: `(B, 2, 224, 224, 3)` uint8

Data is not tracked in git. Use the EgoExo4D CLI and local credentials to download the required parts, then build the NPZ with:

```bash
python scripts/prepare_fact_egoexo_npz.py
```

## Train

Single script training:

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

Dynamic GPU launcher:

```bash
python scripts/launch_fact_dynamic_gpus.py \
  --gpu-ids 0 1 2 3 4 5 6 7 \
  --min-gpus 1 \
  --max-gpus 8
```

The launcher starts with any idle GPU and can restart from checkpoints with more GPUs when they become available. This is checkpoint-based restart, not true DDP hot-plugging.

## Export Tokens

```bash
python scripts/extract_fact_tokens.py \
  --checkpoint outputs/fact_tokenizer/run_name/fact_tokenizer.ckpt \
  --input-npz data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz \
  --output-npz outputs/fact_tokenizer/run_name/ego_tokens.npz
```

Exported tokens contain only the ego shared action branch:

- `indices`
- `soft_probs`
- `confidence`

Private residual latents are not exported.

## Validation

```bash
python scripts/validate_fact_tokens.py \
  --checkpoint outputs/fact_tokenizer/run_name/fact_tokenizer.ckpt \
  --input-npz data/fact_egoexo/shards/train_diverse_500takes_16t_000000.npz
```

Useful metrics to monitor:

- self and swapped DINO feature reconstruction MSE
- VQ/codebook loss
- code usage fraction
- ego/exo assignment KL
- token confidence histogram

Mechanism-level action-token probes:

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

This writes:

- `causality_ablation.csv/json`: correct token vs global shuffle, same-take shuffle, random-take, temporal-offset, zero, and random-code controls.
- `private_leakage_ablation.csv/json`: action-token/private-residual 3x3 ablation plus private-dropout sweeps.
- `semantic_probe.json`: optional label-based purity/NMI/conditional histograms, plus automatic view-invariance and take-leakage controls.
- `probe_summary.json`: compact readout of the most important deltas.

If you have take-level or interval labels, pass them directly:

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
