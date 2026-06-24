# FACT Tokenizer Filtering v0 / Filtered-v6b 95000 Gate Report

更新时间：2026-06-24

本文记录 `filtered-v6b` 主线从 `92000` 继续到 `95000` 的 1k-step gate sweep。结论只覆盖当前 Stage-1 FACT tokenizer，不扩展到 WAM、Action Head、SONIC 或机器人部署。

## 1. 目的

上一轮 `filtering_v0` 实验显示，task-relevant / interaction-heavy subset 能让 `v6b` 在 filtered heldout 上获得更好的 ego action discrimination：

```text
v6b@90000 -> filtered-v6b@92000
ego same-take delta: 0.004939 -> 0.005725
ego random-take delta: 0.005485 -> 0.006105
ego random-code delta: 0.031744 -> 0.035348
```

因此本轮继续验证：

```text
filtered-v6b 从 92000 继续到 95000，ego random-take 是否能继续靠近 0.008？
同时 random-code、used codes 和 take NMI 是否保持安全？
```

## 2. 数据和 Probe 口径

训练数据：

```text
/data_all/intern02/egoexo-task-filter/outputs/filtering_v0_500takes_20260624/filtered_npz/train_by_take.npz
```

heldout 数据：

```text
/data_all/intern02/egoexo-task-filter/outputs/filtering_v0_500takes_20260624/filtered_npz/heldout_by_take.npz
```

semantic / take labels：

```text
data/fact_egoexo_sxh_handoff/splits/diverse_500takes_t0p5_s1_48t_seed123_80_20/heldout_labels.jsonl
```

评估统一追踪：

- `ego_swap same_take_shuffle delta`
- `ego_swap random_take delta`
- `ego_swap random_code delta`
- `exo` 对应三项
- ego / exo used codes
- ego / exo tuple-level `take_uid nmi_sqrt`

## 3. 运行资产

`filtered-v6b@92000` 起点：

```text
outputs/fact_tokenizer/filtered_v6b_from_v6b90000_filtering_v0_interaction_8gpu_gate92000_20260624_172321/fact_tokenizer.ckpt
```

`93000` run：

```text
outputs/fact_tokenizer/filtered_v6b_from92000_to93000_filtering_v0_interaction_8gpu_20260624_185625
```

`94000` run：

```text
outputs/fact_tokenizer/filtered_v6b_from93000_to94000_filtering_v0_interaction_8gpu_20260624_191951
```

`95000` run：

```text
outputs/fact_tokenizer/filtered_v6b_from94000_to95000_filtering_v0_interaction_8gpu_20260624_194328
```

heldout probes：

```text
outputs/fact_tokenizer/filtered_v6b_from_v6b90000_filtering_v0_interaction_8gpu_gate92000_20260624_172321/heldout_probe_20260624_175730
outputs/fact_tokenizer/filtered_v6b_from92000_to93000_filtering_v0_interaction_8gpu_20260624_185625/heldout_probe_20260624_191648
outputs/fact_tokenizer/filtered_v6b_from93000_to94000_filtering_v0_interaction_8gpu_20260624_191951/heldout_probe_20260624_194032
outputs/fact_tokenizer/filtered_v6b_from94000_to95000_filtering_v0_interaction_8gpu_20260624_194328/heldout_probe_20260624_200333
```

take semantic probes：

```text
outputs/fact_tokenizer/v6b90000_on_filtering_v0_interaction_semantic_take_probe_20260624_200803
outputs/fact_tokenizer/filtered_v6b_from_v6b90000_filtering_v0_interaction_8gpu_gate92000_20260624_172321/semantic_take_probe_20260624_180412
```

其中 `93000`、`94000`、`95000` 的 heldout probe 已直接带 `take_uid parent_task_name task_name` 标签，因此不需要额外 semantic-only probe。

## 4. 训练完成情况

本轮为了保证每 1000 step 都有一个明确 checkpoint，拆成三个连续 run：

```text
92000 -> 93000
93000 -> 94000
94000 -> 95000
```

每段 final checkpoint 即对应千步 checkpoint。

完成状态：

| target | exit code | checkpoint | heldout probe | note |
|---|---:|---|---|---|
| 93000 | 0 | saved | done | no NaN / no OOM |
| 94000 | 0 | saved | done | no NaN / no OOM |
| 95000 | 0 | saved | done | no NaN / no OOM |

资源备注：

- 运行期间服务器上存在另一个 8-GPU workload。
- `filtered-v6b` 每卡约 9.2GB 显存，未触发 CUDA OOM。
- 训练结束后无 FACT 训练进程残留。

## 5. Filtered Heldout 结果

| run | step | ego used | exo used | ego same | ego random-take | ego random-code | exo same | exo random-take | exo random-code | ego take NMI | exo take NMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v6b@90000` | 89999 | 24 | 28 | 0.004939 | 0.005485 | 0.031744 | 0.001506 | 0.001565 | 0.026624 | 0.319844 | 0.391946 |
| `filtered-v6b@92000` | 91999 | 27 | 29 | 0.005725 | 0.006105 | 0.035348 | 0.001576 | 0.001503 | 0.024935 | 0.326989 | 0.404612 |
| `filtered-v6b@93000` | 92999 | 26 | 28 | 0.005572 | 0.005832 | 0.034765 | 0.001536 | 0.001588 | 0.024486 | 0.333035 | 0.405884 |
| `filtered-v6b@94000` | 93999 | 26 | 29 | 0.005878 | 0.006146 | 0.034650 | 0.001561 | 0.001548 | 0.024094 | 0.325621 | 0.408737 |
| `filtered-v6b@95000` | 94999 | 27 | 28 | 0.005903 | 0.006196 | 0.033256 | 0.001583 | 0.001601 | 0.023542 | 0.327247 | 0.405139 |

## 6. Delta Summary

`filtered-v6b@95000` vs `filtered-v6b@92000`：

| metric | delta |
|---|---:|
| ego used codes | +0 |
| exo used codes | -1 |
| ego same | +0.000178 |
| ego random-take | +0.000091 |
| ego random-code | -0.002092 |
| exo same | +0.000007 |
| exo random-take | +0.000097 |
| exo random-code | -0.001393 |
| ego take NMI | +0.000258 |
| exo take NMI | +0.000527 |

`filtered-v6b@95000` vs `v6b@90000`：

| metric | delta |
|---|---:|
| ego used codes | +3 |
| exo used codes | +0 |
| ego same | +0.000964 |
| ego random-take | +0.000711 |
| ego random-code | +0.001512 |
| exo same | +0.000078 |
| exo random-take | +0.000036 |
| exo random-code | -0.003082 |
| ego take NMI | +0.007403 |
| exo take NMI | +0.013193 |

## 7. 观察

正向点：

- `ego same` 从 `92000` 到 `95000` 继续小幅提升：`0.005725 -> 0.005903`。
- `ego random-take` 也小幅提升：`0.006105 -> 0.006196`。
- `ego random-code` 仍保持在安全线 `>= 0.025` 以上。
- ego used codes 保持 `27`。
- take NMI 基本持平，没有明显 take leakage 爆炸。

负向点：

- `ego random-take` 到 `95000` 仍只有 `0.006196`，没有明显靠近 `0.008`。
- `ego random-code` 从 `92000` 的 `0.035348` 降到 `95000` 的 `0.033256`。
- exo random-code 从 `92000` 的 `0.024935` 降到 `95000` 的 `0.023542`，低于原始 `v6b@90000` filtered-heldout baseline。
- `93000` 相比 `92000` 有短暂回落，说明 1k 内波动仍然明显。

## 8. 结论

本轮结论：

```text
filtered-v6b 主线从 92000 到 95000 仍有正向但很弱的 ego-side 收益。
92000 之后已经进入边际收益区。
```

更具体地说：

- `95000` 是当前 filtered-v6b sweep 中 ego same / ego random-take 最高的 checkpoint。
- 但提升幅度很小，不足以说明继续单纯延长训练会把 `ego random-take` 推到 `0.008`。
- `random-code` 开始下滑，说明继续延长需要谨慎。
- take leakage 没有明显恶化，因此当前问题不是 leakage 爆炸，而是 action discrimination 增长变慢。

## 9. 当前决策

不建议马上继续当前 recipe 往更长步数硬跑。

建议下一步：

1. 保留 `filtered-v6b@95000` 作为当前 filtered-v6b 最好 checkpoint。
2. 暂时不重启当前 `filtered-v6e` miner，因为 `ego random-take` 还没有足够靠近 `0.008`，且此前 filtered-v6e smoke 没打过 filtered-v6b smoke。
3. 优先做数据筛选人工校准，确认 `filtering_v0` 的 `fact_main` 是否真的是 interaction-rich。
4. 若继续方法线，应设计 data-aware negative，而不是复用当前 DINO-only v6e miner。
5. 后续所有新方法都应继续用同一个 filtered heldout 和 take NMI 口径比较。
