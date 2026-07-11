# FACT v7 continuous effect implementation

This branch is an isolated, VQ-off implementation of a
pose-conditioned, camera-compensated 2D observed interaction transition
representation. It does not claim depth, dense 3D flow, physical causality, or
cross-embodiment validity.

## Hard guards

- `depth_valid` and `flow_3d_valid` are false in every current cache entry.
- Requests for depth/3D flow fail before training.
- Legacy transition codelabels, gold probe samples, locked samples, and the
  Assembly101 probe cannot enter the main trainer.
- The final cross decoder accepts target current context, source continuous
  semantics, and target camera context. It has no private/source-image-geometry
  argument.
- The final state dict contains no VQ/codebook state.

## Migrated NPY transition correction

The filtering-v2 snapshot must not be used directly as a v7 video source.
Raw-video reconstruction showed that the complete no-filter arrays are already
`t+0.5s`, while the independently materialized filtering-v2 arrays use
`t+1.0s` despite the historical directory name containing `t0p5`. Use
`rebuild_fact_npy_transition.py --source-transition-seconds 0.5` to anchor and
hash-bind every no-filter endpoint to exact raw-video frame indices, then use
`materialize_fact_npy_subset.py` to derive filtered rows by the frozen source
row index. The rebuild CLI requires the empirically verified source spacing so
the two legacy families cannot be confused. Both tools publish atomically with
content and raw-video provenance hashes. Any target cache built from the
legacy filtered arrays is diagnostic-only and must be rebuilt.

## Build order

1. Build base manifests with `scripts/build_fact_effect_manifest.py`.
2. Join the human quality strata and separately derived sampling weights with
   `scripts/index_fact_take_quality.py`. Quality is never an effect target.
3. Index camera/text/phase with `scripts/index_fact_effect_annotations.py`.
4. Stream Relations masks with `scripts/index_fact_relations_masks.py`.
   A nonzero annotation-anchor offset requires a decoded anchor-frame image;
   the target builder then runs RAFT from that real anchor to both endpoints.
5. Index sparse body/hand anchors with
   `scripts/index_fact_sparse_pose_anchors.py`.
6. Freeze a 50-row quality/pose/mask-stratified audit selection with
   `scripts/select_fact_target_audit_samples.py`, then build only those rows
   with `scripts/build_fact_effect_targets.py --sample-index-npy ...
   --sample-index-contract .../audit_selection.json`.
7. Create and review the visualization pack with
   `scripts/audit_fact_effect_targets.py`. Validation emits a cache-identity
   bound gate. Every later formal full cache or shard must supply that gate;
   only the hash-bound 50-row audit selection may be built before it.
   bound gate only after at least 45/50 rows pass all alignment checks.
8. Build the full formal cache with `--visual-audit-gate ...`. A full formal
   build fails without that gate. Independent shards are merged only through
   `scripts/merge_fact_effect_target_shards.py`.
9. Freeze the shared P0/P1/P2/P3/P4 eligibility list and P3 donors with
   `scripts/freeze_fact_p3_eligibility.py`; it is bound to the 47-take gold
   exclusion file.
10. Run C0-C2/T1-T2 and P0-P4 through
   `scripts/train_fact_effect_v7.py`. The script enforces the 5k/20k seed,
   optimizer-update count, global-batch, GPU-count, take exclusion, common
   eligibility, and final-experiment contracts. P4 executes one primary and
   one extra-Ego forward per micro-batch.

The frozen numerical configuration is in
`configs/fact_effect_v7.yaml`.

## Locked and gold data

- `scripts/prepare_fact_short73_locked.py` has explicit `provisional` and
  `final` stages. Provisional manifests can only be decoded for the perceptual
  audit; final freeze requires train/heldout/diagnostic/history references and
  completed Ego and Exo perceptual-nearest-neighbor evidence. Candidate arrays,
  sample IDs, the provisional freeze, and every reference NPY are hash-bound.
- `scripts/materialize_fact_short73_locked.py` writes 73 x 8 transitions to a
  staging directory and atomically publishes only after all 584 transitions
  decode. It also writes `role=locked_test`, `training_valid=false`, and a
  trainer-quarantined effect manifest.
- `scripts/prepare_fact_gold300.py` freezes 140/60/100 samples, excludes the
  500 diagnostics, and writes the 47 representation-training take exclusions.
- `scripts/materialize_fact_gold300_review.py` joins the three frozen source
  arrays by exact sample/take/time and renders lossless, opaque-ID Ego/Exo
  endpoint sheets. Locked samples cannot share a pack with train/dev. Each run
  atomically publishes three physically separate artifacts: admin mapping and
  provenance, annotator A tasks, and dual-only annotator B tasks. Every
  image/source/source-contract/freeze hash is recorded; weak, codelabel,
  prediction, or pre-filled columns are rejected.
- `scripts/freeze_fact_npy_source_contract.py` binds RGB color order,
  `t/t+0.5s` endpoint semantics, shapes/dtypes, and all five source NPY hashes.
  A root contract requires a passed `audit_fact_npy_source_semantics.py`
  report: every selected gold endpoint is decoded again from its hash-bound
  raw Ego/Exo video and must match the NPY exactly. Filtered arrays can inherit
  a parent contract only after exact row-by-row subset verification.
- `scripts/validate_fact_gold300.py` requires both effect/contact Cohen kappa
  to reach 0.70 on the 60 dual-annotation rows.
- `scripts/calibrate_fact_weak_semantics.py` is the only way to enable weak
  phase/contact targets. It binds the deterministic verb map to all 60 dev
  rows and requires both measured precisions to reach 0.80.
- `scripts/probe_fact_effect_gold.py` is feature-only: the effect encoder is
  never updated. Dev freezes the actual fitted probe artifacts. Single-model
  locked mode is deliberately disabled; `probe_fact_effect_locked_campaign.py`
  claims the set once and evaluates all 12 final control/seed probes using
  those exact artifact hashes. The one-shot marker lives in the canonical
  locked asset root, so changing output directories cannot bypass it.

## Evaluation

- Frozen semantic features: `scripts/extract_fact_effect_features.py` and
  `scripts/merge_fact_effect_features.py`.
- Paired-value GO gate: `scripts/evaluate_fact_paired_value.py`. Leakage NMI is
  computed from frozen feature NPZs, not supplied as a free numeric argument.
  Produce those NPZs with `scripts/extract_fact_leakage_features.py`; evaluator
  inputs are bound to checkpoint, run fingerprint, prediction report, and
  manifest hashes.
- Campaign gate: `scripts/aggregate_fact_paired_campaign.py` refuses a GO
  unless filtered, unfiltered, and fresh-short73 reports are all present.
- Assembly101: freeze a fixed 8/2 split with
  `scripts/prepare_fact_assembly101_probe.py` (use
  `scripts/unpack_fact_assembly101.py` for the source NPZ), then run the
  one-shot `scripts/probe_fact_assembly101.py` after model/config/threshold
  freeze. Its split and global consumed marker are hash-bound to the canonical
  Assembly asset root.

Filtered, complete unfiltered, and fresh short73 results must remain separate.
Assembly101 must not be used for model or threshold selection.

## Formal-run preconditions

Formal screen/final training additionally requires:

- `--gold-freeze`, plus its exact 47-take exclusion file;
- the common `--paired-eligibility` artifact for P0-P4;
- a formal target cache whose manifest/RGB/DINO/code hashes match the active
  inputs and whose config contains the passed 50-row visual release;
- identical optimizer, sampling, objective, model dimensions, and shared
  control-contract hash across paired runs;
- a disjoint, at-least-equal-size extra Ego pool for P4;
- cache-bound Exo target/quality provenance for P1.
