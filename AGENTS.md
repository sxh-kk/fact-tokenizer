# FACT Tokenizer Project Guidance

## Planning Reference

Before planning, implementing, refactoring, or evaluating work in this repository, consult:

- [docs/fact_tokenizer_plan_3_0.md](docs/fact_tokenizer_plan_3_0.md)

Treat this document as the project-level research plan for FACT tokenizer work unless the user explicitly supersedes it.

## Design Priorities

- Keep the first-stage MVP centered on ego-accessible shared action tokens learned from paired Ego/Exo transitions.
- Preserve the separation between shared action tokens and private view-residual latents.
- Private residuals are tokenizer-training auxiliaries; they should not become WAM labels or exported downstream action signals.
- Prefer changes that support the shared action codebook, swapped future reconstruction, within-view warm-up, confidence-gated Exo alignment, codebook balancing, and tokenizer quality validation described in the plan.
- When discussing or implementing later-stage WAM, Video DiT, Action Head, SONIC alignment, or robot ego adapter work, align interfaces with the staged training/inference flow in the plan.

## Scope Discipline

- This repository currently tracks the FACT tokenizer MVP. Do not expand implementation into WAM, Action Head, SONIC, or full robot deployment modules unless the user asks for that stage explicitly.
- If existing code, experiment results, or a requested change conflicts with the plan, call out the conflict and propose the smallest adjustment that keeps the project direction coherent.
