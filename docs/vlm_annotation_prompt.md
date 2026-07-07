# VLM Annotation Prompt

You are judging whether an Ego-Exo4D take is useful for training the first-stage FACT tokenizer.

Use the contact sheet and metadata to assess action-token training value. Prefer takes where the ego view has hand/object/contact or active interaction cues, the exo view has body/global context, and the take contains phase changes such as approach, reach, contact, carry, place, or release.

Return strict JSON only:

```json
{
  "ego_hand_visibility": 0,
  "exo_body_visibility": 0,
  "object_interaction": 0,
  "phase_diversity": 0,
  "take_relevance": "F_bad_or_unclear",
  "usable_for": "discard",
  "confidence": 0.0,
  "reason": "short reason"
}
```

Allowed integer scores:

- `0`: absent or unusable
- `1`: weak, partial, or uncertain
- `2`: clear and useful

Allowed `take_relevance`:

- `A_interaction_rich`
- `B_loco_body`
- `C_active_view_only`
- `D_scene_only`
- `E_fine_dexterous`
- `F_bad_or_unclear`

Allowed `usable_for`:

- `tokenizer_main`
- `loco_aux`
- `discard`
- `diagnostic_candidate`

Do not mark a take as `tokenizer_main` based only on task name, scene appearance, camera motion, or pretty frames. If hand-object/contact evidence is weak but whole-body locomotion is clear, prefer `loco_aux`. If evidence is ambiguous or fine dexterous beyond the current coarse tokenizer target, prefer `diagnostic_candidate`.
