# Best CoDraft-Diff Combination

Kept as the current most valuable configuration:

- Selector: `all_features`
- Refiner: `10000-step draft_refine_adapter`
- Gate: `error_aware_adapter_gate`
- Refinement ratio: `0.20`

Best measured result:

- final_acc: `0.3327`
- top5: `0.5268`
- correction_precision: `0.2088`
- correction_recall: `0.0593`
- wrong_edit_rate: `0.0362`
- preserve: `0.9638`
- edit_gain: `3080`

For a more conservative gate:

- method: `error_aware_utility_gate`
- gate threshold: `0.8`
- final_acc: `0.3291`
- wrong_edit_rate: `0.0222`
- edit_gain: `2713`

Large training checkpoints were removed to save disk space. The retained CSV,
JSON, PNG, README, and decision files are sufficient for paper analysis and
plot reproduction.
