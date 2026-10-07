# Experiment map

The small-model feasibility phase is concluded as of 2026-10-07. Read the root README and [claim ledger](../docs/RESULTS.md) before interpreting historical tables. Paths are intentionally stable because later studies import earlier modules.

| Directory / entry point | Role | Status |
|---|---|---|
| `dynamic_boundary_rl/run_consistent_dual.py` | Current canonical continuation: content KD, length-policy RL, aligned categorical risk, persistent optimizers | Completed; no low-degradation speedup |
| `dynamic_boundary_rl/run_dynamic_experiment.py` | First dynamic-boundary/dual experiment | Historical; policy/dual mismatch retained for audit |
| `two_model_rl/` | Fixed evaluator plus content-LoRA and policy updates; teacher scoring and model reconstruction utilities | Completed earlier comparison and current dependency |
| `adaptive_k_rl/` | Maximum horizon four, variable actual length one through four; saved rollout scoring | Completed earlier experiment and current dependency |
| `fixed_k_gate/` | Earlier fixed-block/binary gate; finite-mixture continuation implementation | Historical design; still an imported dependency, not the final policy |
| `layer_fusion_probe/` | Read every frozen layer through low-rank projections and a nonlinear MLP | Completed KD-only diagnostic; not all-layer backbone LoRA, not current inference |
| `bpe_probe/` | Early natural-language probes, EOB studies, corrected data preparation | Old result snapshots are historical; use audited data loader only |
| `synthetic/` | Controlled character-level experiment and attributed nanoGPT implementation | Mechanism evidence; not natural-language/GPU performance |

The final continuation deliberately does not add the all-layer MLP. Distillation-only fusion results and RL results must not be attributed to the same actor. No old files were moved into `src/` solely for appearance: that would break the established import and checkpoint contracts.

## Reading order

[Current report](../docs/CONSISTENT_DYNAMIC_BOUNDARY_2026_10_07.md), [replay commands](../docs/REPRODUCIBILITY.md), [theory](../docs/THEORY_JA.md), then the chronological table in [RESULTS.md](../docs/RESULTS.md).

Archived experimental branches and draft discussions remain provenance. The closing integration does not erase failed variants, declare every historical hypothesis true, or schedule more experiments.
