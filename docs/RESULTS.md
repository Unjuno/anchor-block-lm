# Results, evidence levels, and corrections

Snapshot date: 2026-10-07. This page separates observations, mathematical consequences, and untested extrapolations. The closing change introduces no new model training or latency experiment.

## Claim ledger

| Statement | Status | Evidence / limit |
|---|---|---|
| Actual commitment length is context-dependent and learned | Observed | Dynamic and consistent-policy probes; maximum four, actual one through four |
| Boundaries can change as actor training proceeds | Observed | Final continuation: argmax changes in 10/64, 0/64, 5/64 fixed contexts; probabilities can change without an argmax change |
| Frozen AR/evaluator tensors stay unchanged | Verified in recorded runs | Weight hashes, non-LoRA tensor comparison and checked-input AR-logit tests |
| The final method has an exact teacher sequence distribution | Not established; generally false for approximate tails | Frozen anchors do not correct approximate continuation tokens |
| Final continuation preserves quality while becoming faster | Not demonstrated | Progress/quality screen 0/3; all final speed ratios below one |
| Reward hacking occurred | Not established | Reference reward increased while risk decreased; shorter commits explain a tradeoff |
| LoRA-RL is indistinguishable from full-weight RL | Not tested here | Do not infer equivalence from using LoRA |
| Increasing model size guarantees speedup | False as an unconditional inference | Requires favorable auxiliary/cache cost scaling and maintained quality-feasible block lengths |
| Small-model feasibility phase may be closed | Scope decision | A reproducible negative-performance result is a valid stopping point; not a successful accelerator claim |

## Canonical final measurements

Use [CONSISTENT_DYNAMIC_BOUNDARY_2026_10_07.md](CONSISTENT_DYNAMIC_BOUNDARY_2026_10_07.md) and its [unchanged summary](evidence/consistent_dynamic_boundary_2026_10_07.json). Content LoRA receives teacher-sample KD; length LoRA receives REINFORCE. This differs from the earlier content-RL experiment. No all-layer fusion is used in the final continuation.

The final sampled-policy results across three seeds span 1.486–1.541 tokens/backbone call and 0.261–0.341 nat/token augmented-trace discrepancy. Speed ratios are 0.922–0.935 against ordinary AR. These three ranges must not be combined to invent a fourth, better operating point. Within-seed before/after changes are given in the README and original report. One corpus, reused dev windows, three training seeds and Monte Carlo scoring limit the conclusions.

Per-decision train-risk targets remain unchanged. Two final point estimates meet the original budget, but every reported trajectory-cluster interval crosses its budget. A soft adaptive penalty is not a hard per-block or full-sequence guarantee.

## Corrected cost accounting

The earlier conversational estimate combined tokens/call from the 16-prompt quality evaluation with speed from the separate 8-prompt timing workload. That yields only a rough estimate. Use the **timing workload's own committed tokens and calls**:

| Seed | Timed tokens | Timed actor calls | Timed tokens/call | AR median s | Actor median s | Effective overhead ratio | Latency increase |
|---|---:|---:|---:|---:|---:|---:|---:|
| 48017 | 384 | 246 | 1.560976 | 0.272815486 | 0.291640946 | 0.668690 | 6.900% |
| 48018 | 384 | 252 | 1.523810 | 0.272129431 | 0.291124696 | 0.630175 | 6.980% |
| 48019 | 384 | 254 | 1.511811 | 0.267694435 | 0.290371178 | 0.639878 | 8.471% |

The ratio is an aggregate timing identity, **not an isolated head/MLP profile**. It absorbs every difference between the timed actor and AR work. Individual components cannot be identified from one ratio. Time ranges and source-file hashes are in [final_snapshot_2026_10_07.json](evidence/final_snapshot_2026_10_07.json). Run `python scripts/verify_snapshot.py` from the repository root to recompute it.

Throughput and latency percentages differ. At 0.935× throughput, throughput drops about 6.5%, whereas time rises about 7.0%. In a hypothetical model with overhead ratio 0.62 and one-token commits, speed is 1/1.62: throughput drops 38.27%, but latency rises 62%. The old phrase "38% slower" was ambiguous and must not be used as a latency statement.

## Earlier studies are preserved, not pooled

| Study | Reading | Interpretation |
|---|---|---|
| Synthetic character-level mechanism | [synthetic](../experiments/synthetic/README.md) | Controlled mechanism, not a natural-language speed result |
| Early BPE/EOB/preselected/random gating | [audit](RESULTS_AUDIT.md) | Historical data/preprocessing and comparison limitations; not canonical validation |
| Corrected fixed-horizon binary gate | [report](FIXED_K_GATE_RUN_2026_10_05.md) | Historical binary choice, not the final variable-length specification |
| Adaptive length 1–4 | [report](ADAPTIVE_K_RL_2026_10_05.md) | Length selection learned; compression alone does not imply speed |
| Fixed evaluator plus content RL | [report](TWO_MODEL_LORA_INITIAL_RESULT.md) | Distribution fidelity worsened in the tested setting |
| All-layer low-rank MLP | [report](LAYER_FUSION_RESULT_2026_10_06.md) | KD-only; tiny/inconsistent incremental benefit; no new RL result |
| Dynamic-boundary first version | [report](DYNAMIC_BOUNDARY_CONSTRAINED_RL_2026_10_07.md) | Boundary movement observed; policy/dual mismatch diagnosed |
| Consistent categorical continuation | [report](CONSISTENT_DYNAMIC_BOUNDARY_2026_10_07.md) | Current closing evidence; shorter outputs, reduced discrepancy, no successful speedup |

The all-three-seed fusion GO rule was local to that historical architecture probe. It is **not a theorem or a prerequisite** for studying adaptive block learning generally.

## Novelty and comparison boundaries

This repository is a small-model implementation/feasibility record. It does not establish priority over multi-token prediction, blockwise decoding, on-policy distillation, adaptive computation, or speculative decoding. See [REFERENCES.md](REFERENCES.md). Ordinary AR is the primary performance reference. Earlier random/confidence controls are supporting diagnostics, not the research goal.
