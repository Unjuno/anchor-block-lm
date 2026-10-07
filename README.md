# Anchor Block LM

**Research snapshot: small-model feasibility study concluded, 2026-10-07.**

A nanoGPT research prototype that learns **where to emit several tokens and where to return to one-token autoregressive generation**. Actual block boundaries can change with both context and training. The maximum prediction horizon is four total tokens; the committed length is not fixed.

**Established:** a frozen AR anchor, trainable continuation branches, and a LoRA length policy can be combined and trained; learned boundaries move. **Not established:** low-degradation wall-clock acceleration, exact sampling-distribution preservation, or faster performance merely by increasing model size.

[Results and claim boundaries](docs/RESULTS.md) · [Related work / novelty audit](docs/RELATED_WORK_2026.md) · [Theory / 理論整理](docs/THEORY_JA.md) · [Reproduction](docs/REPRODUCIBILITY.md) · [Experiment index](experiments/README.md) · [Research closeout](docs/RESEARCH_CLOSEOUT_JA.md)

## The current method

```text
TRAINING
fixed AR evaluator <--- teacher probabilities on actor-visited states
                         |
actor: frozen AR backbone + continuation LoRA + length-policy LoRA
                         |
recollect current actor states -> update content -> update length policy
                         -> reevaluate the same fixed probe contexts

INFERENCE (no evaluator call)
context -> one backbone pass -> ordinary AR anchor
                            -> continuation distribution + confidence features
                            -> learned length policy -> commit 1, 2, 3, or 4 tokens
```

The current continuation content is trained by unfiltered full-horizon teacher-sample distillation. The length policy is trained by contextual-bandit REINFORCE with an adaptive teacher-risk penalty. Both sets of LoRA weights can change; evaluator, original AR weights, and non-LoRA actor weights stay frozen. Confidence is an observation, not a self-awarded reward. The final continuation uses the same categorical policy for learning, state collection, and expected-risk constraint updates. Argmax deployment is reported separately.

An exact AR anchor means the **same next-token conditional distribution at the same input context**, not that an entire generated sequence remains identical after approximate blocks have been committed. An immediate one-token decision still incurs actor overhead unless the extra branch is explicitly bypassed.

## Final small-model results

Three archived teacher/student seeds; tiny two-layer, width-64 nanoGPT; one corrected Gutenberg-body BPE corpus. These are development results, not an untouched confirmatory test.

| Seed | Generated tokens / backbone call, before → after | Trace-distribution discrepancy, before → after (nat/token) | Final speed / pure AR |
|---|---:|---:|---:|
| 48017 | 1.780 → 1.541 | 0.507 → 0.331 | 0.935 |
| 48018 | 1.527 → 1.499 | 0.282 → 0.261 | 0.935 |
| 48019 | 1.598 → 1.486 | 0.416 → 0.341 | 0.922 |

Quality evaluation: 16 dev prompts, four draws, 48 output tokens. Timing: AMD EPYC 9V74, CPU one thread, FP32, batch 1, no KV cache, unpinned clock; eight prompts × 48 tokens; five repeats, medians, excluded warmup and evaluator replay. Timed trajectories differ from the larger quality sample: do not combine their call counts when inferring overhead.

Final latency is **6.90–8.47% longer** than pure AR in these measurements. The stochastic discrepancy is an augmented token/length-trace reverse-KL estimate: its expectation upper-bounds token-marginal KL; it is not an exact marginal KL or a human quality score. Distribution discrepancy decreased mainly alongside shorter commits. The retained progress/quality development screen passed **0/3** seeds; constraint satisfaction was not statistically established.

Sources: [complete final report](docs/CONSISTENT_DYNAMIC_BOUNDARY_2026_10_07.md), [original numeric summary](docs/evidence/consistent_dynamic_boundary_2026_10_07.json), and [same-timing-trace reanalysis](docs/evidence/final_snapshot_2026_10_07.json).

## Novelty positioning after the 2026 literature audit

A focused primary-source review found substantial overlap with recent work. In particular, **CLP** already combines a backbone-generated first token with a learned span-length predictor and no token-by-token verifier; **AdaMTP** and **EntMTP** adapt prediction/speculation horizon to uncertainty; **K-Forcing** and **PTP** address joint multi-token generation; and **MTP-RL** explicitly adapts MTP competence while an RL policy changes.

Accordingly, this repository does **not** claim novelty for "multi-token prediction", "backbone first token", "adaptive extra-token length", "on-policy distillation", or "MTP during RL" in isolation.

The narrower research object explored here is the combination of **full-horizon continuation distillation on all actor-visited contexts + a separate constrained-RL commit-length LoRA + repeated state recollection as continuation competence changes, with no inference-time verifier**. No identical primary paper was found in the audit, but absence from a search is not proof of priority. The project should therefore be described as an open feasibility study of that combination, not as a world-first decoding paradigm.

See [Related work / novelty audit](docs/RELATED_WORK_2026.md) for the comparison matrix and safe public wording.

## What larger models might change

Larger models can help **if** saved AR work grows faster than the continuation, policy, sampling, and KV-cache maintenance costs, while acceptable block lengths are retained. This is a conditional hypothesis, not a scaling result. More output heads, larger vocabularies, cache catch-up, memory bandwidth, batching, and kernel behavior can erase the benefit. LoRA weight merging does not remove new continuation modules. See the complete cost derivation and assumptions in [Theory](docs/THEORY_JA.md).

## Start here

No training is required to inspect the published claims:

```bash
python scripts/verify_snapshot.py
python -m unittest discover -s tests -v
```

For model tests, create a Python 3.13 environment, install the CPU dependencies and run the seven suites as documented in [Reproduction](docs/REPRODUCIBILITY.md). Training and evaluation commands are separate; do not benchmark while training is running. Archived model inputs are required to reproduce the final continuation.

The existing `experiments/` paths are preserved to avoid breaking imports and reproduction commands. `fixed_k_gate/` is a historical experiment **and a dependency** of later variable-length code; its name does not make the current policy fixed-length. Old EOB, fixed-gate, joint-content-RL and layer-fusion studies are indexed, not silently rewritten as the current design.

## Historical evidence and scope

Early natural-language BPE numbers were affected by Gutenberg-wrapper preprocessing and inadequately matched baselines. They are retained as **historical, not headline evidence**; read [the audit](docs/RESULTS_AUDIT.md). Corrected-data negative results remain published. The all-layer MLP study was distillation-only, not all-layer backbone LoRA or an RL success.

This repository ends this iteration at a documented feasibility/negative-performance snapshot. It is not a production inference engine and does not require a large-model experiment to be a useful, reproducible record. No claim of first-of-its-kind novelty, reward-hacking detection, or guaranteed acceleration is made. Possible future work is separate from the completed scope.

## Prior work and license

Related areas: multi-token/blockwise prediction, on-policy distillation, low-rank adaptation, adaptive computation and speculative decoding. The latter can provide exact distribution-preserving acceleration through verification/correction; this prototype does not implement that guarantee. See [primary references](docs/REFERENCES.md).

MIT License: [LICENSE](LICENSE). The reused nanoGPT model retains its [upstream MIT license](experiments/synthetic/third_party/nanogpt/LICENSE). Project licensing does not relicense third-party source texts. Citation metadata: [CITATION.cff](CITATION.cff).
