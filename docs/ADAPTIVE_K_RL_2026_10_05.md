# Adaptive commit-length RL: completed evaluation, 2026-10-05

## Scope and provenance

This is the user-approved **fixed maximum horizon, variable committed length** experiment. Maximum prediction length is four total tokens including the ordinary anchor; the policy may commit 1, 2, 3, or 4. It is neither the previous fixed-four binary gate nor the EOB experiment.

Three teacher/student/policy training seeds (48017, 48018, 48019) were already completed. This follow-up adds full-sequence evaluation of those exact saved weights; no new training, length rule, reward tuning, or generator change was introduced.

- Training and corrected pure-AR timing: [run 37324518978](https://github.com/Unjuno/anchor-block-lm/actions/runs/37324518978), commit `c059008244c10cdcd952c88c6bab097151e42696`.
- Saved-model sequence evaluation: [run 37326561983](https://github.com/Unjuno/anchor-block-lm/actions/runs/37326561983), commit `608251787e08e4e46201a05b59d2a2025d3f349b`.
- Training artifact: `11351169485`, SHA-256 `14f8dcd6ad5574e39d693135c329f66238d6da0153db773fb01edcd7bedd209e`.
- Evaluation artifact: `11352582171`, SHA-256 `d55843269a13477837876e8c0207cebba627bffda4c14e57f64d7fdb85de9c9e`.

The corrected Gutenberg body-v2 source, frozen tokenizer and test-array hashes were checked when reconstructing evaluation inputs. The tokenizer was not retrained. This is one play, train-only byte-level BPE vocabulary 1024, tiny nanoGPT with 169,728 base parameters, 2 layers, 4 heads, width 64, context 64. Training used 600 teacher steps, 1200 content steps and 800 categorical-policy steps per seed.

## What the model learns

The generator learns the full maximum-horizon teacher samples, without choosing training targets by committed length. Teacher sampling is untruncated at temperature 1 (top-p=1). The parallel continuation uses a four-component mixture to represent some dependence among future slots. Its approximation is not assumed exact.

The complete generator is then frozen. Only the categorical policy's LoRA A/B tensors are reward-trained. It sees context, the realized anchor and distribution confidence features before a continuation sample is drawn. Training is contextual-bandit REINFORCE with a reward for extra tokens and a penalty for estimated prefix distribution divergence. Deployment uses the highest-scoring length action. It is not long-horizon sequence RL and the reward does not directly measure wall-clock time.

## Full free-running evaluation

Each model generated 16 saved test prompts x 8 stochastic draws x 48 tokens: 6,144 adaptive tokens and 6,144 ordinary-AR tokens per seed. No teacher is called to accept or reject generated blocks. Teacher replay happens only after generation for measurement.

| Seed | Emit 1 | Emit 2 | Emit 3 | Emit 4 | Tokens per backbone call |
|---|---:|---:|---:|---:|---:|
| 48017 | 2069 | 1931 | 71 | 0 | 1.509211 |
| 48018 | 1687 | 2218 | 3 | 3 | 1.570954 |
| 48019 | 2085 | 2018 | 5 | 2 | 1.494891 |

All four actions are available. The learned policies mostly choose 1 or 2; longer continuations are rare, not forced. Requested lengths are saved separately from emitted lengths because the final output budget can truncate a requested block.

On the original 64-context diagnostic, mean summed marginal entropy of the three future slots was lower among states assigned length 2 than among length-1 states in all three seeds: 13.541 vs 15.561, 14.662 vs 15.396, and 14.253 vs 15.256 nats. This is an association, not a calibrated probability of correctness or evidence of a monotone rule for every state.

## Full-sequence distribution distortion

The independent scorer sums student anchor and mixture-prefix log likelihoods and compares them to teacher autoregressive likelihood along the actual generated history. The deterministic policy chooses the boundary before sampling its tail, so the recorded segmentation determines the sequence likelihood. Averaging generated-sequence log likelihood ratios estimates reverse KL. Negative individual samples are not clipped.

| Seed | Mean reverse KL estimate, nats / 48-token sequence | Nats / token | Prompt-cluster bootstrap 95% interval, nats / token |
|---|---:|---:|---|
| 48017 | 14.059159 | 0.292899 | [0.267183, 0.320854] |
| 48018 | 14.161558 | 0.295032 | [0.272250, 0.315812] |
| 48019 | 14.092328 | 0.293590 | [0.272068, 0.314611] |

The exact AR control's sequence log ratio was zero in every evaluated draw. The adaptive distribution is measurably different from the teacher; these values are not percentages of text quality lost or preserved. There was no human/task-quality evaluation and no prospectively chosen acceptable distortion threshold.

Intervals resample 16 prompt clusters after averaging the 8 draws within each prompt. They do not include uncertainty across independently trained models, datasets or protocols. Saved test material was reused for this post-hoc diagnostic: this is not a fresh confirmatory holdout.

## Real CPU timing, separate from sequence scoring

Use the corrected **pure AR** implementation, which runs neither the continuation head nor policy. Earlier run 37323486087 put continuation work in the nominal AR baseline; its apparent 1.30–1.37x speed ratios are superseded.

The corrected timing run used AMD EPYC 7763 CPU, one thread, FP32, batch 1, no KV cache, unpinned clock, 8 prompts x 48 output tokens, three timing trials with different sampled continuations. Values below are medians; trials are stochastic workloads, not three identical token traces. CPython 3.13.15, torch 2.10.0+cpu, numpy 2.3.5. These timings are separate from the larger free-running scoring sample above.

| Seed | Pure AR median seconds | Adaptive median seconds | AR/adaptive time ratio |
|---|---:|---:|---:|
| 48017 | 0.359343 | 0.398177 | 0.902472 |
| 48018 | 0.353367 | 0.381177 | 0.927040 |
| 48019 | 0.354445 | 0.394880 | 0.897601 |

Despite fewer backbone calls, this implementation is slower than pure AR. The added head, confidence and policy work must be counted. No GPU or production latency gain is established.

## Verification

Evaluation workflow tests: adaptive 21, fixed-horizon dependency 13, BPE/data integrity 37, synthetic 14 — **85 passed, no skips**. The new scorer had nine observed missing-implementation failures locally before its nine tests passed. Python compilation passed. Its published Git blob matches the locally tested source (`afbdb01d3660e69877016eb86c679bb5f6f9bd8e`).

All three generator hashes matched their training records; generator and policy hashes were unchanged by evaluation. Exact-AR control scoring, prefix-mixture normalization, segmentation validation and the known analytical log-ratio tests passed. CI success is implementation evidence, not a successful speed/quality claim.

## Conclusion

**Dynamic length learning is demonstrated at nanoGPT scale. Low-distortion wall-clock acceleration is not.** Keep the agreed distinction: all-horizon self-distillation and a separate confidence-informed adaptive commit policy. Neither forcing the longest action nor reverting to a fixed block length is justified by this result. The immediate limitations are continuation distribution fidelity, reward calibration and inference overhead. Related aspects are statistical distillation, reinforcement-learning decision costs and inference-system implementation; none can be inferred from backbone-call reduction alone.
