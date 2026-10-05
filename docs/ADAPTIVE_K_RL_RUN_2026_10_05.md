# Adaptive-k RL run — 2026-10-05

## Question

Keep one self-distilled maximum horizon fixed at H=4, but let a reward-trained
policy choose the committed prefix length k in {1,2,3,4} from confidence-related
features. The distillation target distribution itself is not selected or filtered
by k.

## Protocol

- Corrected `gutenberg-body-v2` corpus and train-only byte-level BPE.
- Tiny nanoGPT teacher: 2 layers, 4 heads, width 64, context 64.
- Full teacher categorical sampling at temperature 1 / top-p=1 for content
  distillation; every sampled context contributes the complete H=4 horizon.
- The distilled generator is frozen and hashed before RL.
- Only four LoRA tensors in a categorical commit-length policy are trained.
- Policy actions: k=1,2,3,4.
- k=1 is the exact frozen-AR anchor path.
- Reward: `(k - 1) - lambda * sampled_prefix_log(q/p)`.
- RL is contextual-bandit REINFORCE, not long-horizon sequence RL.
- No EOB and no inference-time teacher verification.

Corrected execution: GitHub Actions run
[37324518978](https://github.com/Unjuno/anchor-block-lm/actions/runs/37324518978).
The first timing run used a non-pure AR timing path and is not used for latency
claims here. A regression test now fails if the AR timing path calls the tail
predictor.

## Held-out common-state result

Reverse KL is Monte Carlo estimated for the prefix selected by the policy.
Lower is better. It is a distribution metric, not an error percentage.

| Seed | k histogram on 64 states | Mean k | Selected prefix reverse KL |
|---|---|---:|---:|
| 48017 | 1:34, 2:29, 3:1, 4:0 | 1.484 | 0.334 nats |
| 48018 | 1:34, 2:30, 3:0, 4:0 | 1.469 | 0.408 nats |
| 48019 | 1:31, 2:33, 3:0, 4:0 | 1.516 | 0.445 nats |

Descriptive mean ± sample standard deviation:

- mean selected k: **1.490 ± 0.024**
- selected prefix reverse KL: **0.396 ± 0.056 nats**

The generator hash was unchanged before/after RL in all three seeds.

## Confidence relation

The policy receives hidden state, realized anchor embedding, marginal maximum
probabilities, marginal entropies, and mixture entropy. The simple sum of the
three marginal entropies was lower on states assigned k=2 than on states
assigned k=1 in all three seeds:

| Seed | Mean uncertainty, k=1 | Mean uncertainty, k=2 |
|---|---:|---:|
| 48017 | 15.561 | 13.541 |
| 48018 | 15.396 | 14.662 |
| 48019 | 15.256 | 14.253 |

This is consistent with the intended behavior: more confident states receive a
longer commit. It does not establish a globally monotone k-vs-confidence law.
There was only one held-out k=3 state in seed 48017 and no held-out k=4 states.

## Unforced generation

Eight prompts, 48 output tokens each.

| Seed | Tokens/backbone call | Call reduction vs AR | Requested k histogram |
|---|---:|---:|---|
| 48017 | 1.542 | 35.16% | 1:114, 2:132, 3:3, 4:0 |
| 48018 | 1.620 | 38.28% | 1:85, 2:152, 3:0, 4:0 |
| 48019 | 1.483 | 32.55% | 1:130, 2:129, 3:0, 4:0 |

Descriptive mean tokens/backbone-call: **1.548 ± 0.069**.

The corrected vanilla-AR timing comparison is negative on this CPU
implementation:

| Seed | AR median s | Adaptive median s | AR/adaptive speed ratio |
|---|---:|---:|---:|
| 48017 | 0.3593 | 0.3982 | 0.902 |
| 48018 | 0.3534 | 0.3812 | 0.927 |
| 48019 | 0.3544 | 0.3949 | 0.898 |

Thus the adaptive path is about 7.9–11.4% slower in wall time despite reducing
backbone calls. This is CPU / FP32 / batch 1 / no KV cache and is not a
production GPU benchmark.

## Why k=3/4 are rare

Full-H=4 teacher-to-student forward KL remains large:

- seed 48017: 4.366 nats per 3-token tail
- seed 48018: 4.156
- seed 48019: 4.256

A post-hoc diagnostic on the preserved held-out risk bank computed the
reward-maximizing action if the Monte Carlo counterfactual risk were known
exactly. Even that oracle mostly prefers k=2:

| Seed | Oracle k histogram | Learned-policy agreement with oracle |
|---|---|---:|
| 48017 | 1:17, 2:42, 3:5, 4:0 | 64.1% |
| 48018 | 1:11, 2:47, 3:5, 4:1 | 51.6% |
| 48019 | 1:10, 2:45, 3:8, 4:1 | 51.6% |

This diagnostic was computed after the primary run and is explanatory, not a
prespecified success metric. It indicates two bottlenecks:

1. the H=4 distilled generator is not accurate enough for k=3/4 to be broadly
   reward-optimal;
2. the learned gate also does not recover all rare longer-prefix opportunities.

## Conclusion

The intended mechanism is now implemented directly:

- H is fixed during self-distillation;
- actual commit length k is dynamic;
- self-distillation is not filtered by the chosen k;
- the generator is frozen during RL;
- confidence-related state features drive k;
- inference does not verify candidates with the teacher.

On the current tiny natural-language model, RL learns a useful **1-vs-2 token
adaptive policy** and reduces backbone calls by roughly one third, but it rarely
uses k=3 and never k=4. The dominant next bottleneck is the fidelity of the
longer fixed-horizon block distribution, followed by the gate's ability to
identify the rare longer safe prefixes. Wall-clock acceleration is not yet
demonstrated.
