# Fixed evaluator + trainable continuation LoRA: first experiment

## Outcome

The requested two-model learning arrangement completed on three archived tiny-nanoGPT teacher/student/policy seeds. A fixed evaluator scores the actor while continuation LoRA and the confidence-informed variable-length policy can change. **This configuration did not improve end-to-end distribution fidelity or beat pure AR latency.** This is a completed negative/uncertain result, not a successful acceleration claim.

## Provenance

Training: [Actions run 37333971033](https://github.com/Unjuno/anchor-block-lm/actions/runs/37333971033), commit `a8754cd3ba01f29fc42890a336e0773efae9618e`.
Content-only audit: [Actions run 37334884783](https://github.com/Unjuno/anchor-block-lm/actions/runs/37334884783), commit `67585b461b473a17828749cada52b678fc647537`.
The audit only scores saved actors; it does not retrain or tune them. Initialization is the archived artifact from run 37324518978. Source/body/tokenizer/split and model hashes were verified.

Training artifacts: 11356195527 (48017), 11355602475 (48018), 11356450036 (48019).
Audit artifacts: 11356211497 (48017), 11355522961 (48018), 11356301232 (48019).

## What actually learned

Maximum prediction remains four total tokens, including the original AR-distribution anchor. Actual committed length remains variable, 1/2/3/4, chosen before sampling the continuation. No EOB, fixed-block forcing, beam search, reranking or inference-time teacher verification is introduced.

The evaluator and original AR backbone remain frozen. Joint training updates 7,200 continuation LoRA A/B parameters and 812 policy LoRA A/B parameters. Original continuation matrices, slot/component embeddings and non-LoRA tensors remain fixed. The control updates only the 812 policy parameters. Both start from identical archived weights and receive four rounds of 64 updates. The joint condition additionally spends evaluator computation on content learning, so this is not an equal-total-training-compute comparison. Each arm recollects actor-visited contexts from training data; trajectories and RNG streams can differ after behavior changes.

Content learning combines full-three-slot reverse-KL score-function REINFORCE with teacher-sampled NLL (weight 0.5). Teacher sampling is temperature 1, top-p 1, without length-dependent sample filtering. A leave-one-out baseline excludes the scored sample. All future slots are trained, not only adopted tokens.

The policy receives a reward for progress minus an initial measured action-cost proxy and a fixed teacher-divergence penalty inherited from the prior training-only run. Costs are shared by both arms; no dev/test checkpoint or hyperparameter search occurred. This is on-policy distillation plus contextual-bandit policy RL, not sequence-return PPO or direct online wall-clock optimization.

## Full sampled-sequence evaluation

Each condition used 16 saved held-out prompts x four stochastic draws x 48 tokens: 3,072 tokens per condition per seed. Teacher replay is outside inference/timing. Reverse KL is estimated by complete generated-sequence log Q/P divided by sequence length; it is not a percentage of semantic quality.

| Seed | Gate-only tokens/call | Joint tokens/call | Gate-only reverse KL, nat/token | Joint reverse KL, nat/token | Joint speed / pure AR |
|---|---:|---:|---:|---:|---:|
| 48017 | 1.4855 | 1.5253 | 0.272459 | 0.295138 | 0.8869 |
| 48018 | 1.5634 | 1.5714 | 0.273965 | 0.297349 | 0.9707 |
| 48019 | 1.5126 | 1.5103 | 0.266288 | 0.289240 | 0.9628 |

Pure AR emits one token per call and has zero sequence log-ratio against the evaluator. Joint training increased the reverse-KL point estimates by 0.02268, 0.02338 and 0.02295 nat/token relative to the gate-only control. Respective prompt-cluster 95% intervals: [-0.01553, 0.05924], [-0.00889, 0.05264], [-0.01371, 0.06085]. All include zero: these sequence comparisons do not establish a statistically resolved difference per seed, and provide no evidence of the proposed improvement.

Joint emitted histograms for lengths 1/2/3/4 were 974/1022/18/0, 840/1113/2/0 and 1001/1028/5/0. Long emission was not forced; length four remained available. Similar average length is not exact equal-throughput matching.

## Shared-content audit, independent of length decisions

Already-trained actors were scored against exactly the same teacher-sampled anchors and full three-token continuations, 16 contexts x 32 samples. The gate-only generator is unchanged and its scores equal the initial generator's scores exactly. Forward KL measures teacher-to-actor distribution coverage, in nats per three-token tail.

| Seed | Gate-only forward KL | Joint forward KL | Joint minus control | Prompt-cluster 95% interval |
|---|---:|---:|---:|---|
| 48017 | 4.278873 | 4.541578 | +0.262705 | [+0.155284, +0.378814] |
| 48018 | 3.931729 | 4.019584 | +0.087855 | [+0.021236, +0.161862] |
| 48019 | 3.913365 | 4.089091 | +0.175726 | [+0.070323, +0.300006] |

All three forward-KL differences and their reported intervals are positive. Under this shared-target diagnostic, generator distribution coverage worsened; this is not merely a consequence of committing more tokens. The audit was added after the first sequence results, without changing any model, and is not a new confirmatory holdout.

## Hardware and latency

Seed 48017: Intel Xeon Platinum 8573C. Seeds 48018/48019: AMD EPYC 9V74. All used one CPU thread, FP32, decode batch 1, no KV cache, unpinned clock, CPython 3.13.15, torch 2.10.0+cpu, numpy 2.3.5, tokenizers 0.22.2. The nanoGPT base remains 169,728 parameters, two layers, four heads, width 64, context 64, vocabulary 1024. Data is corrected Gutenberg body-v2; no larger teacher/model was introduced.

Timing: eight prompts x 48 output tokens, five replays with identical within-condition random traces and excluded warmup. LoRA is merged into separate deployment copies. Pure AR runs neither continuation nor policy heads. Speed ratios are ratios of median times on each seed's own host; absolute cross-host times are not pooled. Joint ratios span 0.887–0.971, while gate-only ratios span 0.867–0.952. Neither demonstrates wall-clock speedup over pure AR. The initial microbenchmark is only a frozen training reward proxy.

## Verification and interpretation

Five CI suites passed: 17 two-model, 21 adaptive, 13 fixed-horizon dependency, 37 BPE/data integrity, 14 synthetic: **102 passed, no skips**, plus compilation. Local new tests failed before implementation. Checks include analytical categorical reverse-KL gradients, exact AR scoring, LoRA merge equivalence, zero-budget generation, full-horizon targets, teacher freezing and genuine content updates. Artifact ZIP hashes and raw rollout arithmetic were independently rechecked. Saved non-LoRA tensors match their source weights, and only the joint condition changes generator content.

Observed fact: the two-model update runs, but current end-to-end behavior does not improve; teacher-to-actor distribution coverage worsens. Potential explanations include loss imbalance, insufficient regularization or limited continuation-branch adaptation. They are hypotheses, not established causes: no loss-component ablation was performed. This result neither falsifies all two-model RL approaches nor supports forcing a fixed committed length or scaling up.

Statistical distillation concerns distribution coverage, reinforcement learning concerns action/credit assignment, and inference engineering concerns actual cost. The next bounded diagnostic is to separate content distillation from its RL term while preserving the fixed evaluator and variable-length architecture. No further training is claimed here.
