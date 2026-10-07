# Fixed-K binary LoRA gate: completed run, 2026-10-05

## Evidence and scope

- Tested implementation: `0b65b3282eac2136cc6b8989e5d76a0684b76b30`.
- Completed execution: [GitHub Actions run 37312360779](https://github.com/Unjuno/anchor-block-lm/actions/runs/37312360779), job `111770623826`.
- [Full evidence artifact](https://github.com/Unjuno/anchor-block-lm/actions/runs/37312360779/artifacts/11345923919): 36 files, including three sets of checkpoints, JSON results, counterfactual arrays, context offsets, and the shared tokenizer/source manifests. Artifact SHA-256: `6318fc877d6d3c8dd96b1b09d916ac1ed86c6687b2df29680881e510829764ac`.
- Numerical values below were read from the completed job log, not from historical EOB results. The local artificial-grammar smoke experiment is separate and is not included in this table.

## What actually trained

Block size remained **4 total tokens**, including the ordinary AR-distribution anchor and three parallel continuation slots. There was no learned length, EOB, candidate reranking, or teacher verification during generation.

The content target was the full frozen teacher distribution at temperature 1 (top-p=1), without discarding difficult contexts or sampled trajectories. The student tail was a normalized four-component mixture, with a shared sampled component across all three future slots. This gives a joint distribution rather than merely multiplying independent marginal heads, but remains a limited approximation.

After content distillation, the entire student was frozen. Only the binary gate's four LoRA tensors (`net.0.A`, `net.0.B`, `net.2.A`, `net.2.B`) were reward-trained. The gate saw context, the already-realized anchor, and distribution summaries, never a sampled candidate tail. Its training was contextual-bandit REINFORCE, not long-horizon RL. Online evaluation used a dev-calibrated threshold on learned logits.

## Data and implementation conditions

Fresh corrected `gutenberg-body-v2`, Project Gutenberg 1513: 142,474 body characters; one train-only BPE tokenizer, vocabulary 1024, reused for all seeds. Token counts: train 44,868 / dev 6,078 / test 5,688. Old checkpoints and banks were not loaded.

- Raw SHA-256: `5a2037a19e60cccb67b3f5fc93a12cb65c58694ce6a1dc61d0d67d0d2502b3b4`
- Body SHA-256: `d4d21b10540438b66a5210034e8e6b058b3d158c0eaae839cc38c2eecae1bc00`
- Tokenizer SHA-256: `d7b17f58b6c5ef29aa10e3a0c7a5fe649351f11a67e7bb93a1253f4e2633b4da`

nanoGPT: 169,728 base parameters, 2 layers, 4 attention heads, width 64, context 64. Each seed used 600 teacher steps, 1200 content steps, 600 gate steps. CPU: AMD EPYC 7763, one thread, FP32, decode batch 1, teacher-training batch 32; clock not pinned, no KV cache. CPython 3.13.15, torch 2.10.0+cpu, numpy 2.3.5, tokenizers 0.22.2.

## Equal-budget common-state diagnostic

Each method selected exactly **8 full blocks among 64 non-overlapping test contexts**. Thus all methods had 64 backbone calls and 88 emitted tokens, or 1.375 tokens/call. The uniform-control entry is the expectation over uniformly chosen exact-eight-block subsets. Each conditional tail KL used 64 Monte Carlo samples. This is an **offline rank-allocation diagnostic**, not an online quota or latency result.

The metric is conditional reverse KL, in nats per three-token continuation; lower is better. It is not an error percentage or a semantic-quality measure.

| Seed | Uniform expected | Entropy-confidence rule | LoRA reward gate | Learned minus uniform, 95% interval |
|---|---:|---:|---:|---|
| 47017 | 5.119112 | 5.107043 | 4.581294 | -0.537818; [-1.166931, +0.113547] |
| 47018 | 5.401567 | 5.312487 | 4.346958 | -1.054608; [-2.121899, -0.095472] |
| 47019 | 5.112958 | 3.644655 | 3.917916 | -1.195042; [-2.533436, -0.043336] |

Across these three seeds, learned selected KL ranged 3.917916–4.581294; uniform expected KL ranged 5.112958–5.401567. Descriptive means were 4.282056 versus 5.211212 (17.83% lower by ratio of means). That percentage is a reduction in the KL metric, not a percentage of output quality retained.

The prespecified per-seed upper-confidence-bound criterion passed against uniform allocation in **2/3 seeds**, not 3/3. Against the entropy-confidence rule it passed in **0/3 seeds**. Learned-minus-confidence intervals were respectively [-1.782500, +0.879043], [-2.306387, +0.658940], and [-0.506621, +1.030687]. The confidence rule had the lower point estimate in seed 47019. Intervals used 1000 nested context/sample bootstrap draws; they do not certify a population effect across training seeds or datasets.

Absolute distribution fidelity is not solved. Teacher-to-student forward KL on independently teacher-sampled test tails was 4.255650, 4.287207, and 3.820699 nats per three-token continuation. Neither these values nor the selected reverse KL establish near-exact teacher preservation.

## Unforced online diagnostic

Eight prompts, 48 output tokens per prompt, three timing repeats per policy. Timing excludes the offline teacher-scoring procedure. Each result uses its own generated history; do not call this an equal-work quality comparison.

| Seed | AR calls | Learned-gate calls | Full four-token calls | Tokens/call | AR median seconds | Learned median seconds | Speed ratio vs AR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 47017 | 384 | 282 | 34 | 1.361702 | 0.359433 | 0.413671 | 0.868888 |
| 47018 | 384 | 294 | 30 | 1.306122 | 0.358173 | 0.428884 | 0.835126 |
| 47019 | 384 | 297 | 29 | 1.292929 | 0.355156 | 0.427456 | 0.830860 |

Every learned-gate emission was either 1 or 4 tokens; no partial blocks or zero-token steps were observed. Calls fell by 22.66–26.56%, but actual CPU speed was only 0.831–0.869 times ordinary AR. Additional-head and gate work outweighed the saved calls in this implementation. Online semantic quality and trajectory-level distribution drift were not evaluated here.

## Verification and conclusion

The canonical repository implementation passed **13/13 tests**, with no skips, and Python compilation. In all three seeds, the original AR backbone equaled the teacher and the complete distilled student's hash was unchanged before/after gate RL. The three generator hashes were `fa9a6ca3b3e5c7b67046a0be659dc9e0517dee5d1709529b1e9488a10e64cba2`, `009a8247c404c8771232eb183ebe7629d066522d50429dbfc1af403e1461d577`, and `34c461c71e0e17fcefa9b668cb1ca58108f4acb44827ddc22807863fa1bc3086`.

**Result:** fixed-K generation plus selective binary LoRA gating is implemented and trainable. This run provides a limited routing signal against uniform allocation, not a robust advantage over the simple confidence baseline, not small absolute distribution distortion, and not a wall-clock speedup. K was not changed in response to results. The immediate research bottleneck is fixed-block joint-distribution fidelity; retaining K and improving the distillation should precede claims of low-loss acceleration.
