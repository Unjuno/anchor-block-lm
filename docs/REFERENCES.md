# Primary references and attribution

These references establish relevant methods, not the success or novelty of this repository. Links and identifiers checked on 2026-10-07. Historical project reports are cited separately in RESULTS.md.

| Reference | Identifier | Relevance and boundary |
|---|---|---|
| Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021; ICLR 2022) | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685), DOI 10.48550/arXiv.2106.09685 | Low-rank weight adaptation and mergeable linear updates; not proof that added block heads are free or LoRA-RL matches full RL |
| Agarwal et al., *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* (2023; ICLR 2024) | [arXiv:2306.13649](https://arxiv.org/abs/2306.13649), DOI 10.48550/arXiv.2306.13649 | GKD and feedback on student-generated states; not validation of this commit-length policy |
| Leviathan, Kalman and Matias, *Fast Inference from Transformers via Speculative Decoding* (2022; ICML 2023) | [arXiv:2211.17192](https://arxiv.org/abs/2211.17192), DOI 10.48550/arXiv.2211.17192 | Distribution-preserving parallel sampling with verification/correction; this project omits that inference mechanism and cannot inherit its guarantee |
| Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023; SOSP 2023) | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180), DOI 10.48550/arXiv.2309.06180 | KV-cache management and serving-system constraints; its benchmark results are not this project's speedup |
| Karpathy, nanoGPT | [upstream repository](https://github.com/karpathy/nanoGPT) | The minimal vendored GPT implementation and preserved MIT notice |

The timing identity, conditional scaling comparison and augmented-trace KL decomposition in THEORY_JA.md are derived explicitly there. They are accounting/probability results, not new acceleration theorems. Broader connections include multi-token prediction, blockwise parallel decoding, adaptive computation and optimal stopping; this list is not an exhaustive priority review.


## 2025–2026 closest work for adaptive multi-token generation

The references below were added after a focused novelty audit on 2026-10-07. They materially narrow the novelty that this repository can claim. See [RELATED_WORK_2026.md](RELATED_WORK_2026.md) for the feature-by-feature comparison.

| Reference | Identifier | Relevance and boundary |
|---|---|---|
| Draxler et al., *Parallel Token Prediction for Language Models* (ICLR 2026) | [arXiv:2512.21323](https://arxiv.org/abs/2512.21323) | Jointly models dependent future tokens by exposing sampling randomness to the model; reported exact-distribution results use error correction/verification. Stronger joint-distribution treatment than this prototype |
| Xie & Zhou, *CLP: Collocation-Length Prediction for Zero-Loss Adaptive Multi-Token Inference* (2026) | [arXiv:2606.10935](https://arxiv.org/abs/2606.10935) | Closest inference architecture: original backbone emits the first token and a lightweight span-length predictor selects how many extra MTP tokens to commit, without token-by-token target verification. Uses supervised correctness labels rather than this repository's RL commit policy |
| Tang et al., *K-Forcing: Joint Next-K-Token Decoding via Push-Forward Language Modeling* (2026) | [arXiv:2606.10820](https://arxiv.org/abs/2606.10820) | Progressive self-forcing distillation of a joint next-k sampler; direct fixed-stride generation and strong batch-serving results. It does not report this repository's learned per-context commit policy |
| Wang et al., *MTP-RL: Acceleration of Reinforcement Learning Rollouts with Policy-Aligned Multi-Token Prediction* (Findings ACL 2026) | [ACL Anthology](https://aclanthology.org/2026.findings-acl.1871/) | Demonstrates that MTP competence/acceptance length must track an RL-updated main policy; remains a speculative draft-and-verify method rather than direct unverified commitment |
| Zhu et al., *Experience-Driven Dynamic Exits for LLMs with Reinforcement Learning (LEDE)* (2026) | [arXiv:2606.03113](https://arxiv.org/abs/2606.03113) | Offline RL dynamically selects both exit layer and speculation length in self-speculative decoding; shows that RL-based variable speculation length is not novel by itself. Still uses target verification, unlike this repository's direct commit policy |
| Chai et al., *Joint Training of Multi-Token Prediction in Reinforcement Learning via Optimal Coefficient Calibration* (2026) | [arXiv:2605.28184](https://arxiv.org/abs/2605.28184) | Studies optimization interference when MTP and RL are trained jointly and adaptively calibrates the MTP coefficient; not a commit-length policy |
| Chen, *EntMTP: Accelerating LLM Inference with Entropy Guided Multi Token Prediction* (2026) | [arXiv:2606.27550](https://arxiv.org/abs/2606.27550) | Training-free per-step speculation-depth/tree scheduling based on entropy; still uses speculative verification |
| Cui et al., *AdaMTP: An Adaptive Training Paradigm for Multi-Token Prediction* (2026) | [arXiv:2608.00434](https://arxiv.org/abs/2608.00434) | Uses entropy-defined semantic boundaries to mask future-token training losses and optionally prune inference candidates; inference remains self-speculative with target verification |
| Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* (ICML 2024) | [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) | Multiple decoding heads / tree-style speculative acceleration; important baseline for MTP-style inference |
| Ankner et al., *Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding* (2024) | [arXiv:2402.05109](https://arxiv.org/abs/2402.05109) | Sequentially dependent MTP draft heads and tree verification; useful adaptive-drafting baseline |

These references mean that **"backbone first token + adaptive extra-token length" is not a novel claim by itself**. Any distinct contribution of this repository must be stated more narrowly: full-horizon continuation distillation on actor-visited contexts plus a separate constrained-RL commit policy, updated together without an inference-time verifier. Even that is a combination-level positioning, not a proven priority claim.
