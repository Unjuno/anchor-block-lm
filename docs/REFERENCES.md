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
